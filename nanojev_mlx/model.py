"""NanoJev 的 MLX 实现：Qwen3-0.6B backbone + 决策头。

权重直接从原 checkpoint 的 best.safetensors 读取，不需要任何格式预转换：
  backbone.*                      -> mlx_lm Qwen3Model（去掉前缀）
  norm / scalar / set_*           -> DecisionHead
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models import qwen3

from .head import DecisionHead, HeadConfig
from .text import group_paths, prepare_examples, validate_request

NEG_INF_LOGIT = -1e9


def build_backbone_args(cfg: dict) -> qwen3.ModelArgs:
    return qwen3.ModelArgs(
        model_type=cfg["model_type"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_theta=cfg["rope_parameters"]["rope_theta"],
        head_dim=cfg["head_dim"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
    )


def map_weights(raw: dict) -> tuple[list, list]:
    """把原 checkpoint 的权重拆成 (backbone 权重, 头部权重) 两组 (name, array)。"""
    backbone, head = [], []
    in_proj_w = in_proj_b = None

    for key, value in raw.items():
        if key.startswith("backbone."):
            backbone.append((key[len("backbone.") :], value))
            continue
        if key == "set_attention.in_proj_weight":
            in_proj_w = np.array(value)
            continue
        if key == "set_attention.in_proj_bias":
            in_proj_b = np.array(value)
            continue
        if key.startswith("set_attention.out_proj."):
            head.append(("out_proj." + key[len("set_attention.out_proj.") :], value))
            continue
        head.append((key, value))

    # PyTorch 融合投影 [3*inner, inner] 按 q/k/v 顺序切开
    if in_proj_w is not None:
        inner = in_proj_w.shape[0] // 3
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            head.append((f"{name}.weight", mx.array(in_proj_w[i * inner : (i + 1) * inner])))
            if in_proj_b is not None:
                head.append((f"{name}.bias", mx.array(in_proj_b[i * inner : (i + 1) * inner])))
    return backbone, head


class DecisionModel(nn.Module):
    def __init__(self, run_config: dict, backbone_config: dict, tokenizer):
        super().__init__()
        self.run_config = run_config
        self.backbone = qwen3.Qwen3Model(build_backbone_args(backbone_config))
        self.head = DecisionHead(
            HeadConfig(hidden_size=backbone_config["hidden_size"], set_head=run_config["set_head"])
        )
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = int(run_config.get("max_length", 512))

    # -- 权重 ---------------------------------------------------------------
    def load_checkpoint(self, weights_path: str | Path):
        backbone, head = map_weights(mx.load(str(weights_path)))
        self.backbone.load_weights(backbone)
        if head:
            self.head.load_weights(head)
        mx.eval(self.parameters())

    # -- 前向 ---------------------------------------------------------------
    def build_batch(self, examples):
        """候选路径右填充成一个矩阵；返回 (tokens, lengths, groupings)。"""
        paths, lengths, groupings = group_paths(examples)
        width = max(lengths)
        tokens = np.full((len(paths), width), self.pad_token_id, dtype=np.int32)
        for i, ids in enumerate(paths):
            tokens[i, : len(ids)] = ids
        return mx.array(tokens), np.array(lengths), groupings

    def logits_for(self, examples) -> tuple[mx.array, mx.array]:
        tokens, lengths, groupings = self.build_batch(examples)
        # 右填充 + 因果掩码：真实位置的注意力不会触及 padding，故与 HF 的
        # causal + padding 双重掩码在数值上等价。
        hidden = self.backbone(tokens)
        path_index = mx.arange(len(lengths))
        leaves = hidden[path_index, mx.array(lengths) - 1]

        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        hidden_size = leaves.shape[-1]
        grouped = mx.zeros((len(examples), kmax, hidden_size))
        valid = np.zeros((len(examples), kmax), dtype=bool)
        for i, ex in enumerate(examples):
            start, n = groupings[i]
            grouped[i, :n] = leaves[start : start + n]
            valid[i, : len(ex["candidate_ids"])] = True

        valid_mx = mx.array(valid)
        choice_rows = mx.array([i for i, ex in enumerate(examples) if ex["type"] == "choice"], dtype=mx.int32)
        z = self.head(grouped, valid_mx, choice_rows)

        rows = []
        for i, ex in enumerate(examples):
            if ex["type"] == "boolean":
                # logits [0, z]：p_true = sigmoid(z)
                head_part = mx.stack([z[i, 0] * 0, z[i, 0]])
                if kmax > 2:
                    head_part = mx.concatenate([head_part, mx.zeros((kmax - 2,))])
                rows.append(head_part)
            else:
                rows.append(z[i])
        logits = mx.stack(rows)
        logits = mx.where(valid_mx, logits, mx.array(NEG_INF_LOGIT, logits.dtype))
        return logits, valid_mx

    def predict(self, payload, temperature: float = 1.0):
        from .predict import run_prediction

        return run_prediction(self, payload, temperature=temperature)


def load_tokenizer(checkpoint_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(checkpoint_dir) / "tokenizer"), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint_dir: str | Path) -> DecisionModel:
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    for name in ("config.json", "best.safetensors"):
        if not (root / name).is_file():
            raise ValueError(f"checkpoint 缺少 {name}")
    for name in ("backbone_config", "tokenizer"):
        if not (root / name).is_dir():
            raise ValueError(f"checkpoint 缺少 {name} 目录")
    run_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    backbone_config = json.loads((root / "backbone_config" / "config.json").read_text(encoding="utf-8"))
    model = DecisionModel(run_config, backbone_config, load_tokenizer(root))
    model.load_checkpoint(root / "best.safetensors")
    model.root = root
    return model


__all__ = [
    "DecisionModel",
    "load_model",
    "load_tokenizer",
    "map_weights",
    "build_backbone_args",
    "prepare_examples",
    "validate_request",
]
