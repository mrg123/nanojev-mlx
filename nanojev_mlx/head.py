"""MLX 版 NanoJev 决策头。

对应原实现 scripts/train_toy_decisions.py 的 DecisionModel：

    norm            LayerNorm(hidden)
    scalar          Linear(hidden, 1)
    set_project     Linear(hidden + 1, inner)
    set_attention   MultiheadAttention(inner, num_heads)
    set_output      Linear(inner, 1)

三种题型：
  boolean  单条语义路径，logits 取 [0, z]，p_true = sigmoid(z)
  choice   候选集合上做 set attention 相对修正后 softmax
  score    有序等级上 softmax，期望值即分数
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class HeadConfig:
    hidden_size: int = 1024
    set_head: str = "attention"
    inner: int = 128
    num_heads: int = 4

    @property
    def head_dim(self) -> int:
        return self.inner // self.num_heads


class DecisionHead(nn.Module):
    """在已分组好的候选表示上产生每条候选的标量 logit。"""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        if cfg.set_head not in {"none", "attention"}:
            raise ValueError(f"不支持的 set_head：{cfg.set_head}")
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.scalar = nn.Linear(cfg.hidden_size, 1)
        if cfg.set_head == "attention":
            self.set_project = nn.Linear(cfg.hidden_size + 1, cfg.inner)
            # PyTorch 用融合的 in_proj_weight[3*inner, inner]；这里拆成 q/k/v 三个投影。
            self.q_proj = nn.Linear(cfg.inner, cfg.inner)
            self.k_proj = nn.Linear(cfg.inner, cfg.inner)
            self.v_proj = nn.Linear(cfg.inner, cfg.inner)
            self.out_proj = nn.Linear(cfg.inner, cfg.inner)
            self.set_output = nn.Linear(cfg.inner, 1)

    def _set_attention(self, u: mx.array, valid: mx.array) -> mx.array:
        """候选集合内做一次自注意力；valid 为 True 的位置参与，False 的位置被屏蔽。

        采用加性掩码（被屏蔽处加 -inf），与 mlx_lm 处理 "causal" 掩码的约定一致。
        """
        batch, length, dims = u.shape
        heads, head_dim = self.cfg.num_heads, self.cfg.head_dim
        reshape = lambda a: a.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        queries = reshape(self.q_proj(u))
        keys = reshape(self.k_proj(u))
        values = reshape(self.v_proj(u))

        # [batch, 1, 1, length]：True 保留、False 置 -inf
        mask = mx.where(valid[:, None, None, :], mx.array(0.0, u.dtype), mx.array(-math.inf, u.dtype))
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0 / math.sqrt(head_dim), mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, dims)
        return self.out_proj(attended)

    def __call__(self, grouped: mx.array, valid: mx.array, choice_rows: mx.array) -> mx.array:
        """grouped: [examples, kmax, hidden]；valid: [examples, kmax]；choice_rows: choice 题的行号。"""
        hidden = self.norm(grouped)
        logits = self.scalar(hidden).squeeze(-1)

        if self.cfg.set_head == "attention" and choice_rows.size > 0:
            rows = hidden[choice_rows]
            rows_valid = valid[choice_rows]
            # log(候选数) 作为集合规模特征，与原实现一致
            counts = rows_valid.sum(-1).astype(mx.float32)
            log_k = mx.log(counts)[:, None, None]
            log_k = mx.broadcast_to(log_k, (rows.shape[0], rows.shape[1], 1))
            projected = self.set_project(mx.concatenate([rows, log_k], axis=-1))
            mixed = self._set_attention(projected, rows_valid)
            delta = self.set_output(mx.tanh(projected + mixed)).squeeze(-1)
            logits = logits.at[choice_rows].add(delta)

        return logits
