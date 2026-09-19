"""批量预测与结果组装（对应原实现的 predict 部分）。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import load_model
from .text import prepare_examples, read_json, validate_request

SCHEMA_VERSION = "nanojev-mlx-inference-v1"


def answer_from_probabilities(example, probabilities):
    ids = example["candidate_ids"]
    if len(probabilities) != len(ids) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("模型产生了无效概率")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("模型概率总和不为1")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result = {"type": example["type"], "probabilities": dict(zip(ids, probabilities))}
    if example["type"] == "boolean":
        result.update(p_true=probabilities[1], value=bool(best))
    elif example["type"] == "choice":
        result.update(choice=ids[best], value=ids[best])
    else:
        score = math.fsum(i * p for i, p in enumerate(probabilities))
        result.update(score=score, level=best, value=score)
    return result


def run_prediction(model, payload, temperature: float = 1.0, batch_questions: int = 0):
    states = validate_request(payload)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature 必须为有限正数")
    if type(batch_questions) is not int or batch_questions < 0:
        raise ValueError("batch_questions 必须为非负整数；0 表示全部问题一次前向")

    examples = prepare_examples(payload, model.tokenizer, model.max_length)
    size = batch_questions or len(examples)
    batches = [examples[i : i + size] for i in range(0, len(examples), size)]

    started = time.perf_counter()
    outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
    forward_passes = 0
    for batch in batches:
        logits, _ = model.logits_for(batch)
        forward_passes += 1
        for i, example in enumerate(batch):
            k = len(example["candidate_ids"])
            scores = logits[i][:k].astype(mx.float32) / temperature
            if not bool(mx.all(mx.isfinite(scores))):
                raise ValueError("模型产生非有限logits，未返回部分预测")
            probabilities = np.array(mx.softmax(scores, axis=-1)).astype(np.float64).tolist()
            outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(
                example, probabilities
            )
    elapsed = time.perf_counter() - started

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "directory": str(getattr(model, "root", "")),
            "base_model": model.run_config.get("model"),
            "base_revision": model.run_config.get("resolved_model_revision"),
            "set_head": model.run_config["set_head"],
        },
        "temperature": {
            "value": float(temperature),
            "fitted_by_this_command": False,
            "note": "显式应用给定标量；默认1不表示模型已校准。",
        },
        "execution": {
            "runtime": "mlx",
            "device": str(mx.default_device()),
            "parameter_storage": "float32",
            "precision": "fp32",
            "forward_autocast": "disabled",
            "states": len(states),
            "questions": len(examples),
            "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
            "forward_passes": forward_passes,
            "batch_questions_limit": batch_questions or "all",
            "autoregressive_decode_steps": 0,
            "prefix_sharing": False,
            "max_length": model.max_length,
            "network_model_calls": 0,
            "persistent_model_load_count": 1,
            "server_evaluation_seconds": elapsed,
        },
        "states": list(outputs.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="NanoJev MLX 批量推理（本地 checkpoint，无网络调用）")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="含 states 数组的 JSON 文件")
    parser.add_argument("--output", help="不设置时输出到 stdout")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0, help="0=全部问题一次前向")
    args = parser.parse_args()

    try:
        model = load_model(args.checkpoint_dir)
        result = run_prediction(
            model, read_json(args.input), temperature=args.temperature, batch_questions=args.batch_questions
        )
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(destination), "execution": result["execution"]}, ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
