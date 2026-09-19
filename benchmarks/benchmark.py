#!/usr/bin/env python3
"""基准测试：复现 README 里的性能数字。

    python benchmarks/benchmark.py --checkpoint-dir checkpoints/NanoJev

默认只测 MLX。要复现 README 中与 PyTorch + MPS 的对比，指向原仓库的克隆：

    python benchmarks/benchmark.py --checkpoint-dir checkpoints/NanoJev \\
        --backend both --nanojev-repo ../NanoJev

每个后端在**独立子进程**中测量，避免两者共存污染内存读数。延迟报告预热后的稳态值，
内存报告峰值 RSS（resident set size），冷启动报告「加载模型 + 首次出结果」的总时长。
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

# 以 `python benchmarks/benchmark.py` 直接运行时，sys.path[0] 是 benchmarks/ 而不是
# 仓库根目录，这里显式补上，子进程同样适用。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 默认使用 README 基准表所用的那个 fixture，这样默认输出可直接与 README 对照。
# 想看更贴近实际使用的多题型例子，用 --request examples/request.json。
DEFAULT_REQUEST = Path(__file__).resolve().parent.parent / "tests" / "reference" / "toy_request.json"


def peak_rss_bytes() -> int:
    """当前进程的峰值常驻内存。macOS 以字节计，Linux 以 KB 计。"""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def bench_mlx(checkpoint: Path, request: Path, rounds: int) -> dict:
    # 冷启动从「导入运行时」开始计，因为用户跑命令行时确实要等这一步。
    started = time.perf_counter()
    import mlx.core as mx

    from nanojev_mlx import load_model, run_prediction
    from nanojev_mlx.text import read_json

    payload = read_json(request)

    mx.reset_peak_memory()
    model = load_model(checkpoint)
    first = run_prediction(model, payload)
    cold_start = time.perf_counter() - started

    latencies = []
    for _ in range(rounds):
        begin = time.perf_counter()
        run_prediction(model, payload)
        latencies.append(time.perf_counter() - begin)

    return {
        "backend": "mlx",
        "cold_start_s": cold_start,
        "latency_ms": [x * 1000 for x in latencies],
        "peak_rss_bytes": peak_rss_bytes(),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "device": str(mx.default_device()),
        "counts": {
            "states": first["execution"]["states"],
            "questions": first["execution"]["questions"],
            "candidate_paths": first["execution"]["candidate_paths"],
        },
    }


def bench_pytorch(checkpoint: Path, request: Path, rounds: int, repo: Path) -> dict:
    started = time.perf_counter()
    scripts = repo / "scripts"
    if not (scripts / "predict_toy_decisions.py").is_file():
        raise SystemExit(f"在 {repo} 下找不到 scripts/predict_toy_decisions.py（用 --nanojev-repo 指定）")
    sys.path.insert(0, str(scripts))

    from predict_toy_decisions import DecisionPredictor, read_json  # type: ignore

    payload = read_json(request)

    engine = DecisionPredictor(str(checkpoint), device_name="mps")
    first = engine.predict(payload)
    cold_start = time.perf_counter() - started

    latencies = []
    for _ in range(rounds):
        begin = time.perf_counter()
        engine.predict(payload)
        latencies.append(time.perf_counter() - begin)

    return {
        "backend": "pytorch+mps",
        "cold_start_s": cold_start,
        "latency_ms": [x * 1000 for x in latencies],
        "peak_rss_bytes": peak_rss_bytes(),
        "device": first["execution"]["device"],
        "counts": {
            "states": first["execution"]["states"],
            "questions": first["execution"]["questions"],
            "candidate_paths": first["execution"]["candidate_paths"],
        },
    }


def summarize(result: dict) -> dict:
    values = result["latency_ms"]
    return {
        "backend": result["backend"],
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "best_ms": min(values),
        "cold_start_s": result["cold_start_s"],
        "peak_rss_gb": result["peak_rss_bytes"] / 1024**3,
        "device": result["device"],
        "counts": result["counts"],
    }


def run_one(args) -> dict:
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    request = Path(args.request).expanduser().resolve()
    if args.backend == "mlx":
        result = bench_mlx(checkpoint, request, args.rounds)
    else:
        result = bench_pytorch(checkpoint, request, args.rounds, Path(args.nanojev_repo).expanduser().resolve())
    summary = summarize(result)
    if args.json:
        print(json.dumps({"summary": summary, "latencies_ms": result["latency_ms"]}, ensure_ascii=False, indent=2))
    else:
        print(f"后端            : {summary['backend']}  ({summary['device']})")
        print(f"负载            : {summary['counts']['states']} states / "
              f"{summary['counts']['questions']} questions / "
              f"{summary['counts']['candidate_paths']} candidate paths")
        print(f"冷启动          : {summary['cold_start_s']:.2f} s")
        print(f"稳态延迟 中位   : {summary['median_ms']:.1f} ms")
        print(f"稳态延迟 平均   : {summary['mean_ms']:.1f} ms")
        print(f"稳态延迟 最快   : {summary['best_ms']:.1f} ms  ({len(result['latency_ms'])} 轮)")
        print(f"峰值常驻内存    : {summary['peak_rss_gb']:.2f} GB")
    return summary


def run_both(args) -> None:
    """在独立子进程中分别测量，避免内存读数互相污染。"""
    summaries = []
    # 先原版后移植版，保证 baseline/ported 的顺序与 README 表格一致。
    for backend in ("pytorch", "mlx"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--backend", backend,
            "--checkpoint-dir", args.checkpoint_dir,
            "--request", args.request,
            "--rounds", str(args.rounds),
            "--nanojev-repo", args.nanojev_repo,
            "--json",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(f"[{backend}] 失败：\n{completed.stderr.strip()}", file=sys.stderr)
            continue
        summaries.append(json.loads(completed.stdout)["summary"])

    if len(summaries) < 2:
        raise SystemExit("需要两个后端都成功才能对比")

    baseline, ported = summaries[0], summaries[1]
    print(f"{'指标':<22}{baseline['backend']:>16}{ported['backend']:>16}{'变化':>14}")
    print("-" * 68)

    def row(label, key, unit, lower_is_better=True):
        left, right = baseline[key], ported[key]
        ratio = left / right if lower_is_better else right / left
        trend = f"{ratio:.2f}x"
        print(f"{label:<22}{left:>13.3f}{unit}{right:>13.3f}{unit}{trend:>14}")

    row("稳态延迟（中位）", "median_ms", " ms")
    row("峰值常驻内存", "peak_rss_gb", " GB")
    row("冷启动", "cold_start_s", " s")
    print()
    print("延迟越低越好；变化列 > 1 表示 nanojev-mlx 更优。")
    print("原始逐轮数据可用 --backend <name> --json 获取。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="含 best.safetensors 的 checkpoint 目录")
    parser.add_argument("--request", default=str(DEFAULT_REQUEST), help="请求 JSON（默认用 examples/request.json）")
    parser.add_argument("--backend", choices=["mlx", "pytorch", "both"], default="mlx")
    parser.add_argument("--rounds", type=int, default=6, help="预热后的计时轮数")
    parser.add_argument("--nanojev-repo", default=".", help="--backend pytorch/both 时，原 NanoJev 仓库路径")
    parser.add_argument("--json", action="store_true", help="输出机器可读的 JSON")
    args = parser.parse_args()

    if args.backend == "both":
        run_both(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
