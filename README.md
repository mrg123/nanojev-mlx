# nanojev-mlx

**English** | [简体中文](README.zh-CN.md)

[![tests](https://github.com/mrg123/nanojev-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/mrg123/nanojev-mlx/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Apple Silicon native port of [NanoJev](https://github.com/TianyuCodings/NanoJev) — a 0.6B parallel decision model running on MLX.**
States and questions in, complete probability distributions out, zero output-token decoding.

NanoJev is CUDA-first: its inference script refuses to run without CUDA, and the recorded
training environment targets an A100 in bf16. To get a like-for-like baseline, the benchmark
below removes **only** that device gate — one change, nothing else in the original touched —
which lets the very same script run on PyTorch MPS. That path still forces PyTorch on you and
leaves memory and startup time on the table. This port keeps the model exactly as trained and
reimplements the decision head in MLX, so the same weights run natively on the Metal GPU with
no PyTorch at all.

> Unofficial community port. Not affiliated with the NanoJev or TypeSafe authors.

---

## Verified, not asserted

The port is checked against the original PyTorch implementation, bit-for-bit close enough to
be the same model. `tests/reference/expected_pytorch.json` is the golden output produced by the
**original** `scripts/predict_toy_decisions.py` on CPU in float32.

| Check | Result |
|---|---|
| Max probability deviation vs PyTorch (fp32) | **1.6e-07** |
| Backbone last-hidden-state max abs error | 1.0e-04 (on values up to 67.6) |
| Probability distributions normalized | all pass |
| Test suite | **51 tests, all passing** (8 need MLX + weights and skip otherwise) |
| Snake decision parity on the game checkpoint | **1.5e-06** max deviation, argmax 12/12 |

Semantics match on every question in the reference fixture:

| Question | PyTorch reference | nanojev-mlx |
|---|---|---|
| Duplicate charge → which team | `billing` @ 0.9999974 | `billing` @ 0.999997 |
| "Has the refund arrived?" (text says no) | `p_true` 0.017378978 | 0.017379 |
| Software error → which team | `technical` @ 0.9994036 | `technical` @ 0.999404 |
| Severity, 4 ordered levels | `score` 1.9999634 | 2.0000 |

Reproduce it yourself:

```bash
NANOJEV_CHECKPOINT=/path/to/checkpoints/NanoJev python -m unittest discover -s tests -v
```

## Benchmarks

Apple M4, 16 GB unified memory, macOS 26.5.1, Python 3.14.6, MLX 0.32.2.
Workload: the reference fixture — 2 states, 6 questions, 15 candidate paths, 1 backbone forward.

| Metric | PyTorch + MPS | **nanojev-mlx** | Change |
|---|---:|---:|---:|
| Steady-state latency | 456 ms | **344 ms** | **1.3× faster** |
| Peak resident memory | 4.91 GB | **2.75 GB** | **−44%** |
| Cold start to first result | 8.2 s | **2.2 s** | **3.8× faster** |
| Runtime dependency footprint | PyTorch ~590 MB | **MLX ~210 MB** | **2.8× smaller** |

Reproduce all of it on your own machine:

```bash
python benchmarks/benchmark.py --checkpoint-dir checkpoints/NanoJev \
    --backend both --nanojev-repo ../NanoJev
```

`--backend both` measures each stack in its own subprocess so the memory readings stay clean.
Omit `--nanojev-repo` and `--backend both` to measure only the MLX port.

**Honest note on the speedup.** 1.3× is real but modest, and the reason is structural: this
model runs the *full 28-layer backbone once per candidate path*, so the workload is
compute-bound on the backbone, not on the 200 K-parameter head this port rewrote. MLX wins
clearly on memory and startup; the raw latency gain is limited by how the model spends its
FLOPs. See [Limitations](#limitations).

**How these numbers were taken.** One machine, one input (the reference fixture), fp32 on both
sides: latency is the median of 6 warm rounds, memory is peak RSS, cold start is runtime import
plus model load plus first result — everything a user waits for when running the CLI. Dependency
footprint is the installed `site-packages` size on Apple Silicon — `torch` versus `mlx` +
`mlx-lm`. Both stacks additionally need `transformers`, `numpy`, and `safetensors`, so that
shared cost cancels out of the comparison.

Cold start includes reading the 2.4 GB weight file, so the very first run on a machine (with
the file not yet in the OS page cache) is roughly 1.5 s slower for each stack and the ratio
narrows accordingly. The values above are the stable, repeatable ones.

## Install

**Requires an Apple Silicon Mac (M1 or newer) running macOS 14 (Sonoma) or later**, and
Python ≥ 3.10. MLX ships wheels for macOS 14, 15 and 26 — macOS 13 and older are not supported,
and an Intel Mac cannot run it at all.

```bash
git clone https://github.com/mrg123/nanojev-mlx.git
cd nanojev-mlx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally install the package itself, which also gives you the `nanojev-mlx` and
`nanojev-mlx-serve` console commands:

```bash
pip install -e .
```

Get the model weights (~2.4 GB) from Hugging Face into `checkpoints/NanoJev`.
`huggingface_hub` is already available as a dependency of `transformers`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
```

No conversion step. The MLX port reads `best.safetensors` directly — the PyTorch file layout is
mapped at load time, so there is no duplicate copy of the weights in a second format.

## Usage

[`examples/request.json`](examples/request.json) is a ready-to-run request covering all three
question types (choice, boolean, score) across two states. Start with it:

```bash
python -m nanojev_mlx.predict \
  --checkpoint-dir checkpoints/NanoJev \
  --input examples/request.json
```

It prints the full result JSON — every candidate's probability, the temperature actually
applied, and an `execution` block reporting the device and that zero decode steps ran.
Summarized to one line per question, the answers for that request are:

| State | Question | Type | Answer |
|---|---|---|---|
| `refund` | `team` | choice | `billing` @ 1.0000 |
| `refund` | `arrived` | boolean | `p_true` 0.0174 → `False` |
| `refund` | `severity` | score | 1.464 (level 1 of 4) |
| `button_error` | `team` | choice | `technical` @ 0.9847 |
| `button_error` | `blocking` | boolean | `p_true` 0.2363 → `False` |

### Python

```python
from nanojev_mlx import load_model, run_prediction

model = load_model("checkpoints/NanoJev")
result = run_prediction(model, {
    "states": [{
        "id": "my_state",
        "state": "订单尚未退款。",
        "questions": {
            "already_refunded": {
                "type": "boolean",
                "instructions": "订单是否已经完成退款？",
            },
        },
    }],
})

print(result["states"][0]["answers"]["already_refunded"])
# {'type': 'boolean', 'probabilities': {'false': 0.985..., 'true': 0.014...},
#  'p_true': 0.0145..., 'value': False}
```

### Local HTTP service

Protocol-compatible with the original `POST /api/evaluate`:

```bash
python -m nanojev_mlx.serve --checkpoint-dir checkpoints/NanoJev --port 8765
```

```bash
curl -X POST http://127.0.0.1:8765/api/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

## Testing

Three layers, split by what they need:

| Suite | Needs | Runs on |
|---|---|---|
| `tests/test_contract.py` — request validation, answer assembly | Python only | any platform |
| `tests/test_demo.py` — Snake environment, planner, episode loop | Python only (stub engine) | any platform |
| `tests/test_live.py` — the live HTTP server, driven over loopback | Python only (stub engine) | any platform |
| `tests/test_equivalence.py` — probabilities vs the PyTorch reference | MLX + a 2.4 GB checkpoint | Apple Silicon |
| `tests/test_snake_equivalence.py` — decision parity on real Snake states | MLX + the Snake checkpoint | Apple Silicon |

```bash
# Everything that needs no MLX, no GPU and no weights — this is what CI runs.
python -m unittest discover -s tests -v

# Everything, including both numerical equivalence checks
NANOJEV_CHECKPOINT=/path/to/checkpoints/NanoJev \
NANOJEV_GAMES_CHECKPOINT=/path/to/variants/games_gold_seed17 \
python -m unittest discover -s tests -v
```

**51 tests** in total: 15 protocol, 17 demo, 11 live, 8 equivalence (4 against the root
fixture, 4 against real Snake decision points). The 8 equivalence tests need MLX plus a
2.4 GB checkpoint and skip rather than fail, so a bare checkout still runs **43 of them**.

CI covers the protocol layer on Linux. The equivalence tests deliberately do not run there:
MLX crashes on import in headless, GPU-less environments
([ml-explore/mlx#3148](https://github.com/ml-explore/mlx/issues/3148)), and GitHub's hosted
macOS runners expose no Metal device. That is why `nanojev_mlx.answers` — the pure-logic half
of the output path — lives in its own module with no MLX import.

## Demo: watch it play Snake

Everywhere else in this repo the model's output is JSON, which is the right output for
a program and a poor first impression for a person. The demo runs one full episode on
your GPU and writes a single self-contained HTML page — no server, no network, no CDN:

```bash
python -m demo.play --checkpoint-dir checkpoints/games/variants/games_gold_seed17 --open
```

The page steps through the episode and shows, at every decision, the probability the
model put on **each** offered move before committing to one. That is the thing a
token-stream model structurally cannot show you.

Prefer it live? `demo.live` serves the same episode over HTTP and updates the board as
each decision is actually computed — play/pause, speed control, and click-to-take-over so
you can play a move yourself. The model is queried first, so you see what it wanted before
you overrode it:

```bash
python -m demo.live --checkpoint-dir checkpoints/games/variants/games_gold_seed17
# open http://127.0.0.1:8770
```

It needs the Snake checkpoint, not the root release — see
[`demo/README.md`](demo/README.md) for the download. Note the architecture it makes
visible: a code planner filters colliding moves and keeps those on a shortest static
path to the food; the model is asked to choose **only when two or more survive**. When
one survives, the move is forced and the model is never called at all.

## How it works

The model is a standard **Qwen3-0.6B** backbone (99.97% of parameters) plus a tiny decision
head (200,578 parameters, 0.034%):

```
grouped hidden states [examples, candidates, 1024]
  └─ LayerNorm
  └─ scalar       Linear(1024 → 1)        base logit per candidate
  └─ set_project  Linear(1025 → 128)      hidden + log(candidate count)
  └─ set_attention 4-head self-attention over the candidate set
  └─ set_output   Linear(128 → 1)         tanh residual → per-candidate correction
```

Question types fall out of the same forward pass:

- **boolean** — one semantic path; logits are `[0, z]`, so `p_true = sigmoid(z)`
- **choice** — softmax over the corrected candidate logits
- **score** — softmax over ordered level descriptions; expected value is the score

Candidate paths are right-padded into a single matrix and processed in **one** backbone call.
Because padding sits to the right of every real token, a plain causal mask yields exactly the
same hidden states as the original's combined causal + padding mask — which is why the port
needs no custom attention kernel.

### What was actually changed

Only the runtime. The model, weights, tokenizer, prompt construction, and probability
semantics are untouched. Concretely:

| Component | Original | Here |
|---|---|---|
| Backbone | HF `Qwen3Model` (PyTorch) | `mlx_lm` `Qwen3Model` |
| Decision head | `nn.MultiheadAttention` | explicit MLX attention with additive mask |
| Fused `in_proj_weight` | kept fused | split into `q_proj` / `k_proj` / `v_proj` |
| Device gate | refuses to run without CUDA (the benchmark removes only this gate) | Metal GPU via MLX, no PyTorch |

`prepare_examples` — the code that turns a state, question, and candidate set into token
paths — is a line-by-line port, because a single differing token would change every
probability downstream.

## Limitations

- **Inference only.** Training, evaluation, and the benchmark pipelines of the original repo
  are not ported. This runs the model; it does not reproduce the paper's training numbers.
- **No prefix sharing.** Every candidate path runs the backbone from scratch, exactly like the
  original (`prefix_sharing: false`). Candidates within a question share a long prefix, so
  caching it is the obvious next optimization and would cut real work meaningfully.
- **No quantization yet.** Weights load in float32 for numerical fidelity. MLX supports 4-bit
  and 8-bit quantization, which would cut memory by ~4× and likely speed things up — but it
  changes the numbers and needs its own validation pass before it can be trusted.
- **Weights are not included.** Download them from Hugging Face as shown above.
- **float32 only.** Upstream trained with bf16 autocast on A100. This port runs fp32, which is
  a *different* numerical path from the original's bf16 — closer to the CPU fp32 reference used
  for the equivalence tests than to the published benchmark figures.

## Troubleshooting

**`pip install` cannot find `mlx` / resolves to a very old version.**
MLX ships wheels for macOS 14, 15 and 26 on Apple Silicon only. On macOS 13 or older, or on an
Intel Mac, there is no wheel — this port cannot run there.

**`ValueError: checkpoint 缺少 best.safetensors` (or `backbone_config`, `tokenizer`).**
`--checkpoint-dir` must point at the directory that directly contains those entries. On the
Hugging Face repo that is either the repository root or one of the `variants/<name>/`
directories — both have the same layout:

```
best.safetensors
config.json
backbone_config/config.json
tokenizer/{tokenizer.json, tokenizer_config.json, chat_template.jinja}
```

So `variants/local_atomic_seed17` works too, as long as you point at the variant directory
itself and not at its parent. Be aware that variants are trained on different data (maze,
Snake) and will not reproduce the numbers in this README, which use the root release.

**`OSError` / `NSRangeException` on import, or a crash before any output.**
MLX needs a Metal GPU. This happens in headless virtual machines, CI containers, and some
remote/SSH sessions. It is not something this port can work around — run it on a normal macOS
desktop session.

**Out of memory on a 16 GB machine.**
This port peaks at ~2.75 GB. Running the PyTorch implementation at the same time adds ~4.9 GB,
so the two together will pressure a 16 GB machine. Nothing else in this repository is
memory-hungry.

## Credits and license

- Original model, training pipeline, and reference implementation:
  [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) — MIT.
- Backbone: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
- MLX port: MIT, see [LICENSE](LICENSE). The original copyright notice is preserved.

MIT licensed. Both the original and this port are independent research work, not affiliated
with TypeSafe or Jev.
