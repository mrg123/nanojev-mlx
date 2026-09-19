# nanojev-mlx

**Apple Silicon native port of [NanoJev](https://github.com/TianyuCodings/NanoJev) — a 0.6B parallel decision model running on MLX.**
States and questions in, complete probability distributions out, zero output-token decoding.

NanoJev's original implementation hard-requires CUDA. This port keeps the model exactly as
trained and reimplements the decision head in MLX, so it runs natively on the Metal GPU of
any Apple Silicon Mac — no CUDA, no PyTorch, no cloud.

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
| Test suite | **19 tests, all passing** |

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
| Steady-state latency | 470 ms | **352 ms** | **1.3× faster** |
| Peak resident memory | 4.91 GB | **2.75 GB** | **−44%** |
| Cold start to first result | 9.9 s | **3.8 s** | **2.6× faster** |
| Dependency weight | PyTorch (~2.5 GB) | MLX (~65 MB) | — |

**Honest note on the speedup.** 1.3× is real but modest, and the reason is structural: this
model runs the *full 28-layer backbone once per candidate path*, so the workload is
compute-bound on the backbone, not on the 200 K-parameter head this port rewrote. MLX wins
clearly on memory and startup; the raw latency gain is limited by how the model spends its
FLOPs. See [Limitations](#limitations).

## Install

Requires an Apple Silicon Mac (M1 or newer) and Python ≥ 3.10.

```bash
git clone https://github.com/mrg123/nanojev-mlx.git
cd nanojev-mlx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get the model weights (~2.4 GB) from Hugging Face into `checkpoints/NanoJev`:

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

### Command line

```bash
python -m nanojev_mlx.predict \
  --checkpoint-dir checkpoints/NanoJev \
  --input request.json
```

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
  --data-binary @request.json
```

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
| Device gate | hard CUDA requirement | Metal GPU via MLX |

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

## Credits and license

- Original model, training pipeline, and reference implementation:
  [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) — MIT.
- Backbone: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
- MLX port: MIT, see [LICENSE](LICENSE). The original copyright notice is preserved.

MIT licensed. Both the original and this port are independent research work, not affiliated
with TypeSafe or Jev.
