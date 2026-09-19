# Snake demo

Watch the MLX port play Snake, one decision at a time.

```bash
python -m demo.play --checkpoint-dir checkpoints/games/variants/games_gold_seed17 --open
```

It runs one full episode on your Mac's GPU and writes `snake-demo.html` — a single
self-contained file with no server, no network and no CDN. Open it and press play.

## Why this exists

Everywhere else in this repository, the model's output is JSON. That is the right
output for a program and a terrible first impression for a person: `p_true: 0.0145`
tells you nothing about what a *decision model* feels like to use.

The demo page shows the thing a token-stream model structurally cannot show you: at
every single step, the full probability the model put on **each** offered move, side
by side, before it commits to one.

## What you need

The Snake checkpoint — **not** the root release:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    local_dir="checkpoints/games",
    allow_patterns=["variants/games_gold_seed17/*"],
)
PY
```

Point `--checkpoint-dir` at `checkpoints/games/variants/games_gold_seed17` (the directory
that directly contains `best.safetensors`), not at its parent.

The root checkpoint is trained for navigation, not Snake. It will still load and run —
the plumbing is identical — but its choices are near-random. Measured on a 10×10 board it
splits two-way decisions at 0.503 / 0.497, which is exactly what a model that knows
nothing about Snake should do. Use it to smoke-test the demo; use the variant to watch
actual play.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint-dir` | required | directory containing `best.safetensors` |
| `--size` | `12` | board side length |
| `--seed` | `61005` | episode seed — the same seed always gives the same game |
| `--controller` | `greedy` | `greedy` takes the argmax, `sample` draws from the distribution |
| `--max-steps` | `256` | horizon; surviving it is a valid outcome |
| `--output` | `snake-demo.html` | where to write the replay |
| `--open` | off | open the result in your browser |

## What the page shows

- the board, with the snake and food, stepping through the recorded episode
- for each decision: every offered move as a probability bar, the argmax marked
- whether that step was a **model decision** or a **code-forced move** — see below
- the running score, step count, and final outcome

## This is a code planner with a model tie-breaker

Worth being precise, because it is the architecture rather than an implementation
detail:

1. **code** filters out every move that would collide on the next step, then keeps the
   moves lying on a shortest static path to the currently visible food
2. **the model** is asked to choose only when **two or more** candidates survive
3. when exactly one survives, the move is forced and **the model is never called**

So the page's "model decision / code forced" counter is not decoration — on a typical
episode a large share of steps never touch the network at all. This is the same
division of labour as the original pipeline (`common_code_planner_with_model_tiebreak`).

## Fidelity

`demo/snake_game.py` is vendored **byte-for-byte** from
[NanoJev](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/snake_game.py), and
the planner plus both prompt strings in `demo/controller.py` are reproduced verbatim
from `evaluate_composed_snake.py`. That is deliberate: the model's answer is a function
of the prompt text, so paraphrasing any of it would change the trajectory and make the
demo incomparable to the reference results.

Only the model backend differs. `MlxEngine` exists solely to adapt
`nanojev_mlx.run_prediction` to the `engine.predict(...)` interface the original
controller already expects.

## Tests

`tests/test_demo.py` covers the environment, the planner and the episode loop with a
stub engine — **no checkpoint and no MLX required**, so it runs in CI:

```bash
python -m unittest discover -s tests -p "test_demo.py" -v
```

The most useful of them asserts that the composed controller **never admits a colliding
action**: that is the entire contract of routing moves through the planner first.

## Honest limitations

- The demo reuses the original author's controller, so it inherits its behaviour — this
  is not a from-scratch learned Snake player.
- The episode is played live but the page is a recording of it; there is no interactive
  game loop, and adding one would not show the probabilities any better.
- A long episode means one model call per non-forced step. On an M4 that is about
  **85 ms per step**, so a full 256-step run is roughly 25 seconds. `--max-steps 60`
  gives a quicker look.
