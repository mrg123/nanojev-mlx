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

## Verified against the published result

The original repository publishes its Snake run as **27 food, 256 steps, alive at
horizon**, with *seed 61005, controller greedy*. Running this port with those exact
parameters reproduces it number for number:

| | Upstream README | This port |
|---|---|---|
| Food collected | **27** | **27** |
| Steps | **256** | **256** |
| Outcome | Alive at horizon | `horizon_survived` |

That was 75 model decisions and 181 code-forced moves, at ~57 ms per step (14.6 s total).

Decision parity is checked separately: on 12 real Snake decision points, MLX and the
original PyTorch implementation agree to **1.5e-06** with identical argmax on all 12
(`tests/test_snake_equivalence.py`).

A useful contrast — the same demo on the **root** checkpoint, which was never trained on
Snake, splits its two-way decisions at 0.503 / 0.497. The game checkpoint is confident
(0.760 / 0.240 on the opening move). That gap is the model actually knowing something.

## Live mode

The page above is a replay: it records an episode first, then plays it back. `demo.live`
runs the same episode **live** instead — the board updates as each decision is actually
computed on your GPU, and you can take over a move yourself:

```bash
python -m demo.live --checkpoint-dir checkpoints/games/variants/games_gold_seed17
# then open http://127.0.0.1:8770
```

- the board updates live, with the probability bars for the decision being made
- play / pause / single-step / speed
- change board size or seed and start again
- **click any candidate bar to play that move yourself.** The model is queried first, so
  you see what it wanted before you overrode it, and your move is validated against the
  planner's candidates — you cannot play a move the planner filtered out
- running counts of model decisions, code-forced moves, and your own overrides

### Why the browser does not own the game

The browser is a display and a control surface only. The rules, the planner and the model
all stay in Python, and the browser asks the server for one step at a time. Porting
`snake_game.py` to JavaScript was the obvious shortcut, but then live mode would no
longer be running the same environment as the reference results, and the fidelity
argument above would simply stop applying to it.

Live mode and recorded mode drive the same `EpisodeSession`. That is checked, not
assumed: driving a full 256-step episode over HTTP with seed 61005 reproduces
**27 food / 256 steps / alive at horizon** — identical to the recorded run and to the
upstream published figure.

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
stub engine, and `tests/test_live.py` starts the real HTTP server on a loopback port and
drives it — **no checkpoint and no MLX required**, so both run in CI:

```bash
python -m unittest discover -s tests -p "test_demo.py" -v
python -m unittest discover -s tests -p "test_live.py" -v
```

The most useful of them assert that the composed controller **never admits a colliding
action** — the entire contract of routing moves through the planner first — and that a
human override cannot bypass that filter either.

## Honest limitations

- The demo reuses the original author's controller, so it inherits its behaviour — this
  is not a from-scratch learned Snake player.
- The episode is played live but the page is a recording of it; there is no interactive
  game loop, and adding one would not show the probabilities any better.
- A long episode means one model call per non-forced step. On an M4 that is about
  **85 ms per step**, so a full 256-step run is roughly 25 seconds. `--max-steps 60`
  gives a quicker look.
