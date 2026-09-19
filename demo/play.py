#!/usr/bin/env python3
"""Play one Snake episode with the MLX port and write a self-contained HTML replay.

    python -m demo.play --checkpoint-dir checkpoints/games/variants/games_gold_seed17

The page needs no server and no network: the whole episode is embedded as JSON.
It shows, for every decision, the probability the model assigned to each offered
move -- which is the thing a token-stream model cannot show you.

Requires the Snake checkpoint (variants/games_gold_seed17), not the root release.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.controller import MlxEngine, run_episode  # noqa: E402


def build_page(episode: dict, checkpoint: str) -> str:
    payload = json.dumps(episode, ensure_ascii=False, separators=(",", ":"))
    title = f"nanojev-mlx · Snake · seed {episode['seed']} · {episode['size']}×{episode['size']}"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg:#fbfbfd; --fg:#1d1d1f; --muted:#6e6e73; --line:#e4e4e8; --card:#fff;
    --snake:#1a7f37; --head:#0a6cff; --food:#d1495b; --accent:#0a6cff; --force:#9a6700;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0e0e11; --fg:#ececf1; --muted:#9a9aa3; --line:#2a2a31; --card:#17171c;
             --snake:#5dd07e; --head:#4d9bff; --food:#ff6b81; --accent:#4d9bff; --force:#e0b64a; }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro SC","PingFang SC",system-ui,sans-serif}}
  .wrap{{max-width:1100px;margin:0 auto;padding:28px 20px 60px}}
  h1{{font-size:22px;margin:0 0 4px}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:22px}}
  .layout{{display:grid;grid-template-columns:minmax(280px,1fr) minmax(320px,1.05fr);gap:26px;align-items:start}}
  @media (max-width:820px){{.layout{{grid-template-columns:1fr}}}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}}
  #board{{display:grid;gap:2px;aspect-ratio:1;width:100%}}
  .cell{{border-radius:3px;background:color-mix(in srgb,var(--line) 45%,transparent);transition:background .08s}}
  .cell.snake{{background:var(--snake)}}
  .cell.head{{background:var(--head)}}
  .cell.food{{background:var(--food);border-radius:50%}}
  .stats{{display:flex;flex-wrap:wrap;gap:14px;margin-top:14px;font-size:13px;color:var(--muted)}}
  .stats b{{color:var(--fg);font-weight:600}}
  .controls{{display:flex;align-items:center;gap:10px;margin:16px 0 6px;flex-wrap:wrap}}
  button{{font:inherit;padding:7px 14px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
  button:hover{{border-color:var(--accent)}}
  button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  input[type=range]{{flex:1;min-width:110px;accent-color:var(--accent)}}
  .step-head{{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:10px}}
  .step-head h2{{font-size:15px;margin:0}}
  .tag{{font-size:12px;padding:2px 9px;border-radius:20px;border:1px solid var(--line);color:var(--muted)}}
  .tag.model{{color:var(--accent);border-color:var(--accent)}}
  .tag.forced{{color:var(--force);border-color:var(--force)}}
  .bar-row{{margin:9px 0}}
  .bar-label{{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}}
  .bar-label .pct{{font-variant-numeric:tabular-nums;color:var(--muted)}}
  .bar-track{{height:9px;border-radius:5px;background:color-mix(in srgb,var(--line) 60%,transparent);overflow:hidden}}
  .bar-fill{{height:100%;background:var(--accent);border-radius:5px}}
  .bar-row.chosen .bar-fill{{background:var(--snake)}}
  .bar-row.chosen .bar-label{{font-weight:650}}
  .note{{font-size:12.5px;color:var(--muted);margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}}
  code{{background:color-mix(in srgb,var(--line) 55%,transparent);padding:1px 5px;border-radius:5px;font-size:12.5px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>NanoJev 在 MLX 上玩贪吃蛇</h1>
  <div class="sub">
    这不是录像回放 —— 下面每一步都是本机 GPU 上实时算出来的。<br>
    checkpoint: <code>{html.escape(checkpoint)}</code> · seed {episode['seed']} · {episode['size']}×{episode['size']}
  </div>

  <div class="layout">
    <div>
      <div class="card"><div id="board"></div></div>
      <div class="controls">
        <button class="primary" id="play">播放</button>
        <button id="prev">上一步</button>
        <button id="next">下一步</button>
        <input type="range" id="seek" min="0" value="0">
        <span class="tag" id="counter"></span>
      </div>
      <div class="stats">
        <span>吃到食物 <b id="s-food">0</b></span>
        <span>步数 <b id="s-steps">0</b></span>
        <span>模型决策 <b id="s-model">0</b></span>
        <span>代码强制 <b id="s-forced">0</b></span>
        <span>结局 <b id="s-outcome"></b></span>
      </div>
    </div>

    <div class="card">
      <div class="step-head">
        <h2 id="step-title">—</h2>
        <span class="tag" id="step-actor">—</span>
      </div>
      <div id="bars"></div>
      <div class="note" id="step-note"></div>
    </div>
  </div>
</div>

<script id="episode" type="application/json">{payload}</script>
<script>
const EP = JSON.parse(document.getElementById('episode').textContent);
const S = EP.size, STEPS = EP.steps;
let i = 0, timer = null;

const board = document.getElementById('board');
board.style.gridTemplateColumns = `repeat(${{S}}, 1fr)`;
const cells = [];
for (let n = 0; n < S * S; n++) {{
  const d = document.createElement('div');
  d.className = 'cell';
  board.appendChild(d);
  cells.push(d);
}}

function draw(state) {{
  for (const c of cells) c.className = 'cell';
  for (const [r, col] of state.body) cells[r * S + col].classList.add('snake');
  const [hr, hc] = state.body[0];
  cells[hr * S + hc].classList.add('head');
  if (state.food) cells[state.food[0] * S + state.food[1]].classList.add('food');
}}

const DIR = {{north:'↑', east:'→', south:'↓', west:'←'}};

function render() {{
  const atEnd = i >= STEPS.length;
  const state = atEnd ? EP.final_state : STEPS[i].state;
  draw(state);

  document.getElementById('s-food').textContent = state.score;
  document.getElementById('s-steps').textContent = atEnd ? STEPS.length : STEPS[i].step_index;
  document.getElementById('s-model').textContent = EP.model_decisions;
  document.getElementById('s-forced').textContent = EP.forced_moves;
  document.getElementById('s-outcome').textContent = EP.outcome;

  document.getElementById('counter').textContent = atEnd ? `结束 · 共 ${{STEPS.length}} 步` : `${{i + 1}} / ${{STEPS.length}}`;
  document.getElementById('seek').max = STEPS.length;
  document.getElementById('seek').value = i;

  const bars = document.getElementById('bars');
  const note = document.getElementById('step-note');
  if (atEnd) {{
    document.getElementById('step-title').textContent = '对局结束';
    document.getElementById('step-actor').textContent = EP.outcome;
    document.getElementById('step-actor').className = 'tag';
    bars.innerHTML = '';
    note.textContent = `共 ${{STEPS.length}} 步，吃到 ${{EP.food_collected}} 个食物。` +
      ` 其中 ${{EP.model_decisions}} 步由模型在候选之间做判断，${{EP.forced_moves}} 步只有唯一选择、代码直接决定（没有调用模型）。`;
    return;
  }}

  const st = STEPS[i];
  document.getElementById('step-title').textContent = `第 ${{st.step_index + 1}} 步 · 选择 ${{DIR[st.action] || ''}} ${{st.action}}`;
  const actor = document.getElementById('step-actor');
  actor.textContent = st.actor === 'model_tiebreak' ? '模型判断' : '代码强制';
  actor.className = 'tag ' + (st.actor === 'model_tiebreak' ? 'model' : 'forced');

  const rows = st.candidate_order.map(a => {{
    const p = st.probabilities[a];
    return `<div class="bar-row ${{a === st.action ? 'chosen' : ''}}">
      <div class="bar-label"><span>${{DIR[a] || ''}} ${{a}}${{a === st.action ? ' ← 实际走的' : ''}}</span>
      <span class="pct">${{(p * 100).toFixed(2)}}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${{(p * 100).toFixed(2)}}%"></div></div>
    </div>`;
  }}).join('');
  bars.innerHTML = rows;

  note.textContent = st.actor === 'model_tiebreak'
    ? `代码规划器先排除了会立刻撞墙/撞身体的走法，并留下到食物静态最短路径上的 ${{st.candidate_order.length}} 个候选；模型只负责在这几个里挑一个。`
    : `这一步只有唯一的安全候选，代码直接决定，没有调用模型。`;
}}

function setStep(n) {{ i = Math.max(0, Math.min(STEPS.length, n)); render(); }}
document.getElementById('next').onclick = () => setStep(i + 1);
document.getElementById('prev').onclick = () => setStep(i - 1);
document.getElementById('seek').oninput = e => setStep(+e.target.value);

const playBtn = document.getElementById('play');
playBtn.onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = '播放'; return; }}
  if (i >= STEPS.length) setStep(0);
  playBtn.textContent = '暂停';
  timer = setInterval(() => {{
    if (i >= STEPS.length) {{ clearInterval(timer); timer = null; playBtn.textContent = '播放'; return; }}
    setStep(i + 1);
  }}, 120);
}};

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') setStep(i + 1);
  if (e.key === 'ArrowLeft') setStep(i - 1);
}});

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="含 best.safetensors 的目录")
    parser.add_argument("--size", type=int, default=12, help="棋盘边长（默认 12）")
    parser.add_argument("--seed", type=int, default=61005, help="对局种子（默认 61005，与原仓库记录一致）")
    parser.add_argument("--controller", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--output", default="snake-demo.html", help="输出的单文件 HTML")
    parser.add_argument("--open", action="store_true", help="生成后用浏览器打开")
    args = parser.parse_args()

    from nanojev_mlx import load_model

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    print(f"加载 checkpoint: {checkpoint}")
    started = time.perf_counter()
    model = load_model(checkpoint)
    print(f"  {time.perf_counter() - started:.2f}s")

    print(f"对局中: {args.size}×{args.size}, seed {args.seed}, controller={args.controller} ...")
    started = time.perf_counter()
    episode = run_episode(
        MlxEngine(model),
        size=args.size,
        seed=args.seed,
        controller=args.controller,
        max_steps=args.max_steps,
    )
    elapsed = time.perf_counter() - started

    out = Path(args.output).expanduser()
    out.write_text(build_page(episode, str(checkpoint)), encoding="utf-8")

    print(
        f"  完成: {episode['survival_steps']} 步, 吃到 {episode['food_collected']} 个食物, "
        f"结局 {episode['outcome']} ({elapsed:.1f}s)"
    )
    print(
        f"  模型判断 {episode['model_decisions']} 次, 代码强制 {episode['forced_moves']} 次, "
        f"平均 {elapsed / max(1, episode['survival_steps']) * 1000:.0f} ms/步"
    )
    print(f"\n打开这个文件查看回放:\n  {out}")
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
