#!/usr/bin/env python3
"""Live Snake demo: play an episode in the browser, one decision at a time.

    python -m demo.live --checkpoint-dir checkpoints/games/variants/games_gold_seed17

Then open http://127.0.0.1:8770

The browser is only a display and a control surface. The game rules, the code planner
and the model all stay in Python. That is deliberate: reimplementing `snake_game.py` in
JavaScript would mean the live demo no longer runs the same environment as the reference
results, and the whole fidelity argument would collapse.

Both this and the recorded demo (`demo.play`) drive the same `EpisodeSession`, so a live
game cannot drift from a recorded one.

Endpoints
    GET  /                    the live page
    GET  /api/health          readiness
    POST /api/snake/new       {"size","seed","controller","max_steps"} -> snapshot
    POST /api/snake/step      {"session","action"?} -> {"step","snapshot"}
    POST /api/snake/state     {"session"} -> snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.controller import EpisodeSession, MlxEngine  # noqa: E402

MAX_REQUEST_BYTES = 100_000
MAX_SESSIONS = 8

# A browser holds connections open (keep-alive) and abandons them without warning on
# reload, navigation or tab close. Two consequences, both handled here:
#
#   * a single-threaded server blocks forever reading the next request off an abandoned
#     connection, freezing the whole app -- hence ThreadingHTTPServer;
#   * an idle keep-alive connection would otherwise hold its thread indefinitely --
#     hence the socket timeout below.
#
# Inference itself is still serialised by a lock: one model, one call at a time.
IDLE_CONNECTION_TIMEOUT = 20


def live_page() -> str:
    """The page. Plain string, not an f-string -- it is mostly CSS and JS braces."""
    return _PAGE


class SnakeDemo:
    """Holds live episodes. One model, many sessions.

    All public methods take a lock. The HTTP server is threaded so that a slow or
    abandoned connection cannot freeze the app, but MLX is not driven concurrently --
    requests queue on the lock and each inference still runs alone.
    """

    def __init__(self, engine):
        self.engine = engine
        self.sessions: dict[str, EpisodeSession] = {}
        self.lock = threading.Lock()

    def _evict(self):
        while len(self.sessions) >= MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))

    def new(self, size=12, seed=61005, controller="greedy", max_steps=256) -> dict:
        with self.lock:
            self._evict()
            session = EpisodeSession(
                self.engine, size=size, seed=seed, controller=controller, max_steps=max_steps
            )
            key = uuid.uuid4().hex[:12]
            self.sessions[key] = session
            return {"session": key, "size": size, "seed": seed, "controller": controller,
                    "max_steps": max_steps, **session.snapshot()}

    def _get(self, key) -> EpisodeSession:
        if not isinstance(key, str) or key not in self.sessions:
            raise ValueError("unknown session; start a new game")
        return self.sessions[key]

    def step(self, key, action=None) -> dict:
        with self.lock:
            session = self._get(key)
            record = session.step(override_action=action)
            return {"session": key, "step": record, **session.snapshot()}

    def state(self, key) -> dict:
        with self.lock:
            return {"session": key, **self._get(key).snapshot()}


def handler_class(demo: SnakeDemo):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = IDLE_CONNECTION_TIMEOUT

        def _send(self, code, body: bytes, mime="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, data):
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request body missing or too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                self._send(200, live_page().encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/health":
                self._json(200, {"ready": True, "runtime": "mlx", "sessions": len(demo.sessions)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            try:
                body = self._body()
                if route == "/api/snake/new":
                    self._json(200, demo.new(
                        size=int(body.get("size", 12)),
                        seed=int(body.get("seed", 61005)),
                        controller=body.get("controller", "greedy"),
                        max_steps=int(body.get("max_steps", 256)),
                    ))
                elif route == "/api/snake/step":
                    action = body.get("action")
                    self._json(200, demo.step(body.get("session"), action if action else None))
                elif route == "/api/snake/state":
                    self._json(200, demo.state(body.get("session")))
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "model inference failed; check the server process"})
                raise

        def log_message(self, fmt, *args):
            # One line per request would drown the log during a 256-step game.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    from nanojev_mlx import load_model

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    print(f"加载 checkpoint: {checkpoint}")
    model = load_model(checkpoint)
    engine = MlxEngine(model)

    server = ThreadingHTTPServer((args.host, args.port), handler_class(SnakeDemo(engine)))
    server.daemon_threads = True
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "ready": True, "runtime": "mlx"}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanojev-mlx · 实时贪吃蛇</title>
<style>
  :root{--bg:#fbfbfd;--fg:#1d1d1f;--muted:#6e6e73;--line:#e4e4e8;--card:#fff;
        --snake:#1a7f37;--head:#0a6cff;--food:#d1495b;--accent:#0a6cff;--human:#9a6700;--force:#8a8a92}
  @media (prefers-color-scheme:dark){:root{--bg:#0e0e11;--fg:#ececf1;--muted:#9a9aa3;--line:#2a2a31;
        --card:#17171c;--snake:#5dd07e;--head:#4d9bff;--food:#ff6b81;--accent:#4d9bff;--human:#e0b64a;--force:#8a8a92}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro SC","PingFang SC",system-ui,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:26px 20px 60px}
  h1{font-size:21px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .layout{display:grid;grid-template-columns:minmax(280px,1fr) minmax(320px,1.05fr);gap:24px;align-items:start}
  @media (max-width:820px){.layout{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
  #board{display:grid;gap:2px;aspect-ratio:1;width:100%}
  .cell{border-radius:3px;background:color-mix(in srgb,var(--line) 45%,transparent);transition:background .06s}
  .cell.snake{background:var(--snake)} .cell.head{background:var(--head)}
  .cell.food{background:var(--food);border-radius:50%}
  .cell.newfood{outline:2px solid var(--food);outline-offset:-2px}
  .controls{display:flex;align-items:center;gap:9px;margin:14px 0 4px;flex-wrap:wrap}
  button{font:inherit;padding:7px 13px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
  button:hover:not(:disabled){border-color:var(--accent)}
  button:disabled{opacity:.45;cursor:default}
  button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  label{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:5px}
  input[type=number]{width:68px;font:inherit;padding:5px 7px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
  input[type=range]{flex:1;min-width:90px;accent-color:var(--accent)}
  .stats{display:flex;flex-wrap:wrap;gap:13px;margin-top:12px;font-size:13px;color:var(--muted)}
  .stats b{color:var(--fg);font-weight:600}
  .step-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:10px}
  .step-head h2{font-size:15px;margin:0}
  .tag{font-size:12px;padding:2px 9px;border-radius:20px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}
  .tag.model{color:var(--accent);border-color:var(--accent)}
  .tag.forced{color:var(--force);border-color:var(--force)}
  .tag.human{color:var(--human);border-color:var(--human)}
  .bar-row{margin:9px 0;border-radius:8px;padding:4px 6px;margin-left:-6px;margin-right:-6px}
  .bar-row.pickable{cursor:pointer}
  .bar-row.pickable:hover{background:color-mix(in srgb,var(--accent) 9%,transparent)}
  .bar-label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}
  .bar-label .pct{font-variant-numeric:tabular-nums;color:var(--muted)}
  .bar-track{height:9px;border-radius:5px;background:color-mix(in srgb,var(--line) 60%,transparent);overflow:hidden}
  .bar-fill{height:100%;background:var(--accent);border-radius:5px;transition:width .15s}
  .bar-row.chosen .bar-fill{background:var(--snake)}
  .bar-row.overridden .bar-fill{background:var(--human)}
  .bar-row.chosen .bar-label{font-weight:650}
  .hint{font-size:12.5px;color:var(--muted);margin-top:12px;padding-top:11px;border-top:1px solid var(--line)}
  code{background:color-mix(in srgb,var(--line) 55%,transparent);padding:1px 5px;border-radius:5px;font-size:12.5px}
  .err{color:#c0392b;font-size:13px;min-height:20px}
</style>
</head>
<body>
<div class="wrap">
  <h1>实时贪吃蛇 · MLX</h1>
  <div class="sub">
    每一步都在你本机的 GPU 上实时算出来——不是回放。<br>
    代码规划器先排除会碰撞的走法，<b>只有剩下两个及以上候选时</b>才问模型。
  </div>

  <div class="layout">
    <div>
      <div class="card"><div id="board"></div></div>

      <div class="controls">
        <button class="primary" id="play">开始</button>
        <button id="single">单步</button>
        <button id="again">新一局</button>
        <input type="range" id="speed" min="0" max="900" value="620" title="每步间隔">
      </div>
      <div class="controls">
        <label>边长 <input type="number" id="size" value="12" min="4" max="32"></label>
        <label>种子 <input type="number" id="seed" value="61005"></label>
      </div>
      <div class="stats">
        <span>吃到食物 <b id="s-food">0</b></span>
        <span>步数 <b id="s-steps">0</b></span>
        <span>模型判断 <b id="s-model">0</b></span>
        <span>代码强制 <b id="s-forced">0</b></span>
        <span>你接管 <b id="s-human">0</b></span>
        <span>结局 <b id="s-outcome">—</b></span>
        <span>上一步耗时 <b id="s-ms">—</b></span>
      </div>
      <div class="err" id="err"></div>
    </div>

    <div class="card">
      <div class="step-head">
        <h2 id="step-title">尚未开始</h2>
        <span class="tag" id="step-actor">—</span>
      </div>
      <div id="bars"><p class="hint">点「开始」让模型自己玩；或点下面出现的候选条自己走一步。</p></div>
      <div class="hint" id="step-note"></div>
    </div>
  </div>
</div>

<script>
const DIR = {north:'↑', east:'→', south:'↓', west:'←'};
let session = null, board = [], S = 12, playing = false, busy = false, timer = null;
const $ = id => document.getElementById(id);

function buildBoard(n) {
  S = n;
  const el = $('board');
  el.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  el.innerHTML = '';
  board = [];
  for (let i = 0; i < n * n; i++) {
    const d = document.createElement('div');
    d.className = 'cell';
    el.appendChild(d);
    board.push(d);
  }
}

function draw(state) {
  for (const c of board) c.className = 'cell';
  for (const [r, col] of state.body) board[r * S + col].classList.add('snake');
  const [hr, hc] = state.body[0];
  board[hr * S + hc].classList.add('head');
  if (state.food) board[state.food[0] * S + state.food[1]].classList.add('food');
}

async function api(path, body) {
  const res = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function applyStats(d) {
  $('s-food').textContent = d.state.score;
  $('s-steps').textContent = d.steps;
  $('s-model').textContent = d.model_decisions;
  $('s-forced').textContent = d.forced_moves;
  $('s-human').textContent = d.human_overrides;
  $('s-outcome').textContent = d.done ? d.outcome : '进行中';
}

function renderBars(step) {
  const bars = $('bars');
  const actor = $('step-actor');
  if (!step) {
    bars.innerHTML = '<p class="hint">点「开始」让模型自己玩；或点下面出现的候选条自己走一步。</p>';
    $('step-title').textContent = '尚未开始';
    actor.textContent = '—'; actor.className = 'tag';
    $('step-note').textContent = '';
    return;
  }
  $('step-title').textContent = `第 ${step.step_index + 1} 步 · 走 ${DIR[step.action] || ''} ${step.action}`;
  const cls = {model_tiebreak: 'model', forced_move: 'forced', human_override: 'human'}[step.actor];
  actor.textContent = {model_tiebreak:'模型判断', forced_move:'代码强制', human_override:'你接管'}[step.actor];
  actor.className = 'tag ' + cls;

  const pickable = step.actor === 'model_tiebreak';
  bars.innerHTML = step.candidate_order.map(a => {
    const p = step.probabilities[a];
    const chosen = a === step.action;
    return `<div class="bar-row ${chosen ? (step.actor === 'human_override' ? 'overridden' : 'chosen') : ''} ${pickable ? 'pickable' : ''}" data-action="${a}">
      <div class="bar-label"><span>${DIR[a] || ''} ${a}${chosen ? ' ← 实际走的' : ''}</span>
      <span class="pct">${(p * 100).toFixed(2)}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(2)}%"></div></div>
    </div>`;
  }).join('');

  if (pickable) {
    bars.querySelectorAll('.bar-row').forEach(row => {
      row.onclick = () => { pause(); takeOver(row.dataset.action); };
    });
  }
  $('step-note').textContent = {
    model_tiebreak: `规划器留下 ${step.candidate_order.length} 个候选，模型从中挑了一个。点任意候选条可以自己走一步（模型仍会被问一次，好让你看它原本怎么想）。`,
    forced_move: '这一步只有唯一的安全候选，代码直接决定，没有调用模型。',
    human_override: '这一步是你接管的。上面的概率是模型原本的判断。'
  }[step.actor];
}

async function newGame() {
  pause();
  $('err').textContent = '';
  const size = Math.max(4, Math.min(32, parseInt($('size').value) || 12));
  const seed = parseInt($('seed').value) || 61005;
  $('size').value = size;
  const d = await api('/api/snake/new', {size, seed, controller: 'greedy', max_steps: 256});
  session = d.session;
  buildBoard(size);
  draw(d.state);
  applyStats(d);
  renderBars(null);
  $('play').textContent = '开始';
}

async function advance(action) {
  if (!session || busy) return false;
  busy = true;
  const t0 = performance.now();
  try {
    const d = await api('/api/snake/step', {session, action});
    $('s-ms').textContent = Math.round(performance.now() - t0) + ' ms';
    draw(d.state);
    applyStats(d);
    renderBars(d.step);
    $('err').textContent = '';
    if (d.done) { pause(); $('play').textContent = '开始'; }
    return !d.done;
  } catch (e) {
    $('err').textContent = e.message;
    pause();
    return false;
  } finally {
    busy = false;
  }
}

async function takeOver(action) {
  await advance(action);
}

function tick() {
  if (!playing) return;
  const delay = Math.max(0, 1000 - parseInt($('speed').value));
  timer = setTimeout(async () => {
    if (!playing) return;
    const more = await advance(null);
    if (more && playing) tick(); else { pause(); $('play').textContent = '开始'; }
  }, delay);
}

function play() {
  if (!session) { newGame().then(() => { playing = true; $('play').textContent = '暂停'; tick(); }); return; }
  playing = true;
  $('play').textContent = '暂停';
  tick();
}
function pause() {
  playing = false;
  if (timer) { clearTimeout(timer); timer = null; }
  const b = $('play'); if (b) b.textContent = '开始';
}

$('play').onclick = () => playing ? pause() : play();
$('single').onclick = () => { pause(); advance(null); };
$('again').onclick = () => newGame();
$('seed').onchange = () => newGame();
document.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
  if (e.key === 'ArrowRight') { pause(); advance(null); }
});

newGame();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
