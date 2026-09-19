"""Snake demo equivalence: MLX must reproduce PyTorch on the game checkpoint.

This is the counterpart of test_equivalence.py for `variants/games_gold_seed17`.
It needs that variant (2.4 GB) plus MLX, so it skips on a bare checkout and in CI:

    NANOJEV_GAMES_CHECKPOINT=/path/to/variants/games_gold_seed17 \\
        python -m unittest discover -s tests -p "test_snake_equivalence.py" -v

The requests in tests/reference/snake_golden.json are real decision points taken
from Snake episodes, and the expected answers were produced by the ORIGINAL
PyTorch implementation on MPS in float32 -- not by this port.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REFERENCE = Path(__file__).resolve().parent / "reference"
TOLERANCE = 1e-5


def find_checkpoint() -> Path | None:
    candidates = []
    if os.environ.get("NANOJEV_GAMES_CHECKPOINT"):
        candidates.append(Path(os.environ["NANOJEV_GAMES_CHECKPOINT"]))
    here = Path(__file__).resolve().parent.parent
    candidates += [
        here / "checkpoints" / "games" / "variants" / "games_gold_seed17",
        here.parent / "NanoJev" / "checkpoints" / "games" / "variants" / "games_gold_seed17",
    ]
    for candidate in candidates:
        if (candidate / "best.safetensors").is_file():
            return candidate
    return None


CHECKPOINT = find_checkpoint()


def mlx_available() -> bool:
    """This suite needs MLX as well as the checkpoint; skip cleanly if either is absent.

    Without this, a machine that has the weights but no working MLX would report a
    confusing ImportError instead of a skip.
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


HAS_MLX = mlx_available()


@unittest.skipIf(CHECKPOINT is None, "未找到贪吃蛇 checkpoint（设置 NANOJEV_GAMES_CHECKPOINT 后启用）")
@unittest.skipIf(not HAS_MLX, "MLX 不可用（仅 Apple 芯片支持）")
class SnakeEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nanojev_mlx import load_model, run_prediction

        cls.golden = json.loads((REFERENCE / "snake_golden.json").read_text(encoding="utf-8"))
        cls.model = load_model(CHECKPOINT)
        cls.result = run_prediction(cls.model, {"states": cls.golden["requests"]})

    def test_probabilities_match_pytorch(self):
        got = {s["id"]: s["answers"]["action"]["probabilities"] for s in self.result["states"]}
        want = {s["id"]: s["answers"]["action"]["probabilities"] for s in self.golden["states"]}
        self.assertEqual(set(got), set(want), "决策点集合不一致")
        worst = 0.0
        for sid in want:
            self.assertEqual(set(got[sid]), set(want[sid]), f"{sid} 的候选集合不一致")
            for key, value in want[sid].items():
                worst = max(worst, abs(got[sid][key] - value))
        self.assertLess(worst, TOLERANCE, f"最大概率偏差 {worst:.3e} 超过容差 {TOLERANCE:g}")

    def test_argmax_matches_pytorch(self):
        for state in self.golden["states"]:
            sid = state["id"]
            want = state["answers"]["action"]["probabilities"]
            got = next(s for s in self.result["states"] if s["id"] == sid)["answers"]["action"]["probabilities"]
            self.assertEqual(max(got, key=got.get), max(want, key=want.get), f"{sid} 选中的走法不一致")

    def test_distributions_are_normalized(self):
        for state in self.result["states"]:
            total = sum(state["answers"]["action"]["probabilities"].values())
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"{state['id']} 概率和 {total}")

    def test_golden_requests_are_real_snake_decisions(self):
        """Guard the fixture itself: every request must be a genuine choice point."""
        from demo import snake_game as snake
        from demo.controller import plan_candidates, render_composed_request

        self.assertGreaterEqual(len(self.golden["requests"]), 8)
        for request in self.golden["requests"]:
            self.assertEqual(set(request), {"id", "state", "questions"})
            question = request["questions"]["action"]
            self.assertEqual(question["type"], "choice")
            self.assertGreaterEqual(len(question["criteria"]), 2, "候选不足两个就不是模型决策点")
            self.assertIn("Snake on a", request["state"])
        # The builder used to make the fixture must still produce the same shape.
        state = snake.make_snake(12, 0)
        plan = plan_candidates(state)
        if len(plan["offered_actions"]) >= 2:
            rebuilt = render_composed_request(state, plan["offered_actions"])
            self.assertEqual(set(rebuilt["questions"]["action"]["criteria"]), set(plan["offered_actions"]))


if __name__ == "__main__":
    unittest.main()
