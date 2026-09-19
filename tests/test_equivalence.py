"""数值等价性测试：MLX 输出必须与 PyTorch 参考实现逐概率一致。

需要一个本地 NanoJev checkpoint（约 2.4GB）。未提供时整个模块跳过：

    NANOJEV_CHECKPOINT=/path/to/checkpoints/NanoJev python -m unittest discover -s tests -v

参考数据 tests/reference/expected_pytorch.json 由原仓库
scripts/predict_toy_decisions.py 在 CPU 上以 float32 生成。
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
    if os.environ.get("NANOJEV_CHECKPOINT"):
        candidates.append(Path(os.environ["NANOJEV_CHECKPOINT"]))
    here = Path(__file__).resolve().parent.parent
    candidates += [here / "checkpoints" / "NanoJev", here.parent / "NanoJev" / "checkpoints" / "NanoJev"]
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


@unittest.skipIf(CHECKPOINT is None, "未找到 checkpoint（设置 NANOJEV_CHECKPOINT 后启用）")
@unittest.skipIf(not HAS_MLX, "MLX 不可用（仅 Apple 芯片支持）")
class EquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nanojev_mlx import load_model, run_prediction
        from nanojev_mlx.text import read_json

        cls.run_prediction = staticmethod(run_prediction)
        cls.request = read_json(REFERENCE / "toy_request.json")
        cls.expected = json.loads((REFERENCE / "expected_pytorch.json").read_text(encoding="utf-8"))
        cls.model = load_model(CHECKPOINT)
        cls.result = run_prediction(cls.model, cls.request)

    def pairs(self):
        got = {s["id"]: s["answers"] for s in self.result["states"]}
        want = {s["id"]: s["answers"] for s in self.expected["states"]}
        self.assertEqual(set(got), set(want), "state 集合不一致")
        for sid in want:
            self.assertEqual(set(got[sid]), set(want[sid]), f"{sid} 的 question 集合不一致")
            for qid in want[sid]:
                yield sid, qid, got[sid][qid], want[sid][qid]

    def test_probabilities_match_pytorch(self):
        worst = 0.0
        for sid, qid, got, want in self.pairs():
            self.assertEqual(got["type"], want["type"], f"{sid}:{qid} 题型不一致")
            self.assertEqual(
                list(got["probabilities"]), list(want["probabilities"]), f"{sid}:{qid} 候选集合不一致"
            )
            for key, value in want["probabilities"].items():
                worst = max(worst, abs(got["probabilities"][key] - value))
        self.assertLess(worst, TOLERANCE, f"最大概率偏差 {worst:.3e} 超过容差 {TOLERANCE:g}")

    def test_decisions_match(self):
        for sid, qid, got, want in self.pairs():
            if got["type"] == "boolean":
                self.assertEqual(got["value"], want["value"], f"{sid}:{qid} 布尔判定不一致")
            elif got["type"] == "choice":
                self.assertEqual(got["choice"], want["choice"], f"{sid}:{qid} 选择不一致")
            else:
                self.assertEqual(got["level"], want["level"], f"{sid}:{qid} 等级不一致")
                self.assertAlmostEqual(got["score"], want["score"], places=5)

    def test_distributions_are_normalized(self):
        for sid, qid, got, _ in self.pairs():
            total = sum(got["probabilities"].values())
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"{sid}:{qid} 概率和 {total}")

    def test_reports_zero_decoding(self):
        execution = self.result["execution"]
        self.assertEqual(execution["autoregressive_decode_steps"], 0)
        self.assertEqual(execution["network_model_calls"], 0)
        self.assertEqual(execution["runtime"], "mlx")


if __name__ == "__main__":
    unittest.main()
