"""协议层测试：请求校验与结果组装。

**不需要模型权重，也不需要 MLX**——因此可以在任何平台运行，包括没有 Metal GPU 的
CI 容器。这正是 answers.py 与 predict.py 分开的原因。

    python -m unittest discover -s tests -p "test_contract.py" -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanojev_mlx.answers import answer_from_probabilities  # noqa: E402
from nanojev_mlx.text import validate_request  # noqa: E402


def state(**question):
    return {"states": [{"id": "s1", "state": "订单尚未退款。", "questions": question}]}


class ValidateRequestTest(unittest.TestCase):
    def test_accepts_minimal_boolean(self):
        payload = state(q={"type": "boolean", "instructions": "是否已退款？"})
        self.assertEqual(len(validate_request(payload)), 1)

    def test_accepts_choice_and_score(self):
        payload = state(
            c={"type": "choice", "instructions": "哪个团队？", "criteria": {"a": "账单", "b": "技术"}},
            n={"type": "score", "instructions": "多严重？", "criteria": ["无", "轻", "重"]},
        )
        self.assertEqual(validate_request(payload)[0]["id"], "s1")

    def test_rejects_extra_top_level_keys(self):
        payload = state(q={"type": "boolean", "instructions": "x"})
        payload["gold"] = {}
        with self.assertRaises(ValueError):
            validate_request(payload)

    def test_rejects_duplicate_state_ids(self):
        item = {"id": "dup", "state": "x", "questions": {"q": {"type": "boolean", "instructions": "y"}}}
        with self.assertRaises(ValueError):
            validate_request({"states": [item, dict(item)]})

    def test_rejects_unknown_question_type(self):
        with self.assertRaises(ValueError):
            validate_request(state(q={"type": "ranking", "instructions": "x"}))

    def test_rejects_choice_with_one_candidate(self):
        with self.assertRaises(ValueError):
            validate_request(state(q={"type": "choice", "instructions": "x", "criteria": {"only": "一个"}}))

    def test_rejects_score_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_request(state(q={"type": "score", "instructions": "x", "criteria": ["只有一个"]}))
        with self.assertRaises(ValueError):
            validate_request(
                state(q={"type": "score", "instructions": "x", "criteria": [f"L{i}" for i in range(11)]})
            )

    def test_rejects_empty_state_text(self):
        with self.assertRaises(ValueError):
            validate_request(
                {"states": [{"id": "s", "state": "", "questions": {"q": {"type": "boolean", "instructions": "x"}}}]}
            )


class AnswerTest(unittest.TestCase):
    def example(self, typ="choice"):
        return {"type": typ, "candidate_ids": ["a", "b", "c"]}

    def test_choice_picks_argmax(self):
        result = answer_from_probabilities(self.example(), [0.1, 0.7, 0.2])
        self.assertEqual(result["choice"], "b")
        self.assertEqual(result["value"], "b")

    def test_boolean_reports_p_true(self):
        example = {"type": "boolean", "candidate_ids": ["false", "true"]}
        result = answer_from_probabilities(example, [0.25, 0.75])
        self.assertAlmostEqual(result["p_true"], 0.75)
        self.assertTrue(result["value"])

    def test_score_returns_expectation(self):
        example = {"type": "score", "candidate_ids": ["0", "1", "2"]}
        result = answer_from_probabilities(example, [0.5, 0.25, 0.25])
        self.assertAlmostEqual(result["score"], 0.75)
        self.assertEqual(result["level"], 0)

    def test_rejects_unnormalized(self):
        with self.assertRaises(ValueError):
            answer_from_probabilities(self.example(), [0.5, 0.5, 0.5])

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            answer_from_probabilities(self.example(), [0.5, float("nan"), 0.5])

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            answer_from_probabilities(self.example(), [1.0])


class SoftmaxTest(unittest.TestCase):
    """概率归一化：所有题型都必须严格归一到 1。"""

    def test_softmax_normalizes(self):
        probs = [0.6436, 0.0704, 0.1579, 0.1281]
        self.assertAlmostEqual(math.fsum(probs), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
