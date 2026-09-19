"""把概率分布组装成最终答案。

**纯 Python，不导入 MLX。** 这一层只做数学与结构组装，因此可以在任何平台上测试——
包括没有 Metal GPU 的 CI 容器（MLX 在无头/无 GPU 环境下导入即崩溃，见
https://github.com/ml-explore/mlx/issues/3148）。

放在独立模块而不是 predict.py，正是为了让协议层测试无需安装 MLX 即可运行。
"""

from __future__ import annotations

import math


def answer_from_probabilities(example, probabilities):
    """按题型把候选概率转成最终判定。

    boolean  取 p_true = probabilities[1]（logits 为 [0, z]，故等价于 sigmoid(z)）
    choice   取 argmax 对应的候选 ID
    score    取概率加权期望作为分数，argmax 作为等级
    """
    ids = example["candidate_ids"]
    if len(probabilities) != len(ids) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("模型产生了无效概率")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("模型概率总和不为1")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result = {"type": example["type"], "probabilities": dict(zip(ids, probabilities))}
    if example["type"] == "boolean":
        result.update(p_true=probabilities[1], value=bool(best))
    elif example["type"] == "choice":
        result.update(choice=ids[best], value=ids[best])
    else:
        score = math.fsum(i * p for i, p in enumerate(probabilities))
        result.update(score=score, level=best, value=score)
    return result


__all__ = ["answer_from_probabilities"]
