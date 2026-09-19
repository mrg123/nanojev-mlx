"""请求校验与候选路径构造。

这里的逻辑是原仓库 scripts/predict_toy_decisions.py 中 validate_request /
prepare_examples 的逐行等价移植。必须完全一致：候选路径的 token 序列只要差一个
token，输出的概率分布就会变，数值对齐验证也就失去意义。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"JSON 含重复键：{key}")
        obj[key] = value
    return obj


def reject_nonfinite(value):
    raise ValueError(f"JSON 不允许非有限数值：{value}")


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_nonfinite,
    )


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_request(payload: Any):
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('输入必须为且仅为 {"states": [...]}，不需要 teacher 或 gold')
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states 必须是非空数组")
    seen_ids = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != {"id", "state", "questions"}:
            raise ValueError("每个 state 项必须只包含 id、state、questions")
        if not nonempty_text(state["id"]) or state["id"] in seen_ids:
            raise ValueError("state id 必须是唯一的非空字符串")
        seen_ids.add(state["id"])
        if not isinstance(state["state"], (str, dict, list)):
            raise ValueError("state 内容须为字符串、JSON对象或数组")
        if not state["state"]:
            raise ValueError("state 内容不得为空")
        questions = state["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions 必须是非空对象")
        for qid, question in questions.items():
            if not nonempty_text(qid) or not isinstance(question, dict):
                raise ValueError("question ID 必须是非空字符串，内容必须为对象")
            if set(question) - {"type", "instructions", "criteria"}:
                raise ValueError(f"{state['id']}:{qid} 含不支持的 question 字段")
            typ = question.get("type")
            if typ not in {"boolean", "choice", "score"} or not nonempty_text(question.get("instructions")):
                raise ValueError(f"{state['id']}:{qid} 题型或 instructions 无效")
            if typ == "boolean":
                if "criteria" in question:
                    criteria = question["criteria"]
                    if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                        raise ValueError("Boolean criteria 只能是含 false 和/或 true 键的对象")
                    if not all(nonempty_text(value) for value in criteria.values()):
                        raise ValueError("Boolean criterion 必须是非空字符串")
            elif typ == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError("Choice criteria 必须是含 2–255 项的对象")
                if not all(nonempty_text(k) and nonempty_text(v) for k, v in criteria.items()):
                    raise ValueError("Choice 候选 ID 和语义描述必须是非空字符串")
            else:
                criteria = question.get("criteria")
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ValueError("Score criteria 必须是含 2–10 项的有序数组")
                if not all(nonempty_text(value) for value in criteria):
                    raise ValueError("Score 等级描述必须是非空字符串")
    return states


def prepare_examples(payload, tokenizer, max_length):
    """逐段 encode，候选文本与 EOS 完全遵循原实现。"""
    states = validate_request(payload)
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length 必须为正整数")
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise ValueError("checkpoint tokenizer 必须有合法 eos_token_id")
    examples = []
    for row in states:
        for qid, q in row["questions"].items():
            typ = q["type"]
            if typ == "boolean":
                ids, texts = ["false", "true"], ["The proposition is true."]
            elif typ == "choice":
                ids = list(q["criteria"])
                texts = [f"{key}: {q['criteria'][key]}" for key in ids]
            else:
                ids = [str(i) for i in range(len(q["criteria"]))]
                texts = q["criteria"]
            segments = [
                f"State:\n{row['state']}\n",
                f"Question type: {typ}\nQuestion:\n{q['instructions']}\n",
            ]
            if typ == "boolean" and "criteria" in q:
                for key, label in (("false", "False"), ("true", "True")):
                    if key in q["criteria"]:
                        segments[1] += f"{label} criterion: {q['criteria'][key]}\n"
            prefix = sum([tokenizer.encode(t, add_special_tokens=False) for t in segments], [])
            leaves = [
                prefix
                + tokenizer.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False)
                + [tokenizer.eos_token_id]
                for t in texts
            ]
            largest = max(map(len, leaves))
            if largest > max_length:
                raise ValueError(
                    f"{row['id']}:{qid} 候选路径为 {largest} token，超过 max_length={max_length}；未截断输入"
                )
            examples.append(
                {
                    "id": f"{row['id']}:{qid}",
                    "state_id": row["id"],
                    "qid": qid,
                    "type": typ,
                    "candidate_ids": ids,
                    "candidate_texts": texts,
                    "leaf_tokens": leaves,
                }
            )
    return examples


def group_paths(examples):
    """把候选路径摊平成 [总路径数, 宽度] 的 token 矩阵，并记录每题的候选切片。

    返回 (tokens, lengths, groupings)；groupings[i] = (start, length) 指向该题
    在摊平后的叶节点数组中的切片。
    """
    paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
    lengths = [len(p) for p in paths]
    groupings = []
    offset = 0
    for ex in examples:
        n = len(ex["leaf_tokens"])
        groupings.append((offset, n))
        offset += n
    return paths, lengths, groupings
