"""nanojev-mlx：NanoJev 的 Apple Silicon / MLX 实现。

NanoJev 是 Jev 的 0.6B 复刻：状态 + 问题 + 候选集输入，一次前向直接输出完整概率
分布，不做任何自回归解码。本包把原 PyTorch 实现移植到 MLX，使其在 Apple 芯片上
原生运行（Metal GPU），无需 CUDA。

属性按需惰性导入：这样 `python -m nanojev_mlx.predict` 不会因为包初始化而重复
加载子模块。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

_EXPORTS = {
    "DecisionHead": "head",
    "HeadConfig": "head",
    "DecisionModel": "model",
    "load_model": "model",
    "load_tokenizer": "model",
    "map_weights": "model",
    "run_prediction": "predict",
    "answer_from_probabilities": "predict",
    "prepare_examples": "text",
    "validate_request": "text",
}

if TYPE_CHECKING:  # pragma: no cover
    from .head import DecisionHead, HeadConfig
    from .model import DecisionModel, load_model, load_tokenizer, map_weights
    from .predict import answer_from_probabilities, run_prediction
    from .text import prepare_examples, validate_request

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return __all__
