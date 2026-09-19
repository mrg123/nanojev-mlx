"""NanoJev Snake demo on MLX.

Reuses the original repository's Snake environment and code planner verbatim, so
the only thing that differs from the reference pipeline is the model backend.
"""

from .controller import MlxEngine, run_episode

__all__ = ["MlxEngine", "run_episode"]
