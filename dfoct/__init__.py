"""DFO-TR for CT reconstruction: the papers' method, a faster equivalent (v1.3) and the fork (v2.3)."""

from .problem import (
    HELDOUT_IMAGES, LOWER, SETTINGS, UPPER, Problem, Run, load_image_problem, load_problem, score, sirt_box,
)  # fmt: skip
from .search import Search, Settings, run

V1_3 = Settings()  # DFO-TR's search: uniform start, uniform restarts at rate 0.001
V2_3 = Settings(spread=16.0, radius=0.05, delta=0.003)  # the fork

METHODS = {"v1.3": V1_3, "v2.3": V2_3}

__all__ = [
    "HELDOUT_IMAGES", "LOWER", "METHODS", "SETTINGS", "UPPER", "V1_3", "V2_3", "Problem", "Run", "Search", "Settings",
    "load_image_problem", "load_problem", "run", "score", "sirt_box",
]  # fmt: skip
