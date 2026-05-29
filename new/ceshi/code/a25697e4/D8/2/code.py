# Variant of a25697e4
# Dimension: D8 — 空间关系：框架朝向与边缘绑定 (meta)
# Change: 每个框架的两个子组件都接触同一条网格边缘（上/下/左/右），凹槽开口朝向网格内部... → 框架接触对面边缘
# Design analysis:
# - The visible rule change is shifted from "每个框架的两个子组件都接触同一条网格边缘（上/下/左/右），凹槽开口朝向网格内部" to "框架接触对面边缘".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 当有2个框架时，它们必须接触对面的两条边缘（一个在左一个在右，或一个在上一个在下）。工具需穿越整个网格对接到对面框架，空间跨度大。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D8_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D8_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
