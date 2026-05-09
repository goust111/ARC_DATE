# Variant of 800d221b
# Dimension: D17 — 干扰项与噪声 (meta)
# Change: 无干扰元素，所有非背景像素都有功能意义... → 添加背景色噪声像素
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "无干扰元素，所有非背景像素都有功能意义" to "添加背景色噪声像素".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 在路径附近放置少量与种子颜色相同但不属于任何种子区域的孤立像素。解题者需要正确判断邻接关系，不被噪声干扰。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D17_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D17_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
