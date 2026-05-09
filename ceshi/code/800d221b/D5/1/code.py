# Variant of 800d221b
# Dimension: D5 — 种子区域结构 (src)
# Change: 种子是矩形区域，内部由两种颜色混合填充（多数色+少数色），外围有1像素P色边框... → 种子为多颜色区域
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "种子是矩形区域，内部由两种颜色混合填充（多数色+少数色），外围有1像素P色边框" to "种子为多颜色区域".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 种子内部包含三种或更多颜色（而非仅两种），多数色判定需要在更多候选中选择。增加颜色计数的复杂度。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D5_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D5_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
