# Variant of 800d221b
# Dimension: D18 — 每个种子的路径数量 (meta)
# Change: 每个种子连接1-3条独立路径到hub... → 种子间共享路径
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "每个种子连接1-3条独立路径到hub" to "种子间共享路径".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 两个种子可以共享一段路径（路径的一部分同时邻接两个种子），该共享段的着色取决于两个种子中面积更大者的多数色。解题者需要处理路径归属冲突。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D18_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D18_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
