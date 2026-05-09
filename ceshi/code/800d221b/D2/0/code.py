# Variant of 800d221b
# Dimension: D2 — hub中心像素着色规则 (transform)
# Change: hub中心像素取与hub边界相邻的重着色分支中出现最多的颜色... → hub中心取最少相邻颜色
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "hub中心像素取与hub边界相邻的重着色分支中出现最多的颜色" to "hub中心取最少相邻颜色".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 取反：hub中心像素取与hub边界相邻重着色分支中出现最少的颜色。解题者需要计数后取最小值而非最大值，改变推理方向。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D2_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D2_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
