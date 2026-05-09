# Variant of 800d221b
# Dimension: D1 — 分支着色规则 (transform)
# Change: 每条分支重着色为其所接触种子区域的多数颜色(majority color)... → 种子面积比例混合着色
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "每条分支重着色为其所接触种子区域的多数颜色(majority color)" to "种子面积比例混合着色".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 分支不再统一着色为一种颜色，而是按种子内部两种颜色的比例交替着色分支像素（例如多数色占70%，则分支70%像素为多数色、30%为少数色，沿路径方向交替排列）。解题者需要计算比例并正确分配，增加数值推理难度。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D1_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D1_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
