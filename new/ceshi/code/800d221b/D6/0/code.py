# Variant of 800d221b
# Dimension: D6 — 分支与种子的匹配条件 (transform)
# Change: 每条分支着色为其物理接触的种子区域的多数颜色（按邻接计数匹配）... → 按颜色相似度匹配
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "每条分支着色为其物理接触的种子区域的多数颜色（按邻接计数匹配）" to "按颜色相似度匹配".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 分支不按物理邻接匹配种子，而是按种子多数颜色与分支路径上某种标记色的对应关系匹配。例如路径上有一个彩色标记像素指示该分支应取哪个种子的颜色。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D6_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D6_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
