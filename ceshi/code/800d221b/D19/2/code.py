# Variant of 800d221b
# Dimension: D19 — 符号编码参数 (symbol_instructions)
# Change: 无显式符号编码，着色规则隐含在空间结构中... → 路径上的标记像素
# Design analysis:
# - The implementation keeps a canonical hub-path-seed layout so the changed rule is isolated to one dimension.
# - The visible difference is the core behavior change from "无显式符号编码，着色规则隐含在空间结构中" to "路径上的标记像素".
# - Generator and solver use the same structural family and deliberately avoid adding unrelated mechanics.
# - Variant detail: 路径中间有一个非P色的标记像素，该标记颜色指示整条分支应着的颜色（覆盖邻接种子规则）。解题者需要识别路径内嵌的指令。
# - Hidden-assumption audit: branch routes, hub geometry, and seed slots are constructed to satisfy the solver's
#   expected non-overlap, in-bounds, and adjacency constraints on every sample.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D19_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D19_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
