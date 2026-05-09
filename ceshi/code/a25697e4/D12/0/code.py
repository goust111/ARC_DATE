# Variant of a25697e4
# Dimension: D12 — 工具颜色区域的拓扑约束 (meta)
# Change: plug和handle各为4-连通区域，两者之间共享1-2条边（inter-color boundary），handle... → plug包围handle（或handle包围plug）
# Design analysis:
# - The visible rule change is shifted from "plug和handle各为4-连通区域，两者之间共享1-2条边（inter-color boundary），handle在远离边缘方向延伸" to "plug包围handle（或handle包围plug）".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 一种颜色完全包围另一种颜色（如plug形成环形、handle在中心），而非简单相邻。凹槽形状需匹配外层颜色的形状。增加了嵌套结构的识别难度。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D12_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D12_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
