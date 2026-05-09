# Variant of a25697e4
# Dimension: D3 — 源对象：工具结构 (src)
# Change: 不接触边界的单个4-连通分量，恰好包含2种颜色（plug色+handle色），每种颜色区域4-9像素，总面积10-15像... → 工具含有空心洞
# Design analysis:
# - The visible rule change is shifted from "不接触边界的单个4-连通分量，恰好包含2种颜色（plug色+handle色），每种颜色区域4-9像素，总面积10-15像素" to "工具含有空心洞".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具内部包含1-2个背景色像素形成的洞（但仍为4-连通整体），对接时洞也被带入框架区域。解题者需处理非实心工具的形状匹配，增加拓扑复杂度。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D3_3", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D3_3"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
