# Variant of a25697e4
# Dimension: D3 — 源对象：工具结构 (src)
# Change: 不接触边界的单个4-连通分量，恰好包含2种颜色（plug色+handle色），每种颜色区域4-9像素，总面积10-15像... → 三色工具
# Design analysis:
# - The visible rule change is shifted from "不接触边界的单个4-连通分量，恰好包含2种颜色（plug色+handle色），每种颜色区域4-9像素，总面积10-15像素" to "三色工具".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具包含3种颜色：plug色、handle色和中间'铰链'色（1-2像素连接plug与handle）。铰链色在对接后变为背景色消失，仅plug和handle保留。解题者需识别三色结构并推断铰链的消失规则。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D3_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D3_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
