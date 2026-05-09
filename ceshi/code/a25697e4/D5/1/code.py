# Variant of a25697e4
# Dimension: D5 — 目标对象：凹槽形状 (dst)
# Change: 阶梯形（staircase/upper-triangular）凹槽，1-3个像素，由间隙行/列与notch组合形成... → 不规则多边形凹槽
# Design analysis:
# - The visible rule change is shifted from "阶梯形（staircase/upper-triangular）凹槽，1-3个像素，由间隙行/列与notch组合形成" to "不规则多边形凹槽".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 凹槽形状随机生成为任意4-连通区域（不限于阶梯形），面积3-6像素。形状更复杂多变，工具的plug需精确吻合不规则轮廓，增加了形状识别的难度。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

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
