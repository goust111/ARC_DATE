# Variant of a25697e4
# Dimension: D1 — 匹配条件：形状匹配方式 (match_cond)
# Change: 工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积... → 边缘颜色匹配
# Design analysis:
# - The visible rule change is shifted from "工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积" to "边缘颜色匹配".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 框架的两个子组件用不同颜色绘制（上半和下半各一种颜色），工具的handle颜色必须与框架某子组件颜色一致才能配对。解题者需同时观察框架的颜色分区和工具的handle颜色。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

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
