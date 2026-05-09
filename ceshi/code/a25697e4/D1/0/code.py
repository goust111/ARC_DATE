# Variant of a25697e4
# Dimension: D1 — 匹配条件：形状匹配方式 (match_cond)
# Change: 工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积... → 颜色匹配替代形状匹配
# Design analysis:
# - The visible rule change is shifted from "工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积" to "颜色匹配替代形状匹配".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 不再根据形状吻合度匹配工具与框架，而是工具的plug颜色必须与框架颜色相同才能配对。凹槽大小固定相同，所有工具形状相同但颜色不同。解题者需从颜色而非几何推断配对关系，推理入口从空间分析变为颜色关联。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D1_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D1_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
