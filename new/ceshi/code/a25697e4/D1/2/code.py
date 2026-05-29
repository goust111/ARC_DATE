# Variant of a25697e4
# Dimension: D1 — 匹配条件：形状匹配方式 (match_cond)
# Change: 工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积... → 凹槽洞数计数匹配
# Design analysis:
# - The visible rule change is shifted from "工具的plug区域形状在D4群变换（旋转/翻转）下与框架凹槽形状一致，plug像素数等于凹槽面积" to "凹槽洞数计数匹配".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 每个框架的凹槽由多个不连通的小洞组成（而非单个阶梯形），工具上标有相应数量的标记像素（如红点）。工具与框架配对条件为标记数等于洞数。解题者需要计数两侧的独立区域/标记数并匹配。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D1_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D1_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
