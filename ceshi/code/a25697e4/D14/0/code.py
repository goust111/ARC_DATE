# Variant of a25697e4
# Dimension: D14 — 框架子组件关系 (meta)
# Change: 两个子组件颜色相同，其中一个是完整矩形（bar），另一个有阶梯形notch，间隔恰好1行或1列... → 子组件颜色不同
# Design analysis:
# - The visible rule change is shifted from "两个子组件颜色相同，其中一个是完整矩形（bar），另一个有阶梯形notch，间隔恰好1行或1列" to "子组件颜色不同".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 框架的两个子组件使用不同颜色（如上半蓝色、下半红色）。plug颜色需与'有notch的子组件'颜色匹配才能正确配对。增加颜色信息的利用维度。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D14_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D14_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
