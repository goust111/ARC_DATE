# Variant of a25697e4
# Dimension: D14 — 框架子组件关系 (meta)
# Change: 两个子组件颜色相同，其中一个是完整矩形（bar），另一个有阶梯形notch，间隔恰好1行或1列... → 两个子组件都有notch
# Design analysis:
# - The visible rule change is shifted from "两个子组件颜色相同，其中一个是完整矩形（bar），另一个有阶梯形notch，间隔恰好1行或1列" to "两个子组件都有notch".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 不再是一个完整矩形+一个有缺口的，而是两个子组件各有一个notch，凹槽由两个notch对向组合形成更复杂的形状。增加凹槽形状的不规则性。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D14_3", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D14_3"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
