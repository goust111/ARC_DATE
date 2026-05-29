# Variant of a25697e4
# Dimension: D4 — 目标对象：框架结构 (dst)
# Change: 由两个同色4-连通子组件组成的边缘框架，一个子组件是完整矩形（bar），另一个有阶梯形缺口（notch），之间间隔1行/... → 单组件带矩形缺口的框架
# Design analysis:
# - The visible rule change is shifted from "由两个同色4-连通子组件组成的边缘框架，一个子组件是完整矩形（bar），另一个有阶梯形缺口（notch），之间间隔1行/列形成凹槽" to "单组件带矩形缺口的框架".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 框架为单个L形或U形连通组件（而非两个分离子组件），凹槽是框架体内挖去的矩形区域。解题者需识别单一组件内的凹槽而非两组件间的间隙，改变了凹槽定位的推理方式。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D4_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D4_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
