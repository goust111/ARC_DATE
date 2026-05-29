# Variant of a25697e4
# Dimension: D16 — 符号编码：变换指示 (symbol_instructions)
# Change: 无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定... → 框架子组件大小关系编码翻转
# Design analysis:
# - The visible rule change is shifted from "无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定" to "框架子组件大小关系编码翻转".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 两个子组件面积大小关系编码翻转信息：上（左）大下（右）小=不翻转，上（左）小下（右）大=需翻转。解题者需比较子组件面积来确定是否翻转。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D16_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D16_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
