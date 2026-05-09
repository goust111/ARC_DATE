# Variant of a25697e4
# Dimension: D2 — 匹配条件：旋转/翻转允许范围 (match_cond)
# Change: 允许完整D4群变换（4种旋转 × 2种翻转 = 最多8种朝向）来对齐工具plug与凹槽... → 限制为90度旋转倍数且需推断旋转角度
# Design analysis:
# - The visible rule change is shifted from "允许完整D4群变换（4种旋转 × 2种翻转 = 最多8种朝向）来对齐工具plug与凹槽" to "限制为90度旋转倍数且需推断旋转角度".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具可旋转但旋转角度由框架边缘的某种标记指示（如框架角落的彩色标记点数表示旋转90度的次数）。解题者需先解码标记再执行旋转，增加一层符号推理。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D2_3", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D2_3"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
