# Variant of a25697e4
# Dimension: D5 — 目标对象：凹槽形状 (dst)
# Change: 阶梯形（staircase/upper-triangular）凹槽，1-3个像素，由间隙行/列与notch组合形成... → T形或十字形凹槽
# Design analysis:
# - The visible rule change is shifted from "阶梯形（staircase/upper-triangular）凹槽，1-3个像素，由间隙行/列与notch组合形成" to "T形或十字形凹槽".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 凹槽呈T形或十字形，plug也为对应形状。这类对称性更强的形状在旋转下的匹配更微妙——某些旋转角度看似吻合但实际不对齐。增加空间推理精度要求。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D5_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D5_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
