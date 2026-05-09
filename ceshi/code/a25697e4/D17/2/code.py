# Variant of a25697e4
# Dimension: D17 — 背景与颜色约束 (meta)
# Change: 背景为最常见颜色（0-9任意），使用3种非背景色（每对需frame_color, plug_color, handle_... → handle颜色与frame颜色相同
# Design analysis:
# - The visible rule change is shifted from "背景为最常见颜色（0-9任意），使用3种非背景色（每对需frame_color, plug_color, handle_color），两个框架可共享颜色" to "handle颜色与frame颜色相同".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: handle使用框架相同颜色，对接后handle与框架在视觉上融合。plug是唯一在凹槽中可见的不同色区域。改变了颜色信息的分布方式。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D17_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D17_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)
