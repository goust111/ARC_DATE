# Variant of 2b83f449
# Dimension: D8 — 洞(hole)的作用 (meta)
# Change: 管道行中的黑色单像素洞将管段分割为更短的独立片段，限制令牌水平移动范围 -> 洞由颜色标记替代

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D8", 2)
