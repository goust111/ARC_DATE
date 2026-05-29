# Variant of 2b83f449
# Dimension: D9 — 多步处理顺序 (multi_step)
# Change: 两步串行：先旋转所有门（in-place修改网格），再模拟所有令牌下落 -> 三步处理：旋转-下落-清除

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D9", 2)
