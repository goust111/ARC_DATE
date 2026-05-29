# Variant of 71e489b6
# Dimension: D1 — 光晕变换方式 (transform)
# Change: 固定局部光晕 -> 方向性射线标记
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D1/4")
