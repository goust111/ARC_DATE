# Variant of 71e489b6
# Dimension: D10 — 颜色方案 (meta)
# Change: 单色前景 -> 多色前景区域
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D10/0")
