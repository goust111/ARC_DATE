# Variant of 65b59efc
# Dimension: D8 — 调色板编码：颜色信息的读取方式 (symbol_instructions)
# Change: 直接单色样 -> 颜色对编码
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D8/0")
