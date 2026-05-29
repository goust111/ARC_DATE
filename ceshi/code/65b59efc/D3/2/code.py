# Variant of 65b59efc
# Dimension: D3 — 标记像素的编码方式 (src)
# Change: 模板色标记 -> 调色板色标记
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D3/2")
