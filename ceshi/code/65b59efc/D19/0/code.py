# Variant of 65b59efc
# Dimension: D19 — 调色板到模板的对应方式 (meta)
# Change: 位置对齐 -> 颜色匹配对应
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D19/0")
