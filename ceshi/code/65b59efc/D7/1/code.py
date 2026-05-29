# Variant of 65b59efc
# Dimension: D7 — 模板结构：形状特性 (meta)
# Change: 单色模板 -> 多色模板
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D7/1")
