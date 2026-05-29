# Variant of 65b59efc
# Dimension: D18 — 输出构建：空块的填充方式 (meta)
# Change: 空块留白 -> 默认图案填充
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D18/0")
