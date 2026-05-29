# Variant of 65b59efc
# Dimension: D4 — 输出网格的铺贴结构 (dst)
# Change: 固定方阵输出 -> 紧凑裁剪输出
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D4/1")
