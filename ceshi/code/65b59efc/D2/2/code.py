# Variant of 65b59efc
# Dimension: D2 — 局部坐标到输出块的映射方式 (match_cond)
# Change: 直接坐标映射 -> 旋转映射
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D2/2")
