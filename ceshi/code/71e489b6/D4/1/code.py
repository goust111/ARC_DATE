# Variant of 71e489b6
# Dimension: D4 — 边界检测：缺陷定位策略 (boundary_detection)
# Change: 邻域边界判断 -> 基于距离的检测
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D4/1")
