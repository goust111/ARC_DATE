# Variant of 65b59efc
# Dimension: D1 — 匹配条件：标记颜色到模板的映射方式 (match_cond)
# Change: 直接颜色匹配 -> 边缘颜色匹配
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D1/1")
