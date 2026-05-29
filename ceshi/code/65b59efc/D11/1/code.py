# Variant of 65b59efc
# Dimension: D11 — 网格布局：三带结构的排列方式 (meta)
# Change: 模板带/调色板带交换位置
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D11/1")
