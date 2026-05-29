# Variant of 65b59efc
# Dimension: D16 — 多步处理：是否需要迭代或链式操作 (meta)
# Change: 单步印章 -> 先印章后擦除
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D16/1")
