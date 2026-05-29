# Variant of 65b59efc
# Dimension: D20 — 输入冗余与干扰项 (meta)
# Change: 仅保留有效模板 -> 添加无用模板干扰
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D20/2")
