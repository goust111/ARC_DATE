# Variant of 36a08778
# Dimension: D14 — 溢出点偏移规则 (meta)
# Change: 两端都作为溢出点... → 溢出点由入水位置决定方向
#
# 设计要点：
# - 从架子左半部分入水时只向左溢出，从右半部分入水时只向右溢出。
# - 只有正中入水时才保留双侧溢出，因此生成器会专门构造少量“正中命中”的样例。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D14_3")
