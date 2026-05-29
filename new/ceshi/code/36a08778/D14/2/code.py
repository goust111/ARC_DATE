# Variant of 36a08778
# Dimension: D14 — 溢出点偏移规则 (meta)
# Change: 两端都作为溢出点... → 仅一侧溢出（交替左右）
#
# 设计要点：
# - 第一次撞架子后只从左侧溢出，第二次只从右侧，之后继续左右交替。
# - 交替规则跟随单条水流分支的命中次数，而不是跟全局行号绑定。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D14_2")
