# Variant of 36a08778
# Dimension: D2 — 架子碰撞后的扩散方式 (transform)
# Change: 双侧溢出... → 仅单侧溢出（最近边缘侧）
#
# 设计要点：
# - 每次撞到架子后，不再向两端都扩散，而是根据当前入水列离哪一侧边界更近，只向那一侧横向扩散并溢出。
# - 生成器同时构造靠左、靠右两类入水，以保证方向规则在样例中可观察。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D2_0")
