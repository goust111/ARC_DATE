# Variant of 36a08778
# Dimension: D7 — 终止条件 (dst)
# Change: 到达底边才停止... → 架子层数限制停止
#
# 设计要点：
# - 每条分支累计命中固定 3 层架子后停止：最后一层仍会画出水平扩散，但不再从两端继续下落。
# - 生成器强制至少存在更深的潜在级联，这样新旧规则在同一输入上会产生可见差异。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D7_3")
