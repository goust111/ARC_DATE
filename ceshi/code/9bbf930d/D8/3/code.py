# Variant of 9bbf930d
# Dimension: D8 — 障碍物：条对结构 (obstacle)
# Change: 彩色条水平铺设在偶数行，从bar_start列延伸到网格右端；真走廊由同色条对构成，假走廊（干扰项）由异色条对构成。原版本主要靠条带/转角墙触发转向；本变体额外在走廊通道内放置孤立彩色阻挡块，粒子撞到这些单格障碍时也会转弯或停止。

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}

"""
design_analysis:
  concrete_variant:
    geometry:
      - 仍然保持条纹布局: 偶数行为彩条行, 奇数行为水平通道行, 第0列全是粒子, 第1列全是背景.
      - 真走廊仍由上下同色条对夹出, 假走廊仍由上下异色条对夹出.
      - 新对象是“孤立阻挡块”: 放在通道中的单个彩色格, 不与相邻四格中的同色格连通, 视觉上明显区别于长条.
      - 阻挡块可以出现在:
        1. 水平通道内, 让粒子提前停下.
        2. 水平通道拐弯口前方, 让粒子因撞块而转入竖直走廊.
        3. 竖直通道内, 让粒子在竖直段中途停下.
    behavior_trace:
      original_small_example:
        - 粒子在奇数行向右走, 直到前方遇到彩色格.
        - 若当前位置的上/下邻格中恰有一个是“可进入的同色夹道”, 就90度转弯; 否则停止.
        - 原生成器里的内部终止很多依赖“上方或下方不再挖空”, 看起来像走廊自然结束.
      variant_small_example:
        - 仍然向右走, 但在原本连续的真走廊中间会看到一个单格彩色块.
        - 如果块前一格旁边恰有合法竖直走廊, 粒子撞块后转弯.
        - 如果块前一格旁边没有合法转弯, 粒子就在块前停止.
      key_difference:
        - 变化不在“如何判定可转弯”, 而在“什么东西会成为障碍”.
        - 旧任务里视觉上更像条带结构本身制造障碍; 新任务里单格块是独立对象, 人眼能直接看到“通道不再畅通”.
  complexity_calibration:
    - 不引入新的颜色映射或多类别规则; 阻挡块仍然只靠“它是彩色、不是背景”这一可见属性起作用.
    - 阻挡块只分两种可见效果: 触发转弯, 或触发停止. 这两个效果都复用原本唯一合法转弯规则, 不增加求解分支爆炸.
    - 至少保留多个真实走廊和假走廊, 并让多个样例反复出现单格块, 使变体在2-4个样例中可归纳.
  implicit_assumption_audit_main:
    - _has_corridor 假设“合法转弯入口格必须是BG/粒子且两侧同色”. 变体不改这一点, 所以孤立块只需要充当前方障碍, 不需要自己编码方向.
    - _trace 原本把“任何非BG/粒子”都视作障碍. 这正好覆盖单格块; 需要避免生成器放出会导致两个垂直方向都可转的歧义局面.
    - main 的入口检测假设真实通道起点仍然由最左侧同色条对定义. 变体不能在入口左侧乱放块, 否则会把“哪几行是可追踪真实走廊”弄乱.
    - main 把输出写到最终停留格, 默认该格原本不是粒子. 变体仍成立, 因为阻挡块只放在前方障碍格而非最终停留格.
  placement_legality:
    - 水平停止块只放在真实奇数行的背景格上, 并避开任何竖直缺口列, 从而保证撞块时没有意外转弯.
    - 拐弯块放在原本的转角前方一格, 与上下条带颜色尽量不同, 保持4邻域同色孤立.
    - 竖直停止块只放在竖直通道经过的奇数行上, 这样它是单格块, 且粒子撞到后左右都没有合法水平走廊.
    - 竖直通道列仍用窗口保留法避免重叠, 各种块都不越界.
  visual_observability:
    - 单格块被刻意做成与紧邻条带不同色, 因此肉眼可区分“长条”和“孤立障碍”.
    - 多个真实通道会包含这种块, 且有的导致转弯、有的导致内部停止, 规则不是偶然个例.
  visual_cleanliness:
    - 仍保持偶数行长条 + 奇数行通道的整齐结构, 只在通道内添加少量单格块.
    - 单格块数量受控, 不会把整个网格填得杂乱.
  color_inference:
    - 不使用颜色id排序或隐藏映射.
    - 任意非背景彩色单格都只扮演“障碍”角色, 颜色本身不承载额外语义.
  solver_generator_contract:
    - solver 的新隐式要求是: 生成器必须保证每个孤立阻挡事件在当前位置最多只有一个合法垂直/水平转向.
    - generator 通过保留竖直走廊窗口、挑选安全停止列、以及最终调用 main 验证, 来保证这些假设成立.
"""


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid real-corridor entrance column for row r."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None
    if int(grid[r, 0]) != PARTICLE:
        return None
    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            entrance_col = c
            break
    if entrance_col is None:
        return None
    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == color), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == color), W)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _is_singleton_block(grid: np.ndarray, r: int, c: int) -> bool:
    """True for a colored obstacle cell with no same-color 4-neighbor."""
    color = int(grid[r, c])
    if color in {BG, PARTICLE}:
        return False
    H, W = grid.shape
    for dr, dc in _DELTAS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == color:
            return False
    return True


def _trace_with_events(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], list[dict[str, object]]]:
    """Trace from (r,c) and record which blockers were hit."""
    H, W = grid.shape
    direction = "right"
    events: list[dict[str, object]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        event = {
            "blocker": (nr, nc),
            "singleton": _is_singleton_block(grid, nr, nc),
            "turned": bool(options),
            "incoming": direction,
        }
        events.append(event)
        if not options:
            break
        r, c, direction = options[0]
    return (r, c), events


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    return _trace_with_events(grid, r, c)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through same-color corridors; any colored blocker can redirect or stop them."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

# Turn / obstacle probabilities
_STRAIGHT_PROB = 0.15
_STRAIGHT_BLOCK_PROB = 0.45
_L_BLOCKED_PROB = 0.45
_U_INTERNAL_STOP_PROB = 0.80

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)
_FAKE_BLOCK_RANGE = (0, 1)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_SINGLETON_EVENTS = 2
_MIN_SINGLETON_STOPS = 1
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _local_forbidden_colors(g: np.ndarray, r: int, c: int) -> set[int]:
    H, W = g.shape
    forbidden = {BG, PARTICLE}
    for dr, dc in _DELTAS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            forbidden.add(int(g[nr, nc]))
    return forbidden


def _paint_single_block(g: np.ndarray, r: int, c: int, extras: list[int], avoid: set[int] | None = None) -> None:
    forbidden = _local_forbidden_colors(g, r, c)
    if avoid:
        forbidden.update(avoid)
    candidates = [color for color in extras if color not in forbidden]
    if not candidates:
        candidates = [color for color in extras if color not in {BG, PARTICLE}]
    g[r, c] = random.choice(candidates)


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if random.random() < 0.5:
                plans[j] = {
                    "kind": "u",
                    "target": i,
                    "direction": "up",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[j] = {"kind": "u_target"}


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            kind = "straight_block" if random.random() < _STRAIGHT_BLOCK_PROB else "straight"
            plans[i] = {"kind": kind}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            kind = "straight_block" if random.random() < _STRAIGHT_BLOCK_PROB else "straight"
            plans[i] = {"kind": kind}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_blocked" if random.random() < _L_BLOCKED_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _promote_blocker_plans(real_indices: list[int], plans: dict[int, dict[str, object]]) -> None:
    def _is_blocker_plan(plan: dict[str, object]) -> bool:
        return plan["kind"] in {"straight_block", "l_blocked"} or (
            plan["kind"] == "u" and bool(plan.get("internal_stop"))
        )

    blocker_count = sum(1 for i in real_indices if _is_blocker_plan(plans[i]))
    if blocker_count >= 2:
        return
    for i in real_indices:
        plan = plans[i]
        if plan["kind"] == "straight":
            plan["kind"] = "straight_block"
            blocker_count += 1
        elif plan["kind"] == "l_edge":
            plan["kind"] = "l_blocked"
            blocker_count += 1
        elif plan["kind"] == "u" and not plan.get("internal_stop"):
            plan["internal_stop"] = True
            blocker_count += 1
        if blocker_count >= 2:
            return


def _safe_horizontal_stop_columns(g: np.ndarray, row: int, top: int, bot: int, bar_start: int, W: int) -> list[int]:
    cols = []
    for c_stop in range(bar_start + 4, W):
        if int(g[row, c_stop]) != BG:
            continue
        prev_c = c_stop - 1
        if prev_c < 0:
            continue
        if int(g[top, prev_c]) == BG or int(g[bot, prev_c]) == BG:
            continue
        cols.append(c_stop)
    return cols


def _add_horizontal_stop_block(
    g: np.ndarray,
    row: int,
    top: int,
    bot: int,
    bar_start: int,
    W: int,
    extras: list[int],
) -> bool:
    cols = _safe_horizontal_stop_columns(g, row, top, bot, bar_start, W)
    if not cols:
        return False
    c_stop = random.choice(cols)
    _paint_single_block(g, row, c_stop, extras, {int(g[top, c_stop]), int(g[bot, c_stop])})
    return True


def _vertical_block_rows(H: int, top: int, bot: int, direction: str) -> list[int]:
    if direction == "up":
        return [r for r in range(1, top, 2)]
    return [r for r in range(bot + 1, H - 1, 2)]


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    extras: list[int],
    blocked: bool,
) -> None:
    _paint_single_block(g, row, c_v, extras, {bar_color[top], bar_color[bot]})

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    if direction == "up":
        for r_br in range(0, top, 2):
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            g[r_br, c_v - 1] = BG

    if blocked:
        candidates = _vertical_block_rows(H, top, bot, direction)
        if candidates:
            r_block = random.choice(candidates)
            _paint_single_block(g, r_block, c_v - 1, extras, {bar_color[top], bar_color[bot]})


def _build_u_turn(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    _paint_single_block(g, row, c_v, extras, {bar_color[top], bar_color[bot]})

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = bar_color[top]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != bar_color[top]]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            c_stop = random.randint(c_stop_min, c_stop_max)
            _paint_single_block(g, j_row, c_stop, extras, {bar_color[j_top], bar_color[j_bot]})


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1

        if plan["kind"] == "straight":
            continue
        if plan["kind"] == "straight_block":
            _add_horizontal_stop_block(g, row, top, bot, bar_start, W, extras)
            continue
        if plan["kind"] == "u_target":
            continue

        c_v = int(plan["c_v"])
        if plan["kind"] in {"l_edge", "l_blocked"}:
            _build_l_turn(
                g,
                H,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                bar_color,
                extras,
                plan["kind"] == "l_blocked",
            )
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _add_fake_decor(g: np.ndarray, bar_rows: list[int], channel_spec: list[str], bar_color: dict[int, int], bar_start: int, W: int, extras: list[int]) -> None:
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1

        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor > 0:
            cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
            for c_decor in cols:
                if int(g[row, c_decor]) == BG:
                    g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

        n_blocks = random.randint(*_FAKE_BLOCK_RANGE)
        safe_cols = [
            c
            for c in range(bar_start + 3, W)
            if int(g[row, c]) == BG and int(g[top, c - 1]) != BG and int(g[bot, c - 1]) != BG
        ]
        random.shuffle(safe_cols)
        for c_block in safe_cols[:n_blocks]:
            _paint_single_block(g, row, c_block, extras, {bar_color[top], bar_color[bot]})


def _collect_variant_stats(g: np.ndarray) -> dict[str, int]:
    H, _ = g.shape
    moved = 0
    singleton_events = 0
    singleton_stops = 0
    singleton_turns = 0
    rows_with_singleton = 0
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        (end_r, end_c), events = _trace_with_events(g, r, 0)
        if (end_r, end_c) != (r, 0):
            moved += 1
        row_has_singleton = False
        for event in events:
            if not bool(event["singleton"]):
                continue
            singleton_events += 1
            row_has_singleton = True
            if bool(event["turned"]):
                singleton_turns += 1
            else:
                singleton_stops += 1
        if row_has_singleton:
            rows_with_singleton += 1
    return {
        "moved": moved,
        "singleton_events": singleton_events,
        "singleton_stops": singleton_stops,
        "singleton_turns": singleton_turns,
        "rows_with_singleton": rows_with_singleton,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _promote_blocker_plans(real_indices, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)
    _add_fake_decor(g, bar_rows, channel_spec, bar_color, bar_start, W, extras)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    stats = _collect_variant_stats(g)
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["singleton_events"] < _MIN_SINGLETON_EVENTS:
        return None
    if stats["singleton_stops"] < _MIN_SINGLETON_STOPS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    if not np.any(out != g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
