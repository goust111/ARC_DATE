# Combo variant of 9bbf930d (index 9)
# combo_spec: D8/3+D6/3
# Name: 走廊内阻挡块+碰假走廊停止

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
      - 仍然使用原题的条纹布局: 偶数行为横条行, 奇数行为水平通道行, 第0列全是粒子, 第1列全是背景.
      - 真走廊仍由上下同色条对定义, 假走廊仍由上下异色条对充当干扰背景.
      - 新增对象是“走廊内孤立阻挡块”: 放在真实水平通道里, 自身是单格彩色像素, 视觉上区别于长条.
      - 融合核心不只是“有块”和“有假走廊”同时存在, 而是构造 trap_turn:
        水平粒子撞到单格块后被迫进入一个竖直口; 转角格本身仍是合法真走廊,
        但下一步进入的竖直口单元左右两侧是异色彩块, 因而该单元是“假走廊”并立刻终止.
    behavior_trace:
      original_small_example:
        - 粒子在真实水平通道内向右走.
        - 前方遇到彩色格时, 只检查上下两个90度转向入口中是否恰有一个是真走廊; 有则转, 无则停.
        - 异色通道只是背景噪声, 即使经过相关区域附近也不会因为“假”而停.
      combo_small_example:
        - 粒子在真实水平通道里先撞到一个单格阻挡块, 这一步复用原题的“前方有彩色格就尝试转弯”.
        - 它转入转角格后继续沿竖直方向走一步, 进入一个左右两侧颜色不匹配的单元.
        - 这个单元相对当前运动方向属于假走廊, 因此立即停住.
      key_difference:
        - D8/3 负责把“什么会触发转向”从条带墙扩展到孤立阻挡块.
        - D6/3 负责把“进入假走廊”从无语义噪声改成终点.
        - 两者协同后的可见模式是: 阻挡块决定是否会被导入假走廊陷阱, 假走廊不再只是摆设.
  complexity_calibration:
    - 计划族保持少量且重复: trap_turn, l_edge/l_blocked, straight/straight_block, 少量u-turn.
    - 每个样本至少包含2条 trap_turn, 保证融合机制不是偶然出现.
    - 仍保留若干非陷阱真实路径, 防止任务退化成“见块就停”这一单一现象.
  implicit_assumption_audit_main:
    - _find_entrance_col 仍假设活动入口在左侧起始几列是水平真走廊; generator 保证陷阱和补丁不会破坏入口段.
    - 原题 _trace 默认“走到BG就继续”; D6/3 破坏了这个前提, 因此现在每次成功移动后都必须重新分类当前格是 real/fake/none.
    - 转向判定仍只接受真走廊; 假走廊是终点而不是合法转向目标, 所以 _has_corridor 不能放宽.
    - 原题把任何非BG/粒子都视作障碍; 这对孤立阻挡块和条带墙都仍成立.
    - assert 依赖“同一时刻至多一个合法垂直/水平转向”; generator 通过非相邻真实通道与vcol窗口保留来保证.
  placement_legality:
    - 活动真实通道优先选成彼此不相邻, 这样 trap_turn 总能朝相邻 fake channel 开口, 不会污染另一条活动入口.
    - 所有竖直转向共用保留列分配器, 避免多个转向/陷阱在同一窗口重叠.
    - straight_block 只放在不会意外暴露竖直转口的位置.
    - trap_turn 的假口只放在转向后的第一格, 左右两侧染成异色, 中间保持BG, 不越界也不覆盖粒子终点.
  visual_observability:
    - 人类能直接看到“单格块挡住了水平通道”.
    - 输出中的终点常常停在一个很短的竖直口中, 且终点左右颜色明显不一致, 与原题的边缘终止/死路终止有稳定差别.
    - 多个样例会重复出现“块 -> 转角 -> 假口停住”的局部模板, 足以在2-4个样例中归纳.
  visual_cleanliness:
    - 假口只是一格宽的局部结构, 不会把整张图变成密集噪声.
    - fake decor 控制得很少, 保留原题条带秩序感.
  color_inference:
    - 只依赖可见的“同色/异色”关系, 不依赖颜色编号大小或排序.
    - 阻挡块颜色本身不编码方向; 它只是一个彩色障碍.
  solver_generator_contract:
    - trap_turn 的转角格必须是real, 其后第一步格必须是fake, 才能实现“被块逼进假走廊即停”.
    - fake mouth 的左右两侧必须都是彩色且不同; generator 直接构造这一点.
    - 任何会被trace访问到的阻挡事件都必须满足至多一个可转方向; 生成后再次用main和trace统计回验.
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


def _corridor_flanks(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int] | None:
    """Return the two flank colors perpendicular to direction, or None if absent."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        return int(grid[r - 1, c]), int(grid[r + 1, c])
    if c == 0 or c == W - 1:
        return None
    return int(grid[r, c - 1]), int(grid[r, c + 1])


def _corridor_kind(grid: np.ndarray, r: int, c: int, direction: str) -> str:
    """Classify a passable cell relative to the current travel direction."""
    flanks = _corridor_flanks(grid, r, c, direction)
    if flanks is None:
        return "none"
    a, b = flanks
    if a in {PARTICLE, BG} or b in {PARTICLE, BG}:
        return "none"
    return "real" if a == b else "fake"


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) is a real corridor cell flanked by same-color walls."""
    return _corridor_kind(grid, r, c, direction) == "real"


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


def _trace_with_events(
    grid: np.ndarray, r: int, c: int
) -> tuple[tuple[int, int], str, list[dict[str, object]]]:
    """Trace one particle and record blocker / fake-entry events."""
    H, W = grid.shape
    direction = "right"
    events: list[dict[str, object]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            return (r, c), "edge", events
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            kind = _corridor_kind(grid, r, c, direction)
            if kind == "fake":
                events.append({"type": "fake_entry", "pos": (r, c), "incoming": direction})
                return (r, c), "fake", events
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        event = {
            "type": "blocker",
            "blocker": (nr, nc),
            "singleton": _is_singleton_block(grid, nr, nc),
            "incoming": direction,
            "turned": bool(options),
        }
        if not options:
            events.append(event)
            return (r, c), "blocked", events
        r, c, direction = options[0]
        event["turn_to"] = direction
        event["turn_cell"] = (r, c)
        events.append(event)
        kind = _corridor_kind(grid, r, c, direction)
        if kind == "fake":
            events.append({"type": "fake_entry", "pos": (r, c), "incoming": direction})
            return (r, c), "fake", events
    return (r, c), "loop_guard", events


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, str]:
    (end_r, end_c), reason, _ = _trace_with_events(grid, r, c)
    return end_r, end_c, reason


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through real corridors with blockers and fake-corridor traps.

    Input structure:
    - bg=7 and particle=6. Column 0 is filled with particles, column 1 is background.
    - Even rows carry long colored bars. Odd rows are horizontal channels. Some neighboring bar pairs match
      (real corridors), others mismatch (fake corridors).
    - Real horizontal corridors may contain isolated colored blocker cells.

    Rule:
    - A particle starts from each odd row whose left entrance is a real horizontal corridor and moves right.
    - It keeps moving through background cells. Any colored cell ahead, including a singleton blocker, acts as
      an obstacle; when blocked, the particle may turn 90 degrees only into a perpendicular real corridor.
    - After every successful move, if the particle's current cell is flanked by mismatched colors relative to its
      current travel direction, it has entered a fake corridor and stops immediately.
    - Otherwise it stops at the grid edge or when a blocker offers no legal real-corridor turn.

    Key insight:
    - The fused behavior is local: isolated blockers redirect motion, while fake corridors are terminal. The
      important examples are the ones where a blocker forces a turn and the very next segment becomes a fake
      corridor mouth, so the particle stops inside a visibly mismatched side channel.
    """
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, _ = _trace(g, r, 0)
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
_N_REAL_OPTIONS = [3, 4, 5]
_N_REAL_WEIGHTS = [5, 3, 1]
_MIN_PALETTE_DIVERSITY = 3

# Plan mix
_MIN_TRAP_TURNS = 2
_MAX_TRAP_TURNS = 3
_STRAIGHT_PROB = 0.20
_STRAIGHT_BLOCK_PROB = 0.30
_L_BLOCKED_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.60

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 1)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_FORCED_FAKE_STOPS = 2
_MAX_RETRIES = 500


def _pick_real_channels(n_channels: int, n_real: int) -> set[int]:
    """Pick mostly non-adjacent real channels so trap mouths stay local."""
    n_real = min(n_real, (n_channels + 1) // 2)
    best: list[int] = []
    for _ in range(80):
        order = list(range(n_channels))
        random.shuffle(order)
        chosen: list[int] = []
        for idx in order:
            if all(abs(idx - prev) > 1 for prev in chosen):
                chosen.append(idx)
                if len(chosen) == n_real:
                    return set(chosen)
        if len(chosen) > len(best):
            best = chosen
    return set(best)


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical turn structure."""
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
    forbidden = {BG, PARTICLE}
    H, W = g.shape
    for dr, dc in _DELTAS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            forbidden.add(int(g[nr, nc]))
    return forbidden


def _paint_single_block(
    g: np.ndarray, r: int, c: int, extras: list[int], avoid: set[int] | None = None
) -> None:
    forbidden = _local_forbidden_colors(g, r, c)
    if avoid:
        forbidden.update(avoid)
    candidates = [color for color in extras if color not in forbidden]
    if not candidates:
        candidates = [color for color in extras if color not in {BG, PARTICLE}]
    g[r, c] = random.choice(candidates)


def _plan_trap_turns(
    real_indices: list[int],
    channel_spec: list[str],
    bar_rows: list[int],
    H: int,
    bar_start: int,
    W: int,
    used_vcols: set[int],
) -> dict[int, dict[str, object]] | None:
    """Choose channels whose singleton block forces a turn into an immediate fake mouth."""
    candidates: list[tuple[int, list[str]]] = []
    for i in real_indices:
        dirs: list[str] = []
        top, bot = bar_rows[i], bar_rows[i + 1]
        if i > 0 and channel_spec[i - 1] == "fake" and top >= 2:
            dirs.append("up")
        if i + 1 < len(channel_spec) and channel_spec[i + 1] == "fake" and bot + 2 < H:
            dirs.append("down")
        if dirs:
            candidates.append((i, dirs))
    if len(candidates) < _MIN_TRAP_TURNS:
        return None
    random.shuffle(candidates)
    target_count = min(_MAX_TRAP_TURNS, len(candidates))
    target_count = random.randint(_MIN_TRAP_TURNS, target_count)

    plans: dict[int, dict[str, object]] = {}
    for i, dirs in candidates:
        if len(plans) >= target_count:
            break
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            break
        plans[i] = {"kind": "trap_turn", "direction": random.choice(dirs), "c_v": c_v}
    if len(plans) < _MIN_TRAP_TURNS:
        return None
    return plans


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Pair same-color real channels into U-turn plans for some remaining rows."""
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
            if i in plans or j in plans:
                continue
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
    """Assign straight, blocked-straight, or L-turn plans to unplanned real channels."""
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


def _safe_horizontal_stop_columns(
    g: np.ndarray, row: int, top: int, bot: int, bar_start: int, W: int
) -> list[int]:
    cols = []
    for c_stop in range(bar_start + 4, W):
        if int(g[row, c_stop]) != BG:
            continue
        prev_c = c_stop - 1
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


def _pick_distinct_pair(preferred_a: int, preferred_b: int, extras: list[int]) -> tuple[int, int]:
    if preferred_a != preferred_b and preferred_a not in {BG, PARTICLE} and preferred_b not in {BG, PARTICLE}:
        return preferred_a, preferred_b
    for color in extras:
        if color != preferred_a:
            return preferred_a, color
    return preferred_a, preferred_b


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
    trap: bool,
) -> None:
    """Build an L-turn triggered by a singleton blocker at (row, c_v)."""
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

    if trap:
        mouth_r = turn_bar - 1 if direction == "up" else turn_bar + 1
        if direction == "up" and top >= 2:
            left_color, right_color = _pick_distinct_pair(bar_color[top], bar_color[top - 2], extras)
        elif direction == "down" and bot + 2 < H:
            left_color, right_color = _pick_distinct_pair(bar_color[bot], bar_color[bot + 2], extras)
        else:
            left_color, right_color = _pick_distinct_pair(bar_color[turn_bar], random.choice(extras), extras)
        g[mouth_r, c_v - 2] = left_color
        g[mouth_r, c_v] = right_color
        return

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
    """Build a U-turn with a singleton blocker at the source row."""
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
    """Render all planned real-channel structures."""
    for i, plan in plans.items():
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        kind = str(plan["kind"])

        if kind in {"straight", "u_target"}:
            continue
        if kind == "straight_block":
            _add_horizontal_stop_block(g, row, top, bot, bar_start, W, extras)
            continue

        c_v = int(plan["c_v"])
        if kind in {"trap_turn", "l_edge", "l_blocked"}:
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
                blocked=(kind == "l_blocked"),
                trap=(kind == "trap_turn"),
            )
            continue

        if kind == "u":
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
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


def _add_fake_decor(
    g: np.ndarray,
    bar_rows: list[int],
    channel_spec: list[str],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
) -> None:
    """Add a little noise inside fake channel rows without disturbing planned paths."""
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if int(g[row, c_decor]) == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])


def _collect_variant_stats(g: np.ndarray) -> dict[str, int]:
    H, _ = g.shape
    moved = 0
    fake_stops = 0
    forced_fake_stops = 0
    singleton_events = 0
    singleton_turns = 0
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        (end_r, end_c), reason, events = _trace_with_events(g, r, 0)
        if (end_r, end_c) != (r, 0):
            moved += 1
        if reason == "fake":
            fake_stops += 1
        row_singleton_turn = False
        for event in events:
            if event.get("type") != "blocker":
                continue
            if bool(event.get("singleton")):
                singleton_events += 1
            if bool(event.get("singleton")) and bool(event.get("turned")):
                singleton_turns += 1
                row_singleton_turn = True
        if reason == "fake" and row_singleton_turn:
            forced_fake_stops += 1
    return {
        "moved": moved,
        "fake_stops": fake_stops,
        "forced_fake_stops": forced_fake_stops,
        "singleton_events": singleton_events,
        "singleton_turns": singleton_turns,
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
    real_set = _pick_real_channels(n_channels, n_real)
    if len(real_set) < _MIN_MOVED:
        return None
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans = _plan_trap_turns(real_indices, channel_spec, bar_rows, H, bar_start, W, used_vcols)
    if plans is None:
        return None
    remaining = [i for i in real_indices if i not in plans]
    _plan_u_turns(remaining, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(remaining, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)
    _add_fake_decor(g, bar_rows, channel_spec, bar_color, bar_start, W)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    stats = _collect_variant_stats(g)
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["forced_fake_stops"] < _MIN_FORCED_FAKE_STOPS:
        return None
    if stats["singleton_turns"] < _MIN_FORCED_FAKE_STOPS:
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
