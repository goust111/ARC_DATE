# Combo variant of 9bbf930d (index 38)
# combo_spec: D10/1+D6/3+D8/3
# Name: 假走廊陷阱+碰假走廊停止+走廊内阻挡块

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
  a_concrete_behavior_and_trace:
    - 保持原题的条纹骨架: 偶数行为横条, 奇数行为水平通道, 第0列全是粒子, 第1列全是背景.
    - 真实走廊仍由同色条对夹出, 假走廊仍由异色条对夹出, 但假走廊不再只是装饰.
    - 新对象是走廊内的孤立彩色阻挡块. 粒子撞到任意彩色格都会触发“尝试向垂直方向找走廊”的原始机制.
    - 融合后的小例子:
      1. 原题: 某粒子沿真实水平走廊向右走, 在 c=9 前方遇到彩色墙, 上方若只有异色夹口则它不会转入, 而是停在 c=8.
      2. D8/3 部分: 现在 c=9 的障碍可以是单格阻挡块而不是整段条带, 粒子一样会在 c=8 处尝试转弯.
      3. D10/1 + D6/3 融合: 如果 c=8 的唯一可转入口是“左右异色”的假竖直口, 粒子会被这个入口骗进去, 并且一进入该假走廊口就立刻停住.
    - 因此输出差异不是“是否撞到块”, 而是“撞块后进入真走廊继续走, 还是被逼进假走廊并停在入口格”.
  b_complexity_calibration:
    - 不引入新的颜色映射, 只使用人眼可见的两个关系: 单格彩色块=障碍, 两侧同色/异色=真/假走廊.
    - 生成器同时制造三类可重复事件:
      1. 单格阻挡块触发真转弯;
      2. 单格阻挡块触发假走廊陷阱;
      3. 单格阻挡块直接造成停止.
    - 这样样例不会简化成“看到假走廊就停”或“看到块就停”这两种过弱规则, 而是保留原题的转弯判定结构.
  c_implicit_assumption_audit:
    - 原 main() 默认“所有可走的 BG 都没有额外语义”; 这里失效, 因为进入假走廊格要立刻停止.
    - 原 _trace 只需要区分“有真转弯 / 没有真转弯”; 这里需要区分“真转弯 / 假转弯陷阱 / 完全无转弯”.
    - 原 generator 主要靠横条本身做阻挡; 这里要保证单格块也能稳定触发且不会制造双向歧义.
    - 原入口检测依赖起始列附近是真水平走廊; 融合后仍保留, 因此生成器不会在入口附近放块或改成假入口.
  d_placement_legality:
    - 所有竖直结构继续用保留窗口分配列, 避免重叠.
    - 单格阻挡块只放在当前为 BG 的通道格, 并尽量避开同色邻居, 使其保持“孤立块”外观.
    - 假陷阱口与真实 L 转角共用同样的几何模板: 一侧开口, 对侧封死, 从而保证每次撞块时最多只有一个垂直候选.
    - 水平停止块只放在不会意外露出真/假转入口的列上, 防止本应停止的粒子误转弯.
  e_visual_observability:
    - 人类可直接观察到: 有些粒子停在普通通道中途, 但另一些粒子会停在“异色夹口”的拐弯入口上.
    - 多组样例里同一种单格块会有两种后果: 逼入真走廊继续走, 或逼入假走廊立刻停. 这正是融合规则的核心对比.
  f_visual_cleanliness:
    - 仍保持规则的横条+通道版式, 只在少量通道位置加入单格块和少量假入口.
    - 真实竖直转弯和假陷阱入口都使用统一的 L 形几何, 视觉上容易并排比较.
  g_inferability:
    - 颜色值大小不参与任何排序或隐藏逻辑.
    - 唯一需要归纳的是“条对同色=真走廊可继续, 条对异色=假走廊且一进入就停”, 以及“单格彩色块会触发这一判定”.
  h_variant_metric:
    - 统计 fake_traps: 被单格阻挡块逼入假走廊并停在入口的次数.
    - 统计 singleton_real_turns: 被单格阻挡块逼入真走廊继续前进的次数.
    - 生成器要求 fake_traps >= 2 且 singleton_real_turns >= 1, 并且输出必须区别于“只认单格块、不认假走廊陷阱”的基线求解器.
  i_solver_generator_contract:
    - solver 的新假设 1: 每次撞到障碍时, 真口和假口加起来至多一个. 生成器通过封死对侧和列窗口保留来保证.
    - solver 的新假设 2: 起步行必须先是水平真走廊. 生成器保证假结构不会破坏 bar_start 附近的真实入口.
    - solver 的新假设 3: 假走廊停止发生在“当前移动方向的垂直两侧均为彩色且异色”时. 生成器只在内部列构造这种格子, 避免边界歧义.
"""


def _corridor_flanks(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int] | None:
    """Return the two cells flanking (r,c) perpendicular to direction."""
    h, w = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return None
        return int(grid[r - 1, c]), int(grid[r + 1, c])
    if c == 0 or c == w - 1:
        return None
    return int(grid[r, c - 1]), int(grid[r, c + 1])


def _corridor_kind(grid: np.ndarray, r: int, c: int, direction: str) -> str:
    """Classify a passable cell as a real corridor, fake corridor, or neither."""
    flanks = _corridor_flanks(grid, r, c, direction)
    if flanks is None:
        return "none"
    a, b = flanks
    if a in {PARTICLE, BG} or b in {PARTICLE, BG}:
        return "none"
    return "real" if a == b else "fake"


def _has_real_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    return _corridor_kind(grid, r, c, direction) == "real"


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first genuine horizontal entrance for row r, if any."""
    h, w = grid.shape
    if not (1 <= r < h - 1) or int(grid[r, 0]) != PARTICLE:
        return None
    for c in range(2, w):
        if int(grid[r, c]) != BG:
            continue
        if _corridor_kind(grid, r, c, "right") != "real":
            continue
        color = int(grid[r - 1, c])
        bar_above = next((k for k in range(w) if int(grid[r - 1, k]) == color), w)
        bar_below = next((k for k in range(w) if int(grid[r + 1, k]) == color), w)
        if bar_above == bar_below and bar_above <= 3:
            return c
    return None


def _is_singleton_block(grid: np.ndarray, r: int, c: int) -> bool:
    """True for a colored obstacle cell with no same-color 4-neighbor."""
    color = int(grid[r, c])
    if color in {BG, PARTICLE}:
        return False
    h, w = grid.shape
    for dr, dc in _DELTAS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
            return False
    return True


def _trace_state(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    allow_fake_turn: bool,
    stop_on_fake: bool,
) -> dict[str, object]:
    """Trace one particle and record why it stopped."""
    h, w = grid.shape
    direction = "right"
    events: list[dict[str, object]] = []
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return {
                "end": (r, c),
                "direction": direction,
                "stop_reason": "edge",
                "stop_context": "move",
                "events": events,
            }

        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            if stop_on_fake and _corridor_kind(grid, r, c, direction) == "fake":
                events.append(
                    {
                        "type": "fake_entry",
                        "at": (r, c),
                        "incoming": direction,
                        "via": "move",
                    }
                )
                return {
                    "end": (r, c),
                    "direction": direction,
                    "stop_reason": "fake",
                    "stop_context": "move",
                    "events": events,
                }
            continue

        real_options = []
        fake_options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            kind = _corridor_kind(grid, tr, tc, turn_dir)
            if kind == "real":
                real_options.append((tr, tc, turn_dir))
            elif kind == "fake":
                fake_options.append((tr, tc, turn_dir))

        total_options = len(real_options) + len(fake_options)
        assert total_options <= 1, "ambiguous turn: multiple perpendicular mouths"

        event = {
            "type": "blocker",
            "blocker": (nr, nc),
            "incoming": direction,
            "singleton": _is_singleton_block(grid, nr, nc),
            "outcome": "stop",
        }

        if real_options:
            r, c, direction = real_options[0]
            event["outcome"] = "real_turn"
            event["turn_dir"] = direction
            events.append(event)
            continue

        if fake_options and allow_fake_turn:
            r, c, direction = fake_options[0]
            event["outcome"] = "fake_trap"
            event["turn_dir"] = direction
            event["trap_at"] = (r, c)
            events.append(event)
            return {
                "end": (r, c),
                "direction": direction,
                "stop_reason": "fake",
                "stop_context": "turn",
                "events": events,
            }

        events.append(event)
        return {
            "end": (r, c),
            "direction": direction,
            "stop_reason": "blocked",
            "stop_context": "blocker",
            "events": events,
        }

    return {
        "end": (r, c),
        "direction": direction,
        "stop_reason": "loop_guard",
        "stop_context": "guard",
        "events": events,
    }


def _solve_with_trace(grid: np.ndarray, *, allow_fake_turn: bool, stop_on_fake: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    out = g.copy()
    for r in range(1, h - 1):
        if _find_entrance_col(g, r) is None:
            continue
        state = _trace_state(g, r, 0, allow_fake_turn=allow_fake_turn, stop_on_fake=stop_on_fake)
        end_r, end_c = state["end"]
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through striped corridors with fake-trap mouths and blocker-driven turns.

    Input structure:
    - bg=7 and particles=6 fill col 0; col 1 is bg; even rows are colored bars and odd rows are
      horizontal channels.
    - Real corridors are flanked by same-color bars; fake corridors are flanked by different colors.
    - Some channels contain isolated colored blocker cells that act as obstacles inside the corridor.

    Rule:
    - A particle starts on a row whose entrance is a real horizontal corridor and moves right through bg.
    - After each move, if the particle is now inside a fake corridor cell relative to its travel
      direction, it stops immediately.
    - When the next cell is colored, the particle tries the two perpendicular directions. A real
      corridor mouth lets it turn and continue. If the unique available mouth is fake, the particle
      enters that fake mouth and stops there. If no mouth is available, it stops in place.

    Key insight:
    - Single-cell blockers do not determine the destination by themselves; they only force a turn
      attempt. The final behavior depends on whether the forced mouth is a true same-color corridor
      or a fake mismatched corridor that functions as a trap.
    """
    return _solve_with_trace(grid, allow_fake_turn=True, stop_on_fake=True)


def _main_blocks_only(grid: np.ndarray) -> np.ndarray:
    """Baseline used in generator validation: singleton blocks exist, fake traps do not."""
    return _solve_with_trace(grid, allow_fake_turn=False, stop_on_fake=False)


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
_TRAP_COUNT_RANGE = (2, 3)
_STRAIGHT_PROB = 0.15
_STRAIGHT_BLOCK_PROB = 0.35
_L_BLOCKED_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.70

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_SINGLETON_EVENTS = 3
_MIN_FAKE_TRAPS = 2
_MIN_REAL_BLOCK_TURNS = 1
_MAX_RETRIES = 500


def _pick_real_channels(n_channels: int, n_real: int) -> set[int]:
    """Pick mostly non-adjacent real channels so local edits stay isolated."""
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


def _allocate_vcol(bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = w - _VCOL_RIGHT_PAD
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
    h, w = g.shape
    forbidden = {BG, PARTICLE}
    for dr, dc in _DELTAS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            forbidden.add(int(g[nr, nc]))
    return forbidden


def _paint_single_block(
    g: np.ndarray,
    r: int,
    c: int,
    extras: list[int],
    avoid: set[int] | None = None,
) -> None:
    forbidden = _local_forbidden_colors(g, r, c)
    if avoid:
        forbidden.update(avoid)
    candidates = [color for color in extras if color not in forbidden]
    if not candidates:
        candidates = [color for color in extras if color not in {BG, PARTICLE}]
    g[r, c] = random.choice(candidates)


def _choose_trap_channels(real_indices: list[int]) -> list[int]:
    if len(real_indices) <= _TRAP_COUNT_RANGE[0]:
        return real_indices[:]
    hi = min(_TRAP_COUNT_RANGE[1], len(real_indices) - 1)
    k = random.randint(_TRAP_COUNT_RANGE[0], hi)
    return random.sample(real_indices, k)


def _plan_traps(
    trap_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    for i in trap_indices:
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            continue
        plans[i] = {"kind": "trap", "direction": random.choice(["up", "down"]), "c_v": c_v}


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    w: int,
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
            if i in plans or j in plans:
                continue
            c_v = _allocate_vcol(bar_start, w, used_vcols)
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
    w: int,
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
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            kind = "straight_block" if random.random() < _STRAIGHT_BLOCK_PROB else "straight"
            plans[i] = {"kind": kind}
            continue
        kind = "l_blocked" if random.random() < _L_BLOCKED_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": random.choice(["up", "down"]), "c_v": c_v}


def _promote_supporting_blockers(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Ensure at least one non-trap singleton block causes a real turn or stop."""
    blocker_support = 0
    for i in real_indices:
        kind = str(plans[i]["kind"])
        if kind in {"trap", "straight_block", "l_blocked"}:
            blocker_support += 1
        elif kind == "u" and bool(plans[i].get("internal_stop", False)):
            blocker_support += 1
    if blocker_support >= 3:
        return
    for i in real_indices:
        kind = str(plans[i]["kind"])
        if kind == "straight":
            plans[i] = {"kind": "straight_block"}
            blocker_support += 1
        elif kind == "l_edge":
            plans[i]["kind"] = "l_blocked"
            blocker_support += 1
        elif kind == "u" and not bool(plans[i].get("internal_stop", False)):
            plans[i]["internal_stop"] = True
            blocker_support += 1
        elif kind == "u_target":
            continue
        elif kind not in {"trap", "straight_block", "l_blocked"}:
            c_v = _allocate_vcol(bar_start, w, used_vcols)
            if c_v is not None:
                plans[i] = {"kind": "l_blocked", "direction": random.choice(["up", "down"]), "c_v": c_v}
                blocker_support += 1
        if blocker_support >= 3:
            return


def _safe_horizontal_stop_columns(
    g: np.ndarray,
    row: int,
    top: int,
    bot: int,
    bar_start: int,
    w: int,
) -> list[int]:
    cols = []
    for c_stop in range(bar_start + 4, w):
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
    w: int,
    extras: list[int],
) -> bool:
    cols = _safe_horizontal_stop_columns(g, row, top, bot, bar_start, w)
    if not cols:
        return False
    c_stop = random.choice(cols)
    _paint_single_block(g, row, c_stop, extras, {int(g[top, c_stop]), int(g[bot, c_stop])})
    return True


def _vertical_block_rows(h: int, top: int, bot: int, direction: str) -> list[int]:
    if direction == "up":
        return [r for r in range(1, top, 2)]
    return [r for r in range(bot + 1, h - 1, 2)]


def _build_real_l_turn(
    g: np.ndarray,
    h: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    extras: list[int],
    blocked_vertical: bool,
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
        for r_br in range(bot + 2, h, 2):
            g[r_br, c_v - 1] = BG

    if blocked_vertical:
        candidates = _vertical_block_rows(h, top, bot, direction)
        if candidates:
            r_block = random.choice(candidates)
            _paint_single_block(g, r_block, c_v - 1, extras, {bar_color[top], bar_color[bot]})


def _build_u_turn(
    g: np.ndarray,
    w: int,
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
        c_stop_min, c_stop_max = c_v + 3, w - 1
        if c_stop_min <= c_stop_max:
            c_stop = random.randint(c_stop_min, c_stop_max)
            _paint_single_block(g, j_row, c_stop, extras, {bar_color[j_top], bar_color[j_bot]})


def _build_fake_trap(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    extras: list[int],
    trap_color: int,
) -> None:
    _paint_single_block(g, row, c_v, extras, {bar_color[top], bar_color[bot], trap_color})
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = trap_color
    g[opp_bar, c_v - 1] = bar_color[opp_bar]


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        kind = str(plan["kind"])

        if kind == "straight":
            continue
        if kind == "straight_block":
            _add_horizontal_stop_block(g, row, top, bot, bar_start, w, extras)
            continue
        if kind == "u_target":
            continue

        c_v = int(plan["c_v"])
        if kind in {"l_edge", "l_blocked"}:
            _build_real_l_turn(
                g,
                h,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                bar_color,
                extras,
                kind == "l_blocked",
            )
        elif kind == "u":
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                w,
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
        elif kind == "trap":
            trap_choices = [c for c in extras if c != color]
            if not trap_choices:
                trap_choices = [c for c in range(10) if c not in {BG, PARTICLE, color}]
            _build_fake_trap(
                g,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                bar_color,
                extras,
                random.choice(trap_choices),
            )


def _add_fake_decor(
    g: np.ndarray,
    bar_rows: list[int],
    channel_spec: list[str],
    bar_color: dict[int, int],
    bar_start: int,
    w: int,
) -> None:
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w), min(n_decor, w - bar_start - 1))
        for c_decor in cols:
            if int(g[row, c_decor]) == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])


def _collect_variant_stats(grid: np.ndarray) -> dict[str, int]:
    h, _ = grid.shape
    stats = {
        "moved": 0,
        "fake_stops": 0,
        "fake_traps": 0,
        "singleton_events": 0,
        "singleton_stops": 0,
        "singleton_real_turns": 0,
        "singleton_fake_traps": 0,
    }
    for r in range(1, h - 1):
        if _find_entrance_col(grid, r) is None:
            continue
        state = _trace_state(grid, r, 0, allow_fake_turn=True, stop_on_fake=True)
        end_r, end_c = state["end"]
        if (end_r, end_c) != (r, 0):
            stats["moved"] += 1
        if state["stop_reason"] == "fake":
            stats["fake_stops"] += 1
            if state["stop_context"] == "turn":
                stats["fake_traps"] += 1
        for event in state["events"]:
            if event.get("type") != "blocker" or not bool(event.get("singleton")):
                continue
            stats["singleton_events"] += 1
            outcome = str(event.get("outcome"))
            if outcome == "stop":
                stats["singleton_stops"] += 1
            elif outcome == "real_turn":
                stats["singleton_real_turns"] += 1
            elif outcome == "fake_trap":
                stats["singleton_fake_traps"] += 1
    return stats


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((h, w), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, h, 2))
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
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    trap_indices = _choose_trap_channels(real_indices)
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_traps(trap_indices, bar_start, w, used_vcols, plans)
    if sum(1 for plan in plans.values() if str(plan["kind"]) == "trap") < _MIN_FAKE_TRAPS:
        return None
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, w, used_vcols, plans)
    _promote_supporting_blockers(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, extras)
    _add_fake_decor(g, bar_rows, channel_spec, bar_color, bar_start, w)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
        baseline = _main_blocks_only(g)
    except Exception:
        return None

    stats = _collect_variant_stats(g)
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["singleton_events"] < _MIN_SINGLETON_EVENTS:
        return None
    if stats["fake_traps"] < _MIN_FAKE_TRAPS:
        return None
    if stats["singleton_real_turns"] < _MIN_REAL_BLOCK_TURNS:
        return None
    if np.array_equal(out, baseline):
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
