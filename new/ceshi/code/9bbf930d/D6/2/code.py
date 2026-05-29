# Variant of 9bbf930d
# Dimension: D6 — 终点条件 (dst)
# Change: 粒子在到达网格边缘或遇到完全阻塞（无有效垂直走廊可转）时停止。... → 穿越固定数量的走廊后停止
#
# Design analysis
# ---------------
# variant_spec:
#   fixed_corridor_limit: 2
#   corridor_counting:
#     - 起始时粒子位于最左列，但一旦进入横向通道，计作第 1 段走廊。
#     - 每次 90 度转弯并进入新的直线段（横向或纵向）时，走廊计数 +1。
#     - 当粒子已经处于第 2 段走廊末端，若前方是障碍且本可继续转入第 3 段，
#       变体要求它停在当前格，不再执行这次转弯。
#   preserved_from_source:
#     - 走廊判定仍然完全依赖“两侧同色条对夹着一个 BG 格”。
#     - 转弯候选仍然只看当前运动方向的两个垂直方向，并保持“最多一个合法转弯”的断言。
#     - 输入布局仍是偶数行条带 / 奇数行水平通道，垂直通道仍由 bar-row 上打洞形成。
#
# concrete_behavior_trace:
#   original:
#     - 水平走廊 -> 第一次拐入竖直走廊 -> 第二次拐回另一条水平走廊 -> 在边界或死路结束。
#   variant:
#     - 水平走廊记作第 1 段，竖直走廊记作第 2 段。
#     - 当粒子到达竖直走廊尽头、理论上可以再拐回第 3 段水平走廊时，它直接停在竖直走廊末端。
#   visual_effect:
#     - L 型路径（总共正好 2 段）与原题终点相同。
#     - U/S 型路径（至少 3 段）会比原题少走最后一段，因此终点常落在网格内部的竖直通道末端。
#
# complexity_calibration:
#   - 若把限制设成 1，则所有可拐弯路径都会在第一次撞墙处停止，转弯结构失去作用，过于简化。
#   - 设成 2 可以保留“会转一次”的原题核心，又能让 3 段路径产生稳定、可见的差异。
#   - generator 因此强制生成至少两个 U-turn source，使样例中反复出现“第二段后停止”的证据。
#
# implicit_assumption_audit_for_main:
#   - _has_corridor() 假设可通行格只能是 BG 或粒子色；变体未改变，仍成立。
#   - 原 _trace() 默认“只要有唯一合法转弯就必须继续”，这是旧终点条件的核心依赖；
#     变体下必须改成“只有当前已进入的段数 < 2 时才能继续转弯”。
#   - main() 的入口检测假设真实通道从列 2/3 附近开始，且上下条带同色；变体不改入口定义，仍成立。
#   - out[end_r, end_c] = PARTICLE 依赖“最终停点一定是可通行格”；新 trace 仍只在 BG/粒子格停下，成立。
#
# placement_legality:
#   - generator 仍使用不重叠的竖直通道列窗口分配，避免两个转弯结构共享同一竖井。
#   - U-turn target 的入口继续被打断，保证该行自己的粒子不会错误地作为独立真实轨迹进入。
#   - 只有通过 solver 的真实追踪和 old-vs-new 差异校验的样本才会被接受。
#
# observability_and_cleanliness:
#   - 不引入任何额外标记或依赖颜色 id 的隐藏排序；人类只需观察“某些粒子在第二段末端就停了”。
#   - 结构仍是整齐条纹 + 少量竖直连接，避免多层覆盖。
#   - 校验要求至少两个粒子命中 corridor limit，防止样本退化回原题。

from __future__ import annotations

import random
from typing import Any

import numpy as np

BG = 7
PARTICLE = 6
MAX_CORRIDORS = 2

_DELTAS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color bars."""
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


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    max_corridors: int | None,
    return_stats: bool = False,
) -> Any:
    """Trace a particle from (r,c) going right.

    If max_corridors is None, this matches the original unlimited behavior.
    Otherwise, the particle may enter at most `max_corridors` straight corridor
    segments. At a would-be turn into segment `max_corridors + 1`, it stops on
    the current cell instead.
    """
    H, W = grid.shape
    direction = "right"
    corridors_used = 1
    turns_taken = 0
    stop_reason = "start"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            stop_reason = "edge"
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
        if not options:
            stop_reason = "blocked"
            break
        if max_corridors is not None and corridors_used >= max_corridors:
            stop_reason = "limit"
            break
        r, c, direction = options[0]
        corridors_used += 1
        turns_taken += 1

    if return_stats:
        return r, c, {
            "corridors_used": corridors_used,
            "turns_taken": turns_taken,
            "stop_reason": stop_reason,
        }
    return r, c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal corridor entrance for row r."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None
    if int(grid[r, 0]) != PARTICLE:
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            bar_above = next((x for x in range(W) if int(grid[r - 1, x]) == above), W)
            bar_below = next((x for x in range(W) if int(grid[r + 1, x]) == below), W)
            if bar_above == bar_below and bar_above <= 3:
                return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through at most two corridor segments."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0, max_corridors=MAX_CORRIDORS)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_SINGLETON_REAL_OPTIONS = [1, 2]
_L_INTERNAL_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.0

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_LIMIT_HITS = 2
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for one vertical corridor."""
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


def _pick_pair_starts(n_channels: int) -> list[int] | None:
    """Pick two length-2 real blocks with at least one fake gap between them."""
    starts = list(range(n_channels - 1))
    random.shuffle(starts)
    for first in starts:
        others = [s for s in starts if s >= first + 3 or s <= first - 3]
        random.shuffle(others)
        for second in others:
            if {first, first + 1}.isdisjoint({second, second + 1}):
                return sorted([first, second])
    return None


def _choose_real_layout(n_channels: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    """Choose two U-pairs plus 1-2 singleton real channels for L-turns."""
    pair_starts = _pick_pair_starts(n_channels)
    if pair_starts is None:
        return None
    pairs = [(s, s + 1) for s in pair_starts]
    blocked = {idx for pair in pairs for idx in pair}
    reserved_for_singletons = set(blocked)
    for idx in blocked:
        if idx - 1 >= 0:
            reserved_for_singletons.add(idx - 1)
        if idx + 1 < n_channels:
            reserved_for_singletons.add(idx + 1)
    singleton_pool = [i for i in range(n_channels) if i not in reserved_for_singletons]
    if not singleton_pool:
        return None

    n_singletons = min(random.choice(_N_SINGLETON_REAL_OPTIONS), len(singleton_pool))
    if n_singletons == 0:
        return None
    singletons = random.sample(singleton_pool, n_singletons)
    return pairs, singletons


def _build_channel_spec(
    n_channels: int,
) -> tuple[list[str], dict[int, dict[str, Any]], list[list[int]]] | None:
    """Return channel roles and movement plans."""
    layout = _choose_real_layout(n_channels)
    if layout is None:
        return None
    pairs, singletons = layout

    channel_spec = ["fake"] * n_channels
    plans: dict[int, dict[str, Any]] = {}
    real_groups: list[list[int]] = []

    for upper, lower in pairs:
        channel_spec[upper] = "real"
        channel_spec[lower] = "real"
        real_groups.append([upper, lower])
        if random.random() < 0.5:
            plans[upper] = {"kind": "u", "target": lower, "direction": "down"}
            plans[lower] = {"kind": "u_target"}
        else:
            plans[lower] = {"kind": "u", "target": upper, "direction": "up"}
            plans[upper] = {"kind": "u_target"}

    for idx in singletons:
        channel_spec[idx] = "real"
        real_groups.append([idx])
        plans[idx] = {
            "kind": "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge",
            "direction": random.choice(["up", "down"]),
        }

    return channel_spec, plans, real_groups


def _build_bar_colors(
    n_channels: int,
    channel_spec: list[str],
    extras: list[int],
    real_groups: list[list[int]],
) -> list[int] | None:
    """Construct bar-row colors while keeping real structures color-separated."""
    if len(real_groups) > len(extras):
        return None

    required: dict[int, int] = {}
    group_colors = random.sample(extras, len(real_groups))
    for group, color in zip(real_groups, group_colors):
        start = min(group)
        end = max(group) + 1
        for row_idx in range(start, end + 1):
            prev = required.get(row_idx)
            if prev is not None and prev != color:
                return None
            required[row_idx] = color

    def _dfs(row_idx: int, acc: list[int]) -> list[int] | None:
        if row_idx == n_channels + 1:
            return acc.copy()

        candidates = [required[row_idx]] if row_idx in required else extras[:]
        random.shuffle(candidates)
        for color in candidates:
            if row_idx > 0:
                prev = acc[row_idx - 1]
                if channel_spec[row_idx - 1] == "real" and color != prev:
                    continue
                if channel_spec[row_idx - 1] != "real" and color == prev:
                    continue
            if row_idx < n_channels and row_idx + 1 in required:
                next_color = required[row_idx + 1]
                if channel_spec[row_idx] == "real" and next_color != color:
                    continue
                if channel_spec[row_idx] != "real" and next_color == color:
                    continue
            acc.append(color)
            result = _dfs(row_idx + 1, acc)
            if result is not None:
                return result
            acc.pop()
        return None

    return _dfs(0, [])


def _collect_trace_stats(grid: np.ndarray) -> list[dict[str, Any]]:
    """Trace all rows that look like real entrances under both rules."""
    H, _ = grid.shape
    stats = []
    for r in range(1, H - 1):
        entrance_col = _find_entrance_col(grid, r)
        if entrance_col is None:
            continue
        old_r, old_c, old_meta = _trace(grid, r, 0, max_corridors=None, return_stats=True)
        new_r, new_c, new_meta = _trace(
            grid, r, 0, max_corridors=MAX_CORRIDORS, return_stats=True
        )
        stats.append(
            {
                "row": r,
                "entrance_col": entrance_col,
                "old_end": (old_r, old_c),
                "new_end": (new_r, new_c),
                "old": old_meta,
                "new": new_meta,
            }
        )
    return stats


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D6 variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    channel_data = _build_channel_spec(n_channels)
    if channel_data is None:
        return None
    channel_spec, plans, real_groups = channel_data

    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_colors = _build_bar_colors(n_channels, channel_spec, extras, real_groups)
    if bar_colors is None or len(set(bar_colors)) < 3:
        return None

    for br, color in zip(bar_rows, bar_colors):
        g[br, bar_start:W] = color

    used_vcols: set[int] = set()
    for idx, plan in plans.items():
        if plan["kind"] == "u_target":
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            return None
        plan["c_v"] = c_v
        if plan["kind"] == "u":
            plan["internal_stop"] = random.random() < _U_INTERNAL_STOP_PROB

    _execute_plans(g, H, W, plans, bar_rows, bar_colors, bar_start, extras)

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
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_colors[i], bar_colors[i + 1]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    trace_stats = _collect_trace_stats(g)
    moved = sum(1 for item in trace_stats if item["new_end"] != (item["row"], 0))
    limit_hits = sum(1 for item in trace_stats if item["new"]["stop_reason"] == "limit")
    changed_vs_old = sum(1 for item in trace_stats if item["new_end"] != item["old_end"])
    interior_stops = sum(
        1
        for item in trace_stats
        if item["new"]["stop_reason"] == "limit"
        and 0 < item["new_end"][0] < H - 1
        and 0 < item["new_end"][1] < W - 1
    )

    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None
    if limit_hits < _MIN_LIMIT_HITS or changed_vs_old < _MIN_LIMIT_HITS:
        return None
    if interior_stops < 1:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    moved_in_output = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if moved_in_output != moved:
        return None
    return g


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, Any]],
    bar_rows: list[int],
    bar_colors: list[int],
    bar_start: int,
    extras: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    color_by_row = {row: bar_colors[i] for i, row in enumerate(bar_rows)}
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = color_by_row[top]
        c_v = plan["c_v"]

        if plan["kind"] in ("l_edge", "l_internal"):
            stop_at = None
            if plan["kind"] == "l_internal":
                if plan["direction"] == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, color, c_v, plan["direction"], color_by_row, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                color_by_row,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    color: int,
    c_v: int,
    direction: str,
    color_by_row: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn, matching the original geometry."""
    row = top + 1
    g[row, c_v] = color

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = color_by_row[turn_bar]
    g[turn_bar, c_v] = color_by_row[turn_bar]
    g[opp_bar, c_v - 1] = color_by_row[opp_bar]

    if direction == "up":
        for r_br in range(0, top, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    color_by_row: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    """Build a U-turn between two same-color real channels."""
    g[row, c_v] = color

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = color_by_row[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = color_by_row[top]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
