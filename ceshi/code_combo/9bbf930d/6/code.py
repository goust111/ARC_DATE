from __future__ import annotations

# Combo variant of 9bbf930d (index 6)
# combo_spec: D6/1+D18/2
# Name: 固定步数停止+装饰计数编码步数
#
# Design analysis
# ===============
# combo_variant:
#   visible_encoding:
#     idea: 不再增加侧栏，而是把原题里的假走廊从“纯干扰项”改成“重复计数器”。
#     implementation: 每一条假走廊（上下条带主导色不同的奇数行）都放入同样数量 N 个装饰像素。
#     decoding: 解题者先识别出这些假走廊，再数出每条里的装饰点个数；重复出现的同一计数就是全局步数上限。
#     why_repetition: 不是让人从单个位置推断抽象映射，而是让 2-5 条假走廊重复展示同一数字，减少偶然性。
#   movement:
#     unchanged:
#       - 真走廊的几何、入口识别、直行和 90 度转弯规则都沿用原题。
#       - 只有上下同色夹住的通道才可进入；上下异色的假走廊仍不可进入。
#     changed:
#       - 粒子每进入一个相邻格都记 1 步，直行和转弯进入都算一步。
#       - 一旦累计步数达到 N，就立刻停在当前格；即使前方还能继续走，也不再移动。
#   concrete_trace:
#     example:
#       - 某真实通道从 (r,0) 出发，先向右走 4 步，在第 5 步拐进竖走廊。
#       - 原题会继续沿竖走廊走到边缘或死路。
#       - 本融合版若所有假走廊里都可见 5 个装饰点，则该粒子在拐入后的那个格就停止。
#       - 差异不再来自侧栏，而来自“假走廊内部的点数”。
#   complexity_calibration:
#     - N 保持在单个数字量级（这里是 3-8 的小范围），保证肉眼可数且不会把假走廊完全填满。
#     - 每张图至少保留 2 条假走廊，且它们都显示同一个 N，给出重复支撑。
#     - 生成器硬性要求至少 2 条真实路径被截断、至少 1 条在转弯后被截断，
#       直接保证“装饰计数决定终点”的融合效应在每个样本里都真正发生。
#   implicit_assumption_review:
#     - 原 `_trace()` 默认“只要还能走就一直走”，这里必须显式维护 `steps < N`。
#     - 原装饰点对求解完全无关；现在 `main()` 先从假走廊读出 N，再执行轨迹。
#     - 原生成器在 U-turn 目标条带起点会故意改单个像素来破坏入口；若 solver 只看“某一列上下异色”，
#       会把这类真实通道误判为假走廊。这里改用“该行上下主导条带色”识别真/假，并要求假走廊的异色支撑
#       在多个列上持续存在，避开这种单点异常。
#   placement_legality:
#     - 装饰只放在假走廊的奇数行背景格里，不覆盖真实转弯墙，也不进入 col0/col1。
#     - 每条假走廊至少保留一个未装饰的背景格，保证 solver 能看见“上下异色夹住的通道”。
#     - 垂直转弯列仍按不重叠窗口分配，避免多个 L/U 结构互相覆盖。
#   visual_observability:
#     - 假走廊本来就因上下异色而显眼；现在其内部再出现重复数量的点，观察者会自然把它们作为一组去数。
#     - 多条假走廊共享同一计数，而多条真实路径在不同位置提前停止，这两类重复共同支撑“最多走 N 步”。
#   visual_cleanliness:
#     - 编码区域就是既有的假走廊，不额外叠加新面板；规则变化集中在原图结构内，版面仍然整洁。
#   color_inference:
#     - N 仅由“装饰点数量”表达，不依赖颜色值大小，也不需要推断复杂颜色映射。

from collections import Counter
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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars perpendicular to direction."""
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


def _dominant_bar_info(grid: np.ndarray, r: int) -> tuple[int, int, int, int] | None:
    """Return dominant top/bottom bar colors around channel row r plus their start columns."""
    H, W = grid.shape
    if not (1 <= r < H - 1):
        return None

    top_vals = [int(grid[r - 1, c]) for c in range(2, W) if int(grid[r - 1, c]) not in {BG, PARTICLE}]
    bot_vals = [int(grid[r + 1, c]) for c in range(2, W) if int(grid[r + 1, c]) not in {BG, PARTICLE}]
    if not top_vals or not bot_vals:
        return None

    top_color = Counter(top_vals).most_common(1)[0][0]
    bot_color = Counter(bot_vals).most_common(1)[0][0]
    top_start = next((c for c in range(W) if int(grid[r - 1, c]) == top_color), W)
    bot_start = next((c for c in range(W) if int(grid[r + 1, c]) == bot_color), W)
    return top_color, bot_color, top_start, bot_start


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Find the first genuine corridor cell for row r."""
    info = _dominant_bar_info(grid, r)
    if info is None:
        return None
    top_color, bot_color, top_start, bot_start = info
    if top_color != bot_color or top_start != bot_start or top_start > 3:
        return None

    for c in range(max(2, top_start), grid.shape[1]):
        if int(grid[r, c]) != BG:
            continue
        if int(grid[r - 1, c]) == top_color and int(grid[r + 1, c]) == bot_color:
            return c
    return None


def _find_fake_corridor_col(grid: np.ndarray, r: int) -> int | None:
    """Find a stable fake-corridor witness cell for row r, if present."""
    info = _dominant_bar_info(grid, r)
    if info is None:
        return None
    top_color, bot_color, top_start, bot_start = info
    if top_color == bot_color or top_start != bot_start or top_start > 3:
        return None

    mismatch_support = 0
    for c in range(max(2, top_start), grid.shape[1]):
        if int(grid[r - 1, c]) == top_color and int(grid[r + 1, c]) == bot_color:
            mismatch_support += 1
    if mismatch_support < 3:
        return None

    for c in range(max(2, top_start), grid.shape[1]):
        if int(grid[r, c]) != BG:
            continue
        if int(grid[r - 1, c]) == top_color and int(grid[r + 1, c]) == bot_color:
            return c
    return None


def _read_step_limit(grid: np.ndarray) -> int:
    """Decode N as the repeated decoration count across fake corridors."""
    counts = []
    H, _ = grid.shape
    for r in range(1, H - 1, 2):
        if _find_fake_corridor_col(grid, r) is None:
            continue
        count = int(np.count_nonzero(np.array(grid[r], dtype=int) != BG))
        if int(grid[r, 0]) == PARTICLE:
            count -= 1
        if count > 0:
            counts.append(count)

    if not counts:
        return 0
    hist = Counter(counts)
    return max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _trace(
    grid: np.ndarray,
    r: int,
    c: int,
    step_limit: int | None,
    *,
    return_meta: bool = False,
) -> tuple[int, int] | dict[str, int | None]:
    """Trace a particle, optionally capped by a visible step limit."""
    H, W = grid.shape
    direction = "right"
    steps = 0
    turns = 0
    first_turn_step: int | None = None
    safety = H * W * 4

    while safety > 0 and (step_limit is None or steps < step_limit):
        safety -= 1
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            steps += 1
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break

        r, c, direction = options[0]
        steps += 1
        turns += 1
        if first_turn_step is None:
            first_turn_step = steps

    if return_meta:
        return {
            "end_r": int(r),
            "end_c": int(c),
            "steps": int(steps),
            "turns": int(turns),
            "first_turn_step": first_turn_step,
        }
    return int(r), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along real corridors, but stop after at most N visible steps.

    Input structure:
    - bg=7, particle=6. Col 0 is filled with particles and col 1 stays background.
    - Even rows carry colored bars; odd rows are channels.
    - Real channels have matching bar colors above/below; fake channels have mismatched bar colors and contain repeated decoration dots.

    Rule:
    - First read N by counting how many decoration pixels appear inside a fake corridor; the same count is repeated across fake corridors.
    - For each particle row that has a genuine same-color corridor entrance, trace the original path rule: move through bg, and when blocked by color, turn into the unique perpendicular same-color corridor if it exists.
    - Each move into an adjacent cell counts as one step, including the move that enters a turn.
    - Stop as soon as either the original geometry would stop the particle or the cumulative step count reaches N.

    Key insight:
    - The fake corridors are no longer ignorable noise: their decoration count is the global step budget that truncates every real trajectory.
    """
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()
    step_limit = _read_step_limit(g)

    for r in range(1, H - 1, 2):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0, step_limit)
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
_MIN_FAKE_ROWS = 2

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Step-limit encoding via fake decorations
_STEP_LIMIT_MIN = 3
_STEP_LIMIT_MAX = 8
_MIN_TRUNCATED = 2
_MIN_TRUNCATED_AFTER_TURN = 1
_MIN_COMPLETED_WITHIN_LIMIT = 0

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans):
    """Pair same-color real channels into U-turn plans."""
    by_color = {}
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


def _plan_remaining(real_indices, bar_start, W, used_vcols, plans):
    """Assign straight or L-turn plans to real channels not yet planned."""
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at):
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
    g[row, c_v] = color
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
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g,
    W,
    top,
    bot,
    row,
    color,
    c_v,
    direction,
    j_top,
    j_bot,
    bar_color,
    bar_start,
    extras,
    internal_stop,
):
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot)."""
    g[row, c_v] = color

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

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras):
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = plan["c_v"]

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = plan["direction"]
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
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
                bar_color,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _collect_row_summaries(g: np.ndarray) -> list[dict[str, int | None]]:
    """Collect unbounded path summaries for all genuine entrance rows."""
    H, _ = g.shape
    rows = []
    for r in range(1, H - 1, 2):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        try:
            meta = _trace(g, r, 0, None, return_meta=True)
        except AssertionError:
            return []
        if (int(meta["end_r"]), int(meta["end_c"])) == (r, 0):
            continue
        rows.append({"row": r, **meta})
    return rows


def _choose_step_limit(g: np.ndarray, fake_indices: list[int], bar_rows: list[int], bar_start: int) -> int | None:
    """Pick an N that is both visually countable and behaviorally non-degenerate."""
    rows = _collect_row_summaries(g)
    moved = len(rows)
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None
    if len(fake_indices) < _MIN_FAKE_ROWS:
        return None

    per_row_slots = []
    W = g.shape[1]
    for i in fake_indices:
        r = bar_rows[i] + 1
        slots = [c for c in range(bar_start + 1, W) if int(g[r, c]) == BG]
        per_row_slots.append(len(slots))
    max_limit = min(_STEP_LIMIT_MAX, min(per_row_slots) - 1)
    if max_limit < _STEP_LIMIT_MIN:
        return None

    candidates = []
    for step_limit in range(_STEP_LIMIT_MIN, max_limit + 1):
        truncated = sum(int(meta["steps"]) > step_limit for meta in rows)
        truncated_after_turn = sum(
            int(meta["steps"]) > step_limit
            and meta["first_turn_step"] is not None
            and int(meta["first_turn_step"]) <= step_limit
            for meta in rows
        )
        completed = sum(int(meta["steps"]) <= step_limit for meta in rows)
        if (
            truncated >= _MIN_TRUNCATED
            and truncated_after_turn >= _MIN_TRUNCATED_AFTER_TURN
            and completed >= _MIN_COMPLETED_WITHIN_LIMIT
        ):
            candidates.append(step_limit)
    if not candidates:
        return None
    return random.choice(candidates)


def _decorate_fake_rows(
    g: np.ndarray,
    fake_indices: list[int],
    step_limit: int,
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
) -> None:
    """Encode the step limit by planting exactly N decorations in every fake corridor."""
    W = g.shape[1]
    for i in fake_indices:
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        slots = [c for c in range(bar_start + 1, W) if int(g[r, c]) == BG]
        cols = sorted(random.sample(slots, step_limit))
        colors = [bar_color[top], bar_color[bot]]
        for j, c in enumerate(cols):
            g[r, c] = colors[j % 2]


def _fusion_stats(g: np.ndarray) -> dict[str, int | tuple[int, ...]]:
    """Measure whether the fake-corridor count truly controls the stopping behavior."""
    H, _ = g.shape
    fake_counts = []
    step_limit = _read_step_limit(g)
    moved = 0
    truncated = 0
    truncated_after_turn = 0
    completed = 0

    for r in range(1, H - 1, 2):
        if _find_fake_corridor_col(g, r) is not None:
            count = int(np.count_nonzero(np.array(g[r], dtype=int) != BG))
            if int(g[r, 0]) == PARTICLE:
                count -= 1
            fake_counts.append(count)

        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        try:
            bounded = _trace(g, r, 0, step_limit, return_meta=True)
            full = _trace(g, r, 0, None, return_meta=True)
        except AssertionError:
            return {
                "step_limit": 0,
                "moved": 0,
                "truncated": 0,
                "truncated_after_turn": 0,
                "completed": 0,
                "fake_rows": 0,
                "uniform_fake_count": 0,
                "fake_count_values": (),
            }
        start = (r, 0)
        bounded_end = (int(bounded["end_r"]), int(bounded["end_c"]))
        full_end = (int(full["end_r"]), int(full["end_c"]))
        if bounded_end != start:
            moved += 1
        if bounded_end != full_end:
            truncated += 1
            if int(bounded["turns"]) > 0:
                truncated_after_turn += 1
        else:
            completed += 1

    return {
        "step_limit": int(step_limit),
        "moved": moved,
        "truncated": truncated,
        "truncated_after_turn": truncated_after_turn,
        "completed": completed,
        "fake_rows": len(fake_counts),
        "uniform_fake_count": int(len(set(fake_counts)) == 1 and bool(fake_counts)),
        "fake_count_values": tuple(sorted(set(fake_counts))),
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the fused distance/counting variant of 9bbf930d."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    max_real = min(max(_N_REAL_OPTIONS), n_channels - _MIN_FAKE_ROWS)
    allowed_real = [n for n in _N_REAL_OPTIONS if n <= max_real]
    if not allowed_real:
        return None
    allowed_weights = _N_REAL_WEIGHTS[: len(allowed_real)]
    n_real = random.choices(allowed_real, weights=allowed_weights, k=1)[0]
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    fake_indices = [i for i, spec in enumerate(channel_spec) if spec == "fake"]

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

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    step_limit = _choose_step_limit(g, fake_indices, bar_rows, bar_start)
    if step_limit is None:
        return None
    _decorate_fake_rows(g, fake_indices, step_limit, bar_rows, bar_color, bar_start)

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    stats = _fusion_stats(g)
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["step_limit"] != step_limit:
        return None
    if stats["fake_rows"] < _MIN_FAKE_ROWS:
        return None
    if not stats["uniform_fake_count"]:
        return None
    if stats["fake_count_values"] != (step_limit,):
        return None
    if stats["truncated"] < _MIN_TRUNCATED:
        return None
    if stats["truncated_after_turn"] < _MIN_TRUNCATED_AFTER_TURN:
        return None
    if stats["completed"] < _MIN_COMPLETED_WITHIN_LIMIT:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if not np.any((g[:, 0] == PARTICLE) & (out[:, 0] != PARTICLE)):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
