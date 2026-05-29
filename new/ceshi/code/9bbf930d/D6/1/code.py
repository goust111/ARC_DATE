from __future__ import annotations

# Variant of 9bbf930d
# Dimension: D6 — 终点条件 (dst)
# Change: 粒子在到达网格边缘或遇到完全阻塞（无有效垂直走廊可转）时停止。... → 走过固定步数停止
#
# Design analysis
# ===============
# variant_spec:
#   visible_encoding:
#     idea: 在原题走廊区右侧增加一个 2 列宽的独立侧栏，作为“距离尺”。
#     ruler_wall: 倒数第二列整列固定为 meter 色；它既把走廊区与侧栏物理隔开，也让尺子一眼可见。
#     ruler_count: 最右列从顶部开始连续 N 个 meter 色像素；其余为背景。N 就是所有粒子的最大步数。
#     reason_for_sidebar: 原题 col0 是粒子列、col1 是通道前的缓冲列，不能拿来编码；把编码挪到最右侧最稳妥，
#       且不会破坏入口识别和转弯几何。为保留原题 14-21 的可行走宽度，整张图总宽度放宽到 16-23。
#   movement:
#     unchanged:
#       - 入口识别仍然是：某个奇数行在左侧有粒子，且从 col>=2 开始能找到上下同色夹住的背景入口。
#       - 直行仍然是“下一个格是背景/粒子色就继续前进”。
#       - 遇到彩色阻挡时仍然只检查两个垂直方向里是否存在唯一合法走廊；若无则停，若有则转 90 度。
#     changed:
#       - 每次进入一个相邻新格都计 1 步。
#       - 直行进入下一格计 1 步；转弯进入垂直相邻格也计 1 步。
#       - 一旦累计步数达到 N，即使还能继续走、也未到边界，也必须停在当前格。
#   concrete_trace:
#     example:
#       - 假设某粒子从 (r,0) 出发，前方直走 4 步到达某堵墙左侧，然后第 5 步向下拐进竖走廊。
#       - 原题: 若该竖走廊还能再走 6 格，粒子会继续走到更远处，直到边界或死路。
#       - 本变体: 若右侧距离尺高度 N=5，粒子在进入竖走廊的那个格就停止；不会继续把该竖段走完。
#       - 差异: 原题的终点由“空间阻塞”决定；本变体的终点由“空间阻塞 与 步数上限”二者取更早者决定。
#   complexity_calibration:
#     - N 采用全局唯一值，而不是每条路径各自不同，避免规则碎片化。
#     - 每张图仍保留 3-6 条真实通道，且至少 2 条路径被步数上限截断，至少 1 条在拐弯后才被截断，
#       这样肉眼能看出“不是简单水平平移固定列数”，而是沿既有路径累计步数。
#     - 同时要求至少 1 条路径在 N 步以内自然走完，样例里会同时出现“受尺子限制”和“未受限制”两类，
#       便于人类归纳“最多走 N 步”而不是“总是正好走 N 步直到某特定方向”。
#   implicit_assumption_review:
#     - 原 `_trace()` 默认“只要还能走就一直走”，这直接依赖旧 dst；现在必须改成显式维护 `steps < N`。
#     - 原 `main()` 中对入口行的判断依赖“右侧所有列都属于走廊区域”；现在最右两列是侧栏，所以入口扫描、
#       同色条起点判断都必须限制在走廊区 `path_w = W-2` 内。
#     - 原 generator 默认整宽都可用于条带；现在若继续画到最右列，会污染距离尺，因此所有 bar/turn/decor
#       的放置上界都要改成 `path_w`。
#     - 原 generator 只验证 solver 不崩；现在还要验证“相同步道上，无步数上限时的终点”和“有限 N 时的终点”
#       确实不同，否则就会退化回原题。
#   placement_legality:
#     - 侧栏固定占用最后两列，因此真实走廊只在前 `W-2` 列构造。
#     - 垂直转弯列的分配仍使用不重叠窗口，避免多个 L/U 转弯互相覆盖。
#     - 侧栏墙色不参与任何走廊颜色集合，solver 也把入口搜索限制在走廊区，因此不会把侧栏误判为通道。
#   visual_observability:
#     - N 直接体现在最右列有多少个连续 meter 像素，完全基于可见长度，不依赖颜色 id 或排序。
#     - 多个粒子共享同一个 N，但沿不同走廊停在不同方位，这给出足够重复支撑。
#   visual_cleanliness:
#     - 侧栏与走廊之间有一整列同色墙，编码区和任务主体区分明确，不与走廊局部装饰混杂。
#   color_inference:
#     - meter 色固定只出现在侧栏，不参与真实/虚假走廊配色；人类无需推断复杂颜色映射，只需数高度。

import random
import numpy as np

BG = 7
PARTICLE = 6
METER = 8

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
    return a == b and a not in {PARTICLE, BG, METER}


def _read_step_limit(grid: np.ndarray) -> int:
    """Read N from the rightmost ruler column."""
    if grid.shape[1] < 2:
        return 0
    return int(np.count_nonzero(np.array(grid[:, -1], dtype=int) == METER))


def _find_entrance_col(grid: np.ndarray, r: int, path_w: int) -> int | None:
    """Find the first real corridor cell for row r inside the walkable region."""
    H, _ = grid.shape
    if not (1 <= r < H - 1):
        return None
    entrance_col = None
    for c in range(2, path_w):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG, METER}:
            entrance_col = c
            break
    if entrance_col is None:
        return None
    color_above = int(grid[r - 1, entrance_col])
    color_below = int(grid[r + 1, entrance_col])
    bar_above = next((c for c in range(path_w) if int(grid[r - 1, c]) == color_above), path_w)
    bar_below = next((c for c in range(path_w) if int(grid[r + 1, c]) == color_below), path_w)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


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
    """Move magenta particles along corridors, but stop after at most N visible steps."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    if W < 3:
        return out

    path_w = W - 2
    step_limit = _read_step_limit(g)

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r, path_w) is None:
            continue

        end_r, end_c = _trace(g, r, 0, step_limit)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

# Grid dimensions. Total width is widened by 2 to host the visible ruler,
# leaving the original 14-21 walkable columns on the left.
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_TOTAL_WIDTH_RANGE = (16, 23)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 3

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

# Vertical column allocation, expressed in walkable-region coordinates.
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Visible ruler validation
_STEP_LIMIT_MIN = 4
_STEP_LIMIT_MAX = 10
_MIN_TRUNCATED = 2
_MIN_TRUNCATED_AFTER_TURN = 1
_MIN_COMPLETED_WITHIN_LIMIT = 1

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 300


def _allocate_vcol(bar_start: int, walk_w: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return its wall column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = walk_w - _VCOL_RIGHT_PAD
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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, walk_w, used_vcols, plans):
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
            c_v = _allocate_vcol(bar_start, walk_w, used_vcols)
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


def _plan_remaining(real_indices, bar_start, walk_w, used_vcols, plans):
    """Assign straight or L-turn plans to real channels not yet planned."""
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, walk_w, used_vcols)
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
    walk_w,
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
        c_stop_min, c_stop_max = c_v + 3, walk_w - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(g, H, walk_w, plans, bar_rows, bar_color, bar_start, extras):
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
                walk_w,
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


def _paint_ruler(g: np.ndarray, step_limit: int) -> None:
    """Paint the visible step-count ruler into the right sidebar."""
    g[:, -2] = METER
    g[:, -1] = BG
    g[:step_limit, -1] = METER


def _collect_row_summaries(g: np.ndarray) -> list[dict[str, int | None]]:
    """Collect unbounded path summaries for all genuine entrance rows."""
    H, W = g.shape
    path_w = W - 2
    rows = []
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r, path_w) is None:
            continue
        try:
            meta = _trace(g, r, 0, None, return_meta=True)
        except AssertionError:
            return []
        if (int(meta["end_r"]), int(meta["end_c"])) == (r, 0):
            continue
        rows.append({"row": r, **meta})
    return rows


def _choose_step_limit(g: np.ndarray) -> int | None:
    """Pick an N that visibly changes the task instead of collapsing to the original stop rule."""
    H, _ = g.shape
    rows = _collect_row_summaries(g)
    moved = len(rows)
    if not (_MIN_MOVED <= moved <= _MAX_MOVED):
        return None

    hi = min(_STEP_LIMIT_MAX, H)
    candidates = []
    for step_limit in range(_STEP_LIMIT_MIN, hi + 1):
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


def _movement_stats(g: np.ndarray) -> dict[str, int]:
    """Measure how strongly the visible step limit affects generated samples."""
    H, W = g.shape
    path_w = W - 2
    step_limit = _read_step_limit(g)
    moved = 0
    truncated = 0
    truncated_after_turn = 0
    completed = 0

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r, path_w) is None:
            continue
        try:
            bounded = _trace(g, r, 0, step_limit, return_meta=True)
            full = _trace(g, r, 0, None, return_meta=True)
        except AssertionError:
            return {"moved": 0, "truncated": 0, "truncated_after_turn": 0, "completed": 0}
        start = (r, 0)
        bounded_end = (int(bounded["end_r"]), int(bounded["end_c"]))
        full_end = (int(full["end_r"]), int(full["end_c"]))
        if bounded_end != start:
            moved += 1
        if bounded_end != full_end:
            truncated += 1
            if bounded["turns"] > 0:
                truncated_after_turn += 1
        else:
            completed += 1

    return {
        "moved": moved,
        "truncated": truncated,
        "truncated_after_turn": truncated_after_turn,
        "completed": completed,
    }


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the distance-limited variant of 9bbf930d."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_TOTAL_WIDTH_RANGE)
    walk_w = W - 2
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE, METER)], _N_EXTRA_COLORS)

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
        g[br, bar_start:walk_w] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, walk_w, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, walk_w, used_vcols, plans)
    _execute_plans(g, H, walk_w, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, walk_w), min(n_decor, walk_w - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG
    g[:, -2] = METER
    g[:, -1] = BG

    step_limit = _choose_step_limit(g)
    if step_limit is None:
        return None
    _paint_ruler(g, step_limit)

    try:
        out = main(g)
    except Exception:
        return None

    stats = _movement_stats(g)
    if not (_MIN_MOVED <= stats["moved"] <= _MAX_MOVED):
        return None
    if stats["truncated"] < _MIN_TRUNCATED:
        return None
    if stats["truncated_after_turn"] < _MIN_TRUNCATED_AFTER_TURN:
        return None
    if stats["completed"] < _MIN_COMPLETED_WITHIN_LIMIT:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    # Cheap sanity check: the solver should actually move at least one particle to a non-start cell.
    if not np.any((g[:, 0] == PARTICLE) & (out[:, 0] != PARTICLE)):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
