from __future__ import annotations

# Variant of 9bbf930d
# Dimension: D15 — 初始方向 (path)
# Change: 所有粒子初始方向固定向右。... → 初始方向由相邻走廊入口方位决定
#
# Design analysis
# variant:
#   core_change:
#     - 原题里每个粒子都从左边界固定向右进入走廊；这里改成“粒子先看自己附近哪个方向
#       最先出现有效走廊入口”，然后沿那个方向起步。
#   concretization:
#     - 走廊与转弯的局部定义不变：可通行单元必须是 BG/粒子，并且其两侧的平行色条同色。
#     - 为了让 D15 在视觉上可见，生成器不改走廊拓扑本身，而是先生成原题那种
#       “左侧进入”的标准底板，再对整张图做 0/90/180/270 度旋转。
#     - 旋转后，粒子会出现在上/右/下/左任一边界，真正的首个有效入口也随之转到
#       下/左/上/右；因此 solver 必须在四个方向做局部射线搜索，不能再假设固定向右。
#   behavior_trace:
#     - 原逻辑：粒子位于 (r,0)，`main()` 只检查该行，直接调用 `_trace(..., direction="right")`。
#     - 新逻辑：粒子位于任一边界格，例如旋转后在底边 `(H-1,c)`；solver 沿上/下/左/右
#       四条短射线寻找“最先遇到的有效走廊单元”，若唯一最近入口在上方，则以 `up`
#       为初始方向，再沿原有转弯规则追踪。
#     - 差异：变化只发生在“起步方向确定”这一环；进入走廊后的直行/撞墙/90 度转弯
#       完全沿用原任务。
#   complexity_calibration:
#     - 不引入新的颜色语义，也不改变 L/U 型走廊构造，否则会把任务从 D15 扩散到
#       其他维度。
#     - 起步方向只允许四个正交方向，且由最近入口唯一决定，避免自由度过多或出现
#       需要复杂 tie-break 的情况。
#   implicit_assumptions_review:
#     - 原 `_trace()` 把 `direction = "right"` 写死；在旋转样本上会直接错。
#     - 原 `main()` 只遍历 `col 0` 的粒子；旋转后粒子可能在任一边界，这个假设失效。
#     - 原 `main()` 用“上/下同色条 + 左侧 bar 起点 <= 3”来识别水平入口；这只适配
#       左边进入的水平走廊，不适配上下/右侧入口。
#     - 原生成器把 `col 0 = particle, col 1 = BG` 当作输入规格的一部分；这里保留其
#       作为标准底板生成规则，但最终样本允许该结构被整体旋转。
#   placement_legality:
#     - 旋转是全局刚体变换，不会引入重叠、越界或局部拓扑破坏。
#     - 标准底板仍由原始 path-first 规则生成，继续保证真实/虚假走廊、L/U 转弯、
#       交叉口封口方式都合法。
#     - 新 solver 的唯一额外假设是：每个会移动的边界粒子都只有一个最近有效入口。
#       生成器通过“单侧发射 + 整体旋转”自动保证这一点。
#   observability:
#     - 从 2-4 组样例里，人可以直接看到品红边界粒子有时从左边进入，有时从上/下/右边
#       进入；输出仍然只是把这些粒子移动到走廊尽头，因此 D15 的变化是肉眼可见的。
#   cleanliness:
#     - 不在同一局部叠加额外标记；仍然只有背景、粒子、色条和少量假装饰，画面整洁，
#       原题的“条纹走廊”视觉风格保留。
#   generator_guarantees:
#     - 先用新 solver 验证标准底板可解，再对其旋转并再次验证。
#     - 外层验证统计“发生移动的粒子数”与“颜色数”，避免旋转后退化成空样本。

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
    """True if (r,c) is a passable cell flanked by same-color bars perpendicular to direction."""
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


def _find_initial_direction(grid: np.ndarray, r: int, c: int) -> str | None:
    """Find the nearest valid launch corridor around a boundary particle.

    Search outward in the four cardinal directions through contiguous particle/bg cells.
    The initial direction is the unique direction whose first valid corridor cell is closest.
    """
    H, W = grid.shape
    candidates: list[tuple[int, str]] = []
    for direction, (dr, dc) in _DELTAS.items():
        rr, cc = r, c
        steps = 0
        while True:
            steps += 1
            rr += dr
            cc += dc
            if not (0 <= rr < H and 0 <= cc < W):
                break
            if int(grid[rr, cc]) not in {PARTICLE, BG}:
                break
            if _has_corridor(grid, rr, cc, direction):
                candidates.append((steps, direction))
                break
    if not candidates:
        return None
    best = min(steps for steps, _ in candidates)
    directions = [direction for steps, direction in candidates if steps == best]
    if len(directions) != 1:
        return None
    return directions[0]


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    """Trace one particle until it exits or can no longer turn."""
    H, W = grid.shape
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
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _is_boundary_cell(r: int, c: int, H: int, W: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def main(grid: np.ndarray) -> np.ndarray:
    """Move boundary particles into the nearest valid corridor entrance.

    Rule:
    - Each magenta boundary particle locally searches for the nearest valid corridor entrance.
    - If exactly one direction wins that search, the particle starts moving in that direction.
    - Once inside the corridor system, it follows the original task's movement rule unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != PARTICLE or not _is_boundary_cell(r, c, H, W):
                continue
            direction = _find_initial_direction(g, r, c)
            if direction is None:
                continue
            end_r, end_c = _trace(g, r, c, direction)
            if (end_r, end_c) != (r, c):
                out[r, c] = BG
                out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

# Grid dimensions
# Restrict both axes to odd values shared by H/W so 90° rotations preserve the striped look.
_HEIGHT_OPTIONS = [17, 19, 21]
_WIDTH_OPTIONS = [17, 19, 21]
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

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
_MAX_RETRIES = 300


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1) if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Pair same-color real channels into U-turn plans."""
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


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
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
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
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
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in {"l_edge", "l_internal"}:
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif kind == "u":
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _count_moved(inp: np.ndarray, out: np.ndarray) -> int:
    """Count particles that left their launch cell."""
    return int(np.sum((inp == PARTICLE) & (out != PARTICLE)))


def _gen_canonical_attempt() -> np.ndarray | None:
    """Generate one standard left-entry candidate before rotation."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.choice(_WIDTH_OPTIONS)
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
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None
    n_moved = _count_moved(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def _gen_attempt() -> np.ndarray | None:
    """Generate one D15 sample by rotating a canonical left-entry instance."""
    base = _gen_canonical_attempt()
    if base is None:
        return None

    k = random.choice([0, 1, 2, 3])
    g = np.rot90(base, k=k).copy()

    try:
        out = main(g)
    except Exception:
        return None
    n_moved = _count_moved(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
