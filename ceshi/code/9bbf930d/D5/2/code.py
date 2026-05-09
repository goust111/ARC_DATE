# Variant of 9bbf930d
# Dimension: D5 — 源对象：粒子定义 (src)
# Change: 品红色粒子位于最左列（col 0），从有效走廊入口的奇数行出发，初始方向固定向右
#   → 控制像素指定起点：输入不再预置品红粒子；改为在左右边界放置红色控制像素，控制像素所在的行
#   指示哪些真实走廊会从左侧被“注入”一个粒子。输出保留红色控制像素，并把品红粒子画到各自终点。

from __future__ import annotations

import random

import numpy as np

# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior
#    - The corridor geometry is unchanged from the original task: even rows are colored bars, odd rows are
#      horizontal channels, and turns are still resolved by looking for a perpendicular bg cell flanked by a
#      same-color bar pair.
#    - The visible change is only the source object. Instead of a full magenta column at col 0, the input has
#      sparse red control pixels on the left/right border, always on odd rows. Each control row requests one
#      logical particle injection from (row, 0) moving right. The particle itself is absent from the input.
#    - Output keeps every red control pixel fixed and adds one magenta particle at the traced destination for
#      each controlled row.
#    - Small example of the affected stage:
#      original: row 5 has g[5,0] == 6, so main() traces from (5,0) and then clears g[5,0].
#      variant:  row 5 has a red border pixel (at (5,0) or (5,W-1)); main() still traces from logical start
#                (5,0), but must NOT erase the control pixel because it is not the moving object.
# b) Complexity calibration
#    - Controls are restricted to side borders and encode rows only. This keeps the innovation focused on D5
#      instead of introducing a second geometric language for top/bottom controls.
#    - A sample still contains 3-6 active injections, with repeated evidence that red marks select rows while
#      magenta appears only in the output.
#    - Controls may appear on either left or right border when safe, so the rule is visibly "red row markers"
#      rather than "red recolors the old source cell".
# c) Implicit assumption audit from the original main()
#    - `if g[r,0] != PARTICLE`: invalid now, because row selection is driven by red border controls, not by a
#      magenta source cell.
#    - entrance scan and bar-start validation still hold; they depend on corridor geometry, not on source color.
#    - `_trace(g, r, 0)` still holds because the logical start remains col 0 and the trace logic only cares
#      about future cells; the start cell may now be bg or a red control pixel.
#    - `out[r,0] = BG` is no longer valid: it would erase a left-edge control pixel. The variant therefore
#      removes any legacy particle color from the copied canvas up front and only adds destination particles.
# d) Placement legality
#    - The generator preserves the original corridor builder and keeps bars starting at col 2 or 3, so border
#      cells remain free for controls.
#    - Right-edge controls are only placed on rows whose traced destination is not `(row, W-1)`, preventing the
#      final magenta particle from overwriting the control marker.
#    - Only rows with a valid entrance and an actual nontrivial trace are eligible for control placement.
# e) Visual observability
#    - In every valid sample there are multiple red border pixels and zero magenta pixels in the input, while
#      the output introduces magenta at remote corridor endpoints. Across 2-3 examples the human-visible rule is
#      that red marks choose which corridors activate.
# f) Visual cleanliness
#    - Controls stay on the empty border and never overlap corridor bars.
#    - Final particles are deduplicated by endpoint during generation, avoiding clutter from multiple injections
#      collapsing onto the same destination.
# g) Color inference
#    - The only new static color role is fixed red for controls. No hidden color ordering or id-based mapping is
#      used; corridor colors remain purely structural as in the original task.

BG = 7
CTRL = 2
PARTICLE = 6
_NON_BAR = {BG, CTRL, PARTICLE}
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
    return a == b and a not in _NON_BAR


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace a logical particle from (r, c) going right. Return the final passable cell."""
    H, W = grid.shape
    direction = "right"
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


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first valid horizontal corridor entrance for row r, or None."""
    H, W = grid.shape
    if r <= 0 or r >= H - 1 or r % 2 == 0:
        return None
    if int(grid[r, 0]) not in {BG, CTRL}:
        return None

    entrance_col = None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in _NON_BAR:
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


def _control_rows(grid: np.ndarray) -> list[int]:
    """Rows selected by red side-border control pixels."""
    H, W = grid.shape
    rows = {
        int(r)
        for r, c in np.argwhere(grid == CTRL)
        if 0 < int(r) < H - 1 and int(c) in {0, W - 1}
    }
    return sorted(rows)


def main(grid: np.ndarray) -> np.ndarray:
    """Trace logical particles for rows indicated by red side-border controls."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    out[out == PARTICLE] = BG

    for r in _control_rows(g):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) == (r, 0):
            continue
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
_MIN_PALETTE_DIVERSITY = 4

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
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
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
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
    color = bar_color[top]
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
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    """Build a U-turn and break the target row's left entrance so it is not independently startable."""
    color = bar_color[top]
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
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
        elif kind == "u":
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


def _collect_movable_rows(g: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    """Rows whose logical particle would move, deduplicated by final endpoint."""
    H, _ = g.shape
    movable: list[tuple[int, tuple[int, int]]] = []
    seen_endpoints: set[tuple[int, int]] = set()
    for r in range(1, H - 1, 2):
        if _find_entrance_col(g, r) is None:
            continue
        try:
            end = _trace(g, r, 0)
        except AssertionError:
            continue
        if end == (r, 0) or end in seen_endpoints:
            continue
        movable.append((r, end))
        seen_endpoints.add(end)
    return movable


def _place_controls(g: np.ndarray, movable_rows: list[tuple[int, tuple[int, int]]]) -> int:
    """Place 3-6 red controls on selected movable rows. Return the count placed."""
    H, W = g.shape
    if len(movable_rows) < _MIN_MOVED:
        return 0
    n_ctrl = random.randint(_MIN_MOVED, min(_MAX_MOVED, len(movable_rows)))
    chosen = random.sample(movable_rows, n_ctrl)

    right_candidates = [idx for idx, (_, end) in enumerate(chosen) if end[1] != W - 1]
    n_right = random.randint(0, len(right_candidates))
    right_indices = set(random.sample(right_candidates, n_right)) if right_candidates else set()

    for idx, (r, _) in enumerate(chosen):
        if idx in right_indices:
            g[r, W - 1] = CTRL
        else:
            g[r, 0] = CTRL
    return n_ctrl


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input for the D5 control-pixel variant."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, CTRL, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
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

    g[:, 0] = BG
    g[:, 1] = BG
    g[:, W - 1] = BG

    movable_rows = _collect_movable_rows(g)
    n_ctrl = _place_controls(g, movable_rows)
    if not (_MIN_MOVED <= n_ctrl <= _MAX_MOVED):
        return None

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = int(np.count_nonzero(out == PARTICLE))
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if int(np.count_nonzero(g == PARTICLE)) != 0:
        return None
    if len(set(map(int, g.flatten()))) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
