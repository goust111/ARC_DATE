# Variant of 9bbf930d
# Dimension: D8 — 障碍物：条对结构 (obstacle)
# Change: 彩色条水平铺设在偶数行，从bar_start列延伸到网格右端；真走廊由同色条对构成，假走廊由异色条对构成
#   → 条对部分断裂：同一条水平走廊的上下两条彩条改为交错断裂的虚线，粒子需要把 1 格缺口视作仍属于同一走廊。

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

# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# variant_spec:
#   core_visual_change:
#     original: every even-row bar was a continuous horizontal strip.
#     variant: each even-row bar becomes a broken strip. Adjacent bar rows use
#       opposite parity, so a real corridor often looks like:
#         top : C . C . C . C
#         path: . . . . . . .
#         bot : . C . C . C .
#       A human can still see "same-colored dashed rails around one hallway",
#       but the old exact-column solver cannot, because above/below rarely match
#       in the same column.
#   concrete_behavior_trace:
#     original_row_test:
#       row r is considered a real horizontal corridor only if some column c has
#       g[r,c]=BG and g[r-1,c]==g[r+1,c] with both cells colored.
#     variant_row_test:
#       row r is a real corridor if, at a BG cell, the rows r-1 and r+1 both
#       show the same bar color within radius 1 of that column. A 1-cell gap is
#       treated as part of the same dashed rail.
#     small_example:
#       top : [1,7,1]
#       mid : [7,7,7]
#       bot : [7,1,7]
#       old_logic_at_center:
#         g[top,1]=7 and g[bot,1]=1, so the center is not a corridor cell.
#       new_logic_at_center:
#         top has color 1 at distance 1, bot has color 1 at distance 0, so the
#         center still belongs to a real corridor.
#   complexity_calibration:
#     keep_small:
#       only horizontal bar recognition changes; particle motion, blockers, and
#       vertical turn mechanics stay the same.
#     keep_visible:
#       the broken-strip pattern is regular and repeated across many columns, so
#       the change is obvious in 2-3 examples rather than hidden in one corner.
#     avoid_excess_freedom:
#       only 1-cell horizontal gaps are tolerated; no color remapping and no
#       multiple competing gap sizes.
#   implicit_assumption_review_of_original_main:
#     - original _has_corridor(direction in left/right) assumes valid corridors
#       must have colored walls in the exact same column. This fails once bars
#       are dashed.
#     - original entrance detection assumes the first exact matching column is
#       enough to identify a real corridor. In this variant, many real rows have
#       no exact top/bottom overlap anywhere, so we must use local support.
#     - original vertical-turn test checks left/right flank cells on the same
#       bar row. That assumption is still valid, so we keep it exact and make
#       the generator preserve those flank cells.
#   placement_legality:
#     - vertical corridors reserve a 3-column window [c_v-2, c_v] to avoid
#       overlap and to keep left/right flank cells inside the grid.
#     - the generator stores must_keep / must_bg constraints per bar row, then
#       renders the dashed strips afterward. This prevents broken bars from
#       accidentally deleting a required vertical wall or sealing cell.
#   visual_observability:
#     - fake corridors still use different colors above and below, so the only
#       new inference burden is "same-colored dashed rails still count".
#     - no color-id-based ordering is used; all decisions come from visible
#       local geometry.
#   cleanliness:
#     - every even row has one bar color only; breaks are BG holes, so the grid
#       stays readable.
#     - L-turns are kept, but U-turns are dropped in the generator to keep the
#       variant focused on fragmented horizontal bars instead of adding extra
#       routing complexity.
#   generator_metric:
#     - during generation we compare against the original exact-column solver and
#       require at least two moved particles that only the new local-gap solver
#       can move. This prevents degeneration back to the original task.
# ---------------------------------------------------------------------------


def _local_bar_support_color(grid: np.ndarray, bar_row: int, c: int, radius: int = 1) -> int | None:
    """Return the unique colored bar visible within a short horizontal window."""
    H, W = grid.shape
    if not (0 <= bar_row < H):
        return None
    colors = set()
    for cc in range(max(0, c - radius), min(W - 1, c + radius) + 1):
        value = int(grid[bar_row, cc])
        if value not in {BG, PARTICLE}:
            colors.add(value)
    if len(colors) == 1:
        return next(iter(colors))
    return None


def _first_visible_bar_col(grid: np.ndarray, bar_row: int, color: int) -> int | None:
    W = grid.shape[1]
    for c in range(W):
        if int(grid[bar_row, c]) == color:
            return c
    return None


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is part of a corridor under the variant's gap-tolerant rule."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a = _local_bar_support_color(grid, r - 1, c, radius=1)
        b = _local_bar_support_color(grid, r + 1, c, radius=1)
        return a == b and a is not None
    if c == 0 or c == W - 1:
        return False
    a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace a particle from (r,c) using the original movement rule."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through real corridors, allowing 1-cell gaps in horizontal bar pairs."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if _has_corridor(g, r, c, "right"):
                entrance_col = c
                break
        if entrance_col is None:
            continue
        color = _local_bar_support_color(g, r - 1, entrance_col, radius=1)
        if color is None:
            continue
        top_start = _first_visible_bar_col(g, r - 1, color)
        bot_start = _first_visible_bar_col(g, r + 1, color)
        if top_start is None or bot_start is None or max(top_start, bot_start) > 4:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# ---------------------------------------------------------------------------
# Original exact-column solver, kept only for generator-side anti-degeneration
# validation. This is the behavior the D8 variant must differ from.
# ---------------------------------------------------------------------------


def _legacy_has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
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


def _legacy_trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
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
            if 0 <= tr < H and 0 <= tc < W and _legacy_has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) > 1:
            raise AssertionError("ambiguous turn: both perpendicular directions valid")
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above != bar_below or bar_above > 3:
            continue
        end_r, end_c = _legacy_trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_STRAIGHT_PROB = 0.35

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_VARIANT_ONLY_MOVED = 2
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW) and c - 2 >= bar_start
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _build_row_colors(
    bar_rows: list[int], channel_spec: list[str], extras: list[int]
) -> dict[int, int]:
    row_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            row_color[bot] = row_color[top]
        else:
            row_color[bot] = random.choice([c for c in extras if c != row_color[top]])
    return row_color


def _register_vertical_segment(
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
    rows: list[int],
    c_v: int,
) -> None:
    for br in rows:
        must_bg[br].add(c_v - 1)
        must_keep[br].update({c_v - 2, c_v})


def _build_l_turn(
    g: np.ndarray,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    g[row, c_v] = color
    if direction == "up":
        turn_bar, opp_bar = top, bot
        must_bg[turn_bar].add(c_v - 1)
        must_keep[turn_bar].update({c_v - 2, c_v})
        must_keep[opp_bar].add(c_v - 1)
        _register_vertical_segment(must_keep, must_bg, list(range(0, top, 2)), c_v)
    else:
        turn_bar, opp_bar = bot, top
        must_bg[turn_bar].add(c_v - 1)
        must_keep[turn_bar].update({c_v - 2, c_v})
        must_keep[opp_bar].add(c_v - 1)
        _register_vertical_segment(must_keep, must_bg, list(range(bot + 2, g.shape[0], 2)), c_v)


def _render_bar_rows(
    g: np.ndarray,
    bar_rows: list[int],
    row_color: dict[int, int],
    bar_start: int,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    W = g.shape[1]
    for idx, br in enumerate(bar_rows):
        color = row_color[br]
        phase = idx % 2
        for c in range(bar_start, W):
            if c in must_bg[br]:
                g[br, c] = BG
            elif c in must_keep[br] or (c - bar_start) % 2 == phase:
                g[br, c] = color
            else:
                g[br, c] = BG


def _variant_only_moved_rows(inp: np.ndarray) -> int:
    new_out = main(inp)
    old_out = _legacy_main(inp)
    rows = 0
    for r in range(inp.shape[0]):
        moved_new = int(inp[r, 0]) == PARTICLE and int(new_out[r, 0]) != PARTICLE
        moved_old = int(inp[r, 0]) == PARTICLE and int(old_out[r, 0]) != PARTICLE
        if moved_new and not moved_old:
            rows += 1
    return rows


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1
    n_real = min(random.choice(_N_REAL_OPTIONS), n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    row_color = _build_row_colors(bar_rows, channel_spec, extras)

    if len(set(row_color.values())) < 3:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    must_keep = {br: set() for br in bar_rows}
    must_bg = {br: set() for br in bar_rows}

    real_indices = [i for i in range(n_channels) if channel_spec[i] == "real"]
    random.shuffle(real_indices)
    used_vcols: set[int] = set()

    for i in real_indices:
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = row_color[top]
        if random.random() < _STRAIGHT_PROB:
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            continue
        direction = random.choice(["up", "down"])
        _build_l_turn(g, top, bot, row, color, c_v, direction, must_keep, must_bg)

    _render_bar_rows(g, bar_rows, row_color, bar_start, must_keep, must_bg)

    protected_cols = {0, 1}
    for cols in must_keep.values():
        protected_cols.update(cols)
    for cols in must_bg.values():
        protected_cols.update(cols)
    protected_cols.update(np.argwhere(g != BG)[:, 1].tolist())

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        row = bar_rows[i] + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        allowed = [c for c in range(bar_start + 1, W) if c not in protected_cols and int(g[row, c]) == BG]
        if not allowed:
            continue
        for c in random.sample(allowed, min(n_decor, len(allowed))):
            g[row, c] = random.choice([row_color[bar_rows[i]], row_color[bar_rows[i + 1]]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if _variant_only_moved_rows(g) < _MIN_VARIANT_ONLY_MOVED:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
