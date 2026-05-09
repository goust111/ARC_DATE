# Variant of 9bbf930d
# Dimension: D9 — 交叉口行为 (obstacle)
# Change: 垂直走廊与水平走廊交叉时，要么两条条完全断开让粒子惯性通过，要么两条条不断让粒子完全阻挡
#         -> 交叉口强制转弯。粒子在水平走廊里遇到“前方仍是背景、但前方格旁边有唯一竖直支路入口”的交叉口时，
#            不再直行穿过，而是立刻转入该竖直支路。

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

# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   preserved_structure:
#     - bg=7, particle=6
#     - even rows are colored bar rows, odd rows are travel rows
#     - real corridors are odd rows whose bars above/below share a color
#     - fake corridors keep mismatched bar colors and act only as distractors
#   variant_core:
#     - Original solver only considered turning when the next cell was colored.
#       Therefore a pass-through crossing (next cell still bg) was inertial.
#     - New solver adds one extra event before the normal "move straight" step:
#       if the particle is moving horizontally and the next cell is bg, but that
#       next cell has exactly one vertical corridor entry (up or down), then the
#       next cell is treated as a forced junction and the particle must enter the
#       vertical branch instead of continuing horizontally.
#   concrete_pixel_encoding:
#     - A forced junction is encoded without adding a blocker in the horizontal
#       channel. The crossing cell stays bg.
#     - Only one adjacent bar-row cell (above or below the crossing cell) is
#       carved to bg, creating a unique branch entry. The other side stays solid.
#     - The vertical branch continues as a carved bg column. This preserves the
#       original striped look while making the D9 change visually observable:
#       under the old logic the particle would keep going right through the bg
#       crossing, but under the new logic it diverts into the carved branch.
#   behavior_trace_example:
#     - Original: ... particle at (r,c-1), (r,c) is bg, so it moves to (r,c) and
#       keeps going right because no colored blocker was hit.
#     - Variant: same input, but before stepping into (r,c) the solver sees that
#       (r,c) has exactly one vertical branch entry at (r-1,c) or (r+1,c), so it
#       turns immediately into that branch.
#   hidden_assumption_audit:
#     - _has_corridor() still assumes only same-color flanks define a legal turn.
#       The generator keeps that invariant for every forced-branch entry cell.
#     - main() still assumes entrances start at col 2 or 3 and that fake rows do
#       not accidentally become real. The generator preserves full-width bars and
#       only carves branch columns away from the entrance zone.
#     - The original trace asserted at most one perpendicular turn option. This
#       remains true here: every forced junction is one-sided, and every vertical
#       branch stops before crossing another real horizontal corridor.
#   placement_legality:
#     - Branch columns are spaced apart so carvings do not merge.
#     - A branch may cross fake rows, but it is terminated before it would create
#       a two-sided crossing on another real row, preventing ambiguous forced
#       turns for other particles.
#   observability:
#     - Each generated grid contains multiple real channels whose final position
#       differs from the original solver on the same input. That guarantees the
#       new behavior is not a degenerate copy of the base task.

# Generator configuration
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_N_REAL_OPTIONS = [3, 4, 5]
_N_REAL_WEIGHTS = [4, 3, 1]
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 400
_MIN_MOVED = 3
_MAX_MOVED = 5
_MIN_DIVERGENT = 2
_BRANCH_COL_PAD = 2


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
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


def _forced_turn_ahead(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int, str] | None:
    """Return forced junction turn when moving horizontally into a crossing."""
    if direction not in {"left", "right"}:
        return None
    H, W = grid.shape
    dr, dc = _DELTAS[direction]
    cr, cc = r + dr, c + dc
    if not (0 <= cr < H and 0 <= cc < W):
        return None
    if int(grid[cr, cc]) not in {PARTICLE, BG}:
        return None

    options = []
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = cr + tdr, cc + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))

    assert len(options) <= 1, "ambiguous forced junction: both vertical turns valid"
    return options[0] if options else None


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace particle from (r,c) using the D9 forced-turn crossing rule."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
        forced = _forced_turn_ahead(grid, r, c, direction)
        if forced is not None:
            r, c, direction = forced
            continue

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
        assert len(options) <= 1, "ambiguous blocked turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _trace_original(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Original tracing rule, kept only for generator-side degeneration checks."""
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
        assert len(options) <= 1, "ambiguous original turn"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first real entrance column for row r, else None."""
    H, W = grid.shape
    if not (0 < r < H - 1) or int(grid[r, 0]) != PARTICLE:
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
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == int(grid[r - 1, entrance_col])), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])), W)
    if bar_above != bar_below or bar_above > 3:
        return None
    return entrance_col


def _solve_with_trace(grid: np.ndarray, trace_fn) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c = trace_fn(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles, with D9 crossings forcing a vertical detour."""
    return _solve_with_trace(grid, _trace)


def _allocate_branch_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    cands = []
    for c in range(bar_start + 4, W - 2):
        if all(abs(c - used) > _BRANCH_COL_PAD for used in used_cols):
            cands.append(c)
    if not cands:
        return None
    c = random.choice(cands)
    used_cols.add(c)
    return c


def _branch_choices(row: int, real_rows: list[int], H: int) -> list[tuple[str, int]]:
    """Possible one-sided branch destinations for this real row."""
    choices = []
    prev_real = max((x for x in real_rows if x < row), default=None)
    next_real = min((x for x in real_rows if x > row), default=None)

    up_internal = list(range((prev_real + 2) if prev_real is not None else 1, row, 2))
    up_internal = [x for x in up_internal if x % 2 == 1 and x <= row - 2]
    if prev_real is None:
        choices.append(("up", 0))
    if up_internal:
        choices.append(("up", random.choice(up_internal)))

    down_internal = list(range(row + 2, (next_real if next_real is not None else H - 1), 2))
    down_internal = [x for x in down_internal if x % 2 == 1]
    if next_real is None:
        choices.append(("down", H - 1))
    if down_internal:
        choices.append(("down", random.choice(down_internal)))

    return choices


def _carve_vertical_branch(g: np.ndarray, start_row: int, end_row: int, c: int) -> None:
    step = 1 if end_row > start_row else -1
    rr = start_row
    while True:
        g[rr, c] = BG
        if rr == end_row:
            break
        rr += step


def _paint_bars(
    g: np.ndarray,
    bar_rows: list[int],
    channel_is_real: dict[int, bool],
    extras: list[int],
    bar_start: int,
) -> dict[int, int] | None:
    """Assign colors to even rows; real channels use same top/bottom color."""
    H, W = g.shape
    del H
    row_color = {bar_rows[0]: random.choice(extras)}
    for i in range(len(bar_rows) - 1):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if channel_is_real[i]:
            row_color[bot] = row_color[top]
        else:
            row_color[bot] = random.choice([c for c in extras if c != row_color[top]])
    if len(set(row_color.values())) < _N_EXTRA_COLORS:
        return None
    for br in bar_rows:
        g[br, bar_start:W] = row_color[br]
    return row_color


def _count_divergent_particles(g: np.ndarray) -> int:
    H, _ = g.shape
    count = 0
    for r in range(1, H - 1):
        if _find_entrance_col(g, r) is None:
            continue
        if _trace(g, r, 0) != _trace_original(g, r, 0):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    bar_rows = list(range(0, H, 2))
    channel_rows = [r for r in range(1, H, 2)]
    n_channels = len(channel_rows)
    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_indices = sorted(random.sample(range(n_channels), n_real))
    real_set = set(real_indices)
    channel_is_real = {i: i in real_set for i in range(n_channels)}

    bar_start = random.choice(_BAR_START_OPTIONS)
    row_color = _paint_bars(g, bar_rows, channel_is_real, extras, bar_start)
    if row_color is None:
        return None

    real_rows = [channel_rows[i] for i in real_indices]
    forced_target = min(random.choice([2, 2, 3]), len(real_indices))
    used_cols: set[int] = set()
    forced_indices: set[int] = set()

    force_order = real_indices[:]
    random.shuffle(force_order)
    for idx in force_order:
        if len(forced_indices) >= forced_target:
            break
        c = _allocate_branch_col(bar_start, W, used_cols)
        if c is None:
            break
        row = channel_rows[idx]
        choices = _branch_choices(row, real_rows, H)
        if not choices:
            used_cols.remove(c)
            continue
        direction, end_row = random.choice(choices)
        top, bot = bar_rows[idx], bar_rows[idx + 1]
        start_row = top if direction == "up" else bot
        _carve_vertical_branch(g, start_row, end_row, c)
        forced_indices.add(idx)

    if len(forced_indices) < _MIN_DIVERGENT:
        return None

    for i in range(n_channels):
        if channel_is_real[i]:
            continue
        row = channel_rows[i]
        top, bot = bar_rows[i], bar_rows[i + 1]
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        usable = [c for c in range(bar_start + 1, W) if c not in used_cols and int(g[row, c]) == BG]
        if not usable:
            continue
        for c in random.sample(usable, min(n_decor, len(usable))):
            g[row, c] = random.choice([row_color[top], row_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = sum(1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(map(int, g.flatten().tolist()))) != 6:
        return None
    if _count_divergent_particles(g) < _MIN_DIVERGENT:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
