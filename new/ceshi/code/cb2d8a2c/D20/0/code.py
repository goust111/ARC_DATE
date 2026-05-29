# Variant of cb2d8a2c
# Dimension: D20 — 障碍条上蓝色像素的角色 (meta)
# Change: 蓝色像素承担单一角色：计数编码距离约束参数，最后统一变为红色... → 蓝色像素标记障碍条的'通行口'

# Design analysis for the D20 variant
# -----------------------------------
# concretization:
#   - The grid still contains a single green start pixel on one border, plus 1-cell-thick side-attached bars.
#   - Red cells are solid barrier cells. Blue cells are no longer "danger counters"; each blue cell is now a hole in the bar.
#   - The green path may step onto blue cells and pass through the bar at exactly those blue coordinates.
#   - Output keeps unused blue holes blue. Only the holes that are actually traversed become green because the path occupies them.
# behavior_trace:
#   - Original task on a bar with three blue cells:
#       input meaning = radius 3 keep-out zone, so the path must detour well outside the bar.
#   - This variant on the same visual bar:
#       input meaning = three candidate crossing holes, so the path is allowed to pierce the bar at those blue positions.
#   - Code impact:
#       1) the Manhattan-radius flood fill is removed entirely;
#       2) blue cells must not be included in `blocked`;
#       3) the old `out[out == BLUE] = RED` would erase the new semantics and is therefore invalid.
# complexity_calibration:
#   - A trivial generator could make one perfectly vertical tunnel through every bar, but that would underuse the task structure.
#   - To keep the task legible yet nontrivial, bars remain long and alternate wall attachment as before, while mandatory gates are placed in
#     an overlapping central band. The path still tends to move mostly across the grid's main axis, but it may need small side steps between bars.
#   - Each bar may carry 1-4 blue gates; extra unused gates remain visible so the "more blue = more options" rule is observable from examples.
# implicit-assumption audit against the original solver:
#   - `bar_comps = components_mask(bar_mask, ...)` was previously used only to count blue pixels per bar for a keep-out radius.
#     That premise no longer holds, so counting blue pixels must not affect motion.
#   - `forbidden` expansion assumed every bar cell repels the path; in the variant, the path can occupy blue cells, so this whole block must go.
#   - `blocked = bar_mask | forbidden` used to block blue and red alike. That is now wrong; only red stays blocked.
#   - `out[out == BLUE] = RED` was harmless before because blue had no persistent identity. It is harmful now because it destroys visible holes.
# placement legality:
#   - The generator still uses 12-29 x 13-26 grids, 1-4 bars, alternating wall sides, and 6-15 cell bar lengths spanning 32-79% of the cross axis.
#   - Bars are chosen to cover more than half of the cross axis so opposite-side bars overlap in a central band.
#   - One mandatory gate per bar is planted in that overlap band, making gate usage cheaper than routing to the open bar end.
#   - All blue gate positions are non-adjacent within a bar, so each hole reads as a distinct opening rather than a blurry blue segment.
# visual observability and cleanliness:
#   - Red cells remain solid bars, blue cells remain sparse portal markers, and green only appears as the path.
#   - Because every accepted sample must actually turn at least one blue cell per bar into green, the defining D20 behavior is always visible.
# color-change inference:
#   - There is no hidden color lookup. Blue does not globally map to red anymore.
#   - The visible rule is local and structural: "green may replace a blue cell exactly when that blue cell lies on the traversed route."
# new main() assumptions and generator guarantees:
#   - main() assumes exactly one green start on the border; the generator enforces that.
#   - main() assumes obstacle cells are only red/blue line bars and that a path to the opposite edge exists; the generator validates this by calling main().
#   - main() allows arbitrary blue placements, but the generator specifically guarantees at least one used gate on every bar so the variant does not collapse back toward the original behavior.

import heapq
import itertools
import random

import numpy as np

from arc_utils import components_mask

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path from one grid edge to the opposite, using blue-marked openings to cross red obstacle bars.

    Input structure:
    - bg = azure (8). Each obstacle is a 1-cell-thick red/blue bar attached to one side wall.
    - A single green (3) pixel sits on a grid border as the path start.
    - Red (2) cells are solid parts of the bar, while blue (1) cells mark the bar's pass-through holes.

    Rule:
    - Grow the green path from the border start to the opposite edge using orthogonal moves.
    - The path may travel through bg cells and blue cells, but it may not enter red cells.
    - When the path uses a blue opening, that blue cell becomes part of the green path; unused blue openings stay blue.

    Key insight: Blue pixels no longer encode a distance parameter; they are the only places where a bar may be pierced, so their positions steer the route directly.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    BLUE = 1
    RED = 2
    GREEN = 3

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    blocked = g == RED
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the opposite edge must exist"

    out = g.copy()
    path: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    path.reverse()

    for r, c in path:
        if out[r, c] != RED:
            out[r, c] = GREEN
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4  # number of gate holes per bar
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
BAR_CROSS_PREFERRED_LO = 55  # keep bars long enough that alternating sides overlap centrally
START_GAP = 2
BAR_ROW_STEP = 2
END_GAP = 1
MAX_TRACK_SHIFT = 2
MAX_GEN_RETRIES = 300


def _min_main_len(count: int) -> int:
    """Minimum main-axis size to place `count` bars with at least one empty row between them."""
    return START_GAP + (count - 1) * BAR_ROW_STEP + END_GAP + 1


def _random_composition(total: int, parts: int) -> list[int]:
    """Distribute `total` into `parts` non-negative integers uniformly (stars-and-bars)."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and gate count."""
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
        (BAR_CROSS_PREFERRED_LO * cross_size + 99) // 100,
    )
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _interval(cross_size: int, side: int, length: int) -> tuple[int, int]:
    """Inclusive covered interval for a bar attached to the given wall side."""
    if side == 0:
        return 0, length - 1
    return cross_size - length, cross_size - 1


def _choose_gate_offsets(length: int, gate_count: int, mandatory_offset: int) -> tuple[int, ...]:
    """Choose non-adjacent gate offsets including the required opening."""
    combos = [
        combo
        for combo in itertools.combinations(range(length), gate_count)
        if mandatory_offset in combo and all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    assert combos, "length bounds should guarantee a valid non-adjacent gate placement"
    return random.choice(combos)


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, gate_offsets: tuple[int, ...]) -> None:
    """Paint a side-attached bar with red body cells and blue gate holes."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    for offset in gate_offsets:
        colors[offset] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length:] = colors


def _choose_lengths_and_track(
    cross_size: int,
    gate_counts: tuple[int, ...],
    first_side: int,
) -> tuple[list[int], list[tuple[int, int]], list[int]] | None:
    """Pick bar lengths plus a mandatory gate column for each bar."""
    lengths: list[int] = []
    intervals: list[tuple[int, int]] = []
    for i, gate_count in enumerate(gate_counts):
        side = (first_side + i) % 2
        bounds = _length_bounds(cross_size, gate_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        intervals.append(_interval(cross_size, side, length))

    band_lo = max(lo for lo, _ in intervals)
    band_hi = min(hi for _, hi in intervals)
    if band_lo > band_hi:
        return None

    gate_cols: list[int] = []
    current = random.randint(band_lo, band_hi)
    for _ in gate_counts:
        low = max(band_lo, current - MAX_TRACK_SHIFT)
        high = min(band_hi, current + MAX_TRACK_SHIFT)
        if low > high:
            low, high = band_lo, band_hi
        current = random.randint(low, high)
        gate_cols.append(current)
    return lengths, intervals, gate_cols


def _canonical_grid(
    main_size: int,
    cross_size: int,
    gate_counts: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
    count = len(gate_counts)
    slack = main_size - _min_main_len(count)
    if slack < 0:
        return None

    choice = _choose_lengths_and_track(cross_size, gate_counts, first_side)
    if choice is None:
        return None
    lengths, intervals, gate_cols = choice

    extras = _random_composition(slack, count + 1)
    row = START_GAP + extras[0]
    rows = [row]
    for i in range(1, count):
        row += BAR_ROW_STEP + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, length, interval, gate_col, gate_count in zip(rows, lengths, intervals, gate_cols, gate_counts):
        side = 0 if interval[0] == 0 else 1
        mandatory_offset = gate_col - interval[0]
        gate_offsets = _choose_gate_offsets(length, gate_count, mandatory_offset)
        _paint_bar(grid, row, side, length, gate_offsets)

    grid[0, gate_cols[0]] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-start) grid so the start edge matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output has green on the edge opposite to `edge`."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _bars_crossed_via_gates(inp: np.ndarray, out: np.ndarray) -> int:
    """Count how many obstacle bars have at least one blue cell used by the green path."""
    bar_mask = (inp == BLUE) | (inp == RED)
    crossed = 0
    for comp in components_mask(bar_mask, connectivity=4):
        if any(inp[r, c] == BLUE and out[r, c] == GREEN for r, c in comp):
            crossed += 1
    return crossed


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build a candidate grid and validate it through the solver."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    gate_counts = tuple(random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count))
    main_size = random.randint(max(main_lo, _min_main_len(count)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, gate_counts, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if _bars_crossed_via_gates(inp, out) != count:
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D20 input")
