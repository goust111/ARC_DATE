# Combo variant of cb2d8a2c (index 8)
# combo_spec: D4/0+D17/1
# Name: 周期性涂色+双像素宽路径
#
# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    - Canonical orientation keeps the original object grammar: azure background, horizontal
#      side-attached red/blue bars, and one green seed on the top border.
#    - The original solver searched a 1-cell-wide centerline. Here the moving object is a
#      2x2 anchor block whose successive placements overlap, so a straight descent becomes a
#      2-pixel-wide ribbon and a turn automatically produces a filled 2x2 corner.
#    - Example: if the canonical anchor path is (0,5)->(1,5)->(2,5)->(2,6), the old task
#      would paint a thin green elbow. The fused task paints 2x2 squares at those anchors:
#      rows 0-2, cols 5-6 form a vertical band, then rows 2-3, cols 5-7 form the horizontal
#      detour. Coloring anchors in order G,Y,G,Y makes the straight part alternate by row and
#      the corner show a mixed checker-like elbow instead of a solid green bend.
# b) Complexity calibration:
#    - The periodic rule uses only two colors, green and yellow, alternating every anchor.
#      This is strong enough to be visible but still easy to infer from 2-4 examples.
#    - The width change is encoded geometrically by the 2x2 sweep rather than by ad hoc extra
#      pixels, so every sample repeats the same corner motif and does not collapse back to a
#      near-original thin path.
# c) Hidden-assumption audit against the original solver:
#    - Original assumption: one search state equals one painted cell. Broken. Now one search
#      state is the top-left of a 2x2 occupied block, and legality must be checked on all four
#      cells of that block.
#    - Original inter-bar spacing and side-gap formulas only protected a centerline. Broken.
#      A thick path needs one extra free row between successive keep-out halos and one extra
#      free cross-axis cell on the open side of each bar.
#    - Original corner handling was implicit because single-cell steps never leave holes.
#      Broken. The 2x2-anchor sweep is chosen specifically so turns stay filled and readable.
# d) Placement legality:
#    - The generator strengthens the canonical layout constraints to row_gap >= blue_a+blue_b+3,
#      top gap >= first_blue+2, bottom gap >= last_blue+3, and open-side gap >= blue+2.
#      These conditions guarantee a legal route for 2x2 anchors without overlap or clipping.
# e) Visual observability / cleanliness:
#    - The output differs from the source task in two independently visible ways: the path is a
#      thick ribbon, and yellow appears only on alternating ribbon slices. Bars remain thin and
#      side-attached, so obstacle objects and trajectory objects stay visually separable.
# f) Generator-enforced assumptions used by main():
#    - After rotation to canonical form, the unique green seed is the left cell of the initial
#      2-cell border mouth, so its column is never the final column.
#    - Red/blue bars remain 4-connected non-touching components, sorted by row, and their
#      stronger spacing prevents the 2x2 route from needing impossible diagonal squeezing.

from __future__ import annotations

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

PATH_CYCLE = (GREEN, YELLOW)
PATH_COLORS = {GREEN, YELLOW}
DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


def _to_canonical_top(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so the unique green start lies on the top edge."""
    g = np.array(grid, copy=True)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    if sr == 0:
        return g, "top"
    if sr == h - 1:
        return np.flipud(g).copy(), "bottom"
    if sc == 0:
        return g.T.copy(), "left"
    if sc == w - 1:
        return np.rot90(g, 1).copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical_top(grid: np.ndarray, edge: str) -> np.ndarray:
    """Inverse of `_to_canonical_top`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.rot90(grid, -1).copy()  # edge == "right"


def _build_blocked_cells(grid: np.ndarray) -> np.ndarray:
    """Cells that a width-2 path may not occupy."""
    g = np.array(grid, copy=False)
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)
    assert bar_comps, "grid must contain at least one red/blue bar"

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        radius = int(sum(1 for r, c in comp if g[r, c] == BLUE))
        assert radius >= 1, "each bar must contain at least one blue pixel"

        dist = np.full((h, w), -1, dtype=np.int16)
        q: deque[tuple[int, int]] = deque()
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    return bar_mask | forbidden


def _square_is_clear(blocked: np.ndarray, r: int, c: int) -> bool:
    """Whether the 2x2 anchor square with top-left `(r,c)` is fully legal."""
    h, w = blocked.shape
    if not (0 <= r < h - 1 and 0 <= c < w - 1):
        return False
    return not bool(np.any(blocked[r : r + 2, c : c + 2]))


def _search_anchor_path(canonical: np.ndarray) -> list[tuple[int, int]]:
    """Find a shortest low-turn path of 2x2 anchors from the top edge to the bottom edge."""
    g = np.array(canonical, copy=False)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the green start on the top edge"
    assert sc <= w - 2, "canonical start must be the left cell of a 2-cell border mouth"

    blocked = _build_blocked_cells(g)
    blocked[sr, sc] = False
    assert _square_is_clear(blocked, sr, sc), "start 2x2 mouth must be clear"

    init_dir = 2  # down
    goal_row = h - 2
    inf = (10**9, 10**9)

    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if r == goal_row and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not _square_is_clear(blocked, rr, cc):
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid width-2 path to the opposite edge must exist"

    anchors: list[tuple[int, int]] = []
    state = goal_state
    while True:
        r, c, _ = state
        anchors.append((r, c))
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    anchors.reverse()
    return anchors


def _paint_periodic_path(grid: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    """Paint the width-2 anchor ribbon with alternating colors."""
    out = np.array(grid, copy=True)
    for i, (r, c) in enumerate(anchors):
        color = int(PATH_CYCLE[i % len(PATH_CYCLE)])
        for rr in range(r, r + 2):
            for cc in range(c, c + 2):
                if out[rr, cc] not in (BLUE, RED):
                    out[rr, cc] = color
    out[out == BLUE] = RED
    return out


def _solve_canonical(canonical: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    anchors = _search_anchor_path(canonical)
    return _paint_periodic_path(canonical, anchors), anchors


def _path_turns(anchors: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir: tuple[int, int] | None = None
    for (r0, c0), (r1, c1) in zip(anchors, anchors[1:]):
        cur_dir = (r1 - r0, c1 - c0)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a two-pixel-wide striped ribbon from one border to the opposite, staying outside the red bars' blue-count keep-out halos.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) remain 1-cell-thick side-attached obstacles.
    - A single green (3) pixel sits on one border edge and marks the left/top cell of the ribbon's starting mouth.

    Rule:
    - Each bar's blue-count still encodes a Manhattan keep-out radius, but the moving object is now a 2x2 anchor square. A legal route is therefore a chain of adjacent 2x2 placements whose occupied cells all stay outside every keep-out halo.
    - Among legal routes, use the shortest one and break ties by fewer turns, exactly as in the source task. Paint each successive 2x2 anchor alternately green, yellow, green, yellow, so straight segments become striped and corners become filled mixed-color elbows.
    - After the ribbon is painted, all blue pixels on the obstacle bars turn red.

    Key insight: The obstacle semantics are unchanged, but the path is no longer a thin centerline. Solving the fused task requires inflating the mover to width 2 and then extending an alternating color cycle across the entire thick ribbon.
    """
    canonical, edge = _to_canonical_top(np.asarray(grid, dtype=int))
    solved, _anchors = _solve_canonical(canonical)
    return _from_canonical_top(solved, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum canonical main-axis size for a legal width-2 route."""
    last_row = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_row += a + b + 3
    return last_row + blues[-1] + 3


def _random_composition(total: int, parts: int) -> list[int]:
    """Distribute `total` into `parts` non-negative integers uniformly."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    assert choices, "no blue-count combination fits the current main-axis bound"
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range for a width-2 detour on the open side."""
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 2, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint one alternating blue/red bar on the given canonical row."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the canonical top-start input grid."""
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 3 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        low = 1
        high = first_len - 2
    else:
        low = max(1, cross_size - first_len)
        high = cross_size - 2
    if low > high:
        return None
    grid[0, random.randint(low, high)] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start grid so the start edge matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.rot90(grid, -1).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output reaches the edge opposite to `edge`."""
    if edge == "top":
        return bool(np.any(np.isin(out[-1, :], list(PATH_COLORS))))
    if edge == "bottom":
        return bool(np.any(np.isin(out[0, :], list(PATH_COLORS))))
    if edge == "left":
        return bool(np.any(np.isin(out[:, -1], list(PATH_COLORS))))
    return bool(np.any(np.isin(out[:, 0], list(PATH_COLORS))))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt, validated by the fused solver."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size_lo = max(main_lo, _min_main_len(blues))
    if main_size_lo > main_hi:
        return None

    main_size = random.randint(main_size_lo, main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None

    try:
        canonical_out, anchors = _solve_canonical(canonical)
    except Exception:
        return None

    if len(anchors) < 6:
        return None
    if _path_turns(anchors) < 2:
        return None
    yellow_count = int(np.sum(canonical_out == YELLOW))
    green_count = int(np.sum(canonical_out == GREEN))
    if yellow_count < 6 or green_count < 6:
        return None

    inp = _orient_grid(canonical, edge)
    out = _orient_grid(canonical_out, edge)
    if not _reaches_opposite_edge(out, edge):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
