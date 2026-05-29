# Combo variant of cb2d8a2c (index 1)
# combo_spec: D1/0+D2/2
# Name: 对角线反射导航

# Design analysis for the fused variant:
# - Concrete behavior: in canonical top-to-bottom orientation, every obstacle is still a
#   horizontal bar attached to alternating side walls, but the green trace is no longer
#   a shortest orthogonal detour. It is a 45-degree ray. Example: if a left-attached bar
#   has blue-count 2 and exposed keep-out side at column 8, then a ray coming from the
#   right descends diagonally until its next step would enter that keep-out rectangle;
#   the current cell sits at column 9, so the collision is against a vertical boundary
#   and only the horizontal component flips. The path therefore keeps moving downward but
#   changes from down-left to down-right. Relative to the original task, the same blue
#   counts still control clearance, yet the visible motion changes from U/R/D/L elbows to
#   diagonal zigzags with mirror bounces.
# - Complexity calibration: the path uses only the four 45-degree directions and each
#   sample contains 2-4 bars, so the reflection rule repeats enough times to be learned.
#   That keeps the geometry visibly different from the original without creating too many
#   free movement modes.
# - Broken original assumptions: the old solver assumed Manhattan buffering, generic
#   Dijkstra search, and turn cost minimization. None of those survive. Here the keep-out
#   zone is modeled with Chebyshev expansion so a bar induces a rectangular forbidden
#   mirror, the path is deterministic once the first side is known, and turning happens
#   only by reflection at forbidden boundaries.
# - Placement legality: the generator works output-first. It first chooses the bounce
#   columns implied by bar lengths, then derives the bar rows from diagonal travel times.
#   This guarantees no overlap, no out-of-bounds placement, and enough vertical spacing
#   between keep-out rectangles: consecutive bounce-row gaps are forced to be at least
#   blue_i + blue_j + 1.
# - Visual observability / cleanliness: bars remain thin and side-attached, blue still
#   counts the danger radius, and the green result becomes a clean sawtooth path. Because
#   the generator requires at least two reflections, the fused behavior is readable from
#   a few examples instead of collapsing into a nearly straight line.
# - New solver assumptions guaranteed by the generator: after rotating to canonical
#   orientation, bars are side-attached horizontal components sorted by row, the first
#   bar determines the initial diagonal direction, and collisions happen on the intended
#   keep-out boundary so the reflected path reaches the opposite border without looping.

import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


def _to_canonical_top(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate/flip so the single green start lies on the top edge."""
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


def _bar_side(comp: np.ndarray, w: int) -> int:
    """0 = touches left wall, 1 = touches right wall."""
    touch_left = bool(np.any(comp[:, 1] == 0))
    touch_right = bool(np.any(comp[:, 1] == w - 1))
    assert touch_left ^ touch_right, "each bar must touch exactly one side wall"
    return 0 if touch_left else 1


def _canonical_blocked(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build rectangular Chebyshev keep-out zones around every bar component."""
    g = np.array(grid, copy=False)
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    comps = components_mask(bar_mask, connectivity=4)
    assert comps, "grid must contain at least one red/blue bar"

    blocked = np.zeros((h, w), dtype=bool)
    bars: list[dict[str, int]] = []
    for comp in comps:
        pts = np.array(comp, dtype=int)
        blue_count = int(sum(1 for r, c in pts if g[r, c] == BLUE))
        assert blue_count >= 1, "each bar must contain at least one blue pixel"

        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        rr0 = max(0, r0 - blue_count)
        rr1 = min(h - 1, r1 + blue_count)
        cc0 = max(0, c0 - blue_count)
        cc1 = min(w - 1, c1 + blue_count)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        bars.append(
            {
                "row": r0,
                "blue": blue_count,
                "side": _bar_side(pts, w),
            }
        )
    return blocked, sorted(bars, key=lambda x: x["row"])


def _trace_canonical_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    """Trace the diagonal ray in canonical top-start orientation."""
    g = np.array(grid, copy=False)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the green start on the top edge"

    blocked, bars = _canonical_blocked(g)
    blocked[sr, sc] = False
    init_dc = -1 if bars[0]["side"] == 0 else 1

    path = [(sr, sc)]
    r, c = sr, sc
    dr, dc = 1, init_dc
    reflections = 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        if r == h - 1 and (r, c) != (sr, sc):
            return path, reflections, len(bars)

        state = (r, c, dr, dc)
        assert state not in seen, "reflection loop encountered"
        seen.add(state)

        nr, nc = r + dr, c + dc
        assert 0 <= nr < h and 0 <= nc < w, "ray left the grid before reaching the opposite edge"

        if not blocked[nr, nc]:
            r, c = nr, nc
            path.append((r, c))
            continue

        vertical_hit = 0 <= c + dc < w and blocked[r, c + dc]
        horizontal_hit = 0 <= r + dr < h and blocked[r + dr, c]

        if vertical_hit:
            dc = -dc
        if horizontal_hit:
            dr = -dr
        if not vertical_hit and not horizontal_hit:
            dr = -dr
            dc = -dc
        reflections += 1


def _solve_canonical(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Solve a canonical top-start instance and return output plus reflection stats."""
    g = np.array(grid, copy=True)
    out = g.copy()
    path, reflections, bar_count = _trace_canonical_path(g)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out, reflections, bar_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border green pixel as a 45-degree ray that ricochets around side-attached bars whose blue pixels encode the mirror offset.

    Input structure:
    - bg = azure (8). Red/blue side-attached bars (colors 2/1) remain the obstacle objects.
    - A single green (3) pixel sits on one border edge and marks the ray source.

    Rule:
    - Each bar's blue-count gives a keep-out radius, but the forbidden region is interpreted in diagonal geometry: expand the bar by Chebyshev distance to get a rectangular no-go mirror.
    - From the border, the green path travels one cell per step on a 45-degree diagonal. When the next step would enter a forbidden region, the path reflects with equal incidence and reflection: flip the horizontal component on a vertical-face hit, the vertical component on a horizontal-face hit, or both on a corner hit.
    - The ray continues until it reaches the opposite border, and all blue pixels turn red in the output.

    Key insight: Blue pixels still specify how far the path must stay from each bar, but now that clearance boundary acts like a mirror, so solving the task means inferring diagonal travel plus bounce geometry rather than orthogonal detours.
    """
    canonical, edge = _to_canonical_top(np.asarray(grid, dtype=int))
    out, _reflections, _bar_count = _solve_canonical(canonical)
    return _from_canonical_top(out, edge)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint one canonical horizontal bar on `row`."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _bounce_col(side: int, cross_size: int, length: int, blue_count: int) -> int:
    """Column of the path cell immediately before reflecting off this bar."""
    if side == 0:
        return length + blue_count
    return cross_size - length - blue_count - 1


def _canonical_grid(
    main_lo: int,
    main_hi: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build a canonical top-start instance by planning diagonal bounce points first."""
    count = len(blues)
    sides = tuple((first_side + i) % 2 for i in range(count))
    length_choices: list[list[int]] = []
    for blue_count in blues:
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length_choices.append(list(range(bounds[0], bounds[1] + 1)))

    for _ in range(320):
        lengths = [random.choice(choices) for choices in length_choices]
        bounce_cols = [_bounce_col(side, cross_size, length, blue) for side, length, blue in zip(sides, lengths, blues)]

        if any(not (1 <= c <= cross_size - 2) for c in bounce_cols):
            continue
        if any(abs(bounce_cols[i] - bounce_cols[i - 1]) < blues[i - 1] + blues[i] + 1 for i in range(1, count)):
            continue

        start_min = blues[0] + 1
        start_max = (cross_size - 1 - bounce_cols[0]) if sides[0] == 0 else bounce_cols[0]
        if start_min > start_max:
            continue

        final_dc = 1 if sides[-1] == 0 else -1
        end_min = 1
        end_max = (cross_size - 1 - bounce_cols[-1]) if final_dc == 1 else bounce_cols[-1]
        if end_min > end_max:
            continue

        inter_sum = sum(abs(bounce_cols[i] - bounce_cols[i - 1]) for i in range(1, count))
        total_min = start_min + inter_sum + end_min + 1
        total_max = start_max + inter_sum + end_max + 1
        if total_max < main_lo or total_min > main_hi:
            continue

        main_size = random.randint(max(main_lo, total_min), min(main_hi, total_max))
        remaining = main_size - inter_sum - 1
        start_lo = max(start_min, remaining - end_max)
        start_hi = min(start_max, remaining - end_min)
        if start_lo > start_hi:
            continue

        start_gap = random.randint(start_lo, start_hi)
        end_gap = remaining - start_gap
        if not (end_min <= end_gap <= end_max):
            continue

        start_col = bounce_cols[0] + start_gap if sides[0] == 0 else bounce_cols[0] - start_gap
        exit_col = bounce_cols[-1] + final_dc * end_gap
        if not (0 <= start_col < cross_size and 0 <= exit_col < cross_size):
            continue

        bounce_rows = [start_gap]
        for i in range(1, count):
            bounce_rows.append(bounce_rows[-1] + abs(bounce_cols[i] - bounce_cols[i - 1]))
        bar_rows = [row + 1 for row in bounce_rows]

        grid = np.full((main_size, cross_size), BG, dtype=int)
        for row, side, length, blue_count in zip(bar_rows, sides, lengths, blues):
            _paint_bar(grid, row, side, length, blue_count)
        grid[0, start_col] = GREEN

        try:
            out, reflections, bar_count = _solve_canonical(grid)
        except Exception:
            continue
        if reflections != count or bar_count != count:
            continue
        if out[-1, exit_col] != GREEN:
            continue
        return grid
    return None


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start grid to the requested starting edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.rot90(grid, -1).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    cross_size = random.randint(cross_lo, cross_hi)
    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = tuple(random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count))
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_lo, main_hi, cross_size, blues, first_side)
    if canonical is None:
        return None

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
