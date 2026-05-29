# Combo variant of cb2d8a2c (index 6)
# combo_spec: D3/0+D14/0
# Name: 反向距离+切比雪夫度量
#
# Design analysis
# - concrete_behavior:
#   The visible scene is still "azure field + alternating side-attached obstacle bars + one green border start",
#   but each bar now encodes its keep-out radius by its red remainder rather than its blue count, and that radius is
#   applied with Chebyshev distance. So a horizontal bar on the left wall blocks a square-cornered halo:
#   rows [bar_row-r, bar_row+r] and columns extending to the open side by r more cells.
# - small_trace:
#   Suppose all bars in one puzzle have shared length 8. A bar with blue_count=3 has red_count=5, so the combo rule
#   forbids every cell whose Chebyshev distance to that bar is <= 5. In the original task the same bar used radius 3
#   and Manhattan distance instead, so the path could squeeze diagonally much closer to the free end. Under the combo,
#   another shared-length bar with blue_count=4 has red_count=4, so it permits the closer pass even though it looks
#   "bluer"; that is the D3 inversion, while the square corner clearance is the D14 change.
# - complexity_calibration:
#   To make the inverted encoding legible, all bars in one generated sample share one total length and at least two
#   distinct blue counts. Then red_count = shared_length - blue_count is visually comparable across bars.
#   To make the Chebyshev metric legible, generation rejects cases where the combo path matches the red-count
#   Manhattan path; that avoids samples whose route never touches a corner-sensitive situation.
# - implicit_assumption_audit:
#   In main(), goal detection, path reconstruction, and blue->red recoloring survive unchanged. The only semantic
#   assumptions that break are "radius equals blue_count" and "distance is Manhattan". Those are isolated inside the
#   forbidden-mask builder and rewritten as parameters. The generator previously used blue radii in start spacing,
#   inter-bar spacing, and open-side feasibility; all three are now rewritten in terms of red_count because the
#   solver's actual safety margin comes from red cells.
# - placement_legality:
#   Bars remain 1-cell-thick, alternate walls, and never overlap because each occupies its own main-axis row/column.
#   Start-to-first-bar spacing is at least first_radius + 1 blank cells. Consecutive bars are separated by at least
#   radius_a + radius_b + 1 blank cells, which is exactly what two Chebyshev halos need to leave a safe traversal row.
#   Cross-axis open gap is at least radius + 1, guaranteeing a legal column/row just beyond the square halo.
# - visual_observability:
#   Every sample uses 2-3 bars with varying blue counts, so "more blue means less detour" can be compared directly.
#   The route bends around squared-off corners rather than diamond-like Manhattan corners, and generation filters out
#   cases where that second change would be invisible.
# - cleanliness_and_inference:
#   No extra markers are introduced. Bars, path, and the final blue->red recolor remain the only visual elements,
#   so the added reasoning burden comes from geometry and encoding, not from clutter.

import heapq
import itertools
import random

import numpy as np

from arc_utils import components_mask

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


def _start_and_goal(g: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        return sr, sc, 2, "row", g.shape[0] - 1
    if sr == g.shape[0] - 1:
        return sr, sc, 0, "row", 0
    if sc == 0:
        return sr, sc, 1, "col", g.shape[1] - 1
    if sc == g.shape[1] - 1:
        return sr, sc, 3, "col", 0
    raise AssertionError("green start must be on a border edge")


def _is_goal(goal_kind: str, goal_val: int, r: int, c: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _build_blocked_mask(
    g: np.ndarray,
    *,
    radius_from_red: bool,
    chebyshev_metric: bool,
) -> np.ndarray:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        coords = [(int(r), int(c)) for r, c in comp]
        blue_count = sum(1 for r, c in coords if g[r, c] == BLUE)
        radius = len(coords) - blue_count if radius_from_red else blue_count
        min_r = max(0, min(r for r, _ in coords) - radius)
        max_r = min(h - 1, max(r for r, _ in coords) + radius)
        min_c = max(0, min(c for _, c in coords) - radius)
        max_c = min(w - 1, max(c for _, c in coords) + radius)

        for rr in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                for br, bc in coords:
                    dr = abs(rr - br)
                    dc = abs(cc - bc)
                    dist = max(dr, dc) if chebyshev_metric else dr + dc
                    if dist <= radius:
                        forbidden[rr, cc] = True
                        break

    return bar_mask | forbidden


def _solve_grid(
    grid: np.ndarray,
    *,
    radius_from_red: bool,
    chebyshev_metric: bool,
) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)

    blocked = _build_blocked_mask(
        g,
        radius_from_red=radius_from_red,
        chebyshev_metric=chebyshev_metric,
    )
    blocked[sr, sc] = False

    inf_cost = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf_cost):
            continue
        if _is_goal(goal_kind, goal_val, r, c) and (r, c) != (sr, sc):
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
            if ncost < best_cost.get(key, inf_cost):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the opposite edge must exist"
    out = g.copy()
    state = goal_state
    while True:
        r, c, _ = state
        if out[r, c] == BG:
            out[r, c] = GREEN
        if (r, c) == (sr, sc):
            break
        state = prev[state]
    out[out == BLUE] = RED
    return out


def _path_mask(grid: np.ndarray) -> np.ndarray:
    return grid == GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green trail from one border to the opposite while obeying an inverted distance code and square keep-out zones around the obstacle bars.

    Input structure:
    - The background is azure (8), with 1-cell-thick bars made from blue (1) and red (2) cells attached to alternating side walls.
    - Exactly one green (3) pixel sits on a border and marks the route's start.

    Rule:
    - Each bar's safety radius is its red-cell count, i.e. total bar length minus blue-cell count, so bars with more blue allow a tighter pass.
    - That radius is measured with Chebyshev distance, so every bar creates a square-cornered forbidden halo rather than a Manhattan diamond.
    - The path uses orthogonal steps to reach the opposite edge, and all blue bar cells turn red in the output.

    Key insight: You must decode distance from the red remainder and then apply that value in the stricter chessboard metric, so the route both inverts the expected blue-distance relation and stays farther from diagonal bar corners.
    """
    return _solve_grid(grid, radius_from_red=True, chebyshev_metric=True)


def _min_main_len(radii: tuple[int, ...]) -> int:
    last_bar = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _length_bounds(cross_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    width = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, width - length :] = colors[::-1]


def _feasible_bar_specs(
    count: int,
    cross_size: int,
    main_hi: int,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    bounds = _length_bounds(cross_size)
    if bounds is None:
        return []

    choices: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for length in range(bounds[0], bounds[1] + 1):
        blue_hi = min(BLUE_PER_BAR_HI, (length + 1) // 2)
        if blue_hi < BLUE_PER_BAR_LO:
            continue
        for blues in itertools.product(range(BLUE_PER_BAR_LO, blue_hi + 1), repeat=count):
            if len(set(blues)) < 2:
                continue
            reds = tuple(length - blue for blue in blues)
            if cross_size - length < max(reds) + 1:
                continue
            if _min_main_len(reds) > main_hi:
                continue
            choices.append((length, blues, reds))
    return choices


def _canonical_grid(
    main_size: int,
    cross_size: int,
    length: int,
    blues: tuple[int, ...],
    reds: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    slack = main_size - _min_main_len(reds)
    if slack < 0:
        return None

    extras = _random_composition(slack, len(reds) + 1)
    row = reds[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(reds)):
        row += reds[i - 1] + reds[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(len(blues)))):
        _paint_bar(grid, row, side, length, blue_count)

    if first_side == 0:
        start_col = random.randint(1, length - 1)
    else:
        start_col = random.randint(cross_size - length, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _combo_observable(inp: np.ndarray, combo_out: np.ndarray) -> bool:
    combo_path = _path_mask(combo_out)

    try:
        manhattan_out = _solve_grid(inp, radius_from_red=True, chebyshev_metric=False)
    except AssertionError:
        manhattan_out = None
    try:
        blue_cheb_out = _solve_grid(inp, radius_from_red=False, chebyshev_metric=True)
    except AssertionError:
        blue_cheb_out = None

    metric_visible = manhattan_out is None or not np.array_equal(combo_path, _path_mask(manhattan_out))
    radius_visible = blue_cheb_out is None or not np.array_equal(combo_path, _path_mask(blue_cheb_out))
    return metric_visible and radius_visible


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    param_choices: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for cross_size in range(cross_lo, cross_hi + 1):
        for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1):
            for length, blues, reds in _feasible_bar_specs(count, cross_size, main_hi):
                param_choices.append((cross_size, length, blues, reds))
    if not param_choices:
        return None

    cross_size, length, blues, reds = random.choice(param_choices)
    main_size = random.randint(max(main_lo, _min_main_len(reds)), main_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, length, blues, reds, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
    except Exception:
        return None
    if not _reaches_opposite_edge(out, edge):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if not _combo_observable(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
