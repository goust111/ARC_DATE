# Combo variant of cb2d8a2c (index 7)
# combo_spec: D4/2+D9/1
# Name: 路径颜色随障碍变化+蓝色变绿

# Design analysis for combo D4/2 + D9/1
# =======================================
# concrete_behavior:
#   original:
#     - The green seed starts on one border and the solver finds the shortest low-turn route
#       to the opposite border while staying outside each bar's Manhattan keep-out radius.
#     - Every background path cell becomes green.
#     - Every blue obstacle pixel becomes red, so each bar ends as a solid red strip.
#   fused_variant:
#     - Path geometry is unchanged: blue counts still define keep-out radii and Dijkstra still
#       minimizes (steps, turns).
#     - Output painting now has two coupled effects:
#       1) Each bar owns a main-axis routing slab
#          [main-blue_count-1, main+blue_count+1]. Background path cells inside that slab are
#          recolored by a fixed palette:
#            1 blue -> yellow(4)
#            2 blue -> gray(5)
#            3 blue -> magenta(6)
#            4 blue -> orange(7)
#       2) Every blue obstacle pixel becomes green(3), not red.
#     - Small trace:
#       If the trip is vertical and a left bar sits on row 8 with blue_count=3, then the
#       detour rows 4..12 are painted magenta(6). The blue cells embedded in that bar also
#       become green in the output, so the same blue-count clue controls both the detour color
#       and the bar recolor.
#
# complexity_calibration:
#   - The path-color map stays global and fixed. A per-instance palette would be too hard to
#     infer from 2-4 ARC examples.
#   - The number of path colors is capped at four, matching the original blue-count range.
#   - The generator forces 2-4 bars and at least two distinct blue counts, so samples usually
#     show repeated count->color evidence instead of collapsing to one detour color.
#   - Blue->green is intentionally simple and global; the structural challenge remains in
#     connecting those green bar pixels to the nearby slab color changes.
#
# hidden_assumptions_audit:
#   original_main_dependencies:
#     - Path search depends only on blocked cells, not on final colors. Safe to reuse.
#     - The original paint loop assumed every background path cell should become green.
#       That is no longer valid because slab ownership can override the path color.
#     - The old `out[out == BLUE] = RED` is invalid because blue must remain visible as green.
#   fused assumptions:
#     - Each obstacle component is still a single 1-cell-thick bar, so it has one unique
#       main-axis coordinate. The generator keeps that invariant.
#     - Consecutive slabs must not overlap ambiguously. The original spacing rule implies
#       disjoint or exactly touching slabs:
#       next_main - prev_main >= prev_blue + next_blue + 2.
#     - Blue->green happens after bar extraction, otherwise the solver would lose the input
#       cue used to count each bar's blue pixels.
#
# placement_legality:
#   - The generator keeps the original wall-attached bar construction, alternating sides,
#     length bounds, and keep-out spacing, so bars never overlap or leave the grid.
#   - No extra input objects are added for the combo; slab ownership is derived only from the
#     placed bars, so there is no new collision surface.
#   - Start placement still lies within the first bar's cross-span, preserving the intended
#     routing shape around the first obstacle.
#
# visual_observability:
#   - A human sees the same blue-count clue twice in the output:
#     more blue markers on a bar imply a wider detour and a specific detour color, while those
#     exact blue markers become green.
#   - The path colors use colors absent from the input obstacles/background, so the multicolor
#     bypass stands out cleanly.
#   - The generator rejects samples that do not show at least two distinct slab colors or that
#     fail to leave any blue->green obstacle conversion.
#
# cleanliness_and_inference:
#   - Output roles stay readable: red = bar body, green = seed / absorbed blue markers /
#     any non-slab path cells, and {4,5,6,7} = obstacle-specific detour bands.
#   - The fused rule is not a loose concatenation: the same blue-count signal controls both
#     the path detour color and the obstacle recolor, so the two changes reinforce one clue.

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
PATH_COLOR_BY_BLUE = {1: 4, 2: 5, 3: 6, 4: 7}


def _collect_bars(
    g: np.ndarray,
    *,
    path_is_vertical: bool,
) -> list[dict[str, object]]:
    """Extract obstacle bars together with their blue counts and main-axis line."""
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, object]] = []
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        assert blue_count in PATH_COLOR_BY_BLUE, "blue_count must stay within palette range"
        main_values = sorted({int(r if path_is_vertical else c) for r, c in comp})
        assert len(main_values) == 1, "each bar must occupy exactly one main-axis line"
        bars.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "blue_count": int(blue_count),
                "main": int(main_values[0]),
            }
        )
    return bars


def _build_blocked(
    h: int,
    w: int,
    bars: list[dict[str, object]],
) -> np.ndarray:
    """Expand every bar by its Manhattan keep-out radius."""
    forbidden = np.zeros((h, w), dtype=bool)
    bar_mask = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["blue_count"])
        coords = list(bar["coords"])

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in coords:
            bar_mask[r, c] = True
            dist[r, c] = 0
            q.append((r, c))

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


def _routing_slabs(
    bars: list[dict[str, object]],
    *,
    axis_limit: int,
) -> list[tuple[int, int, int]]:
    """Return ordered main-axis intervals paired with their detour color."""
    slabs: list[tuple[int, int, int]] = []
    for bar in bars:
        blue_count = int(bar["blue_count"])
        main = int(bar["main"])
        lo = max(0, main - blue_count - 1)
        hi = min(axis_limit - 1, main + blue_count + 1)
        slabs.append((lo, hi, PATH_COLOR_BY_BLUE[blue_count]))
    slabs.sort()
    for (_, prev_hi, _), (next_lo, _, _) in zip(slabs, slabs[1:]):
        assert prev_hi <= next_lo, "routing slabs must stay ordered"
    return slabs


def _paint_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    *,
    path_is_vertical: bool,
    slabs: list[tuple[int, int, int]],
) -> None:
    """Paint background path cells using the bar-controlled slab palette."""
    for r, c in path:
        if out[r, c] != BG:
            continue
        main_value = int(r if path_is_vertical else c)
        color = GREEN
        for lo, hi, slab_color in slabs:
            if lo <= main_value <= hi:
                color = slab_color
                break
        out[r, c] = color


def _path_colors() -> set[int]:
    return {GREEN, *PATH_COLOR_BY_BLUE.values()}


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """Quantify whether both halves of the combo are visibly active."""
    path_palette_used = {int(v) for v in np.unique(out) if int(v) in PATH_COLOR_BY_BLUE.values()}
    return {
        "detour_colors": len(path_palette_used),
        "absorbed_blue_pixels": int(np.count_nonzero((inp == BLUE) & (out == GREEN))),
        "plain_green_bg_path": int(np.count_nonzero((inp == BG) & (out == GREEN))),
        "colored_bg_path": int(
            np.count_nonzero(np.isin(out, list(PATH_COLOR_BY_BLUE.values())) & (inp == BG))
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a border-started route to the opposite edge, coloring each obstacle bypass by that bar's blue-count palette while turning blue markers green.

    Input structure:
    - bg = azure (8). Obstacles are 1-cell-thick bars made of red (2) and blue (1) cells, attached to alternating side walls.
    - A single green (3) pixel lies on one border and marks the route start.

    Rule:
    - The number of blue cells on a bar still sets that bar's Manhattan keep-out radius, so the path geometry is found exactly by staying outside those expanded regions.
    - After the shortest low-turn route is reconstructed, background path cells inside each bar-local routing slab are painted with that bar's fixed palette color; route cells outside all slabs stay green.
    - Finally, every blue obstacle pixel becomes green instead of red.

    Key insight: The same visible blue-count clue now serves two linked roles at once — it controls both how far the route must detour and what color that detour uses, while the blue bar markers themselves are absorbed into green in the output.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    dirs = _DIRS_URDL

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
        path_is_vertical = True
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
        path_is_vertical = True
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
        path_is_vertical = False
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
        path_is_vertical = False
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bars = _collect_bars(g, path_is_vertical=path_is_vertical)
    blocked = _build_blocked(h, w, bars)
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
        for nd, (dr, dc) in enumerate(dirs):
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
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()

    out = g.copy()
    slabs = _routing_slabs(bars, axis_limit=(h if path_is_vertical else w))
    _paint_path(out, path, path_is_vertical=path_is_vertical, slabs=slabs)
    out[out == BLUE] = GREEN
    return out


# --------------- generator ---------------

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all bars with required spacing."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


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
    """Choose visible blue-count combinations that fit in the main-axis budget."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and len(set(combo)) >= 2
    ]
    assert choices, "generator needs multi-count configurations for visible path recoloring"
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and blue count."""
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint a single bar on `row` starting from the given wall side."""
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
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
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
    """Check whether the solved output reaches the edge opposite to `edge`."""
    path_colors = list(_path_colors())
    if edge == "top":
        return bool(np.any(np.isin(out[-1, :], path_colors)))
    if edge == "bottom":
        return bool(np.any(np.isin(out[0, :], path_colors)))
    if edge == "left":
        return bool(np.any(np.isin(out[:, -1], path_colors)))
    return bool(np.any(np.isin(out[:, 0], path_colors)))


def _has_visible_variant_signal(inp: np.ndarray, out: np.ndarray) -> bool:
    """Reject samples where the fused behavior is too weak to observe."""
    metrics = _variant_metrics(inp, out)
    return (
        metrics["detour_colors"] >= 2
        and metrics["absorbed_blue_pixels"] >= 2
        and metrics["colored_bg_path"] >= 2
    )


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build an input and validate with the fused solver."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not _has_visible_variant_signal(inp, out):
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
