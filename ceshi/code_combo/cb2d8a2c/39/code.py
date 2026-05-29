# Combo variant of cb2d8a2c (index 39)
# combo_spec: D5/2+D19/0
# Name: L形起始标记+角落图例

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
YELLOW = 4
ORANGE = 7

LEGEND_H = 8
LEGEND_W = 9
LEGEND_DIST_ROWS = (4, 5, 6)
RADIUS_BY_BLUE = {1: 2, 2: 3, 3: 4}

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 18, 26
COLS_LO, COLS_HI = 18, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 3
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400

# Design analysis for the combo variant
# =====================================
# concrete_variant:
#   legend_panel:
#     - A fixed 8x9 orange-framed panel occupies one corner in canonical orientation.
#     - Its top mini-scene shows a green 3-cell L-marker plus one yellow cell that marks
#       the forward step leaving the L's elbow.
#     - Its bottom three rows are a shuffled count legend:
#         1 blue -> 2 yellow
#         2 blue -> 3 yellow
#         3 blue -> 4 yellow
#       Each row is rendered as B...B then Y...Y, so the mapping is readable as count-to-count.
#   start_marker:
#     - The real source is no longer a single border pixel. It is a 3-cell green L touching
#       the source edge, with its elbow one cell inside the grid.
#     - In canonical orientation the marker sits on the top edge and there are two handedness
#       choices:
#         right-handed: {(0,c), (1,c), (1,c+1)}
#         left-handed:  {(0,c), (1,c), (1,c-1)}
#     - The elbow is the true path origin. The legend contains the matching L-shape and shows
#       that the route leaves the elbow in the yellow-cell direction.
#   behavior_trace:
#     - Original task: find one green border pixel, infer direction directly from the touched edge,
#       and use blue-count == keep-out radius.
#     - Combo task:
#       1. Rotate the whole input until the orange legend frame lands in the canonical corner.
#       2. Read the legend's L example to learn which empty neighbor of the elbow is "forward".
#       3. Read the three blue->yellow count rows to obtain keep-out radii.
#       4. Decode the real green L-marker outside the legend, start from its elbow, and route to
#          the opposite edge with those legend-provided radii.
#     - Difference:
#       the solver must parse two explicit symbolic cues before any pathfinding happens, and the
#       blue counts no longer mean their radius directly.
#
# complexity_calibration:
#   - Only two marker handedness options and four whole-grid rotations are used, so the source cue
#     is visibly different without exploding the state space.
#   - The radius legend uses only three rows with repeated support in every sample; this keeps the
#     mapping easy to read while still changing the original geometry because every radius is blue+1.
#   - Shuffling the three legend rows prevents "top row means 1" shortcuts and forces actual legend
#     reading, but the row count remains small enough for 2-4 ARC examples.
#
# implicit_assumption_audit:
#   original_assumptions_broken:
#     - "there is exactly one green cell" -> false: there is one green legend exemplar and one green
#       start marker, each with three cells.
#     - "all blue/red cells are obstacle bars" -> false: the legend also contains blue sample pixels.
#     - "blue_count itself is the radius" -> false: legend yellow-run lengths define the radius.
#     - "the non-start green cells can be ignored" -> false for an orthogonal L-marker, because its
#       arm cells are 4-neighbors of the elbow and must be blocked from path expansion.
#   code_adjustments:
#     - isolate the legend first and mask it out from marker/bar detection
#     - decode the start marker from the remaining green component
#     - block all marker cells except the elbow search start
#     - convert only obstacle-bar blues to red; legend blues stay blue
#
# placement_legality:
#   - Canonical generation reserves rows 0..7 and cols 0..8 for the legend.
#   - The first obstacle bar is forced to attach to the right wall, so the start marker can be placed
#     inside that bar's cross-axis span while staying well away from the legend block.
#   - Bars begin below the legend and continue with Manhattan-clearance spacing computed from the
#     legend-derived radii, so no bar overlaps the panel and no keep-out zone swallows the start elbow.
#   - Whole-grid rotation is applied only after the canonical layout is complete, preserving all
#     non-overlap guarantees in every orientation.
#
# visual_observability_and_cleanliness:
#   - The orange frame makes the legend an obvious separate information region instead of accidental
#     obstacle clutter.
#   - The start marker is the only green object outside that panel, so the eye can separate "instruction"
#     from "actual source".
#   - The distance rows are plain blue-then-yellow runs, which are countable at a glance and avoid
#     introducing a second obstacle-like red pattern inside the legend.
#   - The path never enters the legend because the panel is treated as blocked informational space.
#
# nondegeneracy_check_plan:
#   - Generator variability should show at least:
#       * both marker handedness choices
#       * all four whole-grid rotations
#       * multiple legend-row permutations
#   - The distance map is intentionally non-original (radius = blue+1), so solving the same bars with
#     the original cb2d8a2c semantics would usually produce a different or invalid route.
#
# post_change_solver_assumptions:
#   - The orange-framed legend exists in exactly one corner after some 90-degree rotation.
#   - The legend contains exactly one green L exemplar and exactly one yellow forward cell in its
#     direction subpanel.
#   - Outside the legend there is exactly one green 3-cell L-marker.
#   - Each obstacle bar contains 1-3 blue pixels so that the legend provides a radius entry.


def _neighbor_degree4(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((rr, cc) in cells for rr, cc in neighbors4(r, c))


def _decode_l_marker(cells_arr: np.ndarray) -> tuple[int, int, frozenset[tuple[int, int]]]:
    """Return (elbow_row, elbow_col, normalized_neighbor_offsets) for a 3-cell L-marker."""
    cells = {tuple(map(int, rc)) for rc in cells_arr}
    assert len(cells) == 3, "L-marker must have exactly three cells"
    elbows = [cell for cell in cells if _neighbor_degree4(cell, cells) == 2]
    assert len(elbows) == 1, "L-marker must have a unique elbow"
    er, ec = elbows[0]
    sig = frozenset((r - er, c - ec) for r, c in cells if (r, c) != (er, ec))
    return er, ec, sig


def _parse_legend(g: np.ndarray) -> tuple[np.ndarray, frozenset[tuple[int, int]], int]:
    """Parse the canonical top-left legend. Returns (legend_mask, exemplar_signature, init_dir)."""
    h, w = g.shape
    assert h >= LEGEND_H and w >= LEGEND_W, "grid is too small for the legend"

    panel = g[:LEGEND_H, :LEGEND_W]
    frame_mask = np.zeros_like(panel, dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True
    assert np.all(panel[frame_mask] == ORANGE), "legend frame must be orange"
    assert np.all(panel[~frame_mask] != ORANGE), "orange may appear only on the legend frame"

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:LEGEND_H, :LEGEND_W] = True

    dir_green = np.argwhere(panel[1:4, 1:5] == GREEN)
    dir_yellow = np.argwhere(panel[1:4, 1:5] == YELLOW)
    assert dir_green.shape[0] == 3, "legend direction example must contain exactly three green cells"
    assert dir_yellow.shape[0] == 1, "legend direction example must contain exactly one yellow forward cell"

    legend_cells = np.array([(int(r + 1), int(c + 1)) for r, c in dir_green], dtype=int)
    er, ec, exemplar_sig = _decode_l_marker(legend_cells)
    yr, yc = map(int, dir_yellow[0] + 1)
    delta = (yr - er, yc - ec)
    dir_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    assert delta in dir_map, "legend yellow cell must be cardinally adjacent to the elbow"
    init_dir = dir_map[delta]

    radius_map: dict[int, int] = {}
    for row in LEGEND_DIST_ROWS:
        seq = [int(panel[row, c]) for c in range(1, LEGEND_W - 1) if panel[row, c] != BG]
        assert seq, "legend distance row must not be empty"
        blue_count = 0
        while blue_count < len(seq) and seq[blue_count] == BLUE:
            blue_count += 1
        radius = len(seq) - blue_count
        assert blue_count > 0 and radius > 0, "legend distance row must be blue-run then yellow-run"
        assert seq == [BLUE] * blue_count + [YELLOW] * radius, "legend row must be a single blue run followed by a single yellow run"
        assert blue_count not in radius_map, "legend distance entries must use distinct blue counts"
        radius_map[blue_count] = radius

    assert radius_map == RADIUS_BY_BLUE, "legend must encode the expected blue-to-radius mapping"
    return legend_mask, exemplar_sig, init_dir


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Rotate the grid until the fixed legend is in canonical top-left orientation."""
    for rot_k in range(4):
        cand = np.rot90(grid, rot_k)
        try:
            _parse_legend(cand)
            return cand, rot_k
        except AssertionError:
            continue
    raise AssertionError("could not find the corner legend in any 90-degree orientation")


def _goal_for_dir(init_dir: int, h: int, w: int) -> tuple[str, int]:
    if init_dir == 0:
        return "row", 0
    if init_dir == 1:
        return "col", w - 1
    if init_dir == 2:
        return "row", h - 1
    return "col", 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a green path from a legend-decoded L-marker to the opposite edge while staying farther from each bar than the legend-prescribed blue-count radius.

    Input structure:
    - Background is azure (8), obstacle bars still use blue/red (1/2), and the path source is now a 3-cell green L-marker near one border.
    - A compact orange-framed corner legend contains a green L example with a yellow forward cell, plus blue-to-yellow count rows that define the bar radii.
    - The entire puzzle may be rotated, so the legend and the real marker turn together.

    Rule:
    - First rotate mentally until the corner legend is upright, then read the yellow forward step attached to the legend's L example and apply that meaning to the real L-marker's elbow.
    - For each red/blue bar, count its blue pixels and consult the legend's blue-run -> yellow-run table to get the Manhattan keep-out radius.
    - Draw the shortest path (breaking ties by fewer turns) from the decoded elbow cell to the edge opposite the decoded direction, and finally turn only the obstacle-bar blue pixels red.

    Key insight: The green L alone is not enough, because its elbow has two empty exits; the corner legend tells you which exit is forward and also upgrades every obstacle's radius from "blue count" to "legend yellow count".
    """
    canonical, rot_k = _to_canonical(np.asarray(grid, dtype=int))
    g = np.array(canonical, copy=True)
    h, w = g.shape

    legend_mask, exemplar_sig, init_dir = _parse_legend(g)

    start_mask = (g == GREEN) & (~legend_mask)
    green_comps = components_mask(start_mask, connectivity=4)
    assert len(green_comps) == 1, "there must be exactly one green L-marker outside the legend"
    start_comp = green_comps[0]
    sr, sc, start_sig = _decode_l_marker(start_comp)
    assert start_sig == exemplar_sig, "start marker must match the legend's L-shape exemplar"

    goal_kind, goal_val = _goal_for_dir(init_dir, h, w)

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bar_mask = ((g == BLUE) | (g == RED)) & (~legend_mask)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        assert blue_count in RADIUS_BY_BLUE, "every bar must have a legend-supported blue count"
        radius = RADIUS_BY_BLUE[blue_count]

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
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

    blocked = bar_mask | forbidden | legend_mask | start_mask
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

    out_can = g.copy()
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    for r, c in path:
        if out_can[r, c] == BG:
            out_can[r, c] = GREEN

    out_can[(out_can == BLUE) & bar_mask] = RED
    return np.rot90(out_can, (4 - rot_k) % 4)


def _min_main_len(radii: tuple[int, ...]) -> int:
    """Minimum canonical main-axis size after reserving the legend block above the bars."""
    total = LEGEND_H
    for a, b in zip(radii, radii[1:]):
        total += a + b + 2
    return total + radii[-1] + 2


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
    """Pick blue counts whose legend-derived radii fit in the available main axis."""
    choices = []
    for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
        radii = tuple(RADIUS_BY_BLUE[b] for b in combo)
        if _min_main_len(radii) <= main_hi:
            choices.append(combo)
    if not choices:
        raise ValueError("no feasible blue-count assignment for the chosen main-axis size")
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and legend-derived radius."""
    radius = RADIUS_BY_BLUE[blue_count]
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - radius - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint a single obstacle bar on `row` from the chosen wall side."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _paint_down_marker(grid: np.ndarray, elbow_col: int, handedness: int) -> None:
    """Paint the canonical top-edge L-marker whose forward step is downward."""
    grid[0, elbow_col] = GREEN
    grid[1, elbow_col] = GREEN
    if handedness == 0:
        grid[1, elbow_col + 1] = GREEN
    else:
        grid[1, elbow_col - 1] = GREEN


def _paint_legend(grid: np.ndarray, handedness: int, row_order: list[int]) -> None:
    """Paint the canonical corner legend."""
    grid[:LEGEND_H, :LEGEND_W] = BG
    grid[0, :LEGEND_W] = ORANGE
    grid[LEGEND_H - 1, :LEGEND_W] = ORANGE
    grid[:LEGEND_H, 0] = ORANGE
    grid[:LEGEND_H, LEGEND_W - 1] = ORANGE

    if handedness == 0:
        green_cells = [(1, 1), (2, 1), (2, 2)]
        yellow_cell = (3, 1)
    else:
        green_cells = [(1, 3), (2, 2), (2, 3)]
        yellow_cell = (3, 3)
    for r, c in green_cells:
        grid[r, c] = GREEN
    grid[yellow_cell] = YELLOW

    for row, blue_count in zip(LEGEND_DIST_ROWS, row_order):
        radius = RADIUS_BY_BLUE[blue_count]
        grid[row, 1 : 1 + blue_count] = BLUE
        grid[row, 1 + blue_count : 1 + blue_count + radius] = YELLOW


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    handedness: int,
) -> np.ndarray | None:
    """Build the canonical grid: top-left legend, top-edge start marker, downward route."""
    radii = tuple(RADIUS_BY_BLUE[b] for b in blues)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    grid = np.full((main_size, cross_size), BG, dtype=int)
    _paint_legend(grid, handedness, random.sample(list(RADIUS_BY_BLUE), len(RADIUS_BY_BLUE)))

    extras = _random_composition(slack, len(blues) + 1)
    row = LEGEND_H + extras[0]
    rows = [row]
    for i in range(1, len(blues)):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    lengths = []
    first_side = 1  # keep the first obstacle on the right so the legend corner stays clear
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(len(blues)))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    low = max(cross_size - first_len, LEGEND_W + 2)
    high = cross_size - 2
    if low > high:
        return None
    elbow_col = random.randint(low, high)
    _paint_down_marker(grid, elbow_col, handedness)
    return grid


def _gen_attempt() -> np.ndarray | None:
    main_size = random.randint(ROWS_LO, ROWS_HI)
    cross_size = random.randint(COLS_LO, COLS_HI)
    handedness = random.randrange(2)
    feasible_counts = []
    for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1):
        try:
            _choose_blues(count, main_size)
            feasible_counts.append(count)
        except ValueError:
            continue
    if not feasible_counts:
        return None
    count = random.choice(feasible_counts)
    blues = _choose_blues(count, main_size)

    canonical = _canonical_grid(main_size, cross_size, blues, handedness)
    if canonical is None:
        return None

    inp = np.rot90(canonical, random.randrange(4))
    try:
        out = main(inp)
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
