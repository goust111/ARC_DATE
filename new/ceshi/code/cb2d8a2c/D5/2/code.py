# Variant of cb2d8a2c
# Dimension: D5 — 源对象类型 (src)
# Change: 网格边缘的单个绿色像素作为路径起点，向网格内部延伸... → L形/箭头形绿色标记作为起点，并由其形状编码初始方向

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Design analysis for the D5 variant
# ==================================
# concrete_variant:
#   marker:
#     original: one green border pixel, and the touched border directly implied the inward direction
#     variant: a 3-pixel green arrowhead/V marker:
#       down  = {(0,c-1), (0,c+1), (1,c)}
#       up    = {(H-1,c-1), (H-1,c+1), (H-2,c)}
#       right = {(r-1,0), (r+1,0), (r,1)}
#       left  = {(r-1,W-1), (r+1,W-1), (r,W-2)}
#     visible_effect:
#       - input now contains two green border cells plus one interior green tip
#       - the tip is the true path origin
#       - the arrowhead orientation, not border contact alone, determines the main travel direction
#   behavior_trace:
#     original_first_step:
#       - solver finds exactly one green cell
#       - it checks which border that pixel lies on
#       - initial direction is forced by that border
#     variant_first_step:
#       - solver finds one 8-connected green component of size 3
#       - inside that component, the unique degree-2 cell is the arrow tip
#       - vector from the two wing cells toward the tip yields one of U/R/D/L
#       - that decoded direction decides the opposite-edge goal
#     difference:
#       - the start is no longer reducible to "the green border pixel"
#       - the solver must parse local shape geometry before pathfinding
#
# complexity_calibration:
#   - only four marker rotations exist, so the new cue has low combinatorial load
#   - every generated sample still contains the same red/blue bar mechanism, keeping the main puzzle identity intact
#   - the marker is visually salient because it always uses 3 green cells instead of 1
#
# implicit_assumption_audit:
#   old_assumptions_broken_by_D5:
#     - "there is exactly one green cell" -> false
#     - "that green cell lies on the border" -> false for the true start tip
#     - "start direction can be inferred from border side alone" -> false by design
#   code_adjustments:
#     - replace single-pixel lookup with 8-connected marker decoding
#     - keep only the decoded tip as the search start state
#     - leave the two wing cells untouched; they are diagonal to the tip, so they cannot create 4-connected detours
#
# placement_legality:
#   - canonical generation keeps the tip one cell inside the source edge and the wings on the source edge
#   - first obstacle bar starts at least 3 rows/cols away from that edge, so the marker never overlaps any bar or keep-out zone
#   - start tip stays within the first bar's covered cross-axis span, preserving the original "must route around bars" structure
#
# observability_and_cleanliness:
#   - humans can read the marker as an arrowhead/V immediately; there is no hidden color-id convention
#   - the marker sits in empty space near the source edge, separated from bars by at least one blank row/col
#   - only green changes role; red/blue obstacle semantics are unchanged, so the grid does not become cluttered
#
# post_change_solver_assumptions:
#   - exactly one green 8-connected component exists, and it has size 3
#   - that component forms a valid arrowhead with one degree-2 tip and two symmetric wing cells
#   - generator enforces those conditions by placing only this marker and no other green cells


def _decode_start_marker(g: np.ndarray, green: int) -> tuple[int, int, int]:
    """Return (start_row, start_col, init_dir) from a 3-pixel arrowhead marker."""
    green_comps = components_mask(g == green, connectivity=8)
    assert len(green_comps) == 1, "grid must contain exactly one green start marker"
    comp = green_comps[0]
    assert comp.shape[0] == 3, "green start marker must have exactly three pixels"

    cells = [tuple(map(int, rc)) for rc in comp]
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = 0
        for rr, cc in cells:
            if (rr, cc) != (r, c) and max(abs(rr - r), abs(cc - c)) == 1:
                deg += 1
        degrees[(r, c)] = deg

    tips = [cell for cell, deg in degrees.items() if deg == 2]
    wings = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(tips) == 1 and len(wings) == 2, "green marker must be an arrowhead"

    sr, sc = tips[0]
    dvr = 2 * sr - sum(r for r, _ in wings)
    dvc = 2 * sc - sum(c for _, c in wings)
    dir_map = {(-2, 0): 0, (0, 2): 1, (2, 0): 2, (0, -2): 3}
    assert (dvr, dvc) in dir_map, "green marker must encode a cardinal direction"
    return sr, sc, dir_map[(dvr, dvc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path from a directional green marker to the opposite edge, routing around red bars at a distance dictated by their blue pixel count.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) are still the only obstacles.
    - The path source is a 3-pixel green arrowhead near one border, not a single pixel.
    - The arrowhead's orientation encodes the path's initial direction; its interior tip is the actual start cell.

    Rule:
    - Decode the arrowhead direction from its shape, then aim for the opposite grid edge along that axis.
    - Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
    - Draw the shortest path with right-angle turns around the forbidden zones, then recolor every blue pixel red.

    Key insight: The green marker is no longer just a location cue; its geometry tells you which way the route begins before the usual bar-avoidance logic takes over.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1
    RED = 2

    dirs = _DIRS_URDL  # U, R, D, L

    # --- decode the unique green arrowhead marker ---
    sr, sc, init_dir = _decode_start_marker(g, GREEN)

    if init_dir == 0:
        goal_kind, goal_val = "row", 0
    elif init_dir == 2:
        goal_kind, goal_val = "row", h - 1
    elif init_dir == 1:
        goal_kind, goal_val = "col", w - 1
    else:
        goal_kind, goal_val = "col", 0

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    # --- find bar components (colors {1,2}) ---
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    # --- build forbidden region: for each bar, forbid cells within Manhattan distance <= blue_count ---
    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        radius = int(blue_count)

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

    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

    # --- Dijkstra: minimize (steps, turns) ---
    INF = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), INF):
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
            if ncost < best_cost.get(key, INF):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    # --- paint path + convert blue -> red ---
    out = g.copy()
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
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


# --------------- generator (o5x + D5 variant) ---------------

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Tunable parameters (CFG) ---
# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Bar counts and blue-per-bar range
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4  # keepout radius encoded per bar
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79  # bar spans 32-79% of cross-axis
# Generation retry limit
MAX_GEN_RETRIES = 300


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all bars with required spacing and the 2-row/col marker depth."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


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


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    """Pick blue counts per bar such that they fit within main_hi."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
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


def _paint_down_marker(grid: np.ndarray, tip_col: int) -> None:
    """Paint canonical marker for a downward-going path."""
    grid[0, tip_col - 1] = GREEN
    grid[0, tip_col + 1] = GREEN
    grid[1, tip_col] = GREEN


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (source side = top, decoded direction = down)."""
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
        tip_col = random.randint(1, first_len - 1)
    else:
        tip_col = random.randint(cross_size - first_len, cross_size - 2)
    _paint_down_marker(grid, tip_col)
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-source/down-marker) grid so the source side matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output has green on the edge opposite to `edge`."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: pick parameters, build grid, validate via solver."""
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D5 input")
