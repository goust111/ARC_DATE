# Variant of cb2d8a2c
# Dimension: D6 — 路径终止条件 (dst)
# Change: 路径绕过所有障碍条后继续延伸至网格边缘（traverse past all red bars）... → 到达指定目标像素终止

# Design analysis for this D6 variant
# -----------------------------------
# concretization:
# - Keep the original scene: azure background, one green border start, and 1-cell-thick
#   red/blue obstacle bars whose blue-count sets a Manhattan keep-out radius.
# - Add exactly one yellow target pixel inside the grid. It is placed beyond the last
#   obstacle along the travel axis, so the path still has to route around every bar
#   before it can stop.
# - Output behavior changes only at the destination: instead of extending all the way
#   to the opposite border, the shortest legal green route ends exactly on the yellow
#   target, consuming that target cell.
#
# small trace:
# - Original top-start example: after clearing the last horizontal bar, the path keeps
#   going downward until row H-1 because `is_goal` checks the opposite edge only.
# - New variant: after the same detours, Dijkstra heads for the yellow cell instead.
#   If the target sits off to the side, the suffix now includes a final horizontal move
#   and then stops in the interior rather than touching the far border.
# - Difference: the obstacle logic is unchanged; only the terminal state and path suffix
#   are goal-directed.
#
# complexity calibration:
# - The target has a single color and a single-pixel shape, so it is easy to identify.
# - It is not always placed directly below the final detour; the generator biases it to
#   the opposite side of the final opening so the target affects the visible suffix.
# - We keep only one target and one start to avoid branching ambiguity.
#
# implicit assumptions audited from the original solver:
# - Original goal selection hard-coded "opposite edge" from the start side. That is the
#   main D6 dependency and must be replaced by explicit target detection.
# - Original painting only recolored background cells, which was harmless when the goal
#   was an empty edge cell. Here the target cell is non-background, so the path painter
#   must also recolor the target itself to green.
# - The original blocked-mask logic remains valid because bars are still exactly the
#   blue/red components; the yellow target must therefore be excluded from obstacle
#   detection and generator placement must keep it outside all forbidden cells.
#
# placement legality:
# - Bars still obey the original spacing and alternating wall attachments.
# - The generator reserves extra main-axis room after the last bar so there is at least
#   one interior row where a target can be placed outside the final keep-out radius.
# - Candidate target cells are validated by the solver itself; only inputs with a legal
#   path to the target are accepted.
#
# visual observability / cleanliness:
# - Yellow is unused elsewhere, so the new destination cue is explicit in the input.
# - The output no longer has green on the opposite edge; instead it terminates at an
#   interior location where the yellow marker used to be. That contrast is easy to see
#   across a few examples.
# - Object layers stay simple: bars, one start, one target, one path.
#
# new solver assumptions and generator guarantees:
# - `main()` now assumes exactly one yellow target pixel exists; `generate_input()`
#   always inserts exactly one.
# - `main()` assumes the target is reachable without crossing forbidden cells;
#   `generate_input()` checks candidates with the same pathfinder before returning.

import heapq
from collections import deque
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
TARGET = 4

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300


def _start_direction(sr: int, sc: int, h: int, w: int) -> int:
    if sr == 0:
        return 2
    if sr == h - 1:
        return 0
    if sc == 0:
        return 1
    if sc == w - 1:
        return 3
    raise AssertionError("green start must be on a border edge")


def _build_blocked(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

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

    return bar_mask | forbidden


def _solve_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, copy=True)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    target_pts = np.argwhere(g == TARGET)
    assert target_pts.shape[0] == 1, "grid must contain exactly one target pixel"
    tr, tc = map(int, target_pts[0])
    assert (tr, tc) != (sr, sc), "target must differ from the start"

    init_dir = _start_direction(sr, sc, h, w)
    dirs = _DIRS_URDL

    blocked = _build_blocked(g)
    blocked[sr, sc] = False
    target_blocked = bool(blocked[tr, tc])
    blocked[tr, tc] = False
    assert not target_blocked, "target must not lie inside a forbidden zone"

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if (r, c) == (tr, tc):
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

    assert goal_state is not None, "a valid path to the target must exist"

    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green path around radius-coded bars until it lands on a marked destination pixel.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are the same wall-attached obstacles as in the base task.
    - One green (3) border pixel is the start, and one yellow (4) interior pixel is the destination.

    Rule:
    - Each bar still forbids all cells within Manhattan distance <= its blue-pixel count.
    - The green path starts by moving inward, turns orthogonally as needed, and follows the shortest legal route to the yellow target rather than to the far edge.
    - After drawing the route, the target cell is absorbed into the green path and all blue pixels turn red.

    Key insight: The bars still encode how wide the path must detour, but the yellow pixel now decides where that detour sequence stops.
    """
    out = np.array(grid, copy=True)
    path = _solve_path(out)

    for r, c in path:
        if out[r, c] in (BG, TARGET):
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size including one interior target row beyond the last bar."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + blues[-1] + 3


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


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
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
) -> tuple[np.ndarray, list[int], list[int]] | None:
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
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid, rows, lengths


def _target_orders(w: int, last_side: int) -> list[list[int]]:
    mid = w // 2
    if last_side == 0:
        preferred = list(range(mid, 0, -1))
        alternate = list(range(mid + 1, w - 1))
    else:
        preferred = list(range(mid, w - 1))
        alternate = list(range(mid - 1, 0, -1))
    return [preferred, alternate]


def _has_horizontal_suffix(path: list[tuple[int, int]], clear_row: int) -> bool:
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if min(r1, r2) >= clear_row and c1 != c2:
            return True
    return False


def _place_target(
    grid: np.ndarray,
    rows: list[int],
    blues: tuple[int, ...],
    first_side: int,
) -> bool:
    h, w = grid.shape
    last_side = (first_side + len(rows) - 1) % 2
    clear_row = rows[-1] + blues[-1] + 1
    if clear_row > h - 2:
        return False

    row_order = list(range(clear_row, h - 1))
    random.shuffle(row_order)
    col_groups = _target_orders(w, last_side)
    for group_idx, cols in enumerate(col_groups):
        shuffled_cols = cols[:]
        random.shuffle(shuffled_cols)
        for r in row_order:
            for c in shuffled_cols:
                if grid[r, c] != BG:
                    continue
                grid[r, c] = TARGET
                try:
                    path = _solve_path(grid)
                except Exception:
                    grid[r, c] = BG
                    continue
                if _has_horizontal_suffix(path, clear_row) or group_idx == len(col_groups) - 1:
                    return True
                grid[r, c] = BG
    return False


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _path_avoids_opposite_edge(path: list[tuple[int, int]], edge: str, shape: tuple[int, int]) -> bool:
    r, c = path[-1]
    h, w = shape
    if edge == "top":
        return r != h - 1
    if edge == "bottom":
        return r != 0
    if edge == "left":
        return c != w - 1
    return c != 0


def _gen_attempt() -> np.ndarray | None:
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

    built = _canonical_grid(main_size, cross_size, blues, first_side)
    if built is None:
        return None
    canonical, rows, _lengths = built
    if not _place_target(canonical, rows, blues, first_side):
        return None

    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        path = _solve_path(inp)
        if not _path_avoids_opposite_edge(path, edge, inp.shape):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D6 input")
