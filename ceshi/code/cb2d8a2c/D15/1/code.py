# Variant of cb2d8a2c
# Dimension: D15 — 路径与障碍条的交互效果 (transform)
# Change: 路径不接触障碍条，只在距离约束范围外通过；蓝色像素事后统一变为红色... → 路径在距离边界处涂上不同颜色标记
#
# Design analysis
# variant:
#   core_change:
#     - Keep the original shortest-path routing and the same red/blue bar encoding.
#     - Add a second path color: any routed path cell whose Manhattan distance to at least
#       one bar is exactly blue_count + 1 is recolored yellow.
#     - Blue bar pixels still become red at the end, so the only visible novelty is the
#       yellow "constraint boundary" traced by the path where the keep-out radius is tight.
#   concrete_behavior:
#     - Original flow: build forbidden cells at distance <= radius, find a shortest path
#       through the remaining cells, paint every traversed background cell green.
#     - New flow: build the same forbidden cells, also record the exact boundary shell at
#       distance == radius + 1 for every bar, find the same shortest path, then paint path
#       cells on that shell yellow instead of green.
#     - Small trace: if a left-attached bar has blue_count=2 on row 5, the forbidden zone
#       reaches row 3..7 around it. In the original task the path segment on row 2 that
#       slides past the open end stays green. In this variant those row-2 path cells are
#       yellow because row 2 is the precise legal boundary (distance 3).
#   complexity_check:
#     - The change is visible but local: the path geometry is unchanged, only a structured
#       subset of path cells changes color.
#     - Yellow is a single extra color and repeats across multiple bars, so the solver can
#       infer "boundary contact" from 2-4 examples instead of memorizing per-instance cases.
#   implicit_assumptions_review:
#     - The old `_reaches_opposite_edge` checked only green on the far edge; that is no
#       longer valid because the terminal cell may now be yellow.
#     - The old path painter could assign green blindly to every background path cell; the
#       variant must inspect whether each path cell lies on the boundary shell first.
#     - The old forbidden BFS stopped at radius. The variant needs one extra BFS layer to
#       expose the exact shell at radius + 1 without changing passability.
#   placement_legality:
#     - Bars remain 1-cell thick, wall-attached, alternating sides, and spaced so their
#       forbidden zones do not merge in a way that blocks the puzzle.
#     - The generator still validates every sample by calling `main`, and now also rejects
#       samples with no yellow output so the variant cannot collapse back to the original.
#   visual_observability:
#     - Yellow appears exactly on tight detour segments, usually the horizontal/vertical
#       row or column where the path turns around a bar. This makes the D15 change obvious:
#       the same route is present, but the "active distance boundary" is highlighted.
#   cleanliness:
#     - Yellow only replaces path cells, never overlaps bars, so each role stays legible:
#       red bars, green ordinary path, yellow boundary-contact path.
#   color_inference:
#     - The color change is governed by one structural condition only: exact-distance path
#       cells become yellow. There are no hidden color-id orderings or per-bar palettes.

from __future__ import annotations

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


def _start_and_goal(g: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", g.shape[0] - 1
    elif sr == g.shape[0] - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", g.shape[1] - 1
    elif sc == g.shape[1] - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    return sr, sc, init_dir, goal_kind, goal_val


def _path_and_boundary_mask(g: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = g.shape
    dirs = _DIRS_URDL
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    boundary = np.zeros((h, w), dtype=bool)
    for comp_arr in bar_comps:
        comp = [(int(r), int(c)) for r, c in comp_arr]
        radius = sum(1 for r, c in comp if g[r, c] == BLUE)

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            if d <= radius:
                forbidden[r, c] = True
            elif d == radius + 1:
                boundary[r, c] = True

            if d == radius + 1:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked = forbidden.copy()
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
    return path, boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green route across the grid and highlight the exact keep-out boundary where the detour becomes tight.

    Input structure:
    - bg = azure (8). Bars made of blue/red cells (1/2) are 1-cell-thick obstacles attached to alternating side walls.
    - A single green (3) pixel lies on one outer edge and marks the route's entry point.

    Rule:
    - For each bar, the number of blue cells gives its Manhattan keep-out radius; the path may only use cells farther away than that radius.
    - The route grows to the opposite edge using shortest orthogonal motion around those forbidden zones.
    - Any routed path cell that sits exactly one step outside a bar's keep-out radius is recolored yellow (4) to mark the active distance boundary, and all blue bar cells become red.

    Key insight: Blue cells do double duty: they still set how far the path must stay away, and now they also determine which parts of the final path are highlighted as exact-boundary contact.
    """
    g = np.array(grid, copy=True)
    path, boundary = _path_and_boundary_mask(g)

    out = g.copy()
    for r, c in path:
        if out[r, c] != BG:
            continue
        out[r, c] = YELLOW if boundary[r, c] else GREEN

    out[out == BLUE] = RED
    return out


# --------------- generator (o5x) ---------------

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300


def _min_main_len(blues: tuple[int, ...]) -> int:
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
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
) -> np.ndarray | None:
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
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    path_mask = (out == GREEN) | (out == YELLOW)
    if edge == "top":
        return bool(np.any(path_mask[-1, :]))
    if edge == "bottom":
        return bool(np.any(path_mask[0, :]))
    if edge == "left":
        return bool(np.any(path_mask[:, -1]))
    return bool(np.any(path_mask[:, 0]))


def _has_boundary_marker(out: np.ndarray) -> bool:
    return bool(np.any(out == YELLOW))


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

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not _has_boundary_marker(out):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c input")
