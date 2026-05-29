# Variant of cb2d8a2c
# Dimension: D10 — 障碍条的空间排列 (obstacle)
# Change: 障碍条在网格中分散排列，路径依次绕过每个障碍条，障碍条之间有足够的空间供路径通过... → 障碍条按嵌套方式排列

# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    - Original canonical layout: horizontal obstacle bars are spread along the
#      path axis and alternate which outer wall they touch; the green path
#      snakes left-right-left between separated keep-out zones.
#    - Variant canonical layout: bars still stay 1-cell thick and still encode
#      their keep-out radius by blue-pixel count, but they are no longer all
#      attached to the global walls. Instead, each new bar is placed inside the
#      open chamber left by the previous bar. If the first bar is left-attached,
#      its keep-out leaves a right-side chamber; the next bar is attached to the
#      right boundary of that chamber, shrinking it; the next bar attaches to
#      the left boundary of the shrunken chamber, and so on. The path therefore
#      enters a progressively narrower corridor and draws nested rectangular
#      detours instead of a loose snake.
#    - Small trace: with width 25, blue counts (1,1,1), and left/right/left
#      bars of lengths (9,6,5), the chamber widths evolve
#      25 -> 15 -> 8 -> 2. The original logic would have placed those three
#      bars on unrelated rows while keeping the full width available between
#      them; the variant keeps reusing the previous chamber so every detour is
#      geometrically contained inside the earlier one.
# b) Complexity calibration:
#    - At least 2 bars are required; 1 bar cannot display nesting.
#    - 3 bars are preferred when width allows it because they produce a visibly
#      floating inner bar and a stronger "concentric corridor" cue.
#    - Blue counts stay in a small range (1-3), and bars keep the same red/blue
#      alternating prefix pattern, so the keep-out rule remains learnable.
# c) Implicit-assumption audit of main():
#    - The border-start detection only assumes one green pixel on the border.
#      This still holds.
#    - Goal selection only assumes the path should reach the opposite edge.
#      This still holds.
#    - Component extraction treats every disconnected red/blue object as one
#      obstacle and never assumes wall attachment or a particular row spacing.
#      Therefore nested/floating bars remain valid inputs.
#    - The keep-out construction depends only on Manhattan distance from each
#      component and on the number of blue cells in that same component. Nested
#      placement does not break this, but it does mean the generator must keep
#      successive bars far enough apart in the main axis to avoid forbidden-zone
#      overlap that would seal the corridor completely.
#    - Dijkstra only assumes there exists some unblocked route from the start to
#      the opposite edge; it does not assume the route is monotone or that bars
#      touch the outer walls.
# d) Placement legality:
#    - The generator tracks the current chamber [left,right] explicitly.
#      A left bar spans from left to some endpoint; after applying its keep-out
#      radius b, the next chamber begins at endpoint + b + 1.
#      A right bar is symmetric.
#    - Length choice is constrained by the minimum chamber width required for
#      the remaining suffix of bars, so later bars never run out of room.
#    - Row placement keeps the original vertical separation formula
#      row gap >= b_i + b_{i+1} + 2, preserving a free band between successive
#      keep-out diamonds.
# e) Visual observability:
#    - Across examples, the distinctive cue is not an invisible sort key; it is
#      the visible containment relation between successive bars and the green
#      path's layer-by-layer penetration into a narrowing channel.
# f) Visual cleanliness:
#    - Bars remain thin, disconnected, and use only the original four colors.
#      No extra markers are introduced, and the chamber logic prevents overlaps.
# g) Color-change inference:
#    - Color behavior is unchanged and remains easy to infer: blue cells belong
#      to obstacle bars and turn red after the route is drawn.
# h) Generator guarantee vs. new solver assumptions:
#    - main() now effectively relies on the generator to provide a nested layout
#      whose successive chamber widths stay positive and whose main-axis spacing
#      leaves a traversable corridor. The helper functions below enforce both.

import heapq
from collections import deque
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path from one border to the opposite, drilling through a nested stack of obstacle bars whose blue pixels set their exclusion radii.

    Input structure:
    - bg = azure (8). Red/blue obstacle bars (2/1) are 1-cell-thick strips, and successive bars are nested inside the chamber left by the previous one.
    - A single green (3) pixel sits on a grid border and marks the route entrance.

    Rule:
    - For each disconnected red/blue obstacle, the keep-out radius equals its number of blue pixels, so the green route must stay at Manhattan distance strictly greater than that value.
    - The path grows toward the opposite edge by taking orthogonal turns that successively enter the shrinking chambers created by the nested bars.
    - Once the path is painted, every blue pixel is recolored red.

    Key insight: Blue pixels still encode each obstacle's danger radius, but the obstacles now nest inside one another, so the shortest valid route naturally becomes a concentric sequence of rectangular detours.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1
    RED = 2

    dirs = _DIRS_URDL  # U, R, D, L

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

    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

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


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 19, 29
COLS_LO, COLS_HI = 19, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 3
BAR_LEN_MIN, BAR_LEN_MAX = 5, 15
MAX_GEN_RETRIES = 400


def _min_bar_len(blue_count: int) -> int:
    return max(BAR_LEN_MIN, 2 * blue_count - 1)


def _min_chamber_needed(blues: tuple[int, ...]) -> int:
    need = 1
    for blue_count in reversed(blues):
        need += blue_count + _min_bar_len(int(blue_count))
    return need


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


def _choose_blues_and_count(cross_size: int) -> tuple[int, ...] | None:
    choices: list[tuple[int, ...]] = []
    weights: list[int] = []
    for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1):
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
            if _min_chamber_needed(combo) > cross_size:
                continue
            choices.append(combo)
            weights.append(5 if count == BAR_COUNT_HI else 2)
    if not choices:
        return None
    return random.choices(choices, weights=weights, k=1)[0]


def _paint_bar(
    grid: np.ndarray,
    row: int,
    start: int,
    end: int,
    side: int,
    blue_count: int,
) -> None:
    length = end - start + 1
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, start : end + 1] = colors
    else:
        grid[row, start : end + 1] = colors[::-1]


def _choose_nested_spans(
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> list[tuple[int, int, int]] | None:
    left = 0
    right = cross_size - 1
    spans: list[tuple[int, int, int]] = []

    for i, blue_count in enumerate(blues):
        side = (first_side + i) % 2
        chamber = right - left + 1
        min_next = _min_chamber_needed(blues[i + 1 :])
        lo = _min_bar_len(int(blue_count))
        hi = min(BAR_LEN_MAX, chamber - blue_count - min_next)
        if lo > hi:
            return None

        prefer_lo = max(lo, hi - 4) if i < len(blues) - 1 else lo
        length = random.randint(prefer_lo, hi)
        if side == 0:
            start = left
            end = left + length - 1
            left = end + blue_count + 1
        else:
            end = right
            start = right - length + 1
            right = start - blue_count - 1
        spans.append((start, end, side))

    return spans


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    spans = _choose_nested_spans(cross_size, blues, first_side)
    if spans is None:
        return None

    extras = _random_composition(slack, len(blues) + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(blues)):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, (start, end, side) in zip(rows, blues, spans):
        _paint_bar(grid, row, start, end, side, int(blue_count))

    start0, end0, side0 = spans[0]
    if side0 == 0:
        lo = start0 + 1
        hi = min(end0, start0 + max(1, (end0 - start0 + 1) - 2))
    else:
        lo = max(start0, end0 - max(1, (end0 - start0 + 1) - 2))
        hi = end0 - 1
    if lo > hi:
        return None
    grid[0, random.randint(lo, hi)] = GREEN
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


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    cross_size = random.randint(cross_lo, cross_hi)
    blues = _choose_blues_and_count(cross_size)
    if blues is None:
        return None

    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
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
    raise RuntimeError("failed to generate a valid nested cb2d8a2c input")
