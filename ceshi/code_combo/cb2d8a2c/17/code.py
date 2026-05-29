# Combo variant of cb2d8a2c (index 17)
# combo_spec: D10/2+D13/1
# Name: 嵌套式障碍+大量障碍条

# Design analysis for the combo variant
# ------------------------------------
# a) Concrete behavior:
#    - Original canonical layout: 1-4 horizontal bars are spread down the main
#      axis and alternate which outer wall they touch. The shortest valid route
#      simply snakes around each keep-out diamond in sequence.
#    - Combo layout: keep the same obstacle semantics, but place 5-6 bars in a
#      strictly nested chamber chain. After a left-attached outer bar blocks the
#      current chamber, the next bar is attached to the right boundary of the
#      surviving inner chamber, then back to the left, and so on. The path no
#      longer uses the full width between bars; it repeatedly penetrates deeper
#      into a shrinking corridor and later exits it, producing a dense zigzag.
#    - Small trace: with cross-size 25, blue counts (1,2,1,1,1), and attached
#      sides L/R/L/R/L, chamber widths can evolve as
#      25 -> 18 -> 11 -> 7 -> 3 -> 1. The original task would have kept the
#      full width available after every detour; this variant reuses the residual
#      chamber, so each later obstacle visibly sits inside the previous route.
# b) Complexity calibration:
#    - The visual distinction must be obvious, so the generator always uses at
#      least 5 bars. 6 bars are allowed only when the grid size and radii still
#      leave a traversable corridor.
#    - To keep the task inferable, radii are limited to {1, 2}. This still
#      yields repeated evidence that blue-count controls distance, while avoiding
#      impossible layouts inside a 24-26 cell cross-axis.
#    - Bar lengths are shortened versus the seed task. That is necessary because
#      5+ nested layers cannot physically fit if every bar keeps the original
#      6-cell minimum.
# c) Implicit-assumption audit of main():
#    - Start detection assumes exactly one green border pixel. The generator
#      still enforces that.
#    - Goal detection assumes the route should reach the opposite border. The
#      combo keeps the same input/output contract.
#    - Component extraction only assumes red/blue obstacle pieces are
#      disconnected 4-neighborhood objects. Nested floating bars remain valid.
#    - Keep-out construction only depends on each component's blue count. The
#      new risk is not solver correctness but blocked geometry: too many nested
#      obstacles can make the chamber collapse. The generator therefore tracks
#      chamber width explicitly and reserves room for every remaining suffix.
#    - Dijkstra does not assume wall attachment, monotone motion, or a bounded
#      number of turns. That makes the original pathfinder reusable here.
# d) Placement legality:
#    - The generator maintains the current open chamber [left, right]. A
#      left-attached bar occupies [left, end] and advances the next chamber to
#      [end + blue + 1, right]; a right-attached bar is symmetric.
#    - `_min_chamber_needed()` reserves enough width for the entire remaining
#      suffix of nested bars, so later layers never overlap or leave the grid.
#    - Main-axis rows keep the original spacing logic
#      row_gap >= blue_i + blue_{i+1} + 2, which preserves a walkable band
#      between the Manhattan keep-out regions of adjacent layers.
# e) Visual observability:
#    - A human can see both changes directly: there are many obstacle bars, and
#      each later bar is visibly contained in the chamber left by earlier ones.
#      The output path mirrors that structure by diving inward layer by layer.
# f) Visual cleanliness:
#    - Obstacles remain thin horizontal or vertical strips in the original color
#      palette. No extra markers are introduced, and chamber bookkeeping avoids
#      overlap between bars or between keep-out bands that would visually merge.
# g) Color-change inference:
#    - Color behavior stays simple and repeated across all samples: blue cells
#      are part of obstacle bars, and every blue cell turns red after routing.
# h) Generator guarantee vs. new solver assumptions:
#    - `main()` itself stays generic, but it now relies on the generator to
#      provide deeply nested yet solvable layouts. The helper functions below
#      guarantee 5+ bars, strict chamber shrinkage, legal spacing, and solver
#      validation before any input is returned.

from __future__ import annotations

import heapq
from collections import deque
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path from one border to the opposite by threading through a deep nested stack of obstacle bars whose blue pixels encode their safety margins.

    Input structure:
    - bg = azure (8). Red/blue obstacle bars (2/1) are 1-cell-thick strips, and 5 or more of them are arranged as nested layers inside successively smaller chambers.
    - A single green (3) pixel lies on one border and marks the entry point of the route.

    Rule:
    - Treat each disconnected red/blue bar as one obstacle whose keep-out radius equals its number of blue pixels; the green route must stay at Manhattan distance strictly greater than that radius.
    - Grow the path toward the opposite border using orthogonal moves, repeatedly turning to enter and exit the shrinking chambers formed by the nested bars.
    - After the route is painted, recolor every blue pixel red.

    Key insight: The blue count still tells you how far the path must stand off from each bar, but the bars now form a long nested chain, so the valid shortest route becomes a multi-layer inward-and-outward weave rather than a simple loose snake.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1

    dirs = _DIRS_URDL

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

    bar_mask = (g == BLUE) | (g == 2)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        radius = sum(1 for r, c in comp if g[r, c] == BLUE)

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

    inf_cost = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf_cost):
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
            if ncost < best_cost.get(key, inf_cost):
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
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = 2
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 24, 29
COLS_LO, COLS_HI = 24, 26
BAR_COUNT_LO, BAR_COUNT_HI = 5, 6
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 2
BAR_LEN_MIN, BAR_LEN_MAX = 3, 11
MAX_GEN_RETRIES = 800


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


def _choose_blues_and_count(cross_size: int, main_hi: int) -> tuple[int, ...] | None:
    choices: list[tuple[int, ...]] = []
    weights: list[int] = []
    for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1):
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
            if _min_chamber_needed(combo) > cross_size:
                continue
            if _min_main_len(combo) > main_hi:
                continue
            twos = sum(1 for blue_count in combo if blue_count == 2)
            variety = len(set(combo))
            base = 6 if count == 5 else 3
            weight = base + 3 * variety + 2 * twos
            if count == 5 and twos == 0:
                weight = 1
            choices.append(combo)
            weights.append(weight)
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

        bias = 2 if i < len(blues) - 2 else 3
        pick_lo = max(lo, hi - bias)
        length = random.randint(pick_lo, hi)

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


def _choose_start_col(start: int, end: int, side: int) -> int:
    length = end - start + 1
    if length == 1:
        return start
    if side == 0:
        lo = start + 1
        hi = end - 1 if length >= 3 else end
    else:
        lo = start
        hi = end - 1
        if length >= 3:
            lo = start + 1
    if lo > hi:
        lo, hi = start, end
    return random.randint(lo, hi)


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
    grid[0, _choose_start_col(start0, end0, side0)] = GREEN
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
    blues = _choose_blues_and_count(cross_size, main_hi)
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
    raise RuntimeError("failed to generate a valid nested-many cb2d8a2c input")
