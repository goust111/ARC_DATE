# Combo variant of cb2d8a2c (index 32)
# combo_spec: D4/3+D11/1+D15/1
# Name: 仅涂转弯处+仅输出路径+距离边界标记

"""
Design analysis for the combo variant D4/3 + D11/1 + D15/1.

a) Concretization and behavior trace
   - Input keeps the original object grammar: azure background, one green border seed, and
     1-cell-thick red/blue bars whose blue counts encode Manhattan keep-out radii.
   - The hidden route is still the original shortest orthogonal path from the seed edge to
     the opposite edge, routed outside every bar's forbidden zone.
   - The visible output changes in three coupled ways at once:
     1. D4/3: straight path cells stop being drawn by default; only turn-local cells can be green.
     2. D15/1: path cells that lie exactly on a bar's legal boundary shell (distance radius+1)
        are specially marked yellow.
     3. D11/1: all input objects disappear from the output, so the final grid is rebuilt from
        scratch as background plus sparse route-derived markers only.
   - Small trace on a route with directions D D D R R R D D:
       source task:      draw all route cells green; keep bars visible as red; blue->red
       combo task:       erase everything, then draw turn-window cells near D->R and R->D in
                         green, draw boundary-shell route cells in yellow, leave all other
                         route cells and all obstacle cells as background.
     The same hidden path is recovered, but the output now exposes only two kinds of evidence:
     elbow markers and exact-distance markers.

b) Complexity calibration
   - Using one extra color only for exact-boundary contact keeps the rule structured rather
     than arbitrary: green means "turn neighborhood", yellow means "tight legal boundary".
   - The route itself is still nontrivial because the solver must reconstruct the whole path
     before it can know which cells survive the D11 projection.
   - The generator rejects samples where yellow disappears or where nearly all route cells are
     visible, preventing collapse into either the pure-turn or near-continuous-path regimes.

c) Implicit-assumption review against the original main()
   - Start detection, goal-edge selection, obstacle-component extraction, forbidden-zone BFS,
     and Dijkstra recovery remain valid because the hidden route itself is unchanged.
   - The original painter assumed `out = input copy` and that every background path cell should
     become green. Both assumptions fail here:
       * D11/1 requires discarding bars, the green seed, and all non-marker cells.
       * D4/3 requires only turn-local green painting.
       * D15/1 requires examining exact boundary-shell membership before painting.
   - The original generator checked for green on the far edge. That also fails here because the
     far edge may end up blank, or yellow, or removed entirely under the sparse projection.

d) Placement legality
   - Markers are painted only on recovered path cells, so they never overlap obstacles or leave
     the board.
   - Original bar spacing remains sufficient for path existence.
   - New generator checks ensure at least one visible green turn marker, at least one visible
     yellow boundary marker outside the turn windows, and at least one hidden route segment.

e) Visual observability
   - Across examples, the output consistently removes the bars and preserves only two sparse
     signal types. Humans can compare input bars with output yellow runs to infer "exact legal
     boundary", and compare output green clusters to infer "turn location".
   - No invisible ordering or color-id trick is used; every decision comes from visible geometry
     and visible blue-count radii.

f) Visual cleanliness
   - Rebuilding the output from an all-background canvas avoids red-bar clutter under D11/1.
   - Green has priority on overlapping cells so the two marker types stay visually separable:
     turns appear as compact green elbows, while surrounding tight detour stretches remain yellow.

g) Change inferability
   - Only two non-background colors appear in the final output, each tied to a strong spatial
     condition. This keeps the color semantics inferable from a few examples.

New main() assumptions introduced by the combo and enforced by generate_input()
   - The recovered path has at least one turn pair, so green turn markers exist.
   - The recovered path hits an exact boundary shell outside those turn windows, so yellow is
     genuinely needed and the combo does not degenerate to D4/3 + D11/1.
   - The recovered path also contains blank route cells, so the sparse output does not collapse
     back toward a continuous path.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

TURN_MARK_RADIUS = 1


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


def _recover_path_and_boundary(g: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = g.shape
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


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    turns: list[int] = []
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns.append(i)
    return turns


def _turn_marker_indices(path_len: int, turn_ids: list[int], radius: int = TURN_MARK_RADIUS) -> set[int]:
    painted: set[int] = set()
    for idx in turn_ids:
        lo = max(0, idx - radius)
        hi = min(path_len - 1, idx + radius)
        painted.update(range(lo, hi + 1))
    return painted


def _variant_metrics(inp: np.ndarray) -> dict[str, int | float]:
    path, boundary = _recover_path_and_boundary(inp)
    turn_ids = _turn_indices(path)
    turn_marker_ids = _turn_marker_indices(len(path), turn_ids)
    boundary_path_ids = {i for i, (r, c) in enumerate(path) if boundary[r, c]}
    boundary_only_ids = boundary_path_ids - turn_marker_ids
    visible_ids = turn_marker_ids | boundary_only_ids
    bg_flags = [inp[r, c] == BG for r, c in path]

    max_blank_run = 0
    cur_blank_run = 0
    for i, is_bg in enumerate(bg_flags):
        if is_bg and i not in visible_ids:
            cur_blank_run += 1
            max_blank_run = max(max_blank_run, cur_blank_run)
        else:
            cur_blank_run = 0

    path_bg_total = sum(bg_flags)
    turn_bg = sum(1 for i in turn_marker_ids if bg_flags[i])
    boundary_only_bg = sum(1 for i in boundary_only_ids if bg_flags[i])
    visible_bg = sum(1 for i in visible_ids if bg_flags[i])

    return {
        "path_len": len(path),
        "turn_count": len(turn_ids),
        "path_bg_total": path_bg_total,
        "turn_marker_bg": turn_bg,
        "boundary_only_bg": boundary_only_bg,
        "visible_bg": visible_bg,
        "hidden_bg": path_bg_total - visible_bg,
        "max_blank_run": max_blank_run,
        "visible_ratio": (visible_bg / path_bg_total) if path_bg_total else 0.0,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the same hidden obstacle-avoiding route as the source task, then compress the output to only turn markers and exact-distance markers.

    Input structure:
    - bg = azure (8). Red/blue 1-cell-thick bars (colors 2/1) are the only obstacles and their blue counts still encode keep-out radii.
    - A single green (3) pixel lies on one border edge and indicates where the hidden route starts.

    Rule:
    - Reconstruct the shortest orthogonal path to the opposite edge while staying strictly farther than each bar's blue-count radius.
    - Discard the original scene in the output: bars, blue->red conversion, the seed, and all non-marker path cells vanish to background.
    - Paint only a short green elbow around each 90-degree turn.
    - Paint any other path cell that lies exactly one step outside some bar's keep-out radius in yellow (4).

    Key insight: The geometry of the route is unchanged, but the output is now a sparse diagnostic projection of that route: green shows where it bends, yellow shows where it is just barely legal.
    """
    g = np.array(grid, copy=True)
    path, boundary = _recover_path_and_boundary(g)
    turn_marker_ids = _turn_marker_indices(len(path), _turn_indices(path))

    out = np.full_like(g, BG)
    for i, (r, c) in enumerate(path):
        if g[r, c] != BG:
            continue
        if i in turn_marker_ids:
            out[r, c] = GREEN
        elif boundary[r, c]:
            out[r, c] = YELLOW
    return out


# --------------- generator ---------------

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400
MIN_HIDDEN_BG = 2
MIN_BOUNDARY_ONLY_BG = 2
MAX_VISIBLE_RATIO = 0.82


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


def _is_good_variant_sample(inp: np.ndarray) -> bool:
    metrics = _variant_metrics(inp)
    if metrics["turn_count"] < 2:
        return False
    if metrics["turn_marker_bg"] <= 0:
        return False
    if metrics["boundary_only_bg"] < MIN_BOUNDARY_ONLY_BG:
        return False
    if metrics["hidden_bg"] < MIN_HIDDEN_BG:
        return False
    if metrics["max_blank_run"] < 2:
        return False
    if metrics["visible_ratio"] > MAX_VISIBLE_RATIO:
        return False

    out = main(inp)
    colors = set(map(int, np.unique(out)))
    if not colors.issubset({BG, GREEN, YELLOW}):
        return False
    if GREEN not in colors or YELLOW not in colors:
        return False
    return True


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
        if not np.array_equal(main(inp), out):
            return None
        if not _is_good_variant_sample(inp):
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
