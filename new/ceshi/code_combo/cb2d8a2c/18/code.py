# Combo variant of cb2d8a2c (index 18)
# combo_spec: D4/3+D11/1
# Name: 仅涂转弯处+仅输出路径

"""
Design analysis for the fused cb2d8a2c variant D4/3 + D11/1.

a) Concretization and behavior trace
   - Input stays close to the source task: azure background, one green border seed, and
     1-cell-thick red/blue obstacle bars. Blue count on each bar still sets a Manhattan
     keep-out radius for the hidden route.
   - Source task pipeline on a small trace:
       hidden path directions: D D D R R D D L L D
       source output:         every background path cell becomes green, and bars stay visible
                              after blue->red conversion.
   - D4/3 alone changes only rendering:
       turn indices might be at the first R and first L, so only short 3-cell elbow windows
       around those bends are painted green; long straight runs remain azure.
   - D11/1 then changes the canvas itself:
       instead of starting from the input grid and overlaying results, the final output is a
       blank azure board of the same size that keeps only the path-derived green pixels.
   - In this fusion, the visible result for the same hidden path is just two or more isolated
     elbow markers on an otherwise empty board. The red/blue bars and the original green seed
     disappear entirely.

b) Complexity calibration
   - If we allowed very short or almost fully painted routes, the output could collapse back
     toward the original continuous-path task. To avoid that, the generator requires multiple
     turns, some unpainted background path cells, and a bounded paint ratio.
   - Using a fixed radius-1 window around each turn yields stable 3-cell elbow marks. This is
     sparse enough to look clearly different, but still structured enough for humans to recover
     the hidden path from repeated examples.
   - The generator prefers 2-4 bars so every sample contains repeated bend evidence rather than
     a single trivial detour.

c) Implicit-assumption review against the original main()
   - Start detection, edge-goal selection, obstacle component extraction, forbidden-zone BFS,
     and Dijkstra path recovery remain valid: they depend on the hidden route, not on how the
     final output is rendered.
   - The original painting loop assumed every recovered path cell should remain visible. D4/3
     breaks that assumption, so we must explicitly compute turn locations and only paint a local
     index window around each bend.
   - The original code also assumed the output begins as a copy of the input, which preserves
     bars and the seed. D11/1 breaks that: the output must instead begin as an all-background
     canvas, otherwise obstacle remnants and the seed would survive.
   - The original generator checked that green reaches the opposite edge. That assumption no
     longer holds visually because the last straight segment is usually unpainted. The generator
     must validate hidden-path existence directly rather than via border-visible green.

d) Placement legality
   - Bars are still generated with the source spacing constraints: first-gap >= b1 + 1,
     inter-bar gap >= bi + b(i+1) + 1, and cross-gap >= blue_count + 1. These conditions ensure
     the hidden path exists and stays outside forbidden regions.
   - Turn markers are painted only on recovered path cells, so they cannot overlap obstacles or
     leave the grid.
   - Because the output board is reconstructed from background, removing obstacles is always
     legal and cannot create collisions between layers.

e) Visual observability
   - Across examples, the user sees detailed obstacle bars in the input but only sparse green
     elbows in the output. That contrast makes both dimensions observable: straight segments are
     omitted, and obstacles vanish.
   - The deciding signal is visible and human-usable: blue counts visibly widen the detours, and
     the surviving green marks sit exactly where those detours bend.
   - No invisible ordering or color-id trick is introduced. All reasoning comes from visible bar
     geometry, spacing, and turn structure.

f) Visual cleanliness
   - Rebuilding the output from a blank azure canvas eliminates the clutter that would result from
     mixing sparse turn marks with full obstacle bars.
   - Radius-1 elbows stay thin and legible. Straight segments are absent rather than partially
     faded, so the output reads as a clean set of discrete waypoints.

g) Change inferability
   - Color semantics stay disciplined: only green survives in the output, and it always denotes a
     path-derived turning event. Blue and red are still meaningful in the input, where they encode
     obstacle radii, but they never appear in the output.
   - The fusion is therefore inferable from few examples: input color variety explains the hidden
     route, while output sparsity and obstacle removal explain the new abstraction level.

New main() assumptions introduced by the fusion and enforced by generate_input()
   - Every generated sample has at least two bars, hence at least four right-angle turns.
   - The recovered route contains background cells outside the turn windows, so the output is
     genuinely sparse rather than accidentally continuous.
   - The output contains no bar colors at all; this is validated by explicit fusion metrics.
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

TURN_MARK_RADIUS = 1

# --- Tunable parameters (layout follows source task, output semantics are fused) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400

MAX_TURN_PAINT_RATIO = 0.55
MIN_BLANK_RUN = 2
MIN_TURN_COUNT = 4


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


def _blocked_mask(g: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
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
    return blocked


def _recover_path(g: np.ndarray) -> list[tuple[int, int]]:
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    blocked = _blocked_mask(g, sr, sc)
    h, w = g.shape

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

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
    return path


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


def _painted_index_set(path_len: int, turn_indices: list[int], radius: int = TURN_MARK_RADIUS) -> set[int]:
    painted: set[int] = set()
    for idx in turn_indices:
        lo = max(0, idx - radius)
        hi = min(path_len - 1, idx + radius)
        painted.update(range(lo, hi + 1))
    return painted


def _render_sparse_turn_path(g: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    turn_ids = _turn_indices(path)
    painted_ids = _painted_index_set(len(path), turn_ids)

    out = np.full(g.shape, BG, dtype=int)
    for idx in painted_ids:
        r, c = path[idx]
        if g[r, c] != BLUE and g[r, c] != RED:
            out[r, c] = GREEN
    return out


def _fusion_metrics(inp: np.ndarray) -> dict[str, int | float]:
    path = _recover_path(inp)
    turn_ids = _turn_indices(path)
    painted_ids = _painted_index_set(len(path), turn_ids)
    bg_path_flags = [inp[r, c] == BG for r, c in path]

    path_bg_total = sum(bg_path_flags)
    painted_bg = sum(1 for i in painted_ids if bg_path_flags[i])

    max_blank_run = 0
    cur_blank_run = 0
    for i, is_bg in enumerate(bg_path_flags):
        if is_bg and i not in painted_ids:
            cur_blank_run += 1
            max_blank_run = max(max_blank_run, cur_blank_run)
        else:
            cur_blank_run = 0

    obstacle_pixels = int(np.count_nonzero((inp == BLUE) | (inp == RED)))
    out = _render_sparse_turn_path(inp, path)
    green_pixels = int(np.count_nonzero(out == GREEN))
    non_bg_output = int(np.count_nonzero(out != BG))

    return {
        "path_len": len(path),
        "turn_count": len(turn_ids),
        "path_bg_total": path_bg_total,
        "painted_bg": painted_bg,
        "max_blank_run": max_blank_run,
        "paint_ratio": (painted_bg / path_bg_total) if path_bg_total else 0.0,
        "obstacle_pixels": obstacle_pixels,
        "green_pixels": green_pixels,
        "non_bg_output": non_bg_output,
        "has_red_or_blue_output": int(np.any((out == RED) | (out == BLUE))),
        "seed_removed": int(np.count_nonzero(out == GREEN) < np.count_nonzero(inp == GREEN) + painted_bg),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the obstacle-avoiding route from the source task, then keep only sparse turn markers on an otherwise blank output board.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) still define the forbidden zones that the hidden route must avoid.
    - A single green (3) pixel lies on one border edge and identifies the route's starting side.

    Rule:
    - For each bar, count its blue pixels and expand a Manhattan keep-out region of that radius around the whole bar.
    - Find the shortest orthogonal route from the seed edge to the opposite edge that stays outside all forbidden cells.
    - Do not draw the full route. Mark only the short elbow neighborhoods around right-angle turns, leaving straight path segments as background.
    - Build the result on a blank azure grid: remove the obstacle bars, remove the original seed, and keep only those turn-local green path pixels.

    Key insight: The hidden path logic is unchanged, but the output is now an abstract waypoint map. Blue still encodes how far the path must detour from each bar, while green no longer means "entire route" and instead marks only the route's turning locations.
    """
    g = np.array(grid, copy=True)
    path = _recover_path(g)
    return _render_sparse_turn_path(g, path)


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
    metrics = _fusion_metrics(inp)
    out = main(inp)
    return (
        metrics["turn_count"] >= MIN_TURN_COUNT
        and metrics["path_bg_total"] > 0
        and 0 < metrics["painted_bg"] < metrics["path_bg_total"]
        and metrics["paint_ratio"] <= MAX_TURN_PAINT_RATIO
        and metrics["max_blank_run"] >= MIN_BLANK_RUN
        and metrics["obstacle_pixels"] > 0
        and metrics["green_pixels"] == metrics["non_bg_output"]
        and metrics["has_red_or_blue_output"] == 0
        and np.count_nonzero(out == GREEN) > 0
        and np.count_nonzero(out == GREEN) < metrics["path_bg_total"]
        and out[tuple(np.argwhere(inp == GREEN)[0])] == BG
    )


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
