# Variant of cb2d8a2c
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 连续涂色——绿色路径经过的每个背景格子都被涂为绿色，形成连续的绿色线条... → 仅在转弯处涂色

"""
Design analysis for the D4 trajectory variant.

a) Concretization and behavior trace
   - Input stays structurally identical to the source task: azure background, one green
     border seed, and 1-cell-thick red/blue obstacle bars whose blue count encodes the
     Manhattan keep-out radius.
   - The hidden route is also unchanged: a shortest orthogonal path goes from the seed edge
     to the opposite edge and bends around the expanded forbidden zones.
   - The changed part is only the visible output trace. In the source task, every background
     cell on the recovered path turns green, so each detour appears as a continuous polyline.
     In this variant, the straight sections remain azure. Only cells in a short window around
     each right-angle bend are painted green.
   - Small trace example on a path index sequence:
       source path indices: 0 1 2 3 4 5 6 7
       directions:          D D D R R D D D
       source painting:     paint all background cells 1..7
       variant painting:    paint only indices around the turn at 3->4, e.g. 2,3,4
     The route is still inferable, but the output becomes a sparse collection of elbow marks.

b) Complexity calibration
   - The task must still require understanding the full route, not merely finding bars.
   - Painting one cell before and one cell after each turn gives a stable 3-cell elbow marker.
     This is visually distinct from the original continuous line and keeps the rule simple.
   - Bars remain 1-4 and alternate wall attachment exactly as before, so samples still show
     repeated evidence that each bar induces a pair of bends.

c) Implicit-assumption review against the original main()
   - Start detection, goal-edge selection, obstacle-component extraction, forbidden-zone BFS,
     and Dijkstra path recovery all depend only on the hidden route and remain valid.
   - The original loop
         for r, c in path:
             if out[r, c] == BG: out[r, c] = GREEN
     assumes every recovered path cell should be visible. That assumption is the exact D4
     property being changed, so it must be replaced by turn-local painting.
   - The original generator check for green reaching the opposite edge is no longer valid,
     because the final straight segment may remain completely unpainted even though the route
     does reach the far border.

d) Placement legality
   - Because markers are painted only on cells already belonging to the recovered path, they
     cannot leave the grid or overlap bars.
   - The source spacing constraints are sufficient for path existence. No new geometric freedom
     is introduced in the input itself.
   - The generator additionally rejects cases where the sparse turn markers would accidentally
     cover almost the whole path, preventing visual collapse back toward the original task.

e) Visual observability
   - A human sees the same bars and the same single green seed, but the output now contains
     isolated green elbow marks rather than a full line. Across 2-4 examples, the repeated
     pattern is that green appears only where the route changes direction.
   - No invisible ordering rule is introduced; all decisions still come from visible obstacle
     geometry and visible blue counts.

f) Visual cleanliness
   - Using a 3-cell window per turn avoids noisy thick blobs while keeping each bend legible.
   - Straight corridors remain azure, so green marks do not mask large regions of the board.

g) Color-change inferability
   - Colors remain minimal and structured: blue still upgrades to red, bars stay red, and green
     continues to denote the route-derived signal. The only color-rule change is the support of
     the green trace, which is backed by consistent bend locations.

New main() assumptions introduced by the variant and enforced by generate_input()
   - There is at least one bend pair in generated samples, because bar count stays >= 1.
   - The recovered path contains some background cells outside all turn windows, so the sparse
     output is observably different from a continuous trace.
"""

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


def _blocked_mask(g: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
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
    return blocked, bar_mask


def _recover_path(g: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    blocked, _ = _blocked_mask(g, sr, sc)

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


def _trajectory_metrics(inp: np.ndarray, path: list[tuple[int, int]]) -> dict[str, int | float]:
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

    return {
        "path_len": len(path),
        "turn_count": len(turn_ids),
        "painted_bg": painted_bg,
        "path_bg_total": path_bg_total,
        "max_blank_run": max_blank_run,
        "paint_ratio": (painted_bg / path_bg_total) if path_bg_total else 0.0,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the same obstacle-avoiding route as the source task, but stamp green only near its right-angle bends.

    Input structure:
    - bg = azure (8). Red/blue bars (colors 2/1) are still the route-defining obstacles.
    - A single green (3) pixel lies on one border edge and marks the route's starting point.

    Rule:
    - Each bar keeps the original meaning: its blue-pixel count sets the Manhattan exclusion radius around that bar.
    - Find the shortest edge-to-edge orthogonal route that respects those forbidden zones.
    - Do not color the full route. Leave straight segments as background and color only the short bend neighborhoods around each 90-degree turn.
    - As before, all blue pixels are converted to red in the final output.

    Key insight: The path geometry is unchanged; only its rendering changes. Blue still tells you how far the hidden route must stay away, while green now highlights the route's turning points instead of tracing every step.
    """
    g = np.array(grid, copy=True)
    path = _recover_path(g)
    turn_ids = _turn_indices(path)
    painted_ids = _painted_index_set(len(path), turn_ids)

    out = g.copy()
    for idx in painted_ids:
        r, c = path[idx]
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


# --- Tunable parameters (source layout preserved; only trajectory rendering changes) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300
MAX_TURN_PAINT_RATIO = 0.60
MIN_BLANK_RUN = 2


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
    path = _recover_path(inp)
    metrics = _trajectory_metrics(inp, path)
    return (
        metrics["turn_count"] >= 2
        and metrics["path_bg_total"] > 0
        and metrics["painted_bg"] < metrics["path_bg_total"]
        and metrics["paint_ratio"] <= MAX_TURN_PAINT_RATIO
        and metrics["max_blank_run"] >= MIN_BLANK_RUN
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
    raise RuntimeError("failed to generate a valid cb2d8a2c input")
