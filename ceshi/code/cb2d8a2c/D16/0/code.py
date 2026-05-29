# Variant of cb2d8a2c
# Dimension: D16 — 障碍条的功能差异 (obstacle)
# Change: 所有障碍条功能相同——都需要保持距离绕行，距离由各自的蓝色像素数决定... → 穿越型障碍条（路径直接穿过但涂色不同）

# Design analysis for the D16 variant
# ----------------------------------
# a) Concrete behavior:
#    - The input still contains 1-cell-thick side-attached bars and one green border start.
#    - A bar with at least one blue pixel is a detour bar. Its blue count is still the Manhattan keep-out radius.
#    - A pure-red bar is now a pass-through bar. The path may step on those red cells directly, and those cells stay red in
#      the output instead of becoming green. Visually this creates a green route that is interrupted exactly at the red bar row/col.
#    - Small trace example in canonical top-start orientation:
#      original logic on a pure-red bar at row 4 would still block row 4 and force a side detour;
#      new logic allows the path to go ... (3,5)->(4,5)->(5,5) with cell (4,5) remaining red.
#      The difference appears exactly in the blocked-region construction and in the final painting step.
# b) Complexity calibration:
#    - Every generated task includes both categories: at least one pure-red pass-through bar and at least one blue-marked detour bar.
#      This avoids degeneration into "all detour" or "all pass-through".
#    - The rule uses only two obstacle classes, so the visual distinction is strong but still learnable from a few examples.
# c) Implicit-assumption audit against the original main():
#    - Original `bar_mask | forbidden` assumed every red/blue cell is unwalkable. That is false now; only blue-marked bars and their
#      halos are blocked.
#    - Original path painting only wrote green on background, which was harmless because bars were unreachable. That remains correct,
#      but now it is essential: pass-through segments must stay red, so we must not recolor traversed red cells.
#    - Original `out[out == BLUE] = RED` still holds, because blue markers continue to disappear after routing.
# d) Placement legality:
#    - Bars remain separated along the main axis using the same conservative radius-based spacing formula, with radius 0 for pure-red bars.
#    - Pure-red bars still leave at least one-cell open gap on the non-attached side; blue-marked bars keep the original wider gap.
#    - The generator validates every candidate by solving it and rejecting cases where the path does not actually cross a pure-red bar.
# e) Visual observability:
#    - Each sample shows both behaviors in the same grid: at least one bar is visibly routed around, and at least one red bar is crossed.
#    - The crossed red bar is not marked by a new color; instead the green path is visibly continuous on both sides of a one-cell-thick
#      red barrier, which is enough to infer that this class is traversable.
# f) Visual cleanliness:
#    - Bars are still parallel, side-attached, and alternating walls, so overlaps do not occur.
#    - Only one overlay interaction exists on pass-through bars: the path uses the red cells but does not repaint them, avoiding clutter.
# g) Color-change inference:
#    - No new colors are introduced. Blue remains a structural marker and still turns red afterward.
#    - The only color-dependent branch is visible: blue-marked bars repel; pure-red bars can be crossed while staying red.

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


def _start_and_goal(grid: np.ndarray) -> tuple[int, int, int, str, int]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    h, w = grid.shape
    if sr == 0:
        return sr, sc, 2, "row", h - 1
    if sr == h - 1:
        return sr, sc, 0, "row", 0
    if sc == 0:
        return sr, sc, 1, "col", w - 1
    if sc == w - 1:
        return sr, sc, 3, "col", 0
    raise AssertionError("green start must be on a border edge")


def _is_goal(goal_kind: str, goal_val: int, r: int, c: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _bar_infos(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    infos: list[dict[str, object]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        cells = [tuple(map(int, cell)) for cell in comp]
        blue_count = sum(1 for r, c in cells if grid[r, c] == BLUE)
        infos.append(
            {
                "cells": cells,
                "cell_set": set(cells),
                "blue_count": blue_count,
                "detour": blue_count > 0,
            }
        )
    return bar_mask, infos


def _build_blocked(
    shape: tuple[int, int],
    infos: list[dict[str, object]],
    start: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)

    for info in infos:
        if not bool(info["detour"]):
            continue
        cells = info["cells"]
        radius = int(info["blue_count"])

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in cells:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            blocked[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked[start] = False
    return blocked


def _shortest_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    goal_kind: str,
    goal_val: int,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    dirs = _DIRS_URDL

    INF = (10**9, 10**9)
    sr, sc = start
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), INF):
            continue
        if _is_goal(goal_kind, goal_val, r, c) and (r, c) != (sr, sc):
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


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, copy=True)
    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    bar_mask, infos = _bar_infos(g)
    blocked = _build_blocked(g.shape, infos, (sr, sc))
    path = _shortest_path(blocked, (sr, sc), init_dir, goal_kind, goal_val)

    pass_infos = [info for info in infos if not bool(info["detour"])]
    path_set = set(path)
    pass_hits = sum(1 for info in pass_infos if path_set & info["cell_set"])
    pass_cells = sum(1 for r, c in path if g[r, c] == RED)

    return {
        "grid": g,
        "bar_mask": bar_mask,
        "infos": infos,
        "path": path,
        "pass_bar_count": len(pass_infos),
        "detour_bar_count": sum(1 for info in infos if bool(info["detour"])),
        "pass_bar_hits": pass_hits,
        "pass_cells": pass_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a border-started path to the opposite side, detouring only around blue-marked bars while passing straight through plain red bars.

    Input structure:
    - bg = azure (8), with one green (3) start pixel on a border.
    - Obstacle bars are 1-cell-thick red/blue components attached to alternating side walls.
    - Bars containing blue (1) markers are detour bars; pure-red (2) bars are pass-through bars.

    Rule:
    - For a blue-marked bar, the path must stay outside the Manhattan halo whose radius equals that bar's blue-pixel count.
    - For a pure-red bar, the path may step directly through the bar, but those traversed cells keep the obstacle's red color.
    - The route still connects the start edge to the opposite edge using the shortest path with fewest turns, and all blue pixels become red.

    Key insight: Blue markers no longer decorate a universal obstacle type; they identify which bars repel the path, while unmarked red bars are traversable recoloring segments.
    """
    details = _solve_details(grid)
    out = np.array(details["grid"], copy=True)
    for r, c in details["path"]:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


# --------------- generator (o5x) ---------------

# --- Tunable parameters (CFG) ---
# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Bar counts and blue-per-bar range (count >= 2 so each sample can show both bar behaviors)
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_HI = 4
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
# Generation retry limit
MAX_GEN_RETRIES = 400


def _min_main_len(radii: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all bars with conservative spacing."""
    last_bar = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
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


def _choose_radii(count: int, main_hi: int) -> tuple[int, ...]:
    """Pick per-bar radii, using 0 for pass-through bars and 1-4 for detour bars."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_HI + 1), repeat=count)
        if any(v == 0 for v in combo)
        and any(v > 0 for v in combo)
        and _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and obstacle type."""
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    if radius > 0:
        lo = max(lo, 2 * radius - 1)
        hi = min(BAR_LEN_MAX, cross_size - radius - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    else:
        hi = min(BAR_LEN_MAX, cross_size - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, radius: int) -> None:
    """Paint a single bar on `row` starting from the given wall side."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    if radius > 0:
        colors[: 2 * radius - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths = []
    for row, radius, side in zip(rows, radii, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, radius)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, radius)

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
    radii = _choose_radii(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, radii, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        details = _solve_details(inp)
        out = main(inp)
        if details["pass_bar_count"] < 1 or details["detour_bar_count"] < 1:
            return None
        if details["pass_bar_hits"] < 1 or details["pass_cells"] < 1:
            return None
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D16 input")
