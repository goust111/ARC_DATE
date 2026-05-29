# Variant of cb2d8a2c
# Dimension: D18 — 背景与颜色方案 (meta)
# Change: 背景为浅灰色(8)，路径为绿色(3)，障碍条为红色(2)和蓝色(1)三色方案... → 路径颜色由经过的障碍条决定

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

# Design analysis for the D18 variant
# ----------------------------------
# a) Concrete behavior and where the variant lives
#    - The geometric layout stays the same: one border start pixel, 1-4 one-cell-thick bars,
#      alternating wall attachments, and Manhattan keepout zones encoded by blue marker count.
#    - The necessary color adaptation is that each bar now has its own visible base color from a
#      small palette, while blue cells remain the keepout markers on the input.
#    - Output behavior changes only after routing is solved:
#      1. The path starts green.
#      2. When the path first reaches the far side of bar i's keepout zone, later path cells switch
#         to bar i's base color.
#      3. Blue marker cells on each bar are recolored to that bar's base color, so the output bar is
#         visually solid and the path segment color can be matched back to the most recently passed bar.
#    - Small example in canonical top-to-bottom orientation:
#      original: start green -> detour around left-attached red bar -> all path cells still green.
#      variant:  start green -> detour around left-attached pink bar -> cells below that detour become
#      pink; after the next yellow bar is passed, later cells become yellow.
#    - This means the solver's path search is unchanged; the variant lives in bar decoding and path
#      painting, not in obstacle geometry.
#
# b) Complexity calibration
#    - If bars stayed uniformly red, the new rule would collapse into "first segment green, everything
#      else red", which is visually too weak. The generator therefore chooses each bar's base color from
#      a small fixed palette `(2, 4, 6, 7)`.
#    - The palette is intentionally small so colors repeat across samples and are learnable, but large
#      enough that multi-bar outputs often contain 2-3 visible segments.
#    - Blue stays semantically unique: it always means radius encoding, never path color.
#
# c) Hidden-assumption audit of the original `main()`
#    - `bar_mask = (g == BLUE) | (g == RED)` assumed every obstacle base cell was red. That breaks once
#      bars can be yellow/pink/orange, so the variant must instead treat every non-bg, non-start cell
#      as bar material in the input.
#    - `for r, c in path: out[r, c] = GREEN` assumed a single path color. In the variant this would
#      overwrite the intended inherited colors, so painting must become ordered and stateful.
#    - `out[out == BLUE] = RED` assumed the target color of every blue pixel was globally red. Now each
#      blue pixel must inherit its own component's base color.
#    - The Dijkstra search itself only depends on a boolean blocked mask, so it survives unchanged once
#      that mask is built from the generalized bar components.
#
# d) Placement legality guaranteed by the generator
#    - Bars remain one component each because the only added freedom is per-bar base color, not shape.
#    - Each bar still fits within the cross axis with cross-gap >= blue_count + 1, ensuring a route exists.
#    - Bars still alternate wall attachment and maintain the original main-axis spacing constraints, so
#      keepout zones do not merge and the solver can order bars unambiguously from start side to goal side.
#
# e) Visual observability
#    - A human can infer the new rule from 2-4 examples because every colored path segment appears only
#      after the path has cleared a bar of the same base color.
#    - The first segment staying green provides a stable "before any bar" reference, and the solidified
#      output bars provide the color source that explains later path segments.
#
# f) Visual cleanliness
#    - Path cells never overlap bars because routing still uses the same blocked mask.
#    - Blue markers disappear in the output, reducing clutter rather than increasing it.
#    - Using only four possible base colors prevents the output from becoming noisy.
#
# g) Color-change inferability
#    - The change is not keyed to invisible ids or arbitrary ordering; it is keyed to the visible sequence
#      of bars encountered along the path.
#    - The generator mildly biases toward multi-bar instances, so the segmented-color behavior appears often.
#
# New implicit assumptions introduced by the variant
#    - Every bar component must contain exactly one non-blue base color. The generator enforces this by
#      painting each bar with one chosen base color plus blue markers.
#    - Bars must remain separable components, because the solver maps each passed obstacle to one inherited
#      color. The original spacing rules already guarantee this, and the generator keeps them unchanged.

BG = 8
BLUE = 1
GREEN = 3
BASE_COLORS = (2, 4, 6, 7)
_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


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


def _extract_bars(g: np.ndarray) -> list[dict[str, object]]:
    bar_mask = (g != BG) & (g != GREEN)
    bar_comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, object]] = []

    for comp in bar_comps:
        pts = [(int(r), int(c)) for r, c in comp]
        base_colors = {int(g[r, c]) for r, c in pts if int(g[r, c]) != BLUE}
        assert len(base_colors) == 1, "each bar must have exactly one non-blue base color"
        base_color = next(iter(base_colors))
        blue_count = sum(1 for r, c in pts if int(g[r, c]) == BLUE)
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        bars.append(
            {
                "cells": pts,
                "base_color": int(base_color),
                "radius": int(blue_count),
                "rmin": int(min(rows)),
                "rmax": int(max(rows)),
                "cmin": int(min(cols)),
                "cmax": int(max(cols)),
            }
        )
    return bars


def _ordered_bars(bars: list[dict[str, object]], init_dir: int) -> list[dict[str, object]]:
    if init_dir == 2:
        return sorted(bars, key=lambda bar: int(bar["rmin"]))
    if init_dir == 0:
        return sorted(bars, key=lambda bar: -int(bar["rmax"]))
    if init_dir == 1:
        return sorted(bars, key=lambda bar: int(bar["cmin"]))
    return sorted(bars, key=lambda bar: -int(bar["cmax"]))


def _passed_bar(bar: dict[str, object], r: int, c: int, init_dir: int) -> bool:
    radius = int(bar["radius"])
    if init_dir == 2:
        return int(r) > int(bar["rmax"]) + radius
    if init_dir == 0:
        return int(r) < int(bar["rmin"]) - radius
    if init_dir == 1:
        return int(c) > int(bar["cmax"]) + radius
    return int(c) < int(bar["cmin"]) - radius


def _blocked_mask(h: int, w: int, bars: list[dict[str, object]]) -> np.ndarray:
    forbidden = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["radius"])
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in bar["cells"]:
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

    blocked = forbidden.copy()
    for bar in bars:
        for r, c in bar["cells"]:
            blocked[r, c] = True
    return blocked


def _find_path(
    g: np.ndarray,
    sr: int,
    sc: int,
    init_dir: int,
    goal_kind: str,
    goal_val: int,
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
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
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a border-start path around keepout bars, recoloring each later segment from the most recently bypassed bar.

    Input structure:
    - Background is color 8. A single green (3) start pixel sits on one border edge.
    - Each obstacle bar is a 1-cell-thick component made of blue (1) radius markers plus one repeated base color.

    Rule:
    - A bar's keepout radius equals its blue-pixel count, so the path must stay at Manhattan distance strictly greater than that from every bar cell.
    - The route geometry is the same as the source task: draw the minimum-step, minimum-turn orthogonal path from the start edge to the opposite edge.
    - The path begins green, then whenever it has fully cleared a bar's keepout zone in the travel direction, later path cells switch to that bar's base color.
    - Blue pixels on each bar are converted to the bar's base color in the output.

    Key insight: Blue still controls how far the route must detour, while the visible base color of each passed bar now controls the color of the next path segment.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    sr, sc, init_dir, goal_kind, goal_val = _start_and_goal(g)
    bars = _extract_bars(g)
    ordered_bars = _ordered_bars(bars, init_dir)

    blocked = _blocked_mask(h, w, bars)
    blocked[sr, sc] = False
    path = _find_path(g, sr, sc, init_dir, goal_kind, goal_val, blocked)

    out = g.copy()
    next_bar = 0
    current_color = GREEN
    for r, c in path:
        while next_bar < len(ordered_bars) and _passed_bar(ordered_bars[next_bar], r, c, init_dir):
            current_color = int(ordered_bars[next_bar]["base_color"])
            next_bar += 1
        if out[r, c] == BG:
            out[r, c] = current_color

    for bar in bars:
        base_color = int(bar["base_color"])
        for r, c in bar["cells"]:
            out[r, c] = base_color

    return out


# --------------- generator ---------------

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


def _choose_bar_count() -> int:
    counts = list(range(BAR_COUNT_LO, BAR_COUNT_HI + 1))
    weights = [1 if count == 1 else 2 for count in counts]
    return int(random.choices(counts, weights=weights, k=1)[0])


def _choose_base_colors(count: int) -> tuple[int, ...]:
    bases = [random.choice(BASE_COLORS) for _ in range(count)]
    if count > 1 and len(set(bases)) == 1:
        idx = random.randrange(1, count)
        alternatives = [color for color in BASE_COLORS if color != bases[0]]
        bases[idx] = random.choice(alternatives)
    return tuple(int(color) for color in bases)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(
    grid: np.ndarray,
    row: int,
    side: int,
    length: int,
    blue_count: int,
    base_color: int,
) -> None:
    w = grid.shape[1]
    colors = np.full(length, base_color, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    base_colors: tuple[int, ...],
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
    for row, blue_count, base_color, side in zip(
        rows,
        blues,
        base_colors,
        ((first_side + i) % 2 for i in range(count)),
    ):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count, base_color)

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
    non_bg_path = out != BG
    if edge == "top":
        return bool(np.any(non_bg_path[-1, :]))
    if edge == "bottom":
        return bool(np.any(non_bg_path[0, :]))
    if edge == "left":
        return bool(np.any(non_bg_path[:, -1]))
    return bool(np.any(non_bg_path[:, 0]))


def _shows_variant(inp: np.ndarray, out: np.ndarray) -> bool:
    path_mask = ((inp == BG) | (inp == GREEN)) & (out != BG)
    path_colors = {int(color) for color in out[path_mask].tolist()}
    return GREEN in path_colors and len(path_colors) >= 2


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = _choose_bar_count()
    blues = _choose_blues(count, main_hi)
    base_colors = _choose_base_colors(count)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, base_colors, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not _shows_variant(inp, out):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D18 variant input")
