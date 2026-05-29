# Combo variant of cb2d8a2c (index 28)
# combo_spec: D4/2+D9/1+D18/2
# Name: 多色路径+蓝色吸收+路径颜色随障碍变化

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and behavior trace
#    - Geometry stays anchored to the source task and spec: one green border seed, 2-4
#      one-cell-thick wall-attached bars, alternating sides, and a shortest low-turn path
#      that must remain at Manhattan distance strictly greater than each bar's blue count.
#    - The fusion resolves the color logic into one visible chain:
#      1. Blue-count still encodes keepout radius.
#      2. The same blue-count also maps the bar to a fixed journey color:
#         1 -> yellow(4), 2 -> gray(5), 3 -> magenta(6), 4 -> orange(7).
#      3. When the path first enters a bar's detour slab
#         [main-blue_count-1, main+blue_count+1], it switches to that bar's journey color.
#      4. That color persists after the detour until the next bar takes over, so the output
#         path is a segmented travel history rather than isolated colored patches.
#      5. In the output, red bar cells also change to the bar's journey color, while every
#         blue marker turns green, visually "absorbed" into the path family.
#    - Small canonical top-to-bottom example:
#      original: a bar on row 7 with 3 blue markers forces a wide detour; all path cells are
#      green and blue cells become red.
#      combo: the same geometry is used, but the path turns magenta(6) when it reaches the
#      row slab [3, 11], stays magenta below that bar until another bar is encountered, the
#      bar's red cells become magenta, and its former blue markers become green.
#    - Difference from the source task: geometry is unchanged, but the output now exposes
#      obstacle identity, detour width, and traversal order through one shared color story.
#
# b) Complexity calibration
#    - The journey palette is fixed and tiny (4 colors), so blue-count -> color can be learned
#      across 2-4 ARC examples. A per-instance random palette would be too unconstrained.
#    - The generator forces 2-4 bars and at least two distinct blue counts so that multiple
#      journey colors appear within most samples; otherwise the variant could collapse into
#      something visually too close to the original.
#    - Persistent post-detour coloring is required in validation, so the D18 contribution is
#      not diluted into the simpler D4-only "color only inside slabs" behavior.
#
# c) Hidden-assumption audit of the original main()
#    - Original `bar_mask = (g == BLUE) | (g == RED)` still holds because the input keeps the
#      original obstacle colors; only the output recolors the bars.
#    - Original Dijkstra search depends only on blocked cells. That assumption survives.
#    - Original `for r, c in path: out[r, c] = GREEN` is invalid because path color now depends
#      on the most recently entered obstacle slab.
#    - Original `out[out == BLUE] = RED` is invalid because blue markers must turn green, and
#      red bar cells must recolor according to blue-count.
#    - New assumption: each bar is still a single 1-cell-thick component on one main-axis line,
#      so it has one unambiguous slab interval and one unambiguous journey color. The generator
#      preserves the source task's spacing/orientation constraints to guarantee this.
#
# d) Placement legality
#    - Bars use the original length and spacing constraints from the spec:
#      cross-gap >= blue_count + 1 and inter-bar spacing >= blue_a + blue_b + 1.
#    - Those same constraints imply that consecutive detour slabs are disjoint or touch only at
#      a clean boundary, so a path segment can be assigned to a single current obstacle color.
#    - The start still lies within the first bar's covered cross-range, which preserves the
#      original solvability pattern.
#
# e) Visual observability
#    - A human can see blue marker counts directly in the input and compare them with the output
#      journey colors near the same bar and on the downstream path segment.
#    - Because the path color persists after each detour, traversal order is visible without any
#      hidden ids or numeric sorting rules.
#    - Green absorbed markers form a local cue tying the obstacle's blue markers to the path.
#
# f) Visual cleanliness
#    - The path still never overlaps obstacle cells, so color roles stay separated by location.
#    - Only four journey colors are introduced, and they appear in coherent segments rather than
#      noisy per-cell patterns.
#    - Green is reused only for the start, path, and absorbed markers, which makes the D9 effect
#      noticeable without flooding the grid.
#
# g) Change inferability and anti-degeneration checks
#    - The generator rejects samples unless:
#      1. at least two journey colors appear on path background cells,
#      2. at least one blue marker is absorbed into green,
#      3. at least one path cell outside every detour slab still carries a journey color.
#    - Condition (3) is the key quantitative guardrail proving the sample is not merely a D4-only
#      recoloring. It forces the D18-style "color persists after passing a bar" behavior to be
#      visible in the produced data.
#
# New implicit assumptions introduced by the combo
#    - Blue counts must stay in {1,2,3,4} because the fixed journey palette covers exactly those
#      categories. The generator enforces that.
#    - The path's main-axis progress must encounter bars in order. The original spacing and the
#      shortest-path geometry guarantee that no sample needs ambiguous bar ordering.

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3
PATH_COLOR_BY_BLUE = {1: 4, 2: 5, 3: 6, 4: 7}
JOURNEY_COLORS = frozenset(PATH_COLOR_BY_BLUE.values())
_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Tunable parameters (CFG) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400


def _start_and_goal(g: np.ndarray) -> tuple[int, int, int, str, int, bool]:
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        return sr, sc, 2, "row", g.shape[0] - 1, True
    if sr == g.shape[0] - 1:
        return sr, sc, 0, "row", 0, True
    if sc == 0:
        return sr, sc, 1, "col", g.shape[1] - 1, False
    if sc == g.shape[1] - 1:
        return sr, sc, 3, "col", 0, False
    raise AssertionError("green start must be on a border edge")


def _extract_bars(
    g: np.ndarray,
    *,
    path_is_vertical: bool,
) -> list[dict[str, object]]:
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)
    axis_limit = g.shape[0] if path_is_vertical else g.shape[1]

    bars: list[dict[str, object]] = []
    for comp in bar_comps:
        cells = [(int(r), int(c)) for r, c in comp]
        blue_count = sum(1 for r, c in cells if int(g[r, c]) == BLUE)
        assert blue_count in PATH_COLOR_BY_BLUE, "blue_count must stay within the fixed palette"
        main_values = {int(r if path_is_vertical else c) for r, c in cells}
        assert len(main_values) == 1, "each bar must occupy exactly one main-axis line"
        main = next(iter(main_values))
        bars.append(
            {
                "cells": cells,
                "blue_count": int(blue_count),
                "main": int(main),
                "slab_lo": int(max(0, main - blue_count - 1)),
                "slab_hi": int(min(axis_limit - 1, main + blue_count + 1)),
                "journey_color": int(PATH_COLOR_BY_BLUE[blue_count]),
            }
        )

    bars.sort(key=lambda bar: int(bar["main"]))
    for prev_bar, next_bar in zip(bars, bars[1:]):
        assert int(prev_bar["slab_hi"]) <= int(next_bar["slab_lo"]), "detour slabs must stay ordered"
    return bars


def _ordered_bars(bars: list[dict[str, object]], init_dir: int) -> list[dict[str, object]]:
    if init_dir in (1, 2):
        return list(bars)
    return list(reversed(bars))


def _build_blocked(h: int, w: int, bars: list[dict[str, object]]) -> np.ndarray:
    forbidden = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["blue_count"])
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in bar["cells"]:
            blocked[r, c] = True
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
    return blocked | forbidden


def _find_path(
    h: int,
    w: int,
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


def _main_value(path_is_vertical: bool, r: int, c: int) -> int:
    return int(r if path_is_vertical else c)


def _enters_slab(bar: dict[str, object], main_value: int, init_dir: int) -> bool:
    if init_dir in (1, 2):
        return main_value >= int(bar["slab_lo"])
    return main_value <= int(bar["slab_hi"])


def _solve_context(
    g: np.ndarray,
) -> tuple[int, int, int, str, int, bool, list[dict[str, object]], list[dict[str, object]], list[tuple[int, int]]]:
    h, w = g.shape
    sr, sc, init_dir, goal_kind, goal_val, path_is_vertical = _start_and_goal(g)
    bars = _extract_bars(g, path_is_vertical=path_is_vertical)
    ordered_bars = _ordered_bars(bars, init_dir)
    blocked = _build_blocked(h, w, bars)
    blocked[sr, sc] = False
    path = _find_path(h, w, sr, sc, init_dir, goal_kind, goal_val, blocked)
    return sr, sc, init_dir, goal_kind, goal_val, path_is_vertical, bars, ordered_bars, path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border-started route with obstacle-specific travel colors, letting blue markers be absorbed into green.

    Input structure:
    - Background is azure (8). A single green (3) start pixel lies on one border edge.
    - Obstacles are the original 1-cell-thick red/blue bars: red (2) supplies the bar body and blue (1) markers encode each bar's keepout radius.

    Rule:
    - The path geometry is still the minimum-step, minimum-turn orthogonal route that stays at Manhattan distance strictly greater than each bar's blue-count radius.
    - Each blue-count maps to a fixed journey color. When the path first enters that bar's detour slab, it switches to the mapped color and keeps it until a later bar replaces it.
    - Red cells on a bar are recolored to that same journey color in the output, while every blue marker turns green to show that it was absorbed into the path narrative.

    Key insight: Blue pixels now do double duty: they still determine how wide the detour must be, and they also choose the color that both the bypassed obstacle and the downstream path segment will share.
    """
    g = np.array(grid, copy=True)
    _, _, init_dir, _, _, path_is_vertical, bars, ordered_bars, path = _solve_context(g)

    out = g.copy()
    next_bar = 0
    current_color = GREEN
    for r, c in path:
        main_value = _main_value(path_is_vertical, r, c)
        while next_bar < len(ordered_bars) and _enters_slab(ordered_bars[next_bar], main_value, init_dir):
            current_color = int(ordered_bars[next_bar]["journey_color"])
            next_bar += 1
        if out[r, c] == BG:
            out[r, c] = current_color

    for bar in bars:
        journey_color = int(bar["journey_color"])
        for r, c in bar["cells"]:
            out[r, c] = GREEN if int(g[r, c]) == BLUE else journey_color
    return out


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
        if _min_main_len(combo) <= main_hi and len(set(combo)) >= 2
    ]
    assert choices, "generator needs at least one multi-count blue configuration"
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
    path_colors = {GREEN, *JOURNEY_COLORS}
    if edge == "top":
        return bool(np.any(np.isin(out[-1, :], list(path_colors))))
    if edge == "bottom":
        return bool(np.any(np.isin(out[0, :], list(path_colors))))
    if edge == "left":
        return bool(np.any(np.isin(out[:, -1], list(path_colors))))
    return bool(np.any(np.isin(out[:, 0], list(path_colors))))


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    (
        _sr,
        _sc,
        _init_dir,
        _goal_kind,
        _goal_val,
        path_is_vertical,
        bars,
        _ordered_bars,
        path,
    ) = _solve_context(inp)

    path_palette_colors = set()
    persistent_colored_cells = 0
    path_green_cells = 0
    for r, c in path:
        if int(inp[r, c]) != BG:
            continue
        out_color = int(out[r, c])
        if out_color == GREEN:
            path_green_cells += 1
        if out_color in JOURNEY_COLORS:
            path_palette_colors.add(out_color)
        main_value = _main_value(path_is_vertical, r, c)
        in_slab = any(int(bar["slab_lo"]) <= main_value <= int(bar["slab_hi"]) for bar in bars)
        if (not in_slab) and out_color in JOURNEY_COLORS:
            persistent_colored_cells += 1

    matched_bars = 0
    for bar in bars:
        journey_color = int(bar["journey_color"])
        bar_support = any(int(inp[r, c]) == RED and int(out[r, c]) == journey_color for r, c in bar["cells"])
        path_support = any(int(inp[r, c]) == BG and int(out[r, c]) == journey_color for r, c in path)
        if bar_support and path_support:
            matched_bars += 1

    absorbed_blue_count = int(np.count_nonzero((inp == BLUE) & (out == GREEN)))
    return {
        "path_palette_count": len(path_palette_colors),
        "persistent_colored_cells": persistent_colored_cells,
        "absorbed_blue_count": absorbed_blue_count,
        "matched_bars": matched_bars,
        "path_green_cells": path_green_cells,
    }


def _shows_fusion_signal(inp: np.ndarray, out: np.ndarray) -> bool:
    metrics = _variant_metrics(inp, out)
    return (
        metrics["path_palette_count"] >= 2
        and metrics["persistent_colored_cells"] > 0
        and metrics["absorbed_blue_count"] > 0
        and metrics["matched_bars"] >= 2
        and metrics["path_green_cells"] > 0
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
    min_main = max(main_lo, _min_main_len(blues))
    preferred_min = min_main + 1 if min_main < main_hi else min_main
    main_size = random.randint(preferred_min, main_hi)
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
        if not _shows_fusion_signal(inp, out):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
