# Variant of cb2d8a2c
# Dimension: D20 — 障碍条上蓝色像素的角色 (meta)
# Change: 蓝色像素承担单一角色：计数编码距离约束参数，最后统一变为红色... → 蓝色像素标记'危险区段'

from __future__ import annotations

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

# Design analysis for the D20 variant
# a) Concrete behavior:
# - The input still has one green border start, azure background, and 1-cell-thick red/blue bars.
# - The bar body is now visually asymmetric: a long red safe section attached to a wall, plus a contiguous blue
#   tip at the open end. The blue tip is the dangerous segment. Red cells remain solid obstacles, but only their
#   own cells are blocked; they no longer radiate danger.
# - The output still extends a green path to the opposite edge and still turns all blue cells red at the end.
# - Small trace:
#   Original: a left-attached bar with 3 blue pixels made the whole bar emit a radius-3 halo, so the path could
#   not even skim above the red portion.
#   Variant: only the 3-cell blue tip emits the radius-3 halo; the path may run right next to the red body, then
#   swing wider only near the blue tip. The key difference is local danger at the tip instead of uniform danger
#   around the full bar.
# b) Complexity calibration:
# - The blue segment length still takes values 1-4, so there are only a few radii to infer.
# - The generator forces each bar to have a noticeable red safe body (`SAFE_RED_MIN`), otherwise the local-vs-global
#   difference would be hard to see and examples could collapse back toward the original behavior.
# - A sample is accepted only if the solved path uses at least one cell that the original whole-bar halo would have
#   forbidden. That makes the D20 change visible instead of merely latent.
# c) Hidden assumptions review against the original main():
# - `blue_count = ...` was previously used as a per-bar radius, which is still acceptable, but only for the blue
#   segment of that bar. The dangerous source set must change from `comp` to `blue_cells`.
# - Seeding BFS from every bar cell was harmless in the original task because every bar cell inherited the same halo.
#   In this variant that would be wrong: red safe cells would incorrectly emit danger.
# - The original alternating B-R-B-R prefix generator encoded a scalar count but did not mark a clean spatial
#   segment. The new generator must instead paint one contiguous blue tip so humans can see which bar region is
#   dangerous.
# - The original start placement only needed to land somewhere inside the first bar span. In the variant it should
#   prefer the first bar's red section; otherwise the start could land inside the hazardous tip and reduce the visual
#   contrast between safe-red hugging and blue-tip avoidance.
# d) Placement legality:
# - Bars still alternate wall attachment and remain separated along the travel axis by at least `a + b + 2`, where
#   `a` and `b` are blue-tip lengths. This keeps their blue-tip halos from merging into an impossible wall.
# - Each bar length also satisfies `cross_gap >= blue_len + 1`, guaranteeing at least one passable column/row beyond
#   the hazardous tip.
# - Blue tips are always on the open end, and every bar has at least `SAFE_RED_MIN` red cells so the safe segment is
#   non-empty and visually distinct.
# e) Visual observability:
# - Across examples, people can compare red body vs blue tip directly: the path often slides beside red and only gives
#   extra clearance near blue. This is visible in 2-3 examples because the bar geometry stays simple and repeated.
# - No hidden ordering by color id is used; only visible positions and lengths matter.
# f) Visual cleanliness:
# - The grid still contains only four colors and one object family. Blue is confined to tip segments, so overlays are
#   easy to parse and the output remains uncluttered.
# g) Color-change inference:
# - Blue still turns red in the output, but only after serving as a hazard marker in the input. Because the blue cells
#   are grouped into tip segments, humans can infer that blue is an annotation on part of the red obstacle rather than
#   a separate object class.

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3


def _build_blocked(grid: np.ndarray, *, whole_bar_sources: bool) -> np.ndarray:
    """Return blocked cells using either the original whole-bar halo or the variant blue-only halo."""
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    blocked = bar_mask.copy()

    for comp in components_mask(bar_mask, connectivity=4):
        blue_cells = [(int(r), int(c)) for r, c in comp if grid[r, c] == BLUE]
        if not blue_cells:
            continue
        radius = len(blue_cells)
        if whole_bar_sources:
            sources = [(int(r), int(c)) for r, c in comp]
        else:
            sources = blue_cells

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in sources:
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

    return blocked


def _variant_only_path_cells(inp: np.ndarray, out: np.ndarray) -> int:
    """Count path cells that are legal only under the new blue-only hazard rule."""
    path_mask = (out == GREEN) & (inp == BG)
    old_blocked = _build_blocked(inp, whole_bar_sources=True)
    return int(np.count_nonzero(path_mask & old_blocked))


def _count_visible_danger_tips(inp: np.ndarray) -> int:
    """Count bars that visibly separate a blue danger tip from a red safe body."""
    bar_mask = (inp == BLUE) | (inp == RED)
    visible = 0
    for comp in components_mask(bar_mask, connectivity=4):
        rows = {int(r) for r, _ in comp}
        cols = sorted(int(c) for _, c in comp)
        if len(rows) != 1:
            continue
        row = next(iter(rows))
        c0, c1 = cols[0], cols[-1]
        if len(comp) != c1 - c0 + 1:
            continue
        segment = inp[row, c0 : c1 + 1]
        blue_count = int(np.count_nonzero(segment == BLUE))
        red_count = int(np.count_nonzero(segment == RED))
        if blue_count < 1 or red_count < SAFE_RED_MIN:
            continue
        if np.all(segment[:blue_count] == BLUE) or np.all(segment[-blue_count:] == BLUE):
            visible += 1
    return visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green route from one border to the opposite, skimming red bar segments but steering wider around blue-marked danger tips.

    Input structure:
    - bg = azure (8). Each obstacle is a 1-cell-thick red/blue bar, with blue cells marking the dangerous segment of that bar.
    - A single green (3) pixel sits on a grid border as the route start.

    Rule:
    - Every bar cell is still a solid obstacle, but only the blue cells radiate a keep-out halo; red cells are safe to approach closely.
    - A bar's halo radius equals how many blue cells that same bar contains, and the green path uses right-angle turns to reach the opposite edge.
    - After drawing the path, all blue pixels turn red.

    Key insight: Blue no longer means "the whole bar is dangerous by this count"; blue identifies the specific bar segment whose neighborhood must be avoided.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

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

    blocked = _build_blocked(g, whole_bar_sources=False)
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

    out = g.copy()
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


# --------------- generator (D20 variant) ---------------

# Grid size bounds (rows x cols from original spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26

# Bar counts and blue-tip lengths
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4

# Bar length bounds: keep the original overall scale but require a visible red safe body
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
SAFE_RED_MIN = 3

MAX_GEN_RETRIES = 400


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size so blue-tip halos do not choke the corridor."""
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
    """Pick blue-tip lengths that fit the main-axis spacing constraints."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and max(combo) >= 2
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range given a blue tip and required open-side corridor."""
    lo = max(
        BAR_LEN_MIN,
        blue_count + SAFE_RED_MIN,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - blue_count - 1,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint a bar whose blue segment sits on the open tip, not on the wall side."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    if side == 0:
        colors[length - blue_count :] = BLUE
        grid[row, :length] = colors
    else:
        colors[:blue_count] = BLUE
        grid[row, w - length :] = colors


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
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
    first_blue = blues[0]
    if first_len - first_blue < SAFE_RED_MIN:
        return None

    if first_side == 0:
        lo = 1
        hi = first_len - first_blue - 1
    else:
        lo = cross_size - first_len + first_blue
        hi = cross_size - 2
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
    except Exception:
        return None

    if not _reaches_opposite_edge(out, edge):
        return None
    if _count_visible_danger_tips(inp) <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D20 input")
