# Variant of cb2d8a2c
# Dimension: D19 — 符号指令编码 (symbol_instructions)
# Change: 无显式图例——蓝色像素数量即为距离参数，规则完全从输入输出对中推断... → 障碍条上的标记形状编码距离

# Design analysis for the D19 variant
# a) Concrete appearance + behavior trace:
#    Each obstacle is still a wall-attached bar, but the free end now widens into a shared 3x3 red cap.
#    The blue cells inside that cap form one of three symbols:
#      - center dot        -> keep-out radius 1
#      - orthogonal plus   -> keep-out radius 2
#      - diagonal X        -> keep-out radius 3
#    Old local behavior: a bar row such as B-R-B-R-B-R encoded radius by counting the blue cells on the line.
#    New local behavior: the bar body is mostly red, and the 3x3 cap carries the instruction. Example:
#      old: `1212222` meant radius 2 because there were two blue cells.
#      new: `222` with a 3x3 cap whose blue pattern is `plus` also means radius 2, while an `X` cap can have the
#      same blue-count as the plus but a different radius. The pathfinder still builds a forbidden Manhattan halo
#      per component, but the halo size now comes from symbol decoding instead of raw blue tallying.
# b) Complexity calibration:
#    The variant uses exactly three symbol classes. That is enough to make D19 visible, but small enough that 2-4
#    training pairs can support the mapping. All symbols live in the same 3x3 cap so the visual difference is the
#    internal blue arrangement, not a large geometry change. Two non-dot symbols both use five blue cells, so the
#    task cannot collapse back to "count the blue cells".
# c) Hidden-assumption audit against the original main():
#    - `blue_count = sum(...)` no longer matches the rule and must be replaced by explicit pattern recognition.
#    - Original bars were 1 cell thick. The new cap is 3 cells thick, so original main-axis spacing formulas are
#      no longer harmless: reusing them can merge components or let keep-out halos touch. The generator therefore
#      increases center-to-center spacing.
#    - The original color alternation along the bar was a count carrier. It is removed; solver must not assume
#      alternating colors or that every blue lies on the centerline.
# d) Placement legality:
#    The generator keeps bars disjoint as 4-connected components, keeps every 3x3 cap strictly inside the grid,
#    preserves wall attachment, and keeps at least `radius + 1` uncovered cells on the open side so a legal route
#    exists around each bar. Main-axis placement leaves a blank safe band between adjacent keep-out halos.
# e) Visual observability:
#    A human can see that every output path bends farther around some caps than others. Because plus and X have the
#    same blue-cell count but produce different detours, the intended rule must be about symbol identity.
# f) Visual cleanliness:
#    All annotation is confined to one 3x3 cap per bar. There is no multi-layer overlap beyond the obstacle itself,
#    so bars, symbols, and the final green path remain easy to parse.
# g) Color-change inferability:
#    The palette stays minimal: background azure, obstacle red, symbol blue, path green. The only recolor remains
#    blue -> red after solving, exactly where the symbols were.
# 9) New main() assumptions and generator guarantees:
#    - Every obstacle component contains exactly one of the three supported cap symbols.
#    - Every component bbox is either 3 x L or L x 3 after global rotation.
#    - The unique green start pixel is on the border and lies within the first bar's span, so the first detour is
#      forced by obstacle geometry rather than being optional.

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DOT = frozenset({(0, 0)})
_PLUS = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
_X = frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)})

_SHAPE_TO_RADIUS = {
    _DOT: 1,
    _PLUS: 2,
    _X: 3,
}

_RADIUS_TO_MASK = {
    1: np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    2: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
}


def _decode_radius(g: np.ndarray, comp: np.ndarray) -> int:
    blue_cells = [(int(r), int(c)) for r, c in comp if g[r, c] == BLUE]
    assert blue_cells, "every obstacle must contain a blue instruction symbol"
    rows = [r for r, _ in blue_cells]
    cols = [c for _, c in blue_cells]
    r0, c0 = min(rows), min(cols)
    shape = frozenset((r - r0, c - c0) for r, c in blue_cells)
    assert shape in _SHAPE_TO_RADIUS, f"unsupported symbol shape: {sorted(shape)}"
    return _SHAPE_TO_RADIUS[shape]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a green path from one border to the opposite while wall-attached obstacle bars announce their clearance with blue symbols.

    Input structure:
    - bg = azure (8). Each obstacle is a red bar (2) attached to a side wall, with a 3x3 red end-cap containing a blue symbol (1).
    - A single green (3) pixel appears on one border edge and marks the path start.

    Rule:
    - The blue symbol inside each cap encodes that bar's keep-out radius: center dot -> 1, orthogonal plus -> 2, diagonal X -> 3.
    - Extend the green path to the opposite edge, using axis-aligned moves and staying outside every bar's Manhattan keep-out zone.
    - After the route is drawn, all blue symbol pixels turn red.

    Key insight: The blue cells are instructions by arrangement, not by quantity; two symbols can use the same number of blue cells while demanding different detours.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

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
        radius = _decode_radius(g, comp)

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            dist[r, c] = 0
            q.append((int(r), int(c)))

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


# --------------- generator (o5x) ---------------

# Grid size bounds (rows x cols from the source spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26

# Bar counts and encoded radii
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
RADIUS_LO, RADIUS_HI = 1, 3

# Bar length bounds: total span including the 3x3 symbol cap
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79

MAX_GEN_RETRIES = 400


def _min_main_len(radii: tuple[int, ...]) -> int:
    last_center = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
        last_center += a + b + 4
    return last_center + radii[-1] + 3


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


def _choose_radii(count: int, main_hi: int) -> tuple[int, ...] | None:
    all_choices = [
        combo
        for combo in itertools.product(range(RADIUS_LO, RADIUS_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    if not all_choices:
        return None
    diverse = [combo for combo in all_choices if len(set(combo)) >= 2]
    pool = diverse if diverse else all_choices
    return random.choice(pool)


def _length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 3, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - radius - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, radius: int) -> None:
    w = grid.shape[1]
    if side == 0:
        stem_end = length - 3
        grid[row, :stem_end] = RED
        cap = grid[row - 1 : row + 2, stem_end:length]
    else:
        cap_start = w - length
        cap = grid[row - 1 : row + 2, cap_start : cap_start + 3]
        grid[row, cap_start + 3 : w] = RED
    cap[:, :] = RED
    cap[_RADIUS_TO_MASK[radius]] = BLUE


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 4 + extras[i]
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
        start_col = random.randint(1, first_len - 2)
    else:
        start_col = random.randint(cross_size - first_len + 1, cross_size - 2)
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

    feasible_counts = [count for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1) if _choose_radii(count, main_hi) is not None]
    if not feasible_counts:
        return None
    count = random.choice(feasible_counts)
    radii = _choose_radii(count, main_hi)
    if radii is None:
        return None

    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, radii, first_side)
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D19 input")
