# Variant of cb2d8a2c
# Dimension: D13 — 障碍条数量 (meta)
# Change: 每个示例包含1-3条障碍条，路径依次绕过所有障碍条... → 障碍条中包含干扰条

# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#   - There is still one green border start, azure background, and 1-cell-wide wall-attached
#     red/blue bars whose blue count encodes the keep-out radius.
#   - The green route still goes to the opposite edge by minimizing steps first and turns second.
#   new_d13_change:
#   - The input now mixes 1-3 functional bars with 1-3 distractor bars.
#   - A functional bar is any red component that contains at least one blue cell.
#   - A distractor bar is a visually similar wall-attached pure-red strip. It is ignored by the
#     routing logic, so the path may pass through it and repaint crossed red cells green.
#   concrete_trace:
#   - Original: every red/blue component was inserted into the forbidden mask, so even a pure-red
#     strip forced a detour because its own cells were blocked.
#   - Variant: a pure-red strip contributes no forbidden cells at all. If it is placed on a row
#     where the optimal path is a straight vertical segment, the new solver keeps that straight
#     segment and the output shows a green crossing through the red strip.
#   visibility:
#   - Humans can tell the rule because bars with blue are always detoured around and end fully red,
#     while pure-red bars are sometimes pierced by green. The blue cells therefore remain the only
#     visible cue for "real obstacle" status.
#   complexity_calibration:
#   - Only one new distinction is introduced: blue-bearing bars matter, pure-red bars do not.
#     This keeps the rule learnable from 2-4 examples and avoids needing multiple distractor taxons.
#   implicit_assumption_audit:
#   - Old assumption: `bar_mask = red or blue` meant every component was a real obstacle.
#     This is false now, so the solver must classify components by blue presence before building
#     the forbidden region.
#   - Old assumption: painting only background cells was enough because the path never touched bars.
#     This is false now, because distractor bars may be crossed; the solver must allow path cells
#     to overwrite pure-red distractor cells with green.
#   placement_legality:
#   - Generator keeps every bar on its own row in canonical orientation and reserves at least one
#     blank row between any two bar rows, so 4-connectivity never merges different bars.
#   - Distractors are inserted only on rows at distance >= 2 from functional bars and other
#     distractors, preventing accidental component fusion.
#   - At least one distractor is placed in the start-to-first-bar gap, attached to the same wall as
#     the first functional bar, which guarantees that the optimal path crosses it.
#   cleanliness:
#   - Distractors share the same simple line shape as functional bars, so the scene is still tidy.
#     The only extra visual event is a green cut through some pure-red strips.
#   generator_contract_for_new_main:
#   - `main()` assumes that any blue-bearing component is a simple isolated bar and that the start
#     gap contains at least one safe row for the forced distractor crossing.
#   - `generate_input()` enforces both conditions by using the original bar spacing logic for
#     functional bars and only adding distractors in rows that are not adjacent to other bars.

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_CODE_PATH = Path(__file__).resolve()
for _candidate in (_CODE_PATH.parent, *_CODE_PATH.parents):
    if (_candidate / "arc_utils.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.append(str(_candidate))
        break

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _component_blue_count(grid: np.ndarray, comp: np.ndarray) -> int:
    return sum(1 for r, c in comp if grid[r, c] == BLUE)


def _functional_bar_components(grid: np.ndarray) -> list[np.ndarray]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    return [comp for comp in comps if _component_blue_count(grid, comp) > 0]


def _solve_path(grid: np.ndarray) -> list[tuple[int, int]]:
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

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in _functional_bar_components(g):
        radius = _component_blue_count(g, comp)
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

    blocked = forbidden
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
    Route the green path to the opposite edge while ignoring pure-red decoy bars.

    Input structure:
    - The background is azure (8), with one green (3) start pixel on a border edge.
    - The grid contains several wall-attached 1-cell-wide strips made from red/blue cells; some
      strips include blue markers, while others are pure red distractors.

    Rule:
    - Only connected strips that contain at least one blue cell are real obstacles.
    - A real strip's blue-cell count gives its Manhattan keep-out radius, and the path must avoid
      the whole forbidden zone while reaching the opposite edge with right-angle turns as needed.
    - Pure-red strips do not block movement, so the path may pass through them and repaint those
      crossed cells green; after routing, all remaining blue cells become red.

    Key insight:
    - Blue pixels now do double duty: they both encode the avoidance radius and reveal which strips
      are genuine obstacles. No blue means the strip is only a visual decoy.
    """
    out = np.array(grid, copy=True)
    for r, c in _solve_path(out):
        out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
FUNC_BAR_COUNT_LO, FUNC_BAR_COUNT_HI = 1, 3
DISTRACTOR_COUNT_LO, DISTRACTOR_COUNT_HI = 1, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400


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


def _functional_length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _distractor_length_bounds(cross_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    w = grid.shape[1]
    if blue_count == 0:
        colors = np.full(length, RED, dtype=int)
    else:
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
) -> tuple[np.ndarray, dict[str, object]] | None:
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
    sides = [(first_side + i) % 2 for i in range(count)]
    for row, blue_count, side in zip(rows, blues, sides):
        bounds = _functional_length_bounds(cross_size, blue_count)
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
    meta = {
        "rows": rows,
        "lengths": lengths,
        "sides": sides,
        "start_col": start_col,
        "first_side": first_side,
    }
    return grid, meta


def _choose_crossing_length(cross_size: int, side: int, col: int) -> int | None:
    bounds = _distractor_length_bounds(cross_size)
    if bounds is None:
        return None
    lo, hi = bounds
    need = col + 1 if side == 0 else cross_size - col
    lo = max(lo, need)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _path_mask(path: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path:
        mask[r, c] = True
    return mask


def _add_distractors(
    canonical: np.ndarray,
    meta: dict[str, object],
    distractor_count: int,
) -> np.ndarray | None:
    grid = canonical.copy()
    path = _solve_path(grid)
    path_mask = _path_mask(path, grid.shape)
    main_size, cross_size = grid.shape
    func_rows = set(int(r) for r in meta["rows"])
    used_rows: set[int] = set()

    first_row = int(meta["rows"][0])
    forced_candidates = list(range(1, first_row - 1))
    if not forced_candidates:
        return None
    forced_row = random.choice(forced_candidates)
    forced_col = int(meta["start_col"])
    forced_side = int(meta["first_side"])
    forced_len = _choose_crossing_length(cross_size, forced_side, forced_col)
    if forced_len is None:
        return None
    _paint_bar(grid, forced_row, forced_side, forced_len, blue_count=0)
    used_rows.add(forced_row)

    candidates: list[tuple[int, int]] = []
    for r in range(1, main_size - 1):
        if r in used_rows:
            continue
        if any(abs(r - rr) <= 1 for rr in func_rows | used_rows):
            continue
        cols = np.flatnonzero(path_mask[r])
        if len(cols) != 1:
            continue
        candidates.append((r, int(cols[0])))
    random.shuffle(candidates)

    target = min(distractor_count, 1 + len(candidates))
    for r, col in candidates:
        if len(used_rows) >= target:
            break
        feasible: list[tuple[int, int]] = []
        for side in (0, 1):
            length = _choose_crossing_length(cross_size, side, col)
            if length is not None:
                feasible.append((side, length))
        if not feasible:
            continue
        side, length = random.choice(feasible)
        _paint_bar(grid, r, side, length, blue_count=0)
        used_rows.add(r)

    if len(used_rows) < 1:
        return None
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


def _red_crossings(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.sum((inp == RED) & (out == GREEN)))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    func_count = random.randint(FUNC_BAR_COUNT_LO, FUNC_BAR_COUNT_HI)
    distractor_count = random.randint(DISTRACTOR_COUNT_LO, DISTRACTOR_COUNT_HI)
    blues = _choose_blues(func_count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    built = _canonical_grid(main_size, cross_size, blues, first_side)
    if built is None:
        return None
    canonical, meta = built
    canonical = _add_distractors(canonical, meta, distractor_count)
    if canonical is None:
        return None

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
    except Exception:
        return None
    if not _reaches_opposite_edge(out, edge):
        return None
    if _red_crossings(inp, out) < 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if len(_functional_bar_components(inp)) != func_count:
        return None
    total_comps = components_mask((inp == BLUE) | (inp == RED), connectivity=4)
    distractors = len(total_comps) - len(_functional_bar_components(inp))
    if distractors < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D13 input")
