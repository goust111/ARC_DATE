# Variant of cb2d8a2c
# Dimension: D2 — 路径转弯规则 (path)
# Change: 遇到障碍条时在距离约束处直角转弯，转弯方向由障碍条位置和路径当前方向决定，路径总是绕向障碍条的远端继续前进... → 固定转弯方向（始终顺时针或逆时针）
#
# Design analysis
# ---------------
# variant:
#   core_change:
#     - The path no longer lets obstacle placement decide which lateral side to use.
#       Instead, the task has a single fixed turn preference: in canonical top-start
#       orientation the first detour around every blocking bar goes to the clockwise side.
#   required_adaptation:
#     - In the original task every bar touches a wall. After Manhattan-radius expansion,
#       that leaves only one viable lateral bypass side, so a fixed clockwise/counter-
#       clockwise preference would be invisible and effectively degenerate.
#     - To make D2 observable without changing the bar/radius/color semantics, bars are
#       changed to floating horizontal/vertical segments with safe gaps on both sides.
#       This is the minimal placement change needed to create a genuine left-vs-right
#       detour choice.
#   behavior_trace:
#     - Canonical input: green start on the top edge, path heading downward, horizontal
#       obstacle bars below it.
#     - Original logic: shortest-path search routes around each bar on whichever side is
#       shorter / structurally forced by wall attachment.
#     - Variant logic: if the current main lane intersects a bar's keepout zone, the path
#       goes down to the row just above that zone, turns clockwise (therefore left in the
#       canonical frame), descends past the bar, then returns to the original lane.
#     - Visual effect: each blocking bar produces a same-handed "left bulge" in canonical
#       orientation; after rotation to another start edge this becomes the corresponding
#       global clockwise bias.
#   complexity_calibration:
#     - Keep 2-4 bars so each sample shows repeated evidence for the fixed preference.
#     - Bars still use 1-4 blue cells to encode radius and still convert blue->red.
#     - The reference lane intersects every bar keepout zone, so the altered turning rule
#       is exercised multiple times instead of appearing only on rare samples.
#   hidden_assumptions_review:
#     - The original `main()` relied on Dijkstra tie-breaking and on wall-attached bars.
#       Those assumptions do not survive the variant, so routing is rewritten as explicit
#       per-bar geometry after rotating to a canonical frame.
#     - The new solver assumes all obstacle components are single-row bars in canonical
#       orientation and that the detour rows/columns stay outside every exact keepout mask.
#       The generator enforces those constraints with spacing and side-gap bounds and the
#       solver validates the painted segments against the exact forbidden mask.
#   placement_legality:
#     - Each floating bar leaves at least `blue_count + 1` empty cells on both sides so
#       both clockwise and counterclockwise detours would be physically possible.
#     - Main-axis gaps satisfy the solver's three-segment detour pattern:
#       next_bar.row - prev_bar.row >= prev_blue + next_blue + 3.
#     - Bars are long enough to cover the reference lane, guaranteeing that each one
#       actually forces a turn.
#   observability_and_cleanliness:
#     - Because all bars are floating and the same lane hits each keepout zone, the train
#       pairs repeatedly show "same-handed detours" rather than the original alternating
#       shortest-route behavior.
#     - Blue pixels only encode radius/count; no hidden ordering by color id is used.

from collections import deque
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

TURN_PREFERENCE = "CW"


def _rotate_to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, 1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, -1).copy()  # edge == "left"


def _rotate_from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, -1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, 1).copy()  # edge == "left"


def _start_edge(grid: np.ndarray) -> tuple[str, int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return "top", r, c
    if r == h - 1:
        return "bottom", r, c
    if c == 0:
        return "left", r, c
    if c == w - 1:
        return "right", r, c
    raise AssertionError("green start must lie on a border")


def _bar_data(grid: np.ndarray) -> list[dict[str, int]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, int]] = []
    for comp in comps:
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert r0 == r1, "canonical bars must be horizontal"
        blue_count = int(np.sum(grid[comp[:, 0], comp[:, 1]] == BLUE))
        bars.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "blue": blue_count,
                "top": r0 - blue_count,
                "bottom": r0 + blue_count,
                "left": c0 - blue_count,
                "right": c1 + blue_count,
            }
        )
    bars.sort(key=lambda d: d["row"])
    return bars


def _forbidden_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    forbidden = np.zeros((h, w), dtype=bool)
    for comp in components_mask(bar_mask, connectivity=4):
        radius = int(np.sum(grid[comp[:, 0], comp[:, 1]] == BLUE))
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))
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
    return forbidden


def _paint_segment(
    out: np.ndarray,
    blocked: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    start: tuple[int, int],
) -> None:
    assert r0 == r1 or c0 == c1, "segments must be axis-aligned"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        coords = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        coords = [(r, c0) for r in range(r0, r1 + step, step)]
    for r, c in coords:
        if blocked[r, c] and (r, c) != start:
            raise AssertionError("path segment entered a forbidden cell")
        if out[r, c] == BG:
            out[r, c] = GREEN


def _preferred_detour(bar: dict[str, int], width: int) -> list[int]:
    left_col = bar["left"] - 1
    right_col = bar["right"] + 1
    preferred = [left_col, right_col] if TURN_PREFERENCE == "CW" else [right_col, left_col]
    return [c for c in preferred if 0 <= c < width]


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the start on the top edge"

    bars = _bar_data(g)
    blocked = _forbidden_mask(g)
    blocked[sr, sc] = False

    out = g.copy()
    lane_col = sc
    cur_r = sr
    cur_c = sc

    for bar in bars:
        top_row = bar["top"] - 1
        bottom_row = bar["bottom"] + 1
        if lane_col < bar["left"] or lane_col > bar["right"]:
            continue
        assert cur_r <= top_row, "bars must be ordered with enough vertical spacing"
        _paint_segment(out, blocked, cur_r, cur_c, top_row, lane_col, (sr, sc))

        detour_candidates = _preferred_detour(bar, w)
        detour_col = None
        for cand in detour_candidates:
            try:
                probe = out.copy()
                _paint_segment(probe, blocked, top_row, lane_col, top_row, cand, (sr, sc))
                _paint_segment(probe, blocked, top_row, cand, bottom_row, cand, (sr, sc))
                _paint_segment(probe, blocked, bottom_row, cand, bottom_row, lane_col, (sr, sc))
                detour_col = cand
                break
            except AssertionError:
                continue
        assert detour_col is not None, "generator must leave a valid detour side"

        _paint_segment(out, blocked, top_row, lane_col, top_row, detour_col, (sr, sc))
        _paint_segment(out, blocked, top_row, detour_col, bottom_row, detour_col, (sr, sc))
        _paint_segment(out, blocked, bottom_row, detour_col, bottom_row, lane_col, (sr, sc))
        cur_r, cur_c = bottom_row, lane_col

    _paint_segment(out, blocked, cur_r, cur_c, h - 1, lane_col, (sr, sc))
    out[out == BLUE] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green border-to-border route whose bar detours follow a fixed clockwise preference.

    Input structure:
    - The background is azure (8), with one green (3) start pixel on a border edge.
    - Red/blue obstacle bars (2/1) are thin floating segments perpendicular to the path direction.

    Rule:
    - Each bar's blue-pixel count is its Manhattan keep-out radius, so the green route must stay strictly outside that expanded zone.
    - Rotate the scene to a canonical top-start view, move straight toward the opposite edge, and whenever a bar blocks the current lane, take the same fixed-handed detour side first (here: clockwise), then return to the lane after clearing the bar.
    - When the route is finished, recolor every blue pixel to red.

    Key insight: Blue pixels still encode how wide each bar's danger zone is, but the side chosen for every detour is no longer inferred from the bar placement; it is a task-wide fixed turning bias that must be learned from examples.
    """
    g = np.asarray(grid, dtype=int)
    edge, _, _ = _start_edge(g)
    canonical = _rotate_to_canonical(g, edge)
    solved = _solve_canonical(canonical)
    return _rotate_from_canonical(solved, edge)


# --------------- generator (o5x) ---------------

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 800


def _min_main_len(blues: tuple[int, ...]) -> int:
    row = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        row += a + b + 3
    return row + blues[-1] + 2


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


def _choose_blues(count: int, main_hi: int, cross_hi: int) -> tuple[int, ...]:
    choices = []
    for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count):
        if _min_main_len(combo) > main_hi:
            continue
        if all(_length_bounds(cross_hi, blue_count) is not None for blue_count in combo):
            choices.append(combo)
    assert choices, "at least one blue-count combination must fit"
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(
        BAR_LEN_MIN,
        2 * blue_count - 1,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - 2 * (blue_count + 1),
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, c0: int, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    grid[row, c0 : c0 + length] = colors


def _choose_bar_start(cross_size: int, path_col: int, length: int, blue_count: int) -> int | None:
    min_c0 = max(blue_count + 1, path_col - (length - 1 + blue_count))
    max_c0 = min(cross_size - blue_count - 2 - length + 1, path_col + blue_count)
    if min_c0 > max_c0:
        return None
    candidates = []
    for c0 in range(min_c0, max_c0 + 1):
        keep_left = c0 - blue_count
        keep_right = c0 + length - 1 + blue_count
        left_detour = keep_left - 1
        right_detour = keep_right + 1
        left_cost = path_col - left_detour
        right_cost = right_detour - path_col
        if left_cost <= right_cost:
            continue
        candidates.append(c0)
    if not candidates:
        return None
    cutoff = candidates[max(0, len(candidates) // 2 - 1)]
    biased = [c0 for c0 in candidates if c0 <= cutoff]
    pool = biased if biased else candidates
    return random.choice(pool)


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
) -> np.ndarray | None:
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    path_lo = max(max(blues) + 3, cross_size // 2)
    path_hi = min(cross_size - 3, cross_size // 2 + 3)
    if path_lo > path_hi:
        return None
    path_col = random.randint(path_lo, path_hi)

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 3 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count in zip(rows, blues, strict=False):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        lengths = list(range(bounds[0], bounds[1] + 1))
        random.shuffle(lengths)
        placed = False
        for length in lengths:
            c0 = _choose_bar_start(cross_size, path_col, length, blue_count)
            if c0 is None:
                continue
            _paint_bar(grid, row, c0, length, blue_count)
            placed = True
            break
        if not placed:
            return None

    grid[0, path_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    if edge == "left":
        return np.rot90(grid, 1).copy()
    return np.rot90(grid, -1).copy()  # edge == "right"


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _original_style_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1
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
    forbidden = _forbidden_mask(g)
    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    import heapq

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
    assert goal_state is not None
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


def _difference_metric(inp: np.ndarray) -> int:
    var_out = main(inp)
    orig_out = _original_style_main(inp)
    return int(np.count_nonzero((var_out == GREEN) != (orig_out == GREEN)))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi, cross_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)

    canonical = _canonical_grid(main_size, cross_size, blues)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if _difference_metric(inp) < 8:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D2 input")
