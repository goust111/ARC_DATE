from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _role_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, _W = g.shape
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected marker color plus orange obstacle color"

    marker_candidates: list[int] = []
    for color in colors:
        coords = np.argwhere(g == color)
        rows = set(map(int, coords[:, 0]))
        if rows == {H - 1}:
            marker_candidates.append(int(color))

    assert len(marker_candidates) == 1, "expected exactly one bottom-edge marker color"
    marker = marker_candidates[0]
    obstacles = [int(color) for color in colors if int(color) != marker]
    assert len(obstacles) == 1, "expected one obstacle color"
    return marker, obstacles[0]


def _right_exit_col(g: np.ndarray, obstacle: int, r: int, c: int) -> int:
    _H, W = g.shape
    exit_c = c
    while exit_c + 1 < W and int(g[r, exit_c + 1]) == obstacle:
        exit_c += 1
    exit_c += 1
    assert exit_c < W, "route must have background space to the right of the hit run"
    return exit_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw upward routes from the bottom markers, passing under exposed orange cells and stopping below orange roof notches.

    Input structure
    - The background is the dominant color.
    - One marker color appears only as one or more bottom-edge pixels.
    - The other foreground color forms orange obstacle pixels that the routes climb toward.

    Rule
    - From each bottom marker, move upward in its current column to the nearest orange pixel in the original input.
    - If that orange pixel roofs a one-cell notch, paint the vertical approach red and put a new marker in the notch below it.
    - Otherwise recolor the hit orange pixel teal, mark the cell below it yellow, draw red under the rest of its rightward row-run, mark the exit cell green, and continue upward from that exit column.
    - If no orange pixel remains above the current column, paint red up to the top edge and put a new marker on the top edge.

    Key insight
    - The route is a deterministic upward scan on the original orange mask; each exposed hit exits just to the right of the contiguous orange run on that row.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    marker, obstacle = _role_colors(g, bg)
    out = g.copy()

    def paint_bg(r: int, c: int, color: int) -> None:
        assert 0 <= r < H and 0 <= c < W, "paint coordinate out of bounds"
        assert int(g[r, c]) == bg, "route paint must land on original background"
        current = int(out[r, c])
        assert current in (bg, int(color)), "conflicting route paint"
        out[r, c] = int(color)

    def paint_vertical(c: int, top_r: int, bottom_r_exclusive: int) -> None:
        assert top_r <= bottom_r_exclusive, "vertical route interval must be ordered"
        for rr in range(top_r, bottom_r_exclusive):
            paint_bg(rr, c, Color.RED)

    def is_notch_roof(r: int, c: int) -> bool:
        if not (r + 1 < H and 0 < c < W - 1):
            return False
        return (
            int(g[r + 1, c]) == bg
            and int(g[r + 1, c - 1]) == obstacle
            and int(g[r + 1, c + 1]) == obstacle
        )

    starts = sorted((int(r), int(c)) for r, c in np.argwhere(g == marker))
    assert starts, "expected at least one bottom marker"

    for start_r, start_c in starts:
        assert start_r == H - 1, "all input markers must be on the bottom edge"
        cursor_r = start_r
        c = start_c
        seen: set[tuple[int, int]] = set()

        while True:
            obstacle_rows = [rr for rr in range(cursor_r - 1, -1, -1) if int(g[rr, c]) == obstacle]
            if not obstacle_rows:
                paint_vertical(c, 1, cursor_r)
                paint_bg(0, c, marker)
                break

            r = obstacle_rows[0]
            assert r + 1 < H, "hit obstacle must have a cell below it"

            if is_notch_roof(r, c):
                paint_vertical(c, r + 2, cursor_r)
                paint_bg(r + 1, c, marker)
                break

            paint_vertical(c, r + 2, cursor_r)
            assert int(out[r, c]) in (obstacle, Color.TEAL), "hit cell must still be orange or teal"
            out[r, c] = Color.TEAL
            paint_bg(r + 1, c, Color.YELLOW)

            exit_c = _right_exit_col(g, obstacle, r, c)
            for cc in range(c + 1, exit_c):
                paint_bg(r + 1, cc, Color.RED)
            paint_bg(r + 1, exit_c, Color.GREEN)

            cursor_r = r + 1
            c = exit_c
            state = (cursor_r, c)
            assert state not in seen, "route must not loop"
            seen.add(state)

    return out

# --------------- generator (o5x) ---------------

_GEN_BG = 0
_GEN_RED = 2
_GEN_GREEN = 3
_GEN_YELLOW = 4
_GEN_MARKER = 6
_GEN_ORANGE = 7
_GEN_TEAL = 8

# Canvas and marker bounds.
_GEN_MIN_MARKERS = 1
_GEN_MAX_MARKERS = 3
_GEN_MIN_HEIGHT = 17
_GEN_MAX_HEIGHT = 28
_GEN_MIN_WIDTH = 10
_GEN_MAX_WIDTH = 29

# Lane and obstacle geometry.
_GEN_MIN_LANE_WIDTH = 3
_GEN_LANE_GAP_WIDTH = 1
_GEN_MIN_ORANGE_RUN = 1
_GEN_MAX_ORANGE_RUN = 7
_GEN_NOTCH_SIDE_MARGIN = 1

# Route structure.
_GEN_MIN_ROUTE_HITS = 1
_GEN_MAX_ROUTE_HITS = 6
_GEN_HIT_ROW_TOP_MARGIN = 2
_GEN_HIT_ROW_BOTTOM_MARGIN = 2
_GEN_NOTCH_PROB = 0.4

# Filler density and retry limits.
_GEN_MIN_OBSTACLE_DENSITY = 0.06
_GEN_MAX_OBSTACLE_DENSITY = 0.15
_GEN_FILLER_ATTEMPTS = 4000
_GEN_RETRY_ATTEMPTS = 300


def _gen_lanes(width: int, count: int) -> list[tuple[int, int]] | None:
    gap_cols = _GEN_LANE_GAP_WIDTH * (count - 1)
    usable = width - gap_cols
    min_usable = _GEN_MIN_LANE_WIDTH * count
    if usable < min_usable:
        return None

    lane_widths = [_GEN_MIN_LANE_WIDTH] * count
    for _ in range(usable - min_usable):
        lane_widths[random.randrange(count)] += 1

    lanes: list[tuple[int, int]] = []
    col = 0
    for lane_w in lane_widths:
        lanes.append((col, col + lane_w - 1))
        col += lane_w + _GEN_LANE_GAP_WIDTH
    return lanes


def _gen_count_notches(grid: np.ndarray) -> int:
    H, W = grid.shape
    if H < 2 or W < 3:
        return 0
    roofs = grid[:-1, 1:-1] == _GEN_ORANGE
    centers = grid[1:, 1:-1] == _GEN_BG
    left_walls = grid[1:, :-2] == _GEN_ORANGE
    right_walls = grid[1:, 2:] == _GEN_ORANGE
    return int(np.count_nonzero(roofs & centers & left_walls & right_walls))


def _gen_rows_run_ok(grid: np.ndarray, rows: set[int] | None = None) -> bool:
    H, W = grid.shape
    if rows is None:
        rows = set(range(H))
    for r in rows:
        run = 0
        for c in range(W):
            if int(grid[r, c]) == _GEN_ORANGE:
                run += 1
                if run > _GEN_MAX_ORANGE_RUN:
                    return False
            else:
                run = 0
    return True


def _gen_choose_canvas(marker_count: int) -> tuple[int, int, list[tuple[int, int]] | None]:
    min_width = max(
        _GEN_MIN_WIDTH,
        _GEN_MIN_LANE_WIDTH * marker_count + _GEN_LANE_GAP_WIDTH * (marker_count - 1),
    )
    height = random.randint(_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT)
    width = random.randint(min_width, _GEN_MAX_WIDTH)
    return height, width, _gen_lanes(width, marker_count)


def _gen_mark_no_orange(no_orange: np.ndarray, r: int, c: int) -> None:
    H, W = no_orange.shape
    if 0 <= r < H and 0 <= c < W:
        no_orange[r, c] = True


def _gen_set_obstacle(
    grid: np.ndarray,
    expected: np.ndarray,
    route_used: set[tuple[int, int]],
    r: int,
    c: int,
) -> bool:
    if (r, c) in route_used:
        return False
    if int(grid[r, c]) not in (_GEN_BG, _GEN_ORANGE):
        return False
    grid[r, c] = _GEN_ORANGE
    if int(expected[r, c]) == _GEN_BG:
        expected[r, c] = _GEN_ORANGE
    return True


def _gen_paint_bg(
    grid: np.ndarray,
    expected: np.ndarray,
    no_orange: np.ndarray,
    route_used: set[tuple[int, int]],
    r: int,
    c: int,
    color: int,
) -> bool:
    if (r, c) in route_used:
        return False
    if int(grid[r, c]) != _GEN_BG or int(expected[r, c]) != _GEN_BG:
        return False
    expected[r, c] = color
    no_orange[r, c] = True
    route_used.add((r, c))
    return True


def _gen_paint_hit(
    grid: np.ndarray,
    expected: np.ndarray,
    route_used: set[tuple[int, int]],
    r: int,
    c: int,
) -> bool:
    if (r, c) in route_used:
        return False
    if int(grid[r, c]) != _GEN_ORANGE or int(expected[r, c]) != _GEN_ORANGE:
        return False
    expected[r, c] = _GEN_TEAL
    route_used.add((r, c))
    return True


def _gen_choose_route_plan(
    height: int,
    width: int,
    lane: tuple[int, int],
    want_notch: bool,
) -> dict[str, object] | None:
    """Choose a lane-local route skeleton; the grid is not mutated here."""
    lane_lo, lane_hi = lane
    effective_hi = lane_hi
    if want_notch:
        effective_hi = min(effective_hi - _GEN_NOTCH_SIDE_MARGIN, width - 1 - _GEN_NOTCH_SIDE_MARGIN)
    if effective_hi - lane_lo < _GEN_MIN_ROUTE_HITS:
        return None

    max_hits = min(_GEN_MAX_ROUTE_HITS, effective_hi - lane_lo)
    hit_count = random.randint(_GEN_MIN_ROUTE_HITS, max_hits)
    start_c = random.randint(lane_lo, effective_hi - hit_count)

    if height - (_GEN_HIT_ROW_TOP_MARGIN + _GEN_HIT_ROW_BOTTOM_MARGIN) < hit_count:
        return None
    hit_rows = sorted(
        random.sample(range(_GEN_HIT_ROW_TOP_MARGIN, height - _GEN_HIT_ROW_BOTTOM_MARGIN), hit_count),
        reverse=True,
    )
    notch_row = None
    if want_notch:
        roof_choices = list(range(0, hit_rows[-1]))
        if not roof_choices:
            return None
        notch_row = random.choice(roof_choices)

    runs: list[tuple[int, int, int, int]] = []
    c = start_c
    for i, r in enumerate(hit_rows):
        remaining_hits = hit_count - i - 1
        max_adv = min(_GEN_MAX_ORANGE_RUN, effective_hi - c - remaining_hits)
        if max_adv < _GEN_MIN_ORANGE_RUN:
            return None
        advance = random.randint(_GEN_MIN_ORANGE_RUN, max_adv)
        run_end = c + advance - 1
        back_limit = min(c - lane_lo, _GEN_MAX_ORANGE_RUN - advance)
        back = random.randint(0, back_limit)
        run_left = c - back
        runs.append((r, c, run_left, run_end))
        c = run_end + 1

    return {
        "start_c": start_c,
        "runs": runs,
        "notch_row": notch_row,
        "final_c": c,
    }


def _gen_choose_route_plans(
    height: int,
    width: int,
    lanes: list[tuple[int, int]],
) -> tuple[list[dict[str, object]], int] | None:
    """Choose one route plan per lane and return the intended notch count."""
    marker_count = len(lanes)
    use_notch = random.random() < _GEN_NOTCH_PROB
    notch_route = random.randrange(marker_count) if use_notch else -1
    plans: list[dict[str, object]] = []
    actual_notches = 0

    for i, lane in enumerate(lanes):
        want_notch = i == notch_route
        plan = _gen_choose_route_plan(height, width, lane, want_notch)
        if plan is None and want_notch:
            plan = _gen_choose_route_plan(height, width, lane, False)
            want_notch = False
        if plan is None:
            return None
        if want_notch:
            actual_notches = 1
        plans.append(plan)

    return plans, actual_notches


def _gen_apply_route(
    grid: np.ndarray,
    expected: np.ndarray,
    no_orange: np.ndarray,
    route_used: set[tuple[int, int]],
    plan: dict[str, object],
) -> bool:
    H, W = grid.shape
    start_c = int(plan["start_c"])
    runs = list(plan["runs"])
    notch_row = plan["notch_row"]
    cursor_r = H - 1
    c = start_c

    if int(grid[H - 1, c]) != _GEN_BG:
        return False
    grid[H - 1, c] = _GEN_MARKER
    expected[H - 1, c] = _GEN_MARKER
    no_orange[H - 1, c] = True

    touched_rows: set[int] = set()
    for r, hit_c, run_left, run_end in runs:
        if hit_c != c or not (run_left <= c <= run_end) or run_end + 1 >= W:
            return False
        exit_c = run_end + 1

        for cc in range(run_left, run_end + 1):
            if not _gen_set_obstacle(grid, expected, route_used, r, cc):
                return False
        touched_rows.add(r)
        _gen_mark_no_orange(no_orange, r, run_left - 1)
        _gen_mark_no_orange(no_orange, r, exit_c)

        for rr in range(r + 2, cursor_r):
            if not _gen_paint_bg(grid, expected, no_orange, route_used, rr, c, _GEN_RED):
                return False
        if not _gen_paint_hit(grid, expected, route_used, r, c):
            return False
        if not _gen_paint_bg(grid, expected, no_orange, route_used, r + 1, c, _GEN_YELLOW):
            return False
        for cc in range(c + 1, exit_c):
            if not _gen_paint_bg(grid, expected, no_orange, route_used, r + 1, cc, _GEN_RED):
                return False
        if not _gen_paint_bg(grid, expected, no_orange, route_used, r + 1, exit_c, _GEN_GREEN):
            return False

        cursor_r = r + 1
        c = exit_c

    if notch_row is None:
        for rr in range(1, cursor_r):
            if not _gen_paint_bg(grid, expected, no_orange, route_used, rr, c, _GEN_RED):
                return False
        if not _gen_paint_bg(grid, expected, no_orange, route_used, 0, c, _GEN_MARKER):
            return False
    else:
        r = int(notch_row)
        if not (0 <= r < cursor_r - 1 and 0 < c < W - 1):
            return False
        for rr in range(r + 2, cursor_r):
            if not _gen_paint_bg(grid, expected, no_orange, route_used, rr, c, _GEN_RED):
                return False
        for rr, cc in ((r, c), (r + 1, c - 1), (r + 1, c + 1)):
            if not _gen_set_obstacle(grid, expected, route_used, rr, cc):
                return False
            touched_rows.add(rr)
        if not _gen_paint_bg(grid, expected, no_orange, route_used, r + 1, c, _GEN_MARKER):
            return False
        _gen_mark_no_orange(no_orange, r, c - 1)
        _gen_mark_no_orange(no_orange, r, c + 1)

    return _gen_rows_run_ok(grid, touched_rows)


def _gen_render_routes(
    height: int,
    width: int,
    plans: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Render planned routes into input, expected output, and no-filler masks."""
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    no_orange = np.zeros((height, width), dtype=bool)
    route_used: set[tuple[int, int]] = set()

    for plan in plans:
        if not _gen_apply_route(grid, expected, no_orange, route_used, plan):
            return None
    return grid, expected, no_orange


def _gen_can_add_orange(
    grid: np.ndarray,
    no_orange: np.ndarray,
    cells: list[tuple[int, int]],
    notch_limit: int,
) -> bool:
    for r, c in cells:
        if int(grid[r, c]) != _GEN_BG or bool(no_orange[r, c]):
            return False
    rows = {r for r, _c in cells}
    old_vals = [int(grid[r, c]) for r, c in cells]
    for r, c in cells:
        grid[r, c] = _GEN_ORANGE
    ok = _gen_rows_run_ok(grid, rows) and _gen_count_notches(grid) <= notch_limit
    for (r, c), old in zip(cells, old_vals):
        grid[r, c] = old
    return ok


def _gen_add_filler(
    grid: np.ndarray,
    expected: np.ndarray,
    no_orange: np.ndarray,
    target_count: int,
    notch_limit: int,
) -> bool:
    H, W = grid.shape
    current = int(np.count_nonzero(grid == _GEN_ORANGE))
    attempts = 0
    while current < target_count and attempts < _GEN_FILLER_ATTEMPTS:
        attempts += 1
        remaining = target_count - current
        run_len = random.randint(_GEN_MIN_ORANGE_RUN, min(_GEN_MAX_ORANGE_RUN, remaining, W))
        r = random.randrange(H)
        c0 = random.randrange(W - run_len + 1)
        cells = [(r, c) for c in range(c0, c0 + run_len)]
        if not _gen_can_add_orange(grid, no_orange, cells, notch_limit):
            continue
        for r2, c2 in cells:
            grid[r2, c2] = _GEN_ORANGE
            expected[r2, c2] = _GEN_ORANGE
        current += run_len

    if current == target_count:
        return True

    free_cells = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(grid[r, c]) == _GEN_BG and not bool(no_orange[r, c])
    ]
    random.shuffle(free_cells)
    for r, c in free_cells:
        if current >= target_count:
            break
        cells = [(r, c)]
        if not _gen_can_add_orange(grid, no_orange, cells, notch_limit):
            continue
        grid[r, c] = _GEN_ORANGE
        expected[r, c] = _GEN_ORANGE
        current += 1
    return current == target_count


def _gen_shape_constraints_ok(grid: np.ndarray, expected_notches: int) -> bool:
    return _gen_count_notches(grid) == expected_notches and _gen_rows_run_ok(grid)


def _gen_choose_obstacle_target(grid: np.ndarray) -> int | None:
    area = grid.shape[0] * grid.shape[1]
    min_obstacles = int(np.ceil(area * _GEN_MIN_OBSTACLE_DENSITY))
    max_obstacles = int(np.floor(area * _GEN_MAX_OBSTACLE_DENSITY))
    existing = int(np.count_nonzero(grid == _GEN_ORANGE))
    if existing > max_obstacles:
        return None
    return random.randint(max(min_obstacles, existing), max_obstacles)


def _gen_matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(_GEN_MIN_MARKERS, _GEN_MAX_MARKERS)
    H, W, lanes = _gen_choose_canvas(marker_count)
    if lanes is None:
        return None

    plan_result = _gen_choose_route_plans(H, W, lanes)
    if plan_result is None:
        return None
    plans, actual_notches = plan_result

    render_result = _gen_render_routes(H, W, plans)
    if render_result is None:
        return None
    grid, expected, no_orange = render_result

    if not _gen_shape_constraints_ok(grid, actual_notches):
        return None

    target_obstacles = _gen_choose_obstacle_target(grid)
    if target_obstacles is None:
        return None
    if not _gen_add_filler(grid, expected, no_orange, target_obstacles, actual_notches):
        return None

    if not _gen_shape_constraints_ok(grid, actual_notches):
        return None
    if not _gen_matches_solver(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 891232d6 input")
