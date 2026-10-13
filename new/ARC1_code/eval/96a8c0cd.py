import numpy as np

from arc_utils import Color, find_components, get_bg, in_bounds, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the red trace from its border marker, detouring around each blue/green bar it hits.

    Input structure
    - The grid has a background, one red marker on the border, and blue/green
      one-cell-thick bars perpendicular to the marker's inward direction.

    Rule
    - The red marker travels straight inward from its border.
    - When the trace hits a blue bar, it passes around the bar on the left side
      relative to its direction of travel.
    - When the trace hits a green bar, it passes around the bar on the right
      side relative to its direction of travel.
    - Existing bars are preserved; only background cells on the trace become red.

    Key insight
    - The trace keeps the same overall forward direction; each hit bar only
      shifts the current lane by forcing a detour one cell before the obstacle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    red = Color.RED
    blue = Color.BLUE
    green = Color.GREEN
    assert set(map(int, np.unique(g))) <= {bg, red, blue, green}

    starts = np.argwhere(g == red)
    assert starts.shape == (1, 2)
    r, c = map(int, starts[0])
    if r == 0:
        dr, dc = 1, 0
    elif r == h - 1:
        dr, dc = -1, 0
    elif c == 0:
        dr, dc = 0, 1
    elif c == w - 1:
        dr, dc = 0, -1
    else:
        raise AssertionError("red marker must be on the border")

    components = find_components(g, bg=bg, connectivity=4, colors=[blue, green])
    component_at = {}
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if dr:
            assert r0 == r1
        else:
            assert c0 == c1
        for rr, cc in comp["coords"]:
            key = (int(rr), int(cc))
            assert key not in component_at
            component_at[key] = comp

    out = g.copy()

    def paint_cell(rr: int, cc: int) -> None:
        assert in_bounds(g.shape, rr, cc)
        assert int(g[rr, cc]) in (bg, red)
        out[rr, cc] = red

    def paint_line(a: tuple[int, int], b: tuple[int, int]) -> None:
        ar, ac = a
        br, bc = b
        if ar == br:
            step = 1 if bc >= ac else -1
            for cc in range(ac, bc + step, step):
                paint_cell(ar, cc)
        elif ac == bc:
            step = 1 if br >= ar else -1
            for rr in range(ar, br + step, step):
                paint_cell(rr, ac)
        else:
            raise AssertionError("trace segments must be horizontal or vertical")

    paint_cell(r, c)
    while True:
        rr, cc = r + dr, c + dc
        hit = None
        while in_bounds(g.shape, rr, cc):
            val = int(g[rr, cc])
            if val in (blue, green):
                hit = (rr, cc)
                break
            assert val in (bg, red)
            rr += dr
            cc += dc

        if hit is None:
            last = (rr - dr, cc - dc)
            if (r, c) != last:
                paint_line((r + dr, c + dc), last)
            break

        hr, hc = hit
        turn_start = (hr - dr, hc - dc)
        if (r, c) != turn_start:
            paint_line((r + dr, c + dc), turn_start)

        comp = component_at[hit]
        color = int(comp["color"])
        normal = (-dc, dr) if color == blue else (dc, -dr)
        r0, c0, r1, c1 = comp["bbox"]
        if normal[0] < 0:
            side = (r0 - 1, hc)
        elif normal[0] > 0:
            side = (r1 + 1, hc)
        elif normal[1] < 0:
            side = (hr, c0 - 1)
        else:
            side = (hr, c1 + 1)
        assert in_bounds(g.shape, side[0], side[1])
        assert int(g[side]) == bg

        turn_end = (side[0] - dr, side[1] - dc)
        assert in_bounds(g.shape, turn_end[0], turn_end[1])
        paint_line(turn_start, turn_end)
        paint_cell(side[0], side[1])
        r, c = side

    return out

# --------------- generator (o5x) ---------------

# Canvas and travel bounds.
MIN_ROWS, MAX_ROWS_EXCL = 12, 23
MIN_COLS, MAX_COLS_EXCL = 12, 30
MIN_AXIS_LEN, ORIENT_DOWN, ORIENT_RIGHT = 12, 0, 1

# Path and bar structure.
MIN_HITS, MAX_HITS_EXCL = 4, 9
MIN_HIT_LEVEL, MAX_FIRST_HIT_LEVEL = 2, 4
MIN_HIT_GAP, MAX_HIT_GAP = 2, 4
MAX_BARS_PER_LEVEL, BAR_GAP_CELLS = 3, 2
MAX_TOTAL_BARS, MAX_SCENERY_BARS = 14, 8
HIT_BAR_LENGTH_CHOICES = (2, 3, 3)
SCENERY_BAR_LENGTH_CHOICES = (2, 3)
SHIFT_DIRECTIONS = (-1, 1)

# Colors and retry policy.
BG, RED, BLUE, GREEN = Color.BLACK, Color.RED, Color.BLUE, Color.GREEN
BAR_COLORS, BAR_COLOR_SET = [BLUE, GREEN], {BLUE, GREEN}
MAX_ATTEMPTS = 300


def _gen_size(orient: int, n_hits: int) -> tuple[int, int] | None:
    min_axis = max(MIN_AXIS_LEN, MIN_HIT_GAP * n_hits + MIN_HIT_LEVEL)
    if orient == ORIENT_DOWN:
        if min_axis >= MAX_ROWS_EXCL:
            return None
        h = int(np.random.randint(min_axis, MAX_ROWS_EXCL))
        w_choices = [w for w in range(MIN_COLS, MAX_COLS_EXCL) if w != h]
        w = int(np.random.choice(w_choices))
    else:
        if min_axis >= MAX_COLS_EXCL:
            return None
        w = int(np.random.randint(min_axis, MAX_COLS_EXCL))
        h_choices = [h for h in range(MIN_ROWS, MAX_ROWS_EXCL) if h != w]
        h = int(np.random.choice(h_choices))
    return h, w


def _gen_levels(axis_len: int, n_hits: int) -> list[int] | None:
    max_last = axis_len - MIN_HIT_LEVEL
    latest_first = max_last - MIN_HIT_GAP * (n_hits - 1)
    if latest_first < MIN_HIT_LEVEL:
        return None
    first = int(np.random.randint(MIN_HIT_LEVEL, min(MAX_FIRST_HIT_LEVEL, latest_first) + 1))
    levels = [first]
    cur = first
    for i in range(1, n_hits):
        remaining = n_hits - 1 - i
        max_gap = min(MAX_HIT_GAP, max_last - cur - MIN_HIT_GAP * remaining)
        if max_gap < MIN_HIT_GAP:
            return None
        cur += int(np.random.randint(MIN_HIT_GAP, max_gap + 1))
        levels.append(cur)
    return levels


def _gen_color_for_shift(orient: int, sign: int) -> int:
    return (BLUE if sign > 0 else GREEN) if orient == ORIENT_DOWN else (GREEN if sign > 0 else BLUE)


def _gen_shift_for_color(orient: int, color: int) -> int:
    return (1 if color == BLUE else -1) if orient == ORIENT_DOWN else (-1 if color == BLUE else 1)


def _gen_bar_starts(lane: int, lat_len: int, sign: int, length: int) -> list[int]:
    if sign > 0:
        lo = max(0, lane - length + 1)
        hi = min(lane, lat_len - length - 1)
    else:
        lo = max(1, lane - length + 1)
        hi = min(lane, lat_len - length)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _gen_place_bar(grid: np.ndarray, orient: int, axis: int, start: int, length: int, color: int) -> None:
    patch = np.full((1, length), color, dtype=int)
    if orient == ORIENT_DOWN:
        paste(grid, patch, axis, start)
    else:
        paste(grid, patch.T, start, axis)


def _gen_range_ok(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    return all(end + BAR_GAP_CELLS < a or b + BAR_GAP_CELLS < start for a, b in ranges)


def _make_base_grid(orient: int, n_hits: int) -> tuple[np.ndarray, list[int], int, int] | None:
    size = _gen_size(orient, n_hits)
    if size is None:
        return None
    h, w = size
    axis_len = h if orient == ORIENT_DOWN else w
    lat_len = w if orient == ORIENT_DOWN else h
    levels = _gen_levels(axis_len, n_hits)
    if levels is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    lane = int(np.random.randint(1, lat_len - 1))
    if orient == ORIENT_DOWN:
        grid[0, lane] = RED
    else:
        grid[lane, 0] = RED
    return grid, levels, lat_len, lane


def _hit_bar_choices(lane: int, lat_len: int) -> list[tuple[int, int, list[int]]]:
    choices = []
    for length in HIT_BAR_LENGTH_CHOICES:
        for sign in SHIFT_DIRECTIONS:
            starts = _gen_bar_starts(lane, lat_len, sign, length)
            if starts:
                choices.append((length, sign, starts))
    return choices


def _gen_hit_bars(
    grid: np.ndarray,
    orient: int,
    levels: list[int],
    lat_len: int,
    lane: int,
) -> dict[int, list[tuple[int, int]]] | None:
    ranges_by_level = {level: [] for level in levels}
    used_colors: set[int] = set()
    for i, level in enumerate(levels):
        choices = _hit_bar_choices(lane, lat_len)
        if i == len(levels) - 1 and len(used_colors) == 1:
            missing = BLUE if BLUE not in used_colors else GREEN
            forced_sign = _gen_shift_for_color(orient, missing)
            choices = [ch for ch in choices if ch[1] == forced_sign]
        if not choices:
            return None

        length, sign, starts = choices[int(np.random.randint(0, len(choices)))]
        start = int(np.random.choice(starts))
        color = _gen_color_for_shift(orient, sign)
        _gen_place_bar(grid, orient, level, start, length, color)
        ranges_by_level[level].append((start, start + length - 1))
        used_colors.add(color)
        lane = start + length if sign > 0 else start - 1

    if used_colors != BAR_COLOR_SET:
        return None
    return ranges_by_level


def _gen_add_scenery(
    grid: np.ndarray,
    orient: int,
    levels: list[int],
    lat_len: int,
    ranges_by_level: dict[int, list[tuple[int, int]]],
    path_mask: np.ndarray,
    target: int,
) -> bool:
    for _ in range(target):
        options = []
        for level in levels:
            if len(ranges_by_level[level]) >= MAX_BARS_PER_LEVEL:
                continue
            for length in SCENERY_BAR_LENGTH_CHOICES:
                for start in range(0, lat_len - length + 1):
                    end = start + length - 1
                    if not _gen_range_ok(ranges_by_level[level], start, end):
                        continue
                    if orient == ORIENT_DOWN:
                        cells = [(level, c) for c in range(start, start + length)]
                    else:
                        cells = [(r, level) for r in range(start, start + length)]
                    if any(path_mask[r, c] or grid[r, c] != BG for r, c in cells):
                        continue
                    options.append((level, start, length))
        if not options:
            return False
        level, start, length = options[int(np.random.randint(0, len(options)))]
        color = int(np.random.choice(BAR_COLORS))
        _gen_place_bar(grid, orient, level, start, length, color)
        ranges_by_level[level].append((start, start + length - 1))
    return True


def _gen_attempt() -> np.ndarray | None:
    orient = int(np.random.randint(ORIENT_DOWN, ORIENT_RIGHT + 1))
    n_hits = int(np.random.randint(MIN_HITS, MAX_HITS_EXCL))
    base = _make_base_grid(orient, n_hits)
    if base is None:
        return None
    grid, levels, lat_len, lane = base

    ranges_by_level = _gen_hit_bars(grid, orient, levels, lat_len, lane)
    if ranges_by_level is None:
        return None

    try:
        base_out = main(grid)
    except Exception:
        return None
    path_mask = (base_out == RED) & (grid == BG)

    scenery_target = int(np.random.randint(0, min(MAX_SCENERY_BARS, MAX_TOTAL_BARS - n_hits) + 1))
    final = grid.copy()
    final_ranges = {k: v.copy() for k, v in ranges_by_level.items()}
    if not _gen_add_scenery(final, orient, levels, lat_len, final_ranges, path_mask, scenery_target):
        return None

    expected = final.copy()
    expected[path_mask] = RED
    try:
        if not np.array_equal(main(final), expected):
            return None
    except Exception:
        return None
    return final


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate 96a8c0cd input")
