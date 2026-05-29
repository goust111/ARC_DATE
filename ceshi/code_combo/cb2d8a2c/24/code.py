# Combo variant of cb2d8a2c (index 24)
# combo_spec: D2/0+D18/1
# Name: 固定转弯方向+障碍颜色编码覆盖
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Original cb2d8a2c used wall-attached mixed-color bars. Their blue-pixel count
#      controlled a keep-out radius, and the path found whichever legal bypass geometry
#      the wall placement left open.
#    - D2/0 requires a task-wide fixed handedness, but wall-attached bars would make that
#      invisible because one side is already forced. The bars therefore need to float so
#      both detour sides are genuinely available.
#    - D18/1 adds a second layer: there is still a default turn bias, but one obstacle
#      color can override it. In canonical top-start orientation used below, the default
#      is clockwise, which means the path first sidesteps to the bar's left gap; blue bars
#      override that and force the mirror-image right sidestep instead.
#    - Small trace: suppose the start is at row 0, col 7, and a red bar spans cols 5..10
#      on row 4. The path goes down to row 3, turns left to col 4, drops to row 5, then
#      returns to col 7. If the same bar is recolored blue, the default is overridden and
#      the path instead uses col 11 before returning to col 7.
#    - Difference from the original: detour side is no longer inferred from wall geometry
#      or shortest-safe routing. Difference from plain D18: red bars do not define an
#      independent new rule; they expose the global default, while blue bars visibly flip it.
# b) Complexity calibration:
#    - Each sample uses 3-4 bars so the default clockwise behavior repeats multiple times.
#    - Every sample also includes at least one blue override bar, so the second layer is
#      exercised inside the same grid rather than across rare examples.
#    - Bars keep the original task's overall size regime (12-29 by 13-26 grids; 6-15 cell
#      bars spanning roughly one-third to four-fifths of the cross-axis) so the scene still
#      looks like the source task rather than a different puzzle family.
# c) Hidden-assumption audit:
#    - The old Dijkstra solver assumed blue cells inside a bar encoded radius and that the
#      path objective could recover the intended route. Neither survives here: bars are now
#      monochrome steering tokens, and the route must follow explicit per-bar turn rules.
#    - The old `out[out == BLUE] = RED` post-processing would erase the new override signal,
#      so the fused solver must preserve obstacle colors.
#    - The new solver assumes canonical bars are single-row contiguous floating segments,
#      sorted by row, each intersecting the shared main lane. The generator enforces all of
#      those assumptions directly.
# d) Placement legality:
#    - Bars are placed on distinct rows with at least one clear row above and below every
#      bar, so the three detour segments (top horizontal, side vertical, bottom horizontal)
#      never collide with other bars.
#    - Each bar leaves gaps on both sides and always covers the lane column, guaranteeing
#      that both the default and override detours are physically possible and observable.
#    - Rotations only move between the four start-edge orientations; because construction is
#      done in a canonical top-start frame first, legality is preserved after reorientation.
# e) Visual observability:
#    - Red bars repeatedly produce same-handed bulges, establishing the default bias.
#    - Blue bars produce the opposite bulge while keeping the same shape family, making the
#      override rule directly legible from color.
#    - The generator biases bars so the chosen detour side is often *not* the shorter side,
#      preventing a misleading "path just takes the nearest gap" explanation.
# f) Visual cleanliness:
#    - Obstacles occupy single rows only; the path uses the immediately adjacent clear rows.
#      That keeps every elbow readable and avoids stacked overlaps.
#    - The lane returns to the same central column after every bar, so the output stays sparse
#      and regular instead of drifting into a tangled zig-zag.
# g) Change inferability:
#    - Only two obstacle colors are used. Red means "follow the global default"; blue means
#      "override the default and flip the turn". Because both recur in each sample and the
#      bar shapes remain comparable, a human can infer the color-conditioned exception from
#      a small set of training pairs.

import random

import numpy as np


BG = 8
BLUE = 1
RED = 2
GREEN = 3

DEFAULT_TURN = "CW"
OVERRIDE_COLOR = BLUE

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
SHIFT_LO, SHIFT_HI = 2, 6
MAX_GEN_RETRIES = 600


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    return int(pts[0, 0]), int(pts[0, 1])


def _start_edge(grid: np.ndarray) -> str:
    r, c = _find_start(grid)
    h, w = grid.shape
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("green start must lie on a border")


def _to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, 1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, -1).copy()  # edge == "left"


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, -1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, 1).copy()  # edge == "left"


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    obstacle_colors = {RED, BLUE}
    for r in range(h):
        cols = [c for c in range(w) if int(grid[r, c]) in obstacle_colors]
        if not cols:
            continue
        c0, c1 = cols[0], cols[-1]
        assert cols == list(range(c0, c1 + 1)), "each bar row must be a single contiguous segment"
        color = int(grid[r, c0])
        assert np.all(grid[r, c0 : c1 + 1] == color), "each bar must be monochrome"
        assert 0 < c0 and c1 < w - 1, "bars must float away from both side walls"
        bars.append({"row": r, "c0": c0, "c1": c1, "color": color})
    assert bars, "at least one bar is required"
    return bars


def _paint_segment(out: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    assert r0 == r1 or c0 == c1, "path segments must be axis-aligned"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        coords = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        coords = [(r, c0) for r in range(r0, r1 + step, step)]
    for r, c in coords:
        assert out[r, c] in (BG, GREEN), "path must stay off obstacle cells"
        out[r, c] = GREEN


def _turn_for_bar(color: int, use_color_override: bool) -> str:
    if use_color_override and color == OVERRIDE_COLOR:
        return "CCW" if DEFAULT_TURN == "CW" else "CW"
    return DEFAULT_TURN


def _detour_candidates(bar: dict[str, int], width: int, use_color_override: bool) -> list[int]:
    left_col = bar["c0"] - 1
    right_col = bar["c1"] + 1
    turn = _turn_for_bar(bar["color"], use_color_override)
    preferred = [left_col, right_col] if turn == "CW" else [right_col, left_col]
    return [col for col in preferred if 0 <= col < width]


def _solve_canonical(grid: np.ndarray, *, use_color_override: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sr, sc = _find_start(g)
    assert sr == 0, "canonical solver expects the start on the top border"

    bars = _extract_bars(g)
    out = g.copy()
    lane_col = sc
    cur_r = sr
    cur_c = sc

    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        assert c0 <= lane_col <= c1, "every bar must intersect the lane column"
        assert row >= cur_r + 2, "each bar needs a clear row above and below for the detour"

        top_row = row - 1
        bottom_row = row + 1
        assert bottom_row < h, "bars must leave room below for the route to clear them"

        _paint_segment(out, cur_r, cur_c, top_row, lane_col)

        detour_col = None
        for cand in _detour_candidates(bar, w, use_color_override):
            try:
                probe = out.copy()
                _paint_segment(probe, top_row, lane_col, top_row, cand)
                _paint_segment(probe, top_row, cand, bottom_row, cand)
                _paint_segment(probe, bottom_row, cand, bottom_row, lane_col)
                detour_col = cand
                break
            except AssertionError:
                continue
        assert detour_col is not None, "generator must leave a valid detour on the required side"

        _paint_segment(out, top_row, lane_col, top_row, detour_col)
        _paint_segment(out, top_row, detour_col, bottom_row, detour_col)
        _paint_segment(out, bottom_row, detour_col, bottom_row, lane_col)
        cur_r, cur_c = bottom_row, lane_col

    _paint_segment(out, cur_r, cur_c, h - 1, lane_col)
    return out


def _solve_with_default_only(grid: np.ndarray) -> np.ndarray:
    edge = _start_edge(grid)
    canonical = _to_canonical(grid, edge)
    solved = _solve_canonical(canonical, use_color_override=False)
    return _from_canonical(solved, edge)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    edge = _start_edge(grid)
    canonical = _to_canonical(grid, edge)
    bars = _extract_bars(canonical)
    default_bars = sum(1 for bar in bars if int(bar["color"]) == RED)
    override_bars = sum(1 for bar in bars if int(bar["color"]) == BLUE)

    nonshort_default = 0
    for bar in bars:
        lane_col = _find_start(canonical)[1]
        left_shift = lane_col - (int(bar["c0"]) - 1)
        right_shift = (int(bar["c1"]) + 1) - lane_col
        if int(bar["color"]) == RED and left_shift > right_shift:
            nonshort_default += 1

    fused = main(grid)
    default_only = _solve_with_default_only(grid)
    override_diff = int(np.count_nonzero((fused == GREEN) != (default_only == GREEN)))
    return {
        "bar_count": len(bars),
        "default_bars": default_bars,
        "override_bars": override_bars,
        "nonshort_default": nonshort_default,
        "override_diff": override_diff,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green border-to-border path with a global turn bias that blue bars can overturn.

    Input structure:
    - The background is azure (8), and there is exactly one green (3) start pixel on a border edge.
    - Obstacles are floating monochrome bars made of red (2) or blue (1), perpendicular to the route in the canonical top-start view.

    Rule:
    - Rotate the grid to the canonical orientation where the start is on the top border and the route travels downward.
    - Process bars from top to bottom. The route normally uses a fixed clockwise detour around each bar, returning to the same main lane after clearing it; blue bars override that default and force the opposite detour direction instead.
    - Paint green only on background cells visited by the route, and preserve every obstacle bar's original color.

    Key insight: Red bars reveal the task-wide default handedness, while blue bars are exceptions that visibly flip the same detour template to the other side.
    """
    g = np.asarray(grid, dtype=int)
    edge = _start_edge(g)
    canonical = _to_canonical(g, edge)
    solved = _solve_canonical(canonical, use_color_override=True)
    return _from_canonical(solved, edge)


def _min_main_len(count: int) -> int:
    return 3 * count + 1


def _choose_bar_colors(count: int) -> list[int]:
    colors = [RED, RED, BLUE]
    while len(colors) < count:
        colors.append(random.choice((RED, BLUE)))
    random.shuffle(colors)
    return colors


def _choose_bar_rows(main_size: int, count: int) -> list[int]:
    slack = main_size - _min_main_len(count)
    assert slack >= 0
    extras = _random_composition(slack, count + 1)

    row = 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += 3 + extras[i]
        rows.append(row)

    assert rows[-1] <= main_size - 2
    return rows


def _length_bounds(cross_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100, cross_size - 2)
    if lo > hi:
        return None
    return lo, hi


def _bar_choices(cross_size: int, lane_col: int, color: int) -> list[tuple[int, int]]:
    bounds = _length_bounds(cross_size)
    if bounds is None:
        return []

    lo, hi = bounds
    strong: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    any_visible: list[tuple[int, int]] = []
    for length in range(lo, hi + 1):
        for c0 in range(1, cross_size - length):
            c1 = c0 + length - 1
            if not (c0 <= lane_col <= c1):
                continue
            left_shift = lane_col - (c0 - 1)
            right_shift = (c1 + 1) - lane_col
            chosen_shift = left_shift if color == RED else right_shift
            other_shift = right_shift if color == RED else left_shift
            if SHIFT_LO <= chosen_shift <= SHIFT_HI:
                if chosen_shift > other_shift:
                    strong.append((c0, c1))
                elif chosen_shift != other_shift:
                    fallback.append((c0, c1))
            if chosen_shift != other_shift:
                any_visible.append((c0, c1))
    if strong:
        return strong
    if fallback:
        return fallback
    return any_visible


def _canonical_grid(main_size: int, cross_size: int, colors: list[int]) -> np.ndarray | None:
    if _length_bounds(cross_size) is None:
        return None

    rows = _choose_bar_rows(main_size, len(colors))
    start_lo = max(2, cross_size // 2 - 2)
    start_hi = min(cross_size - 3, cross_size // 2 + 2)
    start_choices = list(range(start_lo, start_hi + 1))
    if not start_choices:
        return None

    for _ in range(160):
        lane_col = random.choice(start_choices)
        spans: list[tuple[int, int]] = []
        ok = True
        for color in colors:
            choices = _bar_choices(cross_size, lane_col, color)
            if not choices:
                ok = False
                break
            spans.append(random.choice(choices))
        if not ok:
            continue

        grid = np.full((main_size, cross_size), BG, dtype=int)
        grid[0, lane_col] = GREEN
        for row, (c0, c1), color in zip(rows, spans, colors, strict=True):
            grid[row, c0 : c1 + 1] = color
        return grid

    return None


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    return _from_canonical(grid, edge)


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
    main_size = random.randint(max(main_lo, _min_main_len(count)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    colors = _choose_bar_colors(count)

    canonical = _canonical_grid(main_size, cross_size, colors)
    if canonical is None:
        return None

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        metrics = _behavior_metrics(inp)
        if metrics["override_bars"] < 1:
            return None
        if metrics["default_bars"] < 2:
            return None
        if metrics["nonshort_default"] < 1:
            return None
        if metrics["override_diff"] < 8:
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
