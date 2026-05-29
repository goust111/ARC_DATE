# Combo variant of cb2d8a2c (index 22)
# combo_spec: D4/2+D18/1
# Name: 路径颜色随障碍变化+障碍颜色编码转弯方向

# Design analysis for the combo variant
# ====================================
# a) Concrete behavior and trace:
#    - Original cb2d8a2c used wall-attached mixed red/blue bars. Blue-count widened a keep-out
#      zone, and a Dijkstra route stayed green everywhere.
#    - That geometry cannot cleanly support "bar color chooses left/right" because a wall-attached
#      obstacle already forces the bypass side. The fusion therefore keeps the start-to-opposite-edge
#      path idea but switches to floating monochrome bars, as in D18/1.
#    - Canonical top-start example:
#      start column = 8. A red bar spans cols 6..10 on row 4, so the path goes down to row 3,
#      then detours left to col 5 and steps below the bar. In the original task this local turn
#      would be forced by geometry and would stay green. In the fusion, red means "turn left" and
#      the detour row plus the following downward segment become red. If the next bar is blue and
#      spans cols 3..9 on row 8, the path reaches row 7 in red, then switches to blue while
#      detouring right to col 10 and keeps that blue until the next obstacle or the exit edge.
#    - The key visible difference is therefore twofold on the same objects:
#      input side: obstacle color is the steering instruction;
#      output side: the same color is copied onto the path segment created by that obstacle.
#
# b) Complexity calibration:
#    - Only two obstacle/path colors are used for the fused rule: red(2) and blue(1).
#      This keeps the mapping learnable and avoids arbitrary per-instance palettes.
#    - Samples use 3-4 bars and always include both colors, so each example repeats both
#      "red -> left detour + red path" and "blue -> right detour + blue path".
#    - The path stays green only before the first obstacle. After that, each obstacle contributes
#      one long color run, which is visually stronger than coloring only a single bend cell.
#
# c) Hidden-assumption audit:
#    - The original `blue_count` keep-out expansion is removed entirely; monochrome obstacle color
#      now carries the rule. Any solver logic depending on internal blue markers is invalid.
#    - The original `out[out == BLUE] = RED` post-process would destroy both the steering signal and
#      the path-record signal, so obstacle colors must be preserved.
#    - Global shortest-path search is no longer appropriate because the route is not chosen by
#      optimization; it is determined bar-by-bar from color instructions after canonicalization.
#    - The new painter cannot validate path cells using the output grid alone, because red/blue
#      also appear on obstacles. It must check legality against the immutable input grid.
#
# d) Placement legality:
#    - Every bar is a 1-cell-thick horizontal segment in canonical orientation, with at least one
#      empty cell on both sides, so both detour targets exist in-bounds.
#    - Each bar row is at least two rows below the previous path row, leaving one clear row for
#      the horizontal sidestep.
#    - The generator chooses each new bar so that the current path column falls inside the bar span;
#      this guarantees the detour rule actually triggers.
#    - After choosing a bar color, the generator also constrains the resulting side-step distance,
#      preventing degenerate zero-shift or nearly invisible turns.
#
# e) Visual observability:
#    - A human can compare otherwise similar red and blue bars in one example and see that they
#      push the path to opposite sides.
#    - The output then repeats the same color on the corresponding post-bar path segment, making
#      the dual role of color inferable from only a few examples.
#    - Because both colors appear in every sample and at least one color repeats, the rule is not
#      reliant on a one-off case.
#
# f) Visual cleanliness:
#    - Obstacles stay on their own rows; horizontal path segments live on the empty row immediately
#      above a bar, so path and obstacle never overlap.
#    - Reusing the obstacle colors on the path is acceptable here because the path cells are always
#      adjacent to, not on top of, the bars. The geometry still separates the two roles cleanly.
#
# g) Change inferability:
#    - There is no hidden numeric color arithmetic. Red means left and paints red; blue means right
#      and paints blue. The same palette entry has the same meaning in input and output.
#    - The green prefix segment provides a visible "before any obstacle" baseline, making the later
#      color adoption easier to spot.
#
# New implicit assumptions introduced by `main()` and enforced by `generate_input()`:
#    - After canonicalization, every obstacle row contains exactly one contiguous monochrome bar.
#    - Bars are processed strictly from top to bottom.
#    - The current path column intersects every bar before that bar is handled.
#    - The detour target column (left gap for red, right gap for blue) is background on both the
#      row above the bar and the bar row itself.
#    - The final path segment from the last bar to the opposite edge remains obstacle-colored,
#      recording the most recently bypassed bar.

import random

import numpy as np


BG = 8
BLUE = 1
RED = 2
GREEN = 3
OBSTACLE_COLORS = (BLUE, RED)
PATH_COLORS = {GREEN, BLUE, RED}

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
SHIFT_LO, SHIFT_HI = 2, 6
MAX_GEN_RETRIES = 400


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


def _edge_of_start(grid: np.ndarray) -> str:
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
    raise AssertionError("green start must be on a border edge")


def _to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _extract_canonical_bars(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    bars: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = [c for c in range(w) if int(grid[r, c]) in OBSTACLE_COLORS]
        if not cols:
            continue
        c0, c1 = cols[0], cols[-1]
        assert cols == list(range(c0, c1 + 1)), "each obstacle row must be one contiguous bar"
        color = int(grid[r, c0])
        assert color in OBSTACLE_COLORS, "bar color must be red or blue"
        assert np.all(grid[r, c0 : c1 + 1] == color), "each bar must be monochrome"
        assert 0 < c0 and c1 < w - 1, "bars must leave a gap on both sides"
        bars.append((r, c0, c1, color))
    assert bars, "at least one obstacle bar is required"
    return bars


def _paint_line(
    out: np.ndarray,
    base: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
) -> None:
    assert color in PATH_COLORS, "path color must stay within the fused palette"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            assert int(base[r0, c]) in (BG, GREEN), "path row must stay on non-obstacle cells"
            out[r0, c] = color
        return

    assert c0 == c1, "path segments are axis-aligned"
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        assert int(base[r, c0]) in (BG, GREEN), "path column must stay on non-obstacle cells"
        out[r, c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a border-started path around floating obstacle bars, using each bar's color both as the turn instruction and as the path's next color.

    Input structure:
    - Background is azure (8) with exactly one green (3) start pixel on a border edge.
    - Obstacles are monochrome red (2) or blue (1) bars. In canonical orientation they are horizontal, 1-cell thick, and leave empty space on both sides.

    Rule:
    - Reorient the grid so the start sits on the top edge and process the bars from top to bottom.
    - When the current path column meets a red bar, detour on the clear row above it to the left gap; when it meets a blue bar, detour to the right gap.
    - The path stays green only until the first obstacle. As soon as a bar is bypassed, the local turn and the whole following segment take that bar's color, and obstacle colors themselves remain unchanged.

    Key insight: Each obstacle color has a dual role that must be read consistently in both grids — it tells the path which side to use in the input and then marks the resulting route segment in the output.
    """
    g = np.array(grid, copy=True)
    edge = _edge_of_start(g)
    cg = _to_canonical(g, edge)

    h, _w = cg.shape
    sr, sc = _find_start(cg)
    assert sr == 0, "canonical start must be on the top border"
    bars = _extract_canonical_bars(cg)

    out = cg.copy()
    cur_r, cur_c = sr, sc
    cur_color = GREEN

    for bar_r, c0, c1, color in bars:
        assert bar_r >= cur_r + 2, "each bar must have one clear sidestep row above it"
        assert c0 <= cur_c <= c1, "current path column must intersect every bar"

        _paint_line(out, cg, cur_r, cur_c, bar_r - 1, cur_c, cur_color)

        next_c = c0 - 1 if color == RED else c1 + 1
        _paint_line(out, cg, bar_r - 1, cur_c, bar_r - 1, next_c, color)
        _paint_line(out, cg, bar_r - 1, next_c, bar_r, next_c, color)

        cur_r, cur_c = bar_r, next_c
        cur_color = color

    assert cur_r < h - 1, "path must still have room to reach the opposite edge"
    _paint_line(out, cg, cur_r, cur_c, h - 1, cur_c, cur_color)
    return _from_canonical(out, edge)


def _min_main_len(count: int) -> int:
    return 2 * count + 2


def _choose_bar_colors(count: int) -> list[int]:
    colors = [RED, BLUE] + [random.choice(OBSTACLE_COLORS) for _ in range(count - 2)]
    random.shuffle(colors)
    return colors


def _choose_bar_rows(main_size: int, count: int) -> list[int]:
    slack = main_size - _min_main_len(count)
    assert slack >= 0
    extras = _random_composition(slack, count + 1)
    gaps = [1 + extra for extra in extras]

    rows: list[int] = []
    row = 1 + gaps[0]
    rows.append(row)
    for i in range(1, count):
        row += 1 + gaps[i]
        rows.append(row)

    assert rows[-1] + 1 + gaps[-1] == main_size
    return rows


def _length_bounds(cross_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100, cross_size - 2)
    if lo > hi:
        return None
    return lo, hi


def _bar_choices(cross_size: int, current_c: int, color: int) -> list[tuple[int, int, int]]:
    bounds = _length_bounds(cross_size)
    if bounds is None:
        return []

    lo, hi = bounds
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        for c0 in range(1, cross_size - length):
            c1 = c0 + length - 1
            if not (c0 <= current_c <= c1):
                continue
            next_c = c0 - 1 if color == RED else c1 + 1
            shift = abs(next_c - current_c)
            if SHIFT_LO <= shift <= SHIFT_HI:
                choices.append((c0, c1, next_c))
    return choices


def _canonical_grid(main_size: int, cross_size: int, colors: list[int]) -> np.ndarray | None:
    if _length_bounds(cross_size) is None:
        return None

    rows = _choose_bar_rows(main_size, len(colors))
    start_choices = list(range(2, cross_size - 2))
    if not start_choices:
        return None

    for _ in range(120):
        start_c = random.choice(start_choices)
        current_c = start_c
        spans: list[tuple[int, int, int]] = []
        ok = True
        for color in colors:
            choices = _bar_choices(cross_size, current_c, color)
            if not choices:
                ok = False
                break
            c0, c1, next_c = random.choice(choices)
            spans.append((c0, c1, color))
            current_c = next_c
        if not ok:
            continue

        grid = np.full((main_size, cross_size), BG, dtype=int)
        grid[0, start_c] = GREEN
        for row, (c0, c1, color) in zip(rows, spans, strict=True):
            grid[row, c0 : c1 + 1] = color
        return grid

    return None


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    return _from_canonical(grid, edge)


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(np.isin(out[-1, :], list(PATH_COLORS))))
    if edge == "bottom":
        return bool(np.any(np.isin(out[0, :], list(PATH_COLORS))))
    if edge == "left":
        return bool(np.any(np.isin(out[:, -1], list(PATH_COLORS))))
    return bool(np.any(np.isin(out[:, 0], list(PATH_COLORS))))


def _path_colors_used(inp: np.ndarray, out: np.ndarray) -> set[int]:
    changed = inp != out
    return {int(out[r, c]) for r, c in zip(*np.where(changed), strict=False)}


def _has_visible_variant_signal(inp: np.ndarray, out: np.ndarray) -> bool:
    used = _path_colors_used(inp, out)
    return GREEN in used and BLUE in used and RED in used


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
        if not _has_visible_variant_signal(inp, out):
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
