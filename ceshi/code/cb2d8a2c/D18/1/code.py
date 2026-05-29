# Variant of cb2d8a2c
# Dimension: D18 — 背景与颜色方案 (meta)
# Change: 背景为浅灰色(8)，路径为绿色(3)，障碍条为红色(2)和蓝色(1)三色方案... → 障碍条颜色编码转弯方向

import random

import numpy as np


# Design analysis
# a) Concrete behavior and trace:
#    - Original bars were wall-attached and mixed blue/red inside the same component; blue-count widened a keep-out zone.
#    - That geometry cannot support "color chooses left/right" because a wall-attached bar already forces a unique bypass side.
#    - The minimal viable change is therefore: make each obstacle a floating 1-cell-thick bar with a gap on both sides.
#      The path still starts on one border and reaches the opposite border, but when it meets a bar:
#      red(2) means detour to the bar's left side, blue(1) means detour to the bar's right side.
#    - Small trace in canonical top-start orientation:
#      old: start column 7 meets a bar touching the left wall, so geometry alone forces a right bypass.
#      new: a floating bar spans cols 4..9, current column is 7. If the bar is red, the path goes left to col 3;
#      if the same bar is recolored blue, the path instead goes right to col 10. The color now changes the output.
# b) Complexity calibration:
#    - Only two obstacle colors are used, matching the requested meta change and keeping the rule learnable.
#    - Each sample uses 3-4 bars and always includes both colors, so the mapping is repeated inside a single example.
# c) Implicit-assumption audit against the old main():
#    - Old `bar_mask = (g == BLUE) | (g == RED)` is still valid, but `blue_count` as radius no longer is.
#    - Old BFS-forbidden expansion depended on blue pixels being annotations inside a bar; that assumption is removed.
#    - Old `out[out == BLUE] = RED` would destroy the new turn-encoding, so the new solver must preserve obstacle colors.
#    - Old Dijkstra searched any shortest safe route; the new rule is deterministic per bar color, so solver logic changes
#      from global pathfinding to ordered bar-by-bar simulation after canonicalizing orientation.
# d) Placement legality:
#    - Generator keeps bars on separate rows, with at least one clear row before each horizontal sidestep.
#    - Every bar has a left gap and a right gap, stays in-bounds, and covers the current path column so the detour triggers.
#    - The chosen detour column after one bar is used when placing the next bar, guaranteeing reachability.
# e) Visual observability:
#    - Swapping red/blue on the same floating bar would change the side used by the green path, which is directly visible.
#    - Because both colors recur in each grid, a human can infer "same shape, different color => opposite detour".
# f) Visual cleanliness:
#    - Bars occupy single rows only, path horizontal segments occur on the clear row immediately above a bar, and overlaps are
#      avoided by construction. The output remains sparse and easy to parse.
# g) Color-change inference:
#    - No hidden color-id arithmetic is used. The only semantic color distinction is red vs blue on otherwise identical bars.

BG = 8
BLUE = 1
RED = 2
GREEN = 3

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
    return np.fliplr(grid).T.copy()  # edge == "right"


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()  # edge == "right"


def _extract_canonical_bars(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    bars: list[tuple[int, int, int, int]] = []
    obstacle_colors = {BLUE, RED}

    for r in range(h):
        cols = [c for c in range(w) if int(grid[r, c]) in obstacle_colors]
        if not cols:
            continue
        c0, c1 = cols[0], cols[-1]
        assert cols == list(range(c0, c1 + 1)), "each bar row must be a contiguous segment"
        color = int(grid[r, c0])
        assert color in obstacle_colors, "bar color must be red or blue"
        assert np.all(grid[r, c0 : c1 + 1] == color), "each bar must be monochrome"
        assert 0 < c0 and c1 < w - 1, "bars must float away from both side walls"
        bars.append((r, c0, c1, color))

    assert bars, "at least one obstacle bar is required"
    return bars


def _paint_line(out: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            assert out[r0, c] in (BG, GREEN), "path row must stay on background"
            out[r0, c] = GREEN
        return

    assert c0 == c1, "path segments are axis-aligned"
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        assert out[r, c0] in (BG, GREEN), "path column must stay on background"
        out[r, c0] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path to the opposite border, with each obstacle bar's color selecting the detour side.

    Input structure:
    - Background is azure (8) with exactly one green (3) start pixel on a border edge.
    - Obstacles are monochrome red (2) or blue (1) bars. In canonical orientation they are horizontal and leave gaps on both sides.

    Rule:
    - Reorient the grid so the start is on the top edge and the path travels downward.
    - Process bars from top to bottom. When the current column meets a red bar, sidestep on the row above it to the bar's left gap;
      when it meets a blue bar, sidestep to the right gap. Then step below the bar and continue downward.
    - Paint green only on background cells traversed by the route; obstacle colors remain unchanged.

    Key insight: The bar color is now the steering instruction itself, so two bars with the same shape can force opposite detours.
    """
    g = np.array(grid, copy=True)
    edge = _edge_of_start(g)
    cg = _to_canonical(g, edge)

    h, w = cg.shape
    sr, sc = _find_start(cg)
    assert sr == 0, "canonical start must be on the top border"
    bars = _extract_canonical_bars(cg)

    out = cg.copy()
    cur_r, cur_c = sr, sc

    for bar_r, c0, c1, color in bars:
        assert bar_r >= cur_r + 2, "each bar must have at least one clear row above it"
        assert c0 <= cur_c <= c1, "current path column must meet every bar"

        _paint_line(out, cur_r, cur_c, bar_r - 1, cur_c)

        if color == RED:
            next_c = c0 - 1
        else:
            next_c = c1 + 1

        _paint_line(out, bar_r - 1, cur_c, bar_r - 1, next_c)
        _paint_line(out, bar_r - 1, next_c, bar_r, next_c)
        cur_r, cur_c = bar_r, next_c

    assert cur_r < h - 1, "path must still have room to reach the opposite edge"
    _paint_line(out, cur_r, cur_c, h - 1, cur_c)
    return _from_canonical(out, edge)


def _min_main_len(count: int) -> int:
    return 2 * count + 2


def _choose_bar_colors(count: int) -> list[int]:
    colors = [RED, BLUE] + [random.choice((RED, BLUE)) for _ in range(count - 2)]
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

    if choices:
        return choices

    for length in range(lo, hi + 1):
        for c0 in range(1, cross_size - length):
            c1 = c0 + length - 1
            if not (c0 <= current_c <= c1):
                continue
            next_c = c0 - 1 if color == RED else c1 + 1
            if next_c != current_c:
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
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D18 input")
