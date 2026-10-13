import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PALETTE_COLORS = [3, 4, 5, 6, 7, 8, 9]

MIN_ROWS = 8
MAX_ROWS = 12
MIN_CHAIN_RED = 2
MAX_CHAIN_RED = 4
MIN_BAR_HEIGHT = 2
MAX_BAR_HEIGHT = 5
MAX_GRID_SIDE = 30
PALETTE_STRIDE = 2
MIN_PALETTE_GAP = 2
MAX_PALETTE_GAP = 4
RIGHT_MARGIN_RANGE = (1, 3)
DISTRACTOR_COUNT_CHOICES = [0, 1, 2, 3]
DISTRACTOR_COUNT_WEIGHTS = [1, 3, 3, 2]
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _palette_from_top_row(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    cols = [int(c) for c in np.flatnonzero(grid[0] != bg)]
    assert cols, "expected palette pixels on the top row"
    colors = [int(grid[0, c]) for c in cols]
    assert len(colors) == len(set(colors)), "palette colors must be distinct"
    assert BLUE not in colors and RED not in colors, "palette must not reuse bar colors"
    return cols, colors


def _extract_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    baseline = h - 1
    bars = []
    for c in range(w):
        rows = np.flatnonzero(np.isin(grid[:, c], (BLUE, RED)))
        if rows.size == 0:
            continue
        colors = {int(grid[r, c]) for r in rows}
        assert len(colors) == 1, "each bar column must be monochrome"
        color = colors.pop()
        assert color in (BLUE, RED)
        assert rows[0] > 0, "bars must stay below the palette row"
        assert rows[-1] == baseline, "bars must share the bottom baseline"
        assert np.all(np.diff(rows) == 1), "each bar must be one contiguous vertical segment"
        assert int(grid[0, c]) == BG, "palette pixels must not overlap bar columns"
        assert np.all(grid[1 : rows[0], c] == BG), "space above each bar must be background"
        bars.append(
            {
                "col": int(c),
                "height": int(rows.size),
                "color": int(color),
            }
        )
    bars.sort(key=lambda bar: bar["col"])
    assert bars, "expected at least one vertical bar"
    blue_bars = [bar for bar in bars if bar["color"] == BLUE]
    assert len(blue_bars) == 1, "expected exactly one blue start bar"
    return bars


def _follow_chain(bars: list[dict]) -> list[dict]:
    red_by_col = {bar["col"]: bar for bar in bars if bar["color"] == RED}
    blue = next(bar for bar in bars if bar["color"] == BLUE)
    visited_red_cols = set()
    chain = [blue]
    current = blue
    while True:
        c = current["col"]
        jump = current["height"]
        candidates = []
        for next_col in (c - jump, c + jump):
            candidate = red_by_col.get(next_col)
            if candidate is not None and next_col not in visited_red_cols:
                candidates.append(candidate)
        assert len(candidates) <= 1, "each chain step must have at most one legal next red bar"
        if not candidates:
            break
        nxt = candidates[0]
        visited_red_cols.add(nxt["col"])
        chain.append(nxt)
        current = nxt
    return chain


def _paint_bar(grid: np.ndarray, col: int, height: int, color: int) -> None:
    baseline = grid.shape[0] - 1
    r0 = baseline - height + 1
    grid[r0 : baseline + 1, col] = color


def _expected_output(grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    _, palette = _palette_from_top_row(grid, bg)
    bars = _extract_bars(grid)
    chain = _follow_chain(bars)
    assert len(chain) == len(palette) + 1, "palette length must match the number of chained red bars"

    out = grid.copy()
    out[0, :] = bg
    for i, bar in enumerate(chain[1:]):
        _paint_bar(out, bar["col"], bar["height"], palette[i])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each red bar reached by the blue-started height-as-jump chain with the matching palette color.

    Input structure:
    - The top row contains distinct palette pixels ordered from left to right.
    - Below it, the bottom baseline supports one blue one-cell-wide vertical bar and several red one-cell-wide vertical bars.
    - Every bar is bottom-aligned, occupies one column, and has a contiguous height.

    Rule:
    - Start from the blue bar.
    - Repeatedly move to the unique unvisited red bar whose column distance from the current bar equals the current bar's height; the next bar may lie to the left or to the right.
    - The i-th reached red bar is recolored with the i-th palette color, while unreached red bars stay red, the blue bar stays blue, and the whole palette row is cleared to background.

    Key insight:
    - The palette does not map to red bars by left-to-right position; it maps by the zigzag causal chain where each bar's height determines the next jump distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _expected_output(g)


def _intended_chain_ok(cols: list[int], heights: list[int]) -> bool:
    bars = []
    for i, (col, height) in enumerate(zip(cols, heights)):
        bars.append({"col": int(col), "height": int(height), "color": BLUE if i == 0 else RED})
    try:
        actual = _follow_chain(bars)
    except AssertionError:
        return False
    return [bar["col"] for bar in actual] == cols


def _sample_chain_relative() -> tuple[list[int], list[int]] | None:
    red_count = random.randint(MIN_CHAIN_RED, MAX_CHAIN_RED)
    total_bars = red_count + 1
    for _ in range(300):
        heights = [random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT) for _ in range(total_bars)]
        signs = [random.choice((-1, 1)) for _ in range(red_count)]
        if len(set(signs)) < 2:
            continue
        cols = [0]
        ok = True
        for i in range(red_count):
            cols.append(cols[-1] + signs[i] * heights[i])
            if cols[-1] in cols[:-1]:
                ok = False
                break
        if not ok:
            continue
        if not _intended_chain_ok(cols, heights):
            continue
        return cols, heights
    return None


def _try_add_distractor(
    bars: list[dict],
    intended_cols: list[int],
    start_col: int,
    max_height: int,
) -> dict | None:
    used_cols = {bar["col"] for bar in bars}
    chain_min = min(intended_cols)
    chain_max = max(intended_cols)
    candidates = list(range(start_col, MAX_GRID_SIDE - 1))
    random.shuffle(candidates)

    for col in candidates:
        if col in used_cols:
            continue
        if min(abs(col - other) for other in used_cols) < 2:
            continue
        height = random.randint(MIN_BAR_HEIGHT, max_height)
        if col - height in used_cols or col + height in used_cols:
            continue
        distractor = {"col": int(col), "height": int(height), "color": RED}
        trial_bars = bars + [distractor]
        try:
            actual_cols = [bar["col"] for bar in _follow_chain(trial_bars)]
        except AssertionError:
            continue
        if actual_cols != intended_cols:
            continue
        if chain_min < col < chain_max or random.random() < 0.35:
            return distractor
    return None


def _render_input(
    rows: int,
    bars: list[dict],
    palette_colors: list[int],
    palette_gap: int,
    right_margin: int,
) -> np.ndarray:
    palette_end = 1 + PALETTE_STRIDE * (len(palette_colors) - 1)
    width = max(palette_end + 1, max(bar["col"] for bar in bars) + 1 + right_margin)
    assert width <= MAX_GRID_SIDE
    grid = np.zeros((rows, width), dtype=int)

    for i, color in enumerate(palette_colors):
        grid[0, 1 + PALETTE_STRIDE * i] = color
    assert 1 + PALETTE_STRIDE * (len(palette_colors) - 1) + palette_gap < min(bar["col"] for bar in bars)

    for bar in bars:
        _paint_bar(grid, bar["col"], bar["height"], bar["color"])
    return grid


def _nondegenerate_candidate(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    _, palette = _palette_from_top_row(grid, bg)
    bars = _extract_bars(grid)
    chain = _follow_chain(bars)
    chain_cols = [bar["col"] for bar in chain]
    red_chain_cols = chain_cols[1:]
    spatial_red_cols = sorted(red_chain_cols)
    if red_chain_cols == spatial_red_cols:
        return False
    if len(chain) != len(palette) + 1:
        return False
    if len(bars) == len(chain):
        return False
    expected = _expected_output(grid)
    if np.array_equal(expected, grid):
        return False
    recolored_cols = {bar["col"] for bar in chain[1:]}
    unchanged_red = [bar for bar in bars if bar["color"] == RED and bar["col"] not in recolored_cols]
    return len(unchanged_red) >= 1


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    chain_spec = _sample_chain_relative()
    if chain_spec is None:
        return None
    relative_cols, heights = chain_spec
    if max(heights) > rows - 1:
        return None

    red_count = len(relative_cols) - 1
    palette_colors = random.sample(PALETTE_COLORS, red_count)
    palette_gap = random.randint(MIN_PALETTE_GAP, MAX_PALETTE_GAP)
    palette_end = 1 + PALETTE_STRIDE * (red_count - 1)
    start_col = palette_end + 1 + palette_gap

    shift = start_col - min(relative_cols)
    cols = [c + shift for c in relative_cols]
    chain_bars = []
    for i, (col, height) in enumerate(zip(cols, heights)):
        chain_bars.append({"col": int(col), "height": int(height), "color": BLUE if i == 0 else RED})

    if max(cols) + RIGHT_MARGIN_RANGE[0] >= MAX_GRID_SIDE:
        return None

    bars = chain_bars[:]
    intended_cols = [bar["col"] for bar in chain_bars]
    wanted_distractors = random.choices(DISTRACTOR_COUNT_CHOICES, weights=DISTRACTOR_COUNT_WEIGHTS, k=1)[0]
    for _ in range(wanted_distractors):
        distractor = _try_add_distractor(bars, intended_cols, start_col, min(MAX_BAR_HEIGHT, rows - 1))
        if distractor is None:
            return None
        bars.append(distractor)

    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    if max(bar["col"] for bar in bars) + 1 + right_margin > MAX_GRID_SIDE:
        return None

    grid = _render_input(rows, bars, palette_colors, palette_gap, right_margin)
    if not _nondegenerate_candidate(grid):
        return None
    try:
        expected = _expected_output(grid)
    except AssertionError:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-palette ARC input")
