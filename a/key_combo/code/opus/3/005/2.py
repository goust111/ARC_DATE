from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
SEPARATOR = 7
ENCODING_PALETTE = (1, 2, 3, 4, 6)
BAR_HEIGHTS = tuple(range(3, 11))
MIN_BARS = 5
MAX_BARS = 7
MAX_SIZE = 30
RETRY_LIMIT = 400


def _find_separator(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == SEPARATOR, axis=1))
    assert len(rows) == 1, "expected exactly one full separator row"
    sep = int(rows[0])
    assert sep + 1 < grid.shape[0], "separator must have an encoding row below it"
    return sep


def _positive_color_counts(row: np.ndarray) -> dict[int, int]:
    values, freqs = np.unique(row[row != BG], return_counts=True)
    return {int(v): int(f) for v, f in zip(values, freqs)}


def _extract_bars(grid: np.ndarray, sep: int) -> list[dict[str, int]]:
    top = grid[:sep]
    assert top.size > 0, "separator must not be on the first row"
    assert np.all(grid[sep + 1 :] != GRAY), "gray bars only live above the separator"

    base_cols = [int(c) for c in np.flatnonzero(top[-1] == GRAY)]
    assert MIN_BARS <= len(base_cols) <= MAX_BARS, "expected 5-7 bottom-aligned gray bars"
    if len(base_cols) >= 2:
        assert min(b - a for a, b in zip(base_cols, base_cols[1:])) >= 2, "bars need empty columns between them"

    bars: list[dict[str, int]] = []
    gray_pixels = 0
    for c in base_cols:
        r = sep - 1
        while r >= 0 and grid[r, c] == GRAY:
            r -= 1
        height = sep - 1 - r
        row0 = sep - height
        assert row0 >= 0, "bar must stay in bounds"
        assert height in BAR_HEIGHTS, "bar height out of allowed range"
        assert np.all(grid[row0:sep, c] == GRAY), "bar must be contiguous"
        assert not np.any(grid[:row0, c] == GRAY), "bar column must contain a single segment"
        if c > 0:
            assert not np.any(grid[row0:sep, c - 1] == GRAY), "bars must be one column wide"
        if c + 1 < grid.shape[1]:
            assert not np.any(grid[row0:sep, c + 1] == GRAY), "bars must be one column wide"
        gray_pixels += height
        bars.append({"col": c, "height": height, "row0": row0})

    assert int(np.count_nonzero(top == GRAY)) == gray_pixels, "unexpected stray gray pixels"
    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must all be distinct"
    return bars


def _parse_case(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= MAX_SIZE, "grid exceeds ARC size limits"

    sep = _find_separator(g)
    bars = _extract_bars(g, sep)
    heights = [bar["height"] for bar in bars]
    tall_height = max(heights)
    short_height = min(heights)
    tall_bar = next(bar for bar in bars if bar["height"] == tall_height)
    short_bar = next(bar for bar in bars if bar["height"] == short_height)

    code_row = g[sep + 1]
    counts = _positive_color_counts(code_row)
    tall_colors = tuple(sorted(color for color, count in counts.items() if count == tall_height))
    short_colors = tuple(sorted(color for color, count in counts.items() if count == short_height))
    assert tall_colors, "missing tallest-height frequency match"
    assert short_colors, "missing shortest-height frequency match"
    assert set(tall_colors).isdisjoint(short_colors), "a color cannot match both frequencies"

    tall_cols = [int(c) for c in np.flatnonzero(np.isin(code_row, tall_colors))]
    short_cols = [int(c) for c in np.flatnonzero(np.isin(code_row, short_colors))]
    assert tall_bar["col"] not in tall_cols, "tallest bar column may not be reused by blue histogram columns"
    assert short_bar["col"] not in short_cols, "shortest bar column may not be reused by red histogram columns"

    return {
        "grid": g,
        "sep": sep,
        "bars": bars,
        "counts": counts,
        "code_row": code_row,
        "tall_height": tall_height,
        "short_height": short_height,
        "tall_bar": tall_bar,
        "short_bar": short_bar,
        "tall_colors": tall_colors,
        "short_colors": short_colors,
        "tall_cols": tall_cols,
        "short_cols": short_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the tallest and shortest gray bars as frequency targets for blue and red histogram columns.

    Input structure:
    - The background is black.
    - Above one full orange separator row there are 5-7 one-column gray vertical bars with distinct heights and a shared bottom alignment.
    - The row immediately below the separator is an encoding row containing scattered colored pixels from a small palette on black.

    Rule:
    - Find the unique tallest and shortest gray bars above the separator, measuring each bar by pixel height.
    - Recolor the tallest bar blue and the shortest bar red, then erase every other gray bar.
    - In the encoding row, every color whose frequency equals the tallest height marks blue columns at all of its positions, drawn upward from the cell just above the separator with that tallest height.
    - Likewise, every color whose frequency equals the shortest height marks red columns of the shortest height at all of its positions.
    - Leave the separator row and the encoding row unchanged.

    Key insight:
    - Ranking the gray bars is only the first half: the two extreme heights become the counting targets that decide which encoding-row columns should sprout the new histogram bars.
    """
    info = _parse_case(grid)
    g = info["grid"]
    sep = int(info["sep"])
    tall_height = int(info["tall_height"])
    short_height = int(info["short_height"])
    tall_bar = info["tall_bar"]
    short_bar = info["short_bar"]
    tall_cols = info["tall_cols"]
    short_cols = info["short_cols"]

    out = g.copy()
    out[:sep] = np.where(out[:sep] == GRAY, BG, out[:sep])
    out[sep - tall_height : sep, int(tall_bar["col"])] = BLUE
    out[sep - short_height : sep, int(short_bar["col"])] = RED

    if tall_cols:
        out[sep - tall_height : sep, tall_cols] = BLUE
    if short_cols:
        out[sep - short_height : sep, short_cols] = RED
    return out


def _choose_group_plan(tall_height: int, short_height: int) -> tuple[int, int]:
    plans: list[tuple[int, int]] = [(1, 1), (1, 1), (1, 1)]
    if 2 * tall_height + short_height + 1 + (tall_height + 1) + 1 <= 28:
        plans.extend([(2, 1), (2, 1)])
    if tall_height + 2 * short_height + 1 + (tall_height + 1) + 1 <= 28:
        plans.extend([(1, 2), (1, 2)])
    return random.choice(plans)


def _choose_frequency_plan(
    tall_height: int, short_height: int
) -> tuple[dict[int, int], tuple[int, ...], tuple[int, ...]]:
    tall_groups, short_groups = _choose_group_plan(tall_height, short_height)

    colors = list(ENCODING_PALETTE)
    random.shuffle(colors)
    tall_colors = tuple(colors[:tall_groups])
    short_colors = tuple(colors[tall_groups : tall_groups + short_groups])
    distractor_colors = colors[tall_groups + short_groups :]

    counts: dict[int, int] = {}
    for color in tall_colors:
        counts[int(color)] = tall_height
    for color in short_colors:
        counts[int(color)] = short_height

    base_total = sum(counts.values())
    low_color = int(distractor_colors[0])
    high_color = int(distractor_colors[1])
    low_count = random.choice((1, 2))
    max_high = min(tall_height + 3, 28 - base_total - low_count - 1)
    assert max_high > tall_height, "need room for a distractor count above the tallest height"
    high_count = random.randint(tall_height + 1, max_high)
    counts[low_color] = low_count
    counts[high_color] = high_count

    if len(distractor_colors) >= 3 and random.random() < 0.65:
        extra_color = int(distractor_colors[2])
        max_extra = min(6, 28 - sum(counts.values()) - 1)
        extra_candidates = [
            count
            for count in range(1, max_extra + 1)
            if count not in {tall_height, short_height, low_count, high_count}
        ]
        if extra_candidates:
            counts[extra_color] = random.choice(extra_candidates)

    return counts, tall_colors, short_colors


def _sample_spaced_columns(width: int, count: int) -> list[int]:
    anchors = sorted(random.sample(range(width - count + 1), count))
    return [int(anchor + i) for i, anchor in enumerate(anchors)]


def _sample_heights(bar_count: int) -> list[int]:
    if random.random() < 0.6:
        windows = [
            list(BAR_HEIGHTS[start : start + bar_count])
            for start in range(len(BAR_HEIGHTS) - bar_count + 1)
        ]
        return random.choice(windows)
    return random.sample(BAR_HEIGHTS, bar_count)


def _choose_width(bar_count: int, total_counts: int) -> int:
    minimum = max(14, 2 * bar_count - 1, total_counts + 1)
    maximum = min(28, total_counts + 5)
    assert minimum <= maximum, "no feasible width for this frequency plan"
    return random.randint(minimum, maximum)


def _layout_encoding_row(
    width: int,
    counts: dict[int, int],
    tall_colors: tuple[int, ...],
    short_colors: tuple[int, ...],
    bar_cols: list[int],
    tall_col: int,
    short_col: int,
) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    matched_colors = list(tall_colors + short_colors)
    matched_total = sum(counts[color] for color in matched_colors)
    non_extreme_bar_cols = [c for c in bar_cols if c not in {tall_col, short_col}]
    available_match_cols = [c for c in range(width) if c not in {tall_col, short_col}]

    max_overlap = min(len(non_extreme_bar_cols) - 1, matched_total - 1)
    overlap_cols: list[int] = []
    if max_overlap > 0 and random.random() < 0.75:
        overlap_cols = random.sample(non_extreme_bar_cols, random.randint(1, max_overlap))

    chosen_match_cols = set(overlap_cols)
    remaining_needed = matched_total - len(chosen_match_cols)
    fresh_cols = [c for c in available_match_cols if c not in bar_cols and c not in chosen_match_cols]
    assert remaining_needed >= 1 and fresh_cols, "need at least one histogram column away from all source bars"
    fresh_col = random.choice(fresh_cols)
    chosen_match_cols.add(fresh_col)
    remaining_needed -= 1

    pool = [c for c in available_match_cols if c not in chosen_match_cols]
    chosen_match_cols.update(random.sample(pool, remaining_needed))

    free_match_cols = list(chosen_match_cols)
    random.shuffle(free_match_cols)
    random.shuffle(matched_colors)
    for color in matched_colors:
        count = counts[color]
        cols = free_match_cols[:count]
        free_match_cols = free_match_cols[count:]
        row[cols] = color

    distractor_colors = [color for color in counts if color not in tall_colors and color not in short_colors]
    random.shuffle(distractor_colors)
    for color in distractor_colors:
        count = counts[color]
        free_cols = [c for c in range(width) if row[c] == BG]
        cols = random.sample(free_cols, count)
        row[cols] = color

    return row


def _render_input(case: dict[str, object]) -> np.ndarray:
    height = int(case["height"])
    width = int(case["width"])
    sep = int(case["sep"])
    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = SEPARATOR
    grid[sep + 1, :] = np.asarray(case["encoding"], dtype=int)
    for bar in case["bars"]:
        col = int(bar["col"])
        bar_height = int(bar["height"])
        grid[sep - bar_height : sep, col] = GRAY
    return grid


def _render_expected(case: dict[str, object]) -> np.ndarray:
    grid = _render_input(case)
    sep = int(case["sep"])
    tall_height = int(case["tall_height"])
    short_height = int(case["short_height"])
    tall_col = int(case["tall_col"])
    short_col = int(case["short_col"])
    encoding = np.asarray(case["encoding"], dtype=int)

    out = grid.copy()
    out[:sep] = np.where(out[:sep] == GRAY, BG, out[:sep])
    out[sep - tall_height : sep, tall_col] = BLUE
    out[sep - short_height : sep, short_col] = RED

    tall_mask = np.isin(encoding, case["tall_colors"])
    short_mask = np.isin(encoding, case["short_colors"])
    tall_cols = [int(c) for c in np.flatnonzero(tall_mask)]
    short_cols = [int(c) for c in np.flatnonzero(short_mask)]
    if tall_cols:
        out[sep - tall_height : sep, tall_cols] = BLUE
    if short_cols:
        out[sep - short_height : sep, short_cols] = RED
    return out


def _validate_case(grid: np.ndarray) -> bool:
    try:
        info = _parse_case(grid)
        out = main(grid)
    except Exception:
        return False

    sep = int(info["sep"])
    counts = info["counts"]
    tall_height = int(info["tall_height"])
    short_height = int(info["short_height"])
    bar_cols = {int(bar["col"]) for bar in info["bars"]}
    tall_col = int(info["tall_bar"]["col"])
    short_col = int(info["short_bar"]["col"])
    matched_cols = set(info["tall_cols"]) | set(info["short_cols"])
    non_extreme = bar_cols - {tall_col, short_col}

    if not any(count < short_height for count in counts.values()):
        return False
    if not any(count > tall_height for count in counts.values()):
        return False
    if not non_extreme - matched_cols:
        return False
    if not matched_cols - bar_cols:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.any((grid[:sep] == GRAY) & (out[:sep] == BG)):
        return False
    if not np.any((grid[:sep] == BG) & np.isin(out[:sep], (BLUE, RED))):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    bar_count = random.randint(MIN_BARS, MAX_BARS)
    heights = _sample_heights(bar_count)
    tall_height = max(heights)
    short_height = min(heights)

    try:
        counts, tall_colors, short_colors = _choose_frequency_plan(tall_height, short_height)
        width = _choose_width(bar_count, sum(counts.values()))
    except AssertionError:
        return None

    bar_cols = _sample_spaced_columns(width, bar_count)
    shuffled_heights = heights[:]
    random.shuffle(shuffled_heights)
    bars = [
        {"col": int(col), "height": int(height)}
        for col, height in zip(bar_cols, shuffled_heights)
    ]
    tall_col = next(bar["col"] for bar in bars if bar["height"] == tall_height)
    short_col = next(bar["col"] for bar in bars if bar["height"] == short_height)

    try:
        encoding = _layout_encoding_row(width, counts, tall_colors, short_colors, bar_cols, tall_col, short_col)
    except Exception:
        return None

    top_margin = random.randint(0, 4)
    sep = tall_height + top_margin
    height = sep + 2 + random.randint(1, 4)
    if height > MAX_SIZE:
        return None

    case = {
        "height": height,
        "width": width,
        "sep": sep,
        "bars": bars,
        "encoding": encoding,
        "tall_height": tall_height,
        "short_height": short_height,
        "tall_col": tall_col,
        "short_col": short_col,
        "tall_colors": tall_colors,
        "short_colors": short_colors,
    }
    inp = _render_input(case)
    expected = _render_expected(case)

    if not np.array_equal(main(inp), expected):
        return None
    if not _validate_case(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {RETRY_LIMIT} attempts")
