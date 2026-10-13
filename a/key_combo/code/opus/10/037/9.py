from __future__ import annotations

import numpy as np


OUTPUT_SIZE = 15
BACKGROUND = 0
BAR_HEIGHT_VALUES = (2, 3, 4, 5, 6)
BAR_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    occupied_cols = [int(c) for c in np.flatnonzero(np.any(g != BACKGROUND, axis=0))]
    assert 4 <= len(occupied_cols) <= 6, "expected 4 to 6 vertical bars"
    assert all(occupied_cols[i + 1] - occupied_cols[i] >= 2 for i in range(len(occupied_cols) - 1)), (
        "bar columns must be separated by at least one empty column"
    )

    bars: list[dict[str, int]] = []
    for c in occupied_cols:
        rows = np.flatnonzero(g[:, c] != BACKGROUND)
        assert rows.size > 0, "every occupied column must contain a bar"
        r0, r1 = int(rows[0]), int(rows[-1])
        colors = np.unique(g[rows, c])
        assert colors.size == 1, "each bar must be monochromatic"
        color = int(colors[0])
        assert r1 == h - 1, "bars must be bottom-aligned"
        assert rows.size == r1 - r0 + 1, "bars must be vertically solid"
        assert np.all(g[r0 : r1 + 1, c] == color), "bar body must be contiguous"
        bars.append({"col": c, "row0": r0, "height": int(rows.size), "color": color})

    heights = [bar["height"] for bar in bars]
    colors = [bar["color"] for bar in bars]
    assert all(height in BAR_HEIGHT_VALUES for height in heights), "bar heights must be within 2..6"
    assert len(set(heights)) == len(heights), "all bar heights must be distinct"
    assert len(set(colors)) == len(colors), "bar colors must be distinct"

    nonzero_mask = g != BACKGROUND
    rebuilt_mask = np.zeros_like(nonzero_mask, dtype=bool)
    for bar in bars:
        rebuilt_mask[h - bar["height"] : h, bar["col"]] = True
    assert np.array_equal(nonzero_mask, rebuilt_mask), "foreground must consist only of the bars"
    return bars


def _extreme_bars(bars: list[dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    heights = [bar["height"] for bar in bars]
    tallest_height = max(heights)
    shortest_height = min(heights)
    assert tallest_height > shortest_height, "tallest and shortest bars must differ"

    tallest = [bar for bar in bars if bar["height"] == tallest_height]
    shortest = [bar for bar in bars if bar["height"] == shortest_height]
    assert len(tallest) == 1, "tallest bar must be unique"
    assert len(shortest) == 1, "shortest bar must be unique"
    return tallest[0], shortest[0]


def _build_output(tallest_height: int, tallest_color: int, shortest_height: int, shortest_color: int) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)

    horizontal_rows = np.arange(tallest_height, OUTPUT_SIZE, tallest_height + 1)
    vertical_cols = np.arange(shortest_height, OUTPUT_SIZE, shortest_height + 1)

    out[horizontal_rows, :] = tallest_color
    out[:, vertical_cols] = shortest_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Convert extreme-height vertical bars into an asymmetric 15x15 lattice.

    Input structure:
    - The input is a black-background grid containing 4 to 6 one-cell-wide colored vertical bars.
    - Every bar is bottom-aligned, occupies exactly one column, has a distinct color, and has a distinct height in 2..6.

    Rule:
    - Find the unique tallest bar and the unique shortest bar.
    - Create a 15x15 black output grid.
    - Draw full-width horizontal lines at rows `H_max, H_max + (H_max + 1), ...` in the tallest bar's color.
    - Then draw full-height vertical lines at columns `H_min, H_min + (H_min + 1), ...` in the shortest bar's color, so intersections keep the vertical-line color.

    Key insight:
    - The output geometry depends on two ranked extremes: the tallest height sets row spacing while the shortest height sets column spacing, and their colors separately encode horizontal versus vertical line direction.
    """
    bars = _extract_bars(np.asarray(grid, dtype=int))
    tallest, shortest = _extreme_bars(bars)
    return _build_output(
        tallest_height=tallest["height"],
        tallest_color=tallest["color"],
        shortest_height=shortest["height"],
        shortest_color=shortest["color"],
    )


def _sample_columns(rng: np.random.Generator, width: int, count: int) -> list[int]:
    bases = sorted(rng.choice(width - count + 1, size=count, replace=False).tolist())
    return [int(base + i) for i, base in enumerate(bases)]


def _sample_heights(rng: np.random.Generator, count: int) -> list[int]:
    if count == 5:
        heights = list(BAR_HEIGHT_VALUES)
    else:
        mode = int(rng.integers(0, 3))
        if mode == 0:
            omitted = 2
        elif mode == 1:
            omitted = 6
        else:
            omitted = int(rng.choice((3, 4, 5)))
        heights = [height for height in BAR_HEIGHT_VALUES if height != omitted]
    rng.shuffle(heights)
    return [int(height) for height in heights]


def _oracle_output(tallest_height: int, tallest_color: int, shortest_height: int, shortest_color: int) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    row = tallest_height
    while row < OUTPUT_SIZE:
        out[row, :] = tallest_color
        row += tallest_height + 1

    col = shortest_height
    while col < OUTPUT_SIZE:
        out[:, col] = shortest_color
        col += shortest_height + 1
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, tallest: dict[str, int], shortest: dict[str, int]) -> None:
    recovered = _extract_bars(inp)
    recovered_tallest, recovered_shortest = _extreme_bars(recovered)
    assert recovered_tallest["height"] == tallest["height"]
    assert recovered_tallest["color"] == tallest["color"]
    assert recovered_shortest["height"] == shortest["height"]
    assert recovered_shortest["color"] == shortest["color"]

    out = main(inp)
    assert np.array_equal(out, expected), "solver output must match the independently computed lattice"

    rows = list(range(tallest["height"], OUTPUT_SIZE, tallest["height"] + 1))
    cols = list(range(shortest["height"], OUTPUT_SIZE, shortest["height"] + 1))
    mask = np.zeros_like(out, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    assert np.all(out[~mask] == BACKGROUND), "only lattice lines may be colored"
    for r in rows:
        for c in cols:
            assert out[r, c] == shortest["color"], "vertical lines must overwrite horizontal ones at intersections"


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        height = int(rng.integers(9, 13))
        bar_count = int(rng.choice((4, 5)))
        width = int(rng.integers(max(11, bar_count * 2 - 1), 15))

        columns = _sample_columns(rng, width, bar_count)
        heights = _sample_heights(rng, bar_count)
        colors = [int(color) for color in rng.choice(BAR_COLOR_POOL, size=bar_count, replace=False)]

        grid = np.zeros((height, width), dtype=int)
        bars: list[dict[str, int]] = []
        for col, bar_height, color in zip(columns, heights, colors):
            grid[height - bar_height : height, col] = color
            bars.append({"col": int(col), "height": int(bar_height), "color": int(color)})

        tallest, shortest = _extreme_bars(bars)
        expected = _oracle_output(
            tallest_height=tallest["height"],
            tallest_color=tallest["color"],
            shortest_height=shortest["height"],
            shortest_color=shortest["color"],
        )

        try:
            _validate_semantics(grid, expected, tallest, shortest)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input after repeated attempts")
