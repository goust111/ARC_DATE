from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LINE_COLOR_BY_CROSSINGS = {
    1: BLUE,
    2: RED,
    3: GREEN,
}

ENDPOINT_COLORS = (4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _full_grey_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]


def _full_grey_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _region_spans(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for i, size in enumerate(lengths):
        spans.append((int(pos), int(pos + size - 1)))
        pos += size
        if i != len(lengths) - 1:
            separators.append(int(pos))
            pos += 1
    return spans, separators


def _choose_endpoint_positions(
    spans: list[tuple[int, int]],
    crossings: int,
) -> tuple[int, int]:
    left_region = random.randint(0, len(spans) - 1 - crossings)
    right_region = left_region + crossings
    left_span = spans[left_region]
    right_span = spans[right_region]

    options: list[tuple[int, int]] = []
    for c0 in range(left_span[0], left_span[1] + 1):
        for c1 in range(right_span[0], right_span[1] + 1):
            if c1 - c0 - crossings >= 1:
                options.append((int(c0), int(c1)))
    assert options, "expected at least one placement with visible black connector cells"
    return random.choice(options)


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    grey_rows = _full_grey_rows(grid)
    grey_cols = _full_grey_cols(grid)
    assert grey_rows or grey_cols, "expected at least one grey separator line"

    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY)]
    assert colors, "expected colored endpoint pairs"

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each non-black, non-grey color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        same_row = r0 == r1
        same_col = c0 == c1
        assert same_row ^ same_col, "each pair must be axis-aligned"

        if same_row:
            assert r0 not in grey_rows
            used_rows.add(int(r0))
            c_lo, c_hi = sorted((int(c0), int(c1)))
            segment = grid[r0, c_lo + 1 : c_hi]
            crossing_count = sum(c in grey_cols for c in range(c_lo + 1, c_hi))
        else:
            assert c0 not in grey_cols
            used_cols.add(int(c0))
            r_lo, r_hi = sorted((int(r0), int(r1)))
            segment = grid[r_lo + 1 : r_hi, c0]
            crossing_count = sum(r in grey_rows for r in range(r_lo + 1, r_hi))

        assert crossing_count in LINE_COLOR_BY_CROSSINGS, "crossing count must be 1, 2, or 3"
        assert np.all((segment == BLACK) | (segment == GREY)), "connectors may pass only through black and grey cells"
        assert int(np.count_nonzero(segment == BLACK)) >= 1, "each connector must paint at least one black cell"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair with a straight segment whose fill color depends on how many grey region boundaries it crosses.

    Input structure:
    - The background is black.
    - Some complete grey rows and columns partition the grid into rectangular regions.
    - Every other color appears exactly twice as a same-color endpoint pair, and each pair lies on one shared row or one shared column.

    Rule:
    - For each same-color pair, draw a straight connector along its shared row or column between the two endpoints.
    - Recolor only the black cells on that path; keep the grey separator cells unchanged and keep both endpoints in their original color.
    - Count how many full grey separator lines lie between the endpoints: crossing 1 makes the connector blue, crossing 2 makes it red, and crossing 3 makes it green.

    Key insight:
    - The endpoint colors identify which pixels belong together, but the connector color is determined only by the number of grey partition boundaries crossed along the path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_rows = set(_full_grey_rows(g))
    grey_cols = set(_full_grey_cols(g))
    assert grey_rows or grey_cols, "expected full grey partition lines"

    out = g.copy()
    non_special_colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]

    for color in non_special_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each endpoint color must occur exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()

        if r0 == r1:
            r = int(r0)
            c_lo, c_hi = sorted((int(c0), int(c1)))
            between = g[r, c_lo + 1 : c_hi]
            assert np.all((between == BLACK) | (between == GREY))
            crossings = sum(c in grey_cols for c in range(c_lo + 1, c_hi))
            line_color = LINE_COLOR_BY_CROSSINGS[crossings]
            black_mask = between == BLACK
            assert int(np.count_nonzero(black_mask)) >= 1
            target = out[r, c_lo + 1 : c_hi]
            assert np.all((target == BLACK) | (target == line_color) | (target == GREY))
            target[black_mask] = line_color
        else:
            assert c0 == c1, "each pair must share either a row or a column"
            c = int(c0)
            r_lo, r_hi = sorted((int(r0), int(r1)))
            between = g[r_lo + 1 : r_hi, c]
            assert np.all((between == BLACK) | (between == GREY))
            crossings = sum(r in grey_rows for r in range(r_lo + 1, r_hi))
            line_color = LINE_COLOR_BY_CROSSINGS[crossings]
            black_mask = between == BLACK
            assert int(np.count_nonzero(black_mask)) >= 1
            target = out[r_lo + 1 : r_hi, c]
            assert np.all((target == BLACK) | (target == line_color) | (target == GREY))
            target[black_mask] = line_color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        orientation = random.choice(("horizontal", "vertical"))
        pair_count = random.randint(3, len(ENDPOINT_COLORS))

        if orientation == "horizontal":
            row_region_count = random.randint(3, 5)
            col_region_count = random.randint(4, 5)
        else:
            row_region_count = random.randint(4, 5)
            col_region_count = random.randint(3, 5)

        row_lengths = [random.randint(2, 5) for _ in range(row_region_count)]
        col_lengths = [random.randint(2, 5) for _ in range(col_region_count)]

        row_spans, grey_rows = _region_spans(row_lengths)
        col_spans, grey_cols = _region_spans(col_lengths)
        height = sum(row_lengths) + len(grey_rows)
        width = sum(col_lengths) + len(grey_cols)
        if max(height, width) > 30:
            continue

        free_rows = [r for span in row_spans for r in range(span[0], span[1] + 1)]
        free_cols = [c for span in col_spans for c in range(span[0], span[1] + 1)]
        if orientation == "horizontal" and len(grey_cols) < 3:
            continue
        if orientation == "vertical" and len(grey_rows) < 3:
            continue
        if orientation == "horizontal" and len(free_rows) < pair_count:
            continue
        if orientation == "vertical" and len(free_cols) < pair_count:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        grid[grey_rows, :] = GREY
        grid[:, grey_cols] = GREY

        counts = [1, 2, 3]
        while len(counts) < pair_count:
            counts.append(random.randint(1, 3))
        random.shuffle(counts)

        colors = random.sample(list(ENDPOINT_COLORS), pair_count)
        if orientation == "horizontal":
            chosen_rows = random.sample(free_rows, pair_count)
            for color, row, crossings in zip(colors, chosen_rows, counts):
                c0, c1 = _choose_endpoint_positions(col_spans, crossings)
                grid[row, c0] = color
                grid[row, c1] = color
        else:
            chosen_cols = random.sample(free_cols, pair_count)
            for color, col, crossings in zip(colors, chosen_cols, counts):
                r0, r1 = _choose_endpoint_positions(row_spans, crossings)
                grid[r0, col] = color
                grid[r1, col] = color

        try:
            _validate_input(grid)
            main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample within retry budget")
