from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
ROW_COLORS = (1, 2, 3)
LINE_COLOR = 4
HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (11, 19)
GENERATION_RETRIES = 300


def _row_main_color(row: np.ndarray) -> int:
    counts = Counter(int(row[c]) for c in range(0, row.size, 2) if int(row[c]) != BG)
    assert counts, "each row must expose its main color on even columns"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "row main color must be unique"
    return int(ranked[0][0])


def _decode_row(row: np.ndarray, main_color: int) -> tuple[np.ndarray, list[int]]:
    decoded = row.copy()
    marker_cols: list[int] = []
    for c in range(1, row.size, 2):
        value = int(row[c])
        if value == BG:
            decoded[c] = main_color
            marker_cols.append(c)
        elif value == main_color:
            decoded[c] = BG
        else:
            decoded[c] = value
    return decoded, marker_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal odd-column hidden row markers by undoing a parity-based inversion, then connect each row's marker pairs.

    Input structure:
    - The grid has a black background, and each row uses one visible main color chosen from a small foreground palette.
    - Even columns show that row's main color normally, while odd columns encode a black/main-color inversion of the true row pattern.
    - Hidden markers are the odd-column cells that appear black in the input but become the row's main color after the inversion is undone.

    Rule:
    - For each row, infer the main color from the most frequent non-black value on even columns.
    - Invert only the odd columns of that row by swapping black with the row's main color and leaving even columns unchanged.
    - Read the newly revealed odd-column main-color cells from left to right and connect each consecutive pair with a horizontal yellow segment, inclusive.

    Key insight:
    - Column parity is the spatial mask that tells you where the row-wise inversion happened; once the odd columns are restored, the former black holes become the only meaningful markers to connect.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    markers_by_row: list[list[int]] = []
    for r in range(g.shape[0]):
        main_color = _row_main_color(g[r])
        decoded_row, marker_cols = _decode_row(g[r], main_color)
        assert marker_cols and len(marker_cols) % 2 == 0, "each row must reveal an even number of markers"
        out[r] = decoded_row
        markers_by_row.append(marker_cols)

    for r, marker_cols in enumerate(markers_by_row):
        for start, end in zip(marker_cols[0::2], marker_cols[1::2]):
            assert end - start >= 4, "paired markers must be separated"
            out[r, start : end + 1] = LINE_COLOR

    return out


def _sample_even_foreground(width: int) -> set[int]:
    even_cols = list(range(0, width, 2))
    assert len(even_cols) >= 4

    if random.random() < 0.5:
        count = random.randint(2, len(even_cols) - 1)
        return set(random.sample(even_cols, count))

    unit_count = len(even_cols)
    run_count = 1 if unit_count < 6 else random.randint(1, 2)
    covered_units: set[int] = set()
    for _ in range(run_count):
        max_len = min(3, unit_count)
        run_len = random.randint(1, max_len)
        start = random.randint(0, unit_count - run_len)
        covered_units.update(range(start, start + run_len))

    while len(covered_units) < 2:
        covered_units.add(random.randrange(unit_count))
    if len(covered_units) == unit_count:
        covered_units.remove(random.choice(tuple(covered_units)))

    return {even_cols[idx] for idx in covered_units}


def _sample_marker_columns(width: int, *, force_double: bool = False) -> list[int]:
    odd_cols = list(range(1, width, 2))
    single_pairs = [(a, b) for i, a in enumerate(odd_cols) for b in odd_cols[i + 1 :] if b - a >= 4]
    assert single_pairs, "width must allow at least one marker pair"

    double_pairs: list[tuple[int, int, int, int]] = []
    for a, b in single_pairs:
        for c, d in single_pairs:
            if b < c and c - b >= 4:
                double_pairs.append((a, b, c, d))

    if double_pairs and (force_double or random.random() < 0.35):
        return list(random.choice(double_pairs))
    return list(random.choice(single_pairs))


def _row_colors(height: int) -> list[int]:
    colors: list[int] = []
    while len(colors) < height:
        chunk = list(ROW_COLORS)
        random.shuffle(chunk)
        colors.extend(chunk)
    return colors[:height]


def _build_input(height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    force_double_row = None
    if width >= 15 and random.random() < 0.6:
        force_double_row = random.randrange(height)

    for r, main_color in enumerate(_row_colors(height)):
        even_foreground = _sample_even_foreground(width)
        marker_cols = _sample_marker_columns(width, force_double=r == force_double_row)

        for c in even_foreground:
            grid[r, c] = main_color
        for c in range(1, width, 2):
            grid[r, c] = main_color
        for c in marker_cols:
            grid[r, c] = BG
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    for row in grid:
        try:
            main_color = _row_main_color(row)
        except AssertionError:
            return False
        if main_color not in ROW_COLORS:
            return False

        odd_values = {int(row[c]) for c in range(1, row.size, 2)}
        if not odd_values <= {BG, main_color}:
            return False

        marker_cols = [c for c in range(1, row.size, 2) if int(row[c]) == BG]
        if not marker_cols or len(marker_cols) % 2 != 0:
            return False
        for start, end in zip(marker_cols[0::2], marker_cols[1::2]):
            if end - start < 4:
                return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if not np.any(out == LINE_COLOR):
        return False
    if np.array_equal(out, grid):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    grid = _build_input(height, width)
    if not _valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 11 input")
