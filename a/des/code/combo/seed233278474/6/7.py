from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the colored headers on the top row and left column to locate the cells
    that belong to each (column-color, row-color) pair, then infer the pair's
    output color from already-colored clue intersections and fill the rest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _dominant_color(g)
    height, width = g.shape

    col_headers = {c: int(g[0, c]) for c in range(1, width) if int(g[0, c]) != bg}
    row_headers = {r: int(g[r, 0]) for r in range(1, height) if int(g[r, 0]) != bg}
    assert col_headers and row_headers, "expected non-background headers on the top row and left column"

    pair_clues: dict[tuple[int, int], set[int]] = defaultdict(set)
    for r, row_color in row_headers.items():
        for c, col_color in col_headers.items():
            value = int(g[r, c])
            if value != bg:
                pair_clues[(col_color, row_color)].add(value)

    pair_to_output: dict[tuple[int, int], int] = {}
    for pair, seen in pair_clues.items():
        assert len(seen) == 1, "every color pair must map to a single output color"
        pair_to_output[pair] = next(iter(seen))

    out = g.copy()
    missing_pairs: set[tuple[int, int]] = set()
    for r, row_color in row_headers.items():
        for c, col_color in col_headers.items():
            pair = (col_color, row_color)
            if pair not in pair_to_output:
                missing_pairs.add(pair)
                continue
            out[r, c] = pair_to_output[pair]

    assert not missing_pairs, "every active row/column color pair needs at least one clue in the input"
    return out


# --------------- generator ---------------

_PALETTE = (1, 2, 3, 4)
_BG_CHOICES = (0, 5, 6, 7, 8, 9)
_MAX_RETRIES = 300

_PAIR_OUTPUT = {
    (1, 1): 2,
    (2, 1): 3,
    (3, 1): 4,
    (4, 1): 1,
    (1, 2): 4,
    (2, 2): 1,
    (3, 2): 2,
    (4, 2): 3,
    (1, 3): 3,
    (2, 3): 4,
    (3, 3): 1,
    (4, 3): 2,
    (1, 4): 1,
    (2, 4): 2,
    (3, 4): 3,
    (4, 4): 4,
}


def _header_instances(colors: list[int]) -> list[int]:
    items: list[int] = []
    bonus_count = random.randint(0, max(1, len(colors) // 2))
    bonus_colors = set(random.sample(colors, bonus_count)) if bonus_count else set()
    for color in colors:
        repeats = 2 + int(color in bonus_colors)
        items.extend([int(color)] * repeats)
    random.shuffle(items)
    return items


def _expected_output(grid: np.ndarray, row_headers: dict[int, int], col_headers: dict[int, int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, row_color in row_headers.items():
        for c, col_color in col_headers.items():
            out[r, c] = _PAIR_OUTPUT[(col_color, row_color)]
    return out


def _input_is_valid(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.flat)
    bg_count = counts.pop(bg, 0)
    return bg_count > max(counts.values(), default=0)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)

    n_col_colors = random.randint(2, len(_PALETTE))
    n_row_colors = random.randint(2, len(_PALETTE))
    active_col_colors = random.sample(list(_PALETTE), n_col_colors)
    active_row_colors = random.sample(list(_PALETTE), n_row_colors)

    col_header_values = _header_instances(active_col_colors)
    row_header_values = _header_instances(active_row_colors)

    extra_cols = random.randint(2, 5)
    extra_rows = random.randint(2, 5)
    width = 1 + len(col_header_values) + extra_cols
    height = 1 + len(row_header_values) + extra_rows

    col_positions = sorted(random.sample(range(1, width), len(col_header_values)))
    row_positions = sorted(random.sample(range(1, height), len(row_header_values)))

    grid = np.full((height, width), bg, dtype=int)
    col_headers = {c: color for c, color in zip(col_positions, col_header_values)}
    row_headers = {r: color for r, color in zip(row_positions, row_header_values)}

    for c, color in col_headers.items():
        grid[0, c] = color
    for r, color in row_headers.items():
        grid[r, 0] = color

    cols_by_color: dict[int, list[int]] = defaultdict(list)
    rows_by_color: dict[int, list[int]] = defaultdict(list)
    for c, color in col_headers.items():
        cols_by_color[int(color)].append(int(c))
    for r, color in row_headers.items():
        rows_by_color[int(color)].append(int(r))

    used_output_colors = {_PAIR_OUTPUT[(col_color, row_color)] for col_color in cols_by_color for row_color in rows_by_color}
    if len(used_output_colors) < 2:
        return None

    total_clues = 0
    for row_color, row_list in rows_by_color.items():
        for col_color, col_list in cols_by_color.items():
            cells = [(r, c) for r in row_list for c in col_list]
            if len(cells) < 2:
                return None
            max_clues = 2 if len(cells) >= 6 else 1
            clue_count = random.randint(1, min(max_clues, len(cells) - 1))
            for r, c in random.sample(cells, clue_count):
                grid[r, c] = _PAIR_OUTPUT[(col_color, row_color)]
                total_clues += 1

    if total_clues == 0:
        return None

    expected = _expected_output(grid, row_headers, col_headers)
    if np.array_equal(grid, expected):
        return None
    if not _input_is_valid(grid, bg):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
