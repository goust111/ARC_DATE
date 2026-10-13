from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PAYLOAD_PALETTE = (1, 2, 3, 4, 6, 7, 8)


def _solid_line_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]


def _separator_color(grid: np.ndarray) -> int:
    solid_rows = _solid_line_indices(grid, axis=0)
    solid_cols = _solid_line_indices(grid, axis=1)
    counts = Counter(int(grid[r, 0]) for r in solid_rows)
    counts.update(int(grid[0, c]) for c in solid_cols)
    candidates = [
        color
        for color in counts
        if any(int(grid[r, 0]) == color for r in solid_rows)
        and any(int(grid[0, c]) == color for c in solid_cols)
    ]
    if not candidates:
        raise AssertionError("expected separator lines in both directions")
    return max(candidates, key=lambda color: (counts[color], color == GREY))


def _content_bands(separator_indices: list[int], size: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(set(separator_indices)):
        if start < idx:
            bands.append((start, idx))
        start = idx + 1
    if start < size:
        bands.append((start, size))
    return bands


def _parse_cells(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    separator = _separator_color(g)

    row_separators = [r for r in _solid_line_indices(g, axis=0) if int(g[r, 0]) == separator]
    col_separators = [c for c in _solid_line_indices(g, axis=1) if int(g[0, c]) == separator]
    row_bands = _content_bands(row_separators, g.shape[0])
    col_bands = _content_bands(col_separators, g.shape[1])

    assert row_bands and col_bands, "expected at least one row and one column of cells"

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "cell bands must be regular"
    assert heights == widths, "cells must be square"

    cells = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            patch = g[r0:r1, c0:c1]
            value = int(patch[0, 0])
            assert np.all(patch == value), "each cell must be a uniform solid block or blank"
            assert value != separator, "separator color cannot occupy a cell interior"
            cells[i, j] = value

    return separator, row_bands, col_bands, cells


def _dominant_color(values: np.ndarray) -> int | None:
    counts = Counter(int(v) for v in values if int(v) != BLACK)
    if not counts:
        return None
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _symbolic_output(cells: np.ndarray) -> np.ndarray:
    out = np.array(cells, copy=True)
    row_dominants = [_dominant_color(out[i, :]) for i in range(out.shape[0])]
    col_dominants = [_dominant_color(out[:, j]) for j in range(out.shape[1])]

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if int(out[i, j]) != BLACK:
                continue
            row_color = row_dominants[i]
            col_color = col_dominants[j]
            out[i, j] = row_color if row_color is not None and row_color == col_color else MAROON
    return out


def _layout_bands(count: int, cell_size: int, separator_width: int, outer_border: bool) -> tuple[list[tuple[int, int]], int]:
    bands: list[tuple[int, int]] = []
    cursor = separator_width if outer_border else 0
    for idx in range(count):
        start = cursor
        end = start + cell_size
        bands.append((start, end))
        cursor = end
        if idx != count - 1 or outer_border:
            cursor += separator_width
    return bands, cursor


def _render_cells(cells: np.ndarray, cell_size: int, separator_width: int, outer_border: bool) -> np.ndarray:
    row_bands, height = _layout_bands(cells.shape[0], cell_size, separator_width, outer_border)
    col_bands, width = _layout_bands(cells.shape[1], cell_size, separator_width, outer_border)

    grid = np.full((height, width), BLACK, dtype=int)
    row_mask = np.zeros(height, dtype=bool)
    col_mask = np.zeros(width, dtype=bool)
    for r0, r1 in row_bands:
        row_mask[r0:r1] = True
    for c0, c1 in col_bands:
        col_mask[c0:c1] = True

    grid[~row_mask, :] = GREY
    grid[:, ~col_mask] = GREY

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            grid[r0:r1, c0:c1] = int(cells[i, j])
    return grid


def _expand_counts(total: int, colors: list[int], special: int) -> list[int]:
    counts = {color: 1 for color in colors}
    counts[special] = 2
    remaining = total - (len(colors) + 1)
    for _ in range(remaining):
        counts[random.choice(colors)] += 1

    values: list[int] = []
    for color in colors:
        values.extend([color] * counts[color])
    random.shuffle(values)
    return values


def _modes_match_targets(cells: np.ndarray, row_targets: list[int], col_targets: list[int]) -> bool:
    row_modes = [_dominant_color(cells[i, :]) for i in range(cells.shape[0])]
    col_modes = [_dominant_color(cells[:, j]) for j in range(cells.shape[1])]
    return row_modes == row_targets and col_modes == col_targets


def _has_multicolor_evidence(cells: np.ndarray) -> bool:
    for i in range(cells.shape[0]):
        colors = {int(v) for v in cells[i, :] if int(v) != BLACK}
        if len(colors) >= 2:
            return True
    for j in range(cells.shape[1]):
        colors = {int(v) for v in cells[:, j] if int(v) != BLACK}
        if len(colors) >= 2:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blank cells in a separator-defined cell matrix from row/column dominant colors.

    Each row votes for its most common non-black cell color, each column does the
    same, and every blank cell is filled by the shared vote when they agree or by
    maroon when they conflict. Existing fills and grey separators remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    separator, row_bands, col_bands, cells = _parse_cells(g)
    assert separator == _separator_color(g)

    row_dominants = [_dominant_color(cells[i, :]) for i in range(cells.shape[0])]
    col_dominants = [_dominant_color(cells[:, j]) for j in range(cells.shape[1])]

    out = g.copy()
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            if int(cells[i, j]) != BLACK:
                continue
            row_color = row_dominants[i]
            col_color = col_dominants[j]
            fill = row_color if row_color is not None and row_color == col_color else MAROON
            out[r0:r1, c0:c1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(3, 6)
    cols = random.randint(3, 6)
    max_colors = min(3, rows - 1, cols - 1)
    color_count = random.randint(2, max_colors)
    dominant_colors = random.sample(list(PAYLOAD_PALETTE), color_count)
    special_color = random.choice(dominant_colors)

    row_targets = _expand_counts(rows, dominant_colors, special_color)
    col_targets = _expand_counts(cols, dominant_colors, special_color)

    cells = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if row_targets[i] == col_targets[j]:
                cells[i, j] = row_targets[i]

    match_positions = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if row_targets[i] == col_targets[j] == special_color
    ]
    conflict_positions = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if row_targets[i] != col_targets[j]
    ]
    if not match_positions or not conflict_positions:
        return None

    match_blank = random.choice(match_positions)
    cells[match_blank] = BLACK
    conflict_blank = random.choice(conflict_positions)

    agreement_positions = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if cells[i, j] != BLACK and (i, j) != match_blank
    ]
    random.shuffle(agreement_positions)
    for i, j in agreement_positions:
        if random.random() >= 0.25:
            continue
        candidate = np.array(cells, copy=True)
        candidate[i, j] = BLACK
        if _modes_match_targets(candidate, row_targets, col_targets):
            cells = candidate

    disagreement_positions = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if row_targets[i] != col_targets[j] and (i, j) != conflict_blank
    ]
    random.shuffle(disagreement_positions)

    extra_pool = [color for color in PAYLOAD_PALETTE if color not in dominant_colors]
    for i, j in disagreement_positions:
        if random.random() >= 0.5:
            continue
        options = [row_targets[i], col_targets[j]]
        if dominant_colors:
            options.append(random.choice(dominant_colors))
        if extra_pool:
            options.append(random.choice(extra_pool))
        random.shuffle(options)
        for color in dict.fromkeys(options):
            candidate = np.array(cells, copy=True)
            candidate[i, j] = color
            if _modes_match_targets(candidate, row_targets, col_targets):
                cells = candidate
                break

    if not _has_multicolor_evidence(cells):
        for i, j in disagreement_positions:
            options = [col_targets[j], row_targets[i]]
            for color in options:
                candidate = np.array(cells, copy=True)
                candidate[i, j] = color
                if _modes_match_targets(candidate, row_targets, col_targets) and _has_multicolor_evidence(candidate):
                    cells = candidate
                    break
            if _has_multicolor_evidence(cells):
                break

    if not _modes_match_targets(cells, row_targets, col_targets):
        return None

    if int(cells[match_blank]) != BLACK or int(cells[conflict_blank]) != BLACK:
        return None

    symbolic_output = _symbolic_output(cells)
    if symbolic_output[match_blank] != row_targets[match_blank[0]]:
        return None
    if symbolic_output[conflict_blank] != MAROON:
        return None

    cell_size = random.randint(2, 4)
    separator_width = 1
    outer_border = random.random() < 0.5

    grid = _render_cells(cells, cell_size, separator_width, outer_border)
    expected = _render_cells(symbolic_output, cell_size, separator_width, outer_border)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
