from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LABEL_COLORS = (BLUE, RED, GREEN)


def _bands(size: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(int(v) for v in separators):
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _divider_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    shared = row_colors & col_colors
    assert len(shared) == 1, "expected exactly one full-row/full-column divider color"
    return next(iter(shared))


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _divider_color(g)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]
    assert len(sep_rows) == 3 and len(sep_cols) == 3, "expected 3 separator rows and 3 separator columns"

    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert len(row_bands) == 4 and len(col_bands) == 4, "expected 3 cell bands plus 1 label band per axis"

    cell_heights = [r1 - r0 + 1 for r0, r1 in row_bands[:3]]
    cell_widths = [c1 - c0 + 1 for c0, c1 in col_bands[:3]]
    assert len(set(cell_heights)) == 1 and len(set(cell_widths)) == 1, "cell sizes must be uniform"
    cell_size = cell_heights[0]
    assert cell_size == cell_widths[0], "cells must be square"
    assert row_bands[3][1] - row_bands[3][0] == 0, "bottom label band must be 1 pixel tall"
    assert col_bands[3][1] - col_bands[3][0] == 0, "right label band must be 1 pixel wide"

    label_col = col_bands[3][0]
    label_row = row_bands[3][0]
    row_labels: list[int] = []
    col_labels: list[int] = []

    for r0, r1 in row_bands[:3]:
        strip = g[r0 : r1 + 1, label_col]
        coords = np.argwhere(strip != BLACK)
        assert coords.shape[0] == 1, "each row label band must contain exactly one label pixel"
        color = int(strip[coords[0, 0]])
        assert color in LABEL_COLORS, "row label color must be blue, red, or green"
        row_labels.append(color)

    for c0, c1 in col_bands[:3]:
        strip = g[label_row, c0 : c1 + 1]
        coords = np.argwhere(strip != BLACK)
        assert coords.shape[0] == 1, "each column label band must contain exactly one label pixel"
        color = int(strip[coords[0, 0]])
        assert color in LABEL_COLORS, "column label color must be blue, red, or green"
        col_labels.append(color)

    template = g[row_bands[0][0] : row_bands[0][1] + 1, col_bands[0][0] : col_bands[0][1] + 1]
    assert set(map(int, np.unique(template))) <= {BLACK, YELLOW}, "template cell must use only black and yellow"
    template_mask = template == YELLOW
    template_pixels = int(template_mask.sum())
    assert 3 <= template_pixels <= 5, "template must contain 3 to 5 yellow pixels"
    assert _is_4_connected(template_mask), "template yellow shape must be 4-connected"
    return sep, cell_size, row_bands, col_bands, row_labels, col_labels, template_mask


def _is_4_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each damaged template cell according to whether its row label matches its column label.

    Input structure:
    - A 3x3 lattice of equal square cells is separated by full grey divider rows and columns.
    - A one-pixel-wide label column sits to the right of the lattice and a one-pixel-tall label row sits below it.
    - The top-left cell is a complete black-and-yellow template, while every other cell shows a partial black-and-yellow version of that template.

    Rule:
    - Detect the grey divider lines and split the grid into the 3x3 cell array plus the edge labels.
    - Use the top-left cell's yellow mask as the template.
    - For each non-template cell at position (r, c), compare its row label with its column label:
      if they match, paint every template foreground position in that cell with the shared label color;
      if they differ, fill only the currently missing template positions with grey and keep the surviving yellow pixels unchanged.
    - Leave the template cell, divider lines, and label row/column unchanged.

    Key insight:
    - The template tells where foreground may exist, but the row and column labels choose between two repair modes: recolored restoration on matches, or grey gap-filling on mismatches.
    """
    g = np.asarray(grid, dtype=int)
    sep, _cell_size, row_bands, col_bands, row_labels, col_labels, template_mask = _parse_grid(g)
    out = g.copy()

    for ri, (r0, r1) in enumerate(row_bands[:3]):
        for ci, (c0, c1) in enumerate(col_bands[:3]):
            if ri == 0 and ci == 0:
                continue
            cell = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            assert set(map(int, np.unique(cell))) <= {BLACK, YELLOW}, "damaged cells must use only black and yellow"
            assert not np.any((cell == YELLOW) & ~template_mask), "non-template cells cannot contain extra foreground outside the template"

            if row_labels[ri] == col_labels[ci]:
                repaired = np.full_like(cell, BLACK)
                repaired[template_mask] = row_labels[ri]
            else:
                missing = template_mask & (cell == BLACK)
                cell[missing] = sep
                repaired = cell
            out[r0 : r1 + 1, c0 : c1 + 1] = repaired

    return out


def _random_template_mask(cell_size: int) -> np.ndarray:
    for _ in range(200):
        count = random.randint(3, min(5, cell_size * cell_size - 1))
        start = (random.randrange(cell_size), random.randrange(cell_size))
        shape = {start}
        while len(shape) < count:
            frontier = []
            for r, c in shape:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in shape:
                        frontier.append((nr, nc))
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) != count:
            continue
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in shape:
            mask[r, c] = True
        if _is_4_connected(mask):
            return mask
    raise RuntimeError("failed to generate a valid template mask")


def _sample_labels() -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    for _ in range(400):
        row_labels = [random.choice(LABEL_COLORS) for _ in range(3)]
        col_labels = [random.choice(LABEL_COLORS) for _ in range(3)]
        if len(set(row_labels)) < 2 or len(set(col_labels)) < 2:
            continue
        matches = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0) and row_labels[r] == col_labels[c]]
        mismatches = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0) and row_labels[r] != col_labels[c]]
        match_colors = {row_labels[r] for r, c in matches}
        if matches and mismatches and len(match_colors) >= 2:
            return row_labels, col_labels, matches, mismatches, sorted(match_colors)
    raise RuntimeError("failed to sample informative labels")


def _random_subset(template_cells: list[tuple[int, int]], force_size: int | None = None) -> set[tuple[int, int]]:
    size = force_size if force_size is not None else random.randint(1, len(template_cells) - 1)
    return set(random.sample(template_cells, size))


def _render_grid(
    cell_size: int,
    template_mask: np.ndarray,
    row_labels: list[int],
    col_labels: list[int],
    visible_cells: dict[tuple[int, int], set[tuple[int, int]]],
) -> np.ndarray:
    total = 3 * cell_size + 4
    grid = np.full((total, total), BLACK, dtype=int)
    separators = [cell_size, 2 * cell_size + 1, 3 * cell_size + 2]
    grid[separators, :] = GREY
    grid[:, separators] = GREY

    for ri in range(3):
        for ci in range(3):
            r0 = ri * (cell_size + 1)
            c0 = ci * (cell_size + 1)
            if (ri, ci) == (0, 0):
                cell = np.where(template_mask, YELLOW, BLACK)
            else:
                cell = np.zeros((cell_size, cell_size), dtype=int)
                for r, c in visible_cells[(ri, ci)]:
                    cell[r, c] = YELLOW
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell

    label_col = total - 1
    label_row = total - 1
    for ri in range(3):
        center = ri * (cell_size + 1) + cell_size // 2
        grid[center, label_col] = row_labels[ri]
    for ci in range(3):
        center = ci * (cell_size + 1) + cell_size // 2
        grid[label_row, center] = col_labels[ci]
    return grid


def _expected_output_from_spec(
    grid: np.ndarray,
    template_mask: np.ndarray,
    row_labels: list[int],
    col_labels: list[int],
    cell_size: int,
) -> np.ndarray:
    out = grid.copy()
    for ri in range(3):
        for ci in range(3):
            if (ri, ci) == (0, 0):
                continue
            r0 = ri * (cell_size + 1)
            c0 = ci * (cell_size + 1)
            cell = out[r0 : r0 + cell_size, c0 : c0 + cell_size]
            if row_labels[ri] == col_labels[ci]:
                cell[:] = BLACK
                cell[template_mask] = row_labels[ri]
            else:
                missing = template_mask & (cell == BLACK)
                cell[missing] = GREY
    return out


def _validate_generated(
    grid: np.ndarray,
    template_mask: np.ndarray,
    row_labels: list[int],
    col_labels: list[int],
    visible_cells: dict[tuple[int, int], set[tuple[int, int]]],
    shared_match: tuple[int, int],
    shared_mismatch: tuple[int, int],
) -> bool:
    try:
        sep, cell_size, row_bands, col_bands, parsed_rows, parsed_cols, parsed_template = _parse_grid(grid)
    except AssertionError:
        return False

    if sep != GREY or parsed_rows != row_labels or parsed_cols != col_labels:
        return False
    if not np.array_equal(parsed_template, template_mask):
        return False

    for ri, (r0, r1) in enumerate(row_bands[:3]):
        for ci, (c0, c1) in enumerate(col_bands[:3]):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if (ri, ci) == (0, 0):
                if not np.array_equal(cell == YELLOW, template_mask):
                    return False
            else:
                expected = np.zeros((cell_size, cell_size), dtype=bool)
                for r, c in visible_cells[(ri, ci)]:
                    expected[r, c] = True
                if not np.array_equal(cell == YELLOW, expected):
                    return False
                if expected.sum() < 1 or expected.sum() >= template_mask.sum():
                    return False

    if visible_cells[shared_match] != visible_cells[shared_mismatch]:
        return False

    expected_out = _expected_output_from_spec(grid, template_mask, row_labels, col_labels, cell_size)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return False

    match_colors = set()
    changed_match = False
    changed_mismatch = False
    mismatch_grey = False

    for ri in range(3):
        for ci in range(3):
            if (ri, ci) == (0, 0):
                continue
            r0 = ri * (cell_size + 1)
            c0 = ci * (cell_size + 1)
            inp_cell = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            out_cell = actual_out[r0 : r0 + cell_size, c0 : c0 + cell_size]
            if row_labels[ri] == col_labels[ci]:
                match_colors.add(row_labels[ri])
                if np.array_equal(inp_cell, out_cell):
                    return False
                if np.any(out_cell == row_labels[ri]) and np.all(out_cell[template_mask] == row_labels[ri]):
                    changed_match = True
                if np.any(out_cell == YELLOW) or np.any(out_cell == GREY):
                    return False
            else:
                missing = template_mask & (inp_cell == BLACK)
                if not np.any(missing):
                    return False
                if not np.array_equal(out_cell[inp_cell == YELLOW], np.full(int(np.sum(inp_cell == YELLOW)), YELLOW)):
                    return False
                if np.any(out_cell[missing] != GREY):
                    return False
                changed_mismatch = True
                mismatch_grey = mismatch_grey or np.any(out_cell == GREY)

    return len(match_colors) >= 2 and changed_match and changed_mismatch and mismatch_grey


def generate_input() -> np.ndarray:
    for _ in range(400):
        cell_size = random.choice((4, 5, 6))
        template_mask = _random_template_mask(cell_size)
        template_cells = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
        row_labels, col_labels, matches, mismatches, _match_colors = _sample_labels()

        shared_subset = _random_subset(template_cells)
        shared_match = random.choice(matches)
        shared_mismatch = random.choice(mismatches)

        visible_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
        visible_cells[(0, 0)] = set(template_cells)
        for pos in [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]:
            if pos == shared_match or pos == shared_mismatch:
                visible_cells[pos] = set(shared_subset)
            else:
                visible_cells[pos] = _random_subset(template_cells)

        match_colors_seen = {row_labels[r] for r, c in matches}
        if len(match_colors_seen) < 2:
            continue

        grid = _render_grid(cell_size, template_mask, row_labels, col_labels, visible_cells)
        if _validate_generated(grid, template_mask, row_labels, col_labels, visible_cells, shared_match, shared_mismatch):
            return grid

    raise RuntimeError("failed to generate a valid input")
