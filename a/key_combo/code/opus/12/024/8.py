from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MAROON = 9
DIVIDER = GREY
INPUT_COLORS = (1, 3, 4, 7)


SYMMETRIC_TEMPLATES = (
    {
        "name": "horizontal_bar",
        "row_span": 1,
        "col_span": 3,
        "cells": ((0, 0), (0, 2)),
    },
    {
        "name": "vertical_bar",
        "row_span": 3,
        "col_span": 1,
        "cells": ((0, 0), (2, 0)),
    },
    {
        "name": "plus",
        "row_span": 3,
        "col_span": 3,
        "cells": ((1, 0), (1, 2), (0, 1), (2, 1)),
    },
    {
        "name": "double_horizontal",
        "row_span": 3,
        "col_span": 3,
        "cells": ((0, 0), (0, 2), (2, 0), (2, 2)),
    },
)


ASYMMETRIC_TEMPLATES = (
    {
        "name": "L_corner",
        "row_span": 3,
        "col_span": 3,
        "cells": ((0, 0), (0, 2), (2, 0)),
    },
    {
        "name": "reverse_L_corner",
        "row_span": 3,
        "col_span": 3,
        "cells": ((0, 2), (2, 0), (2, 2)),
    },
    {
        "name": "T_shape",
        "row_span": 3,
        "col_span": 3,
        "cells": ((0, 0), (0, 2), (1, 1), (2, 1)),
    },
    {
        "name": "inverted_T_shape",
        "row_span": 3,
        "col_span": 3,
        "cells": ((0, 1), (1, 1), (2, 0), (2, 2)),
    },
)


def _find_divider_color(grid: np.ndarray) -> int:
    candidates: list[int] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == grid[r, 0]):
            candidates.append(int(grid[r, 0]))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]):
            candidates.append(int(grid[0, c]))
    candidates = [color for color in candidates if color != BLACK]
    if not candidates:
        raise AssertionError("expected at least one full divider row or column")
    values, counts = np.unique(np.array(candidates, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _segments_between_dividers(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted(divider_indices) + [length]:
        start = prev + 1
        end = idx - 1
        if start <= end:
            segments.append((start, end))
        prev = idx
    if not segments:
        raise AssertionError("divider layout does not leave any cell segments")
    return segments


def _extract_cell_grid(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> np.ndarray:
    cell_grid = np.zeros((len(row_segments), len(col_segments)), dtype=int)
    for r_idx, (r0, r1) in enumerate(row_segments):
        for c_idx, (c0, c1) in enumerate(col_segments):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = np.unique(block)
            if len(colors) != 1:
                raise AssertionError("each logical cell must be uniformly colored")
            cell_grid[r_idx, c_idx] = int(colors[0])
    return cell_grid


def _connect_seed_cells(seed_cells: np.ndarray, divider_color: int) -> np.ndarray:
    connected = seed_cells.copy()
    colors = [
        int(color)
        for color in np.unique(seed_cells)
        if int(color) not in (BLACK, divider_color)
    ]
    for color in colors:
        positions = np.argwhere(seed_cells == color)
        if positions.size == 0:
            continue

        for row in sorted(set(map(int, positions[:, 0]))):
            cols = sorted(int(c) for c in positions[positions[:, 0] == row, 1])
            for left, right in zip(cols, cols[1:]):
                for c in range(left + 1, right):
                    if connected[row, c] not in (BLACK, color):
                        raise AssertionError("conflicting horizontal connection")
                    connected[row, c] = color

        for col in sorted(set(map(int, positions[:, 1]))):
            rows = sorted(int(r) for r in positions[positions[:, 1] == col, 0])
            for top, bottom in zip(rows, rows[1:]):
                for r in range(top + 1, bottom):
                    if connected[r, col] not in (BLACK, color):
                        raise AssertionError("conflicting vertical connection")
                    connected[r, col] = color
    return connected


def _render_cell_grid(
    cell_grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    shape: tuple[int, int],
    divider_rows: list[int],
    divider_cols: list[int],
    divider_color: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if divider_rows:
        out[divider_rows, :] = divider_color
    if divider_cols:
        out[:, divider_cols] = divider_color
    for r_idx, (r0, r1) in enumerate(row_segments):
        for c_idx, (c0, c1) in enumerate(col_segments):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(cell_grid[r_idx, c_idx])
    return out


def _recolor_by_horizontal_symmetry(grid: np.ndarray, divider_color: int) -> np.ndarray:
    out = grid.copy()
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, divider_color)]
    for color in colors:
        mask = grid == color
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        for r, c in coords:
            mirror_r = r0 + r1 - int(r)
            if grid[mirror_r, int(c)] == color:
                out[int(r), int(c)] = RED
            else:
                out[int(r), int(c)] = MAROON
    return out


def _transform_template(template: dict[str, object]) -> dict[str, object]:
    row_span = int(template["row_span"])
    col_span = int(template["col_span"])
    cells = [(int(r), int(c)) for r, c in template["cells"]]
    if random.random() < 0.5:
        cells = [(row_span - 1 - r, c) for r, c in cells]
    if random.random() < 0.5:
        cells = [(r, col_span - 1 - c) for r, c in cells]
    return {
        "name": str(template["name"]),
        "row_span": row_span,
        "col_span": col_span,
        "cells": tuple(cells),
    }


def _logical_capacity(cell_size: int) -> int:
    return min(7, 31 // (cell_size + 1))


def _contiguous_blocks(total: int, spans: list[int]) -> list[list[int]]:
    extra = total - sum(spans)
    gaps = [0] * (len(spans) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    blocks: list[list[int]] = []
    cursor = gaps[0]
    for idx, span in enumerate(spans):
        blocks.append(list(range(cursor, cursor + span)))
        cursor += span + gaps[idx + 1]
    return blocks


def _build_seed_cells(
    logical_rows: int,
    logical_cols: int,
    templates: list[dict[str, object]],
    colors: tuple[int, ...],
) -> np.ndarray:
    seed_cells = np.zeros((logical_rows, logical_cols), dtype=int)

    row_template_order = list(range(len(templates)))
    col_template_order = list(range(len(templates)))
    random.shuffle(row_template_order)
    random.shuffle(col_template_order)

    row_blocks_unsorted = _contiguous_blocks(
        logical_rows,
        [int(templates[idx]["row_span"]) for idx in row_template_order],
    )
    col_blocks_unsorted = _contiguous_blocks(
        logical_cols,
        [int(templates[idx]["col_span"]) for idx in col_template_order],
    )

    row_blocks: list[list[int]] = [[] for _ in templates]
    col_blocks: list[list[int]] = [[] for _ in templates]
    for block, template_idx in zip(row_blocks_unsorted, row_template_order):
        row_blocks[template_idx] = block
    for block, template_idx in zip(col_blocks_unsorted, col_template_order):
        col_blocks[template_idx] = block

    for idx, (template, color) in enumerate(zip(templates, colors)):
        chosen_rows = row_blocks[idx]
        chosen_cols = col_blocks[idx]
        for rel_r, rel_c in template["cells"]:
            seed_cells[chosen_rows[int(rel_r)], chosen_cols[int(rel_c)]] = int(color)
    return seed_cells


def _row_col_segments_from_uniform_cells(
    logical_count: int,
    cell_size: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    segments: list[tuple[int, int]] = []
    divider_indices: list[int] = []
    cursor = 0
    for idx in range(logical_count):
        start = cursor
        end = cursor + cell_size - 1
        segments.append((start, end))
        cursor = end + 1
        if idx != logical_count - 1:
            divider_indices.append(cursor)
            cursor += 1
    return segments, divider_indices


def _evaluate_color_symmetry(grid: np.ndarray, color: int) -> bool:
    mask = grid == color
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected color to be present")
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    for r, c in coords:
        if grid[r0 + r1 - int(r), int(c)] != color:
            return False
    return True


def _independent_expected_from_seed(seed_cells: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    connected = _connect_seed_cells(seed_cells, DIVIDER)
    row_segments, divider_rows = _row_col_segments_from_uniform_cells(seed_cells.shape[0], cell_h)
    col_segments, divider_cols = _row_col_segments_from_uniform_cells(seed_cells.shape[1], cell_w)
    pixel_grid = _render_cell_grid(
        connected,
        row_segments,
        col_segments,
        (
            seed_cells.shape[0] * cell_h + (seed_cells.shape[0] - 1),
            seed_cells.shape[1] * cell_w + (seed_cells.shape[1] - 1),
        ),
        divider_rows,
        divider_cols,
        DIVIDER,
    )
    return _recolor_by_horizontal_symmetry(pixel_grid, DIVIDER)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored cells across a divider grid, then recolor each connected color by horizontal reflection symmetry.

    Input structure:
    - The grid has black background cells separated by full grey divider rows and columns.
    - Each logical cell between dividers is a uniform rectangle, either black or one of a few foreground colors.
    - Same-colored seed cells lie in shared logical rows and/or shared logical columns with black gaps between them.

    Rule:
    - Identify the divider color from full monochromatic rows and columns, then recover the logical cell grid.
    - For each non-black, non-divider color, connect seed cells across every shared logical row and shared logical column by filling the black cells between consecutive seeds.
    - Render the connected logical grid back to pixels, then within each color's tight bounding box recolor pixels to red when their top-bottom mirror is the same color and to maroon otherwise.

    Key insight:
    - The reflection test is not applied to the sparse seeds themselves but to the larger shape created by the row/column connections, so the linking step determines which pixels become symmetric.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    divider_color = _find_divider_color(g)
    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == divider_color)]
    divider_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == divider_color)]
    if not divider_rows or not divider_cols:
        raise AssertionError("expected both horizontal and vertical divider lines")

    row_segments = _segments_between_dividers(g.shape[0], divider_rows)
    col_segments = _segments_between_dividers(g.shape[1], divider_cols)
    seed_cells = _extract_cell_grid(g, row_segments, col_segments)
    connected_cells = _connect_seed_cells(seed_cells, divider_color)
    connected_pixels = _render_cell_grid(
        connected_cells,
        row_segments,
        col_segments,
        g.shape,
        divider_rows,
        divider_cols,
        divider_color,
    )
    return _recolor_by_horizontal_symmetry(connected_pixels, divider_color)


def generate_input() -> np.ndarray:
    for _ in range(500):
        cell_h = random.randint(2, 4)
        cell_w = random.randint(2, 4)
        max_rows = _logical_capacity(cell_h)
        max_cols = _logical_capacity(cell_w)

        sym_template = _transform_template(random.choice(SYMMETRIC_TEMPLATES))
        asym_template = _transform_template(random.choice(ASYMMETRIC_TEMPLATES))
        templates = [sym_template, asym_template]

        min_rows = sum(int(template["row_span"]) for template in templates)
        min_cols = sum(int(template["col_span"]) for template in templates)
        if min_rows > max_rows or min_cols > max_cols:
            continue

        logical_rows = random.randint(min_rows, max_rows)
        logical_cols = random.randint(min_cols, max_cols)
        colors = tuple(random.sample(INPUT_COLORS, k=2))

        seed_cells = _build_seed_cells(logical_rows, logical_cols, templates, colors)
        row_segments, divider_rows = _row_col_segments_from_uniform_cells(logical_rows, cell_h)
        col_segments, divider_cols = _row_col_segments_from_uniform_cells(logical_cols, cell_w)
        shape = (
            logical_rows * cell_h + (logical_rows - 1),
            logical_cols * cell_w + (logical_cols - 1),
        )
        grid = _render_cell_grid(
            seed_cells,
            row_segments,
            col_segments,
            shape,
            divider_rows,
            divider_cols,
            DIVIDER,
        )

        connected_cells = _connect_seed_cells(seed_cells, DIVIDER)
        connected_pixels = _render_cell_grid(
            connected_cells,
            row_segments,
            col_segments,
            shape,
            divider_rows,
            divider_cols,
            DIVIDER,
        )

        per_color = {
            int(color): _evaluate_color_symmetry(connected_pixels, int(color))
            for color in colors
        }
        if not any(per_color.values()):
            continue
        if not any(not is_sym for is_sym in per_color.values()):
            continue

        expected = _independent_expected_from_seed(seed_cells, cell_h, cell_w)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        divider_row_count = sum(np.all(grid[r, :] == DIVIDER) for r in range(grid.shape[0]))
        divider_col_count = sum(np.all(grid[:, c] == DIVIDER) for c in range(grid.shape[1]))
        if divider_row_count != logical_rows - 1 or divider_col_count != logical_cols - 1:
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
