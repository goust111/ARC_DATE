from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
ORDER = 4
GEN_ATTEMPTS = 500
CLUE_SEARCH_ATTEMPTS = 600

BASE_SHAPE_SQUARE = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)

BASE_COLOR_SQUARE = (
    (0, 1, 2, 3),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
)

SHAPE_LIBRARY = (
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
)


def _ordered_unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _permute_square(square, row_perm, col_perm, symbol_perm):
    return [
        [symbol_perm[square[row_perm[r]][col_perm[c]]] for c in range(ORDER)]
        for r in range(ORDER)
    ]


def _count_latin_solutions(partial, limit=2):
    row_used = [
        {partial[r][c] for c in range(ORDER) if partial[r][c] is not None}
        for r in range(ORDER)
    ]
    col_used = [
        {partial[r][c] for r in range(ORDER) if partial[r][c] is not None}
        for c in range(ORDER)
    ]
    for idx in range(ORDER):
        if len(row_used[idx]) != sum(partial[idx][c] is not None for c in range(ORDER)):
            return 0
        if len(col_used[idx]) != sum(partial[r][idx] is not None for r in range(ORDER)):
            return 0

    empties = [(r, c) for r in range(ORDER) for c in range(ORDER) if partial[r][c] is None]
    count = 0

    def search():
        nonlocal count
        if count >= limit:
            return
        if not empties:
            count += 1
            return

        best_idx = None
        best_domain = None
        for idx, (r, c) in enumerate(empties):
            domain = [v for v in range(ORDER) if v not in row_used[r] and v not in col_used[c]]
            if not domain:
                return
            if best_domain is None or len(domain) < len(best_domain):
                best_idx = idx
                best_domain = domain
                if len(best_domain) == 1:
                    break

        r, c = empties.pop(best_idx)
        for value in best_domain:
            row_used[r].add(value)
            col_used[c].add(value)
            partial[r][c] = value
            search()
            partial[r][c] = None
            row_used[r].remove(value)
            col_used[c].remove(value)
            if count >= limit:
                break
        empties.insert(best_idx, (r, c))

    search()
    return count


def _solve_joint(shape_grid, color_grid, limit=2):
    row_shapes = [
        {shape_grid[r][c] for c in range(ORDER) if shape_grid[r][c] is not None}
        for r in range(ORDER)
    ]
    col_shapes = [
        {shape_grid[r][c] for r in range(ORDER) if shape_grid[r][c] is not None}
        for c in range(ORDER)
    ]
    row_colors = [
        {color_grid[r][c] for c in range(ORDER) if color_grid[r][c] is not None}
        for r in range(ORDER)
    ]
    col_colors = [
        {color_grid[r][c] for r in range(ORDER) if color_grid[r][c] is not None}
        for c in range(ORDER)
    ]
    used_pairs = {
        (shape_grid[r][c], color_grid[r][c])
        for r in range(ORDER)
        for c in range(ORDER)
        if shape_grid[r][c] is not None
    }
    clue_count = sum(shape_grid[r][c] is not None for r in range(ORDER) for c in range(ORDER))
    if len(used_pairs) != clue_count:
        return []

    empties = [(r, c) for r in range(ORDER) for c in range(ORDER) if shape_grid[r][c] is None]
    solutions = []

    def search():
        if len(solutions) >= limit:
            return
        if not empties:
            solutions.append(
                (
                    tuple(tuple(row) for row in shape_grid),
                    tuple(tuple(row) for row in color_grid),
                )
            )
            return

        best_idx = None
        best_domain = None
        for idx, (r, c) in enumerate(empties):
            shape_domain = [s for s in range(ORDER) if s not in row_shapes[r] and s not in col_shapes[c]]
            color_domain = [k for k in range(ORDER) if k not in row_colors[r] and k not in col_colors[c]]
            domain = [(s, k) for s in shape_domain for k in color_domain if (s, k) not in used_pairs]
            if not domain:
                return
            if best_domain is None or len(domain) < len(best_domain):
                best_idx = idx
                best_domain = domain
                if len(best_domain) == 1:
                    break

        r, c = empties.pop(best_idx)
        for shape_id, color_id in best_domain:
            row_shapes[r].add(shape_id)
            col_shapes[c].add(shape_id)
            row_colors[r].add(color_id)
            col_colors[c].add(color_id)
            used_pairs.add((shape_id, color_id))
            shape_grid[r][c] = shape_id
            color_grid[r][c] = color_id
            search()
            shape_grid[r][c] = None
            color_grid[r][c] = None
            row_shapes[r].remove(shape_id)
            col_shapes[c].remove(shape_id)
            row_colors[r].remove(color_id)
            col_colors[c].remove(color_id)
            used_pairs.remove((shape_id, color_id))
            if len(solutions) >= limit:
                break
        empties.insert(best_idx, (r, c))

    search()
    return solutions


def _row_col_spans(grid):
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == BLACK)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]
    assert len(sep_rows) == ORDER + 1 and len(sep_cols) == ORDER + 1
    row_spans = [(sep_rows[i] + 1, sep_rows[i + 1]) for i in range(ORDER)]
    col_spans = [(sep_cols[i] + 1, sep_cols[i + 1]) for i in range(ORDER)]
    cell_h = row_spans[0][1] - row_spans[0][0]
    cell_w = col_spans[0][1] - col_spans[0][0]
    assert cell_h == cell_w
    assert all(r1 - r0 == cell_h for r0, r1 in row_spans)
    assert all(c1 - c0 == cell_w for c0, c1 in col_spans)
    return row_spans, col_spans, cell_h


def _render_grid(shape_grid, color_grid, shape_masks, palette, bg, cell_size):
    side = ORDER * cell_size + ORDER + 1
    out = np.zeros((side, side), dtype=int)
    for r in range(ORDER):
        for c in range(ORDER):
            r0 = 1 + r * (cell_size + 1)
            c0 = 1 + c * (cell_size + 1)
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg
            shape_id = shape_grid[r][c]
            color_id = color_grid[r][c]
            if shape_id is None:
                continue
            color = palette[color_id]
            for dr, dc in shape_masks[shape_id]:
                out[r0 + dr, c0 + dc] = color
    return out


def _parse_grid(grid):
    row_spans, col_spans, cell_size = _row_col_spans(grid)
    interior_values = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            interior_values.extend(map(int, grid[r0:r1, c0:c1].ravel()))
    bg = Counter(interior_values).most_common(1)[0][0]
    assert bg != BLACK

    shape_keys = []
    colors = []
    cell_data = [[None for _ in range(ORDER)] for _ in range(ORDER)]
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0:r1, c0:c1]
            values = [int(v) for v in np.unique(patch) if int(v) != bg]
            assert len(values) <= 1
            if not values:
                continue
            color = values[0]
            mask = tuple(map(tuple, np.argwhere(patch == color)))
            colors.append(color)
            shape_keys.append(mask)
            cell_data[r][c] = (mask, color)

    shape_masks = _ordered_unique(shape_keys)
    palette = _ordered_unique(colors)
    assert len(shape_masks) == ORDER
    assert len(palette) == ORDER

    shape_grid = []
    color_grid = []
    for r in range(ORDER):
        shape_row = []
        color_row = []
        for c in range(ORDER):
            datum = cell_data[r][c]
            if datum is None:
                shape_row.append(None)
                color_row.append(None)
                continue
            mask, color = datum
            shape_row.append(shape_masks.index(mask))
            color_row.append(palette.index(color))
        shape_grid.append(shape_row)
        color_grid.append(color_row)

    return shape_grid, color_grid, shape_masks, palette, bg, cell_size


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the blank sprite cells by jointly solving Latin constraints on shape and color.

    Input structure:
    - Black separator lines and a black outer border divide the grid into a 4x4 table of equal cells.
    - Each non-blank cell contains one monochrome sprite chosen from four reusable templates and one of four colors.
    - Blank cells contain only the cell background; among the given cells, shapes are row/column-unique, colors are row/column-unique, and no shape-color pair repeats.

    Rule:
    - Preserve every given cell exactly.
    - Fill each blank cell with one template and one color so that every row and column contains each of the four shapes exactly once and each of the four colors exactly once.
    - Also enforce that each shape-color combination appears at most once in the whole 4x4 table.

    Key insight:
    - Shape and color form two independent Latin-square constraints, but the blanks are only forced once those two systems are coupled by the global no-repeated-combination constraint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shape_grid, color_grid, shape_masks, palette, bg, cell_size = _parse_grid(g)
    solutions = _solve_joint([row[:] for row in shape_grid], [row[:] for row in color_grid], limit=2)
    assert len(solutions) == 1
    solved_shapes, solved_colors = solutions[0]
    return _render_grid(solved_shapes, solved_colors, shape_masks, palette, bg, cell_size)


def _make_shape_masks(cell_size):
    base_masks = random.sample(SHAPE_LIBRARY, ORDER)
    anchor_r = random.randint(0, cell_size - 3)
    anchor_c = random.randint(0, cell_size - 3)
    return [
        tuple(sorted((anchor_r + r, anchor_c + c) for r, c in mask))
        for mask in base_masks
    ]


def _full_assignment():
    row_perm = list(range(ORDER))
    col_perm = list(range(ORDER))
    shape_perm = list(range(ORDER))
    color_perm = list(range(ORDER))
    random.shuffle(row_perm)
    random.shuffle(col_perm)
    random.shuffle(shape_perm)
    random.shuffle(color_perm)
    shape_square = _permute_square(BASE_SHAPE_SQUARE, row_perm, col_perm, shape_perm)
    color_square = _permute_square(BASE_COLOR_SQUARE, row_perm, col_perm, color_perm)
    return shape_square, color_square


def _choose_clues(shape_square, color_square):
    cells = [(r, c) for r in range(ORDER) for c in range(ORDER)]
    for _ in range(CLUE_SEARCH_ATTEMPTS):
        blank_count = random.randint(6, 10)
        blanks = set(random.sample(cells, blank_count))
        shape_grid = [
            [shape_square[r][c] if (r, c) not in blanks else None for c in range(ORDER)]
            for r in range(ORDER)
        ]
        color_grid = [
            [color_square[r][c] if (r, c) not in blanks else None for c in range(ORDER)]
            for r in range(ORDER)
        ]

        seen_shapes = {shape_grid[r][c] for r in range(ORDER) for c in range(ORDER) if shape_grid[r][c] is not None}
        seen_colors = {color_grid[r][c] for r in range(ORDER) for c in range(ORDER) if color_grid[r][c] is not None}
        if len(seen_shapes) != ORDER or len(seen_colors) != ORDER:
            continue
        if len(_solve_joint([row[:] for row in shape_grid], [row[:] for row in color_grid], limit=2)) != 1:
            continue
        if _count_latin_solutions([row[:] for row in shape_grid], limit=2) < 2:
            continue
        if _count_latin_solutions([row[:] for row in color_grid], limit=2) < 2:
            continue
        return shape_grid, color_grid
    return None


def _gen_attempt():
    cell_size = random.randint(4, 5)
    shape_masks = _make_shape_masks(cell_size)
    bg = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    palette = random.sample([color for color in range(1, 10) if color != bg], ORDER)
    shape_square, color_square = _full_assignment()
    clues = _choose_clues(shape_square, color_square)
    if clues is None:
        return None
    shape_grid, color_grid = clues
    return _render_grid(shape_grid, color_grid, shape_masks, palette, bg, cell_size)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-constraint Latin sprite grid")
