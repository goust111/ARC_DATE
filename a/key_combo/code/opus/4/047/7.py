import random
from collections import Counter

import numpy as np


GREY = 5
BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7)
ROW_COUNT = 3
COL_COUNT = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_separator_color(grid: np.ndarray) -> int:
    candidates: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r, :]
        if np.all(row == row[0]):
            candidates.append(int(row[0]))
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]):
            candidates.append(int(col[0]))
    assert candidates, "expected full separator rows/cols"
    return int(Counter(candidates).most_common(1)[0][0])


def _segments_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in separators + [length]:
        if idx > start:
            spans.append((start, idx))
        start = idx + 1
    return spans


def _split_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    sep = _find_separator_color(grid)
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_spans = _segments_from_separators(grid.shape[0], sep_rows)
    col_spans = _segments_from_separators(grid.shape[1], sep_cols)
    assert len(row_spans) == ROW_COUNT and len(col_spans) == COL_COUNT
    cell_shapes = {(r1 - r0, c1 - c0) for r0, r1 in row_spans for c0, c1 in col_spans}
    assert len(cell_shapes) == 1, "expected a regular cell grid"
    return row_spans, col_spans, sep


def _recolor_cell(cell: np.ndarray) -> np.ndarray:
    out = np.array(cell, dtype=int, copy=True)
    colors = [int(color) for color in np.unique(cell) if int(color) != BLACK]
    assert colors, "cell must contain at least one rectangle"
    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        b = _bbox(cell == color)
        assert b is not None
        rects.append((color, b))

    outer_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color, b in rects:
        if not any(_strictly_contains(other_b, b) for _, other_b in rects if other_b != b):
            outer_candidates.append((color, b))
    assert len(outer_candidates) == 1, "expected exactly one outermost rectangle"
    outer_color, (r0, c0, r1, c1) = outer_candidates[0]
    region = out[r0 : r1 + 1, c0 : c1 + 1]
    region[region != BLACK] = int(outer_color)
    return out


def _signature(cell: np.ndarray) -> bytes:
    return cell.astype(np.int8, copy=False).tobytes() + str(cell.shape).encode()


def _build_output(cells: list[np.ndarray], sep: int) -> np.ndarray:
    cell_h, cell_w = cells[0].shape
    out = np.full((ROW_COUNT * cell_h + (ROW_COUNT - 1), cell_w), int(sep), dtype=int)
    for row, cell in enumerate(cells):
        r0 = row * (cell_h + 1)
        out[r0 : r0 + cell_h, :] = cell
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangle cells by outer-frame inheritance, then extract each row's unique post-recolor cell.

    Input structure:
    - A black-background grid is split by full grey separator lines into a 3x4 array of equal cells.
    - Each cell contains one or two strictly nested one-pixel rectangle frames with colors from the non-grey ARC palette.
    - Within a row, the four cells share the same outer-frame color but may differ in inner-frame colors or geometry.

    Rule:
    - For each cell, find the unique rectangle not strictly contained in any other rectangle of that cell.
    - Recolor every non-black pixel inside that outer rectangle's bounding box to the outer rectangle's color, leaving black background unchanged.
    - Compare the four recolored cells in each row pixel-by-pixel, select the unique odd cell, and stack those three odd cells vertically with grey separator rows.

    Key insight:
    - Inner-frame colors are not the true identity signal: after inheriting the outer color, cells that looked different can collapse into the same pattern, and only that post-recolor equivalence determines the odd cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_spans, col_spans, sep = _split_cells(g)

    selected: list[np.ndarray] = []
    for r0, r1 in row_spans:
        recolored_row: list[np.ndarray] = []
        for c0, c1 in col_spans:
            recolored_row.append(_recolor_cell(g[r0:r1, c0:c1]))
        counts = Counter(_signature(cell) for cell in recolored_row)
        odd_indices = [idx for idx, cell in enumerate(recolored_row) if counts[_signature(cell)] == 1]
        assert len(odd_indices) == 1, "each row must have exactly one post-recolor odd cell"
        selected.append(recolored_row[odd_indices[0]])

    return _build_output(selected, sep)


def _draw_frame(cell: np.ndarray, inset: int, color: int) -> None:
    h, w = cell.shape
    r0 = inset
    c0 = inset
    r1 = h - 1 - inset
    c1 = w - 1 - inset
    assert r0 <= r1 and c0 <= c1
    cell[r0, c0 : c1 + 1] = color
    cell[r1, c0 : c1 + 1] = color
    cell[r0 : r1 + 1, c0] = color
    cell[r0 : r1 + 1, c1] = color


def _valid_geometries(cell_h: int, cell_w: int) -> tuple[list[tuple[int, int | None]], list[tuple[int, int | None]]]:
    majority: list[tuple[int, int | None]] = []
    odd: list[tuple[int, int | None]] = []
    candidates = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    for outer_inset, inner_inset in candidates:
        inner_h = cell_h - 2 * inner_inset
        inner_w = cell_w - 2 * inner_inset
        if inner_h >= 2 and inner_w >= 2 and outer_inset < inner_inset:
            majority.append((outer_inset, inner_inset))
    odd_candidates = [(0, None), (1, None), (0, 2), (1, 2)]
    for outer_inset, inner_inset in odd_candidates:
        outer_h = cell_h - 2 * outer_inset
        outer_w = cell_w - 2 * outer_inset
        if outer_h < 2 or outer_w < 2:
            continue
        if inner_inset is None:
            odd.append((outer_inset, None))
            continue
        inner_h = cell_h - 2 * inner_inset
        inner_w = cell_w - 2 * inner_inset
        if inner_h >= 2 and inner_w >= 2 and outer_inset < inner_inset:
            odd.append((outer_inset, inner_inset))
    return majority, odd


def _make_cell(cell_h: int, cell_w: int, outer_color: int, geom: tuple[int, int | None], inner_color: int | None) -> np.ndarray:
    cell = np.zeros((cell_h, cell_w), dtype=int)
    outer_inset, inner_inset = geom
    _draw_frame(cell, outer_inset, outer_color)
    if inner_inset is not None:
        assert inner_color is not None
        _draw_frame(cell, inner_inset, inner_color)
    return cell


def _row_raw_class_sizes(cells: list[np.ndarray]) -> list[int]:
    counts = Counter(_signature(cell) for cell in cells)
    return sorted(counts.values())


def _row_post_class_sizes(cells: list[np.ndarray]) -> list[int]:
    recolored = [_recolor_cell(cell) for cell in cells]
    counts = Counter(_signature(cell) for cell in recolored)
    return sorted(counts.values())


def _compose_grid(rows: list[list[np.ndarray]], sep: int) -> np.ndarray:
    cell_h, cell_w = rows[0][0].shape
    grid_h = ROW_COUNT * cell_h + (ROW_COUNT - 1)
    grid_w = COL_COUNT * cell_w + (COL_COUNT - 1)
    grid = np.full((grid_h, grid_w), int(sep), dtype=int)
    for rr in range(ROW_COUNT):
        for cc in range(COL_COUNT):
            r0 = rr * (cell_h + 1)
            c0 = cc * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = rows[rr][cc]
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    row_spans, col_spans, sep = _split_cells(grid)
    if sep != GREY:
        return False
    odd_cols: list[int] = []
    recolored_selected: list[np.ndarray] = []
    changed_partition = False
    for r0, r1 in row_spans:
        raw_row = [grid[r0:r1, c0:c1] for c0, c1 in col_spans]
        raw_sizes = _row_raw_class_sizes(raw_row)
        post_row = [_recolor_cell(cell) for cell in raw_row]
        post_counts = Counter(_signature(cell) for cell in post_row)
        odd_indices = [idx for idx, cell in enumerate(post_row) if post_counts[_signature(cell)] == 1]
        if len(odd_indices) != 1:
            return False
        odd_cols.append(odd_indices[0])
        recolored_selected.append(post_row[odd_indices[0]])
        if raw_sizes != [1, 1, 2]:
            return False
        if _row_post_class_sizes(raw_row) != [1, 3]:
            return False
        if raw_sizes != [1, 3]:
            changed_partition = True
    if len(set(odd_cols)) != ROW_COUNT:
        return False
    expected = _build_output(recolored_selected, GREY)
    if not np.array_equal(expected, main(grid)):
        return False
    return changed_partition


def _generate_attempt() -> np.ndarray | None:
    cell_h = random.choice((5, 6))
    cell_w = random.choice((5, 6))
    if max(ROW_COUNT * cell_h + (ROW_COUNT - 1), COL_COUNT * cell_w + (COL_COUNT - 1)) > 30:
        return None

    majority_geoms, odd_geoms = _valid_geometries(cell_h, cell_w)
    if not majority_geoms or len(odd_geoms) < 2:
        return None

    odd_columns = random.sample(range(COL_COUNT), ROW_COUNT)
    outer_colors = random.sample(PALETTE, ROW_COUNT)
    rows: list[list[np.ndarray]] = []

    for row in range(ROW_COUNT):
        outer_color = outer_colors[row]
        majority_geom = random.choice(majority_geoms)
        odd_geom_choices = [geom for geom in odd_geoms if geom != majority_geom]
        if not odd_geom_choices:
            return None
        odd_geom = random.choice(odd_geom_choices)

        inner_palette = [color for color in PALETTE if color != outer_color]
        inner_major = random.choice(inner_palette)
        inner_alt_choices = [color for color in inner_palette if color != inner_major]
        if not inner_alt_choices:
            return None
        inner_alt = random.choice(inner_alt_choices)
        odd_inner = None
        if odd_geom[1] is not None:
            odd_inner = random.choice([color for color in inner_palette if color != inner_major or color != inner_alt])

        cells: list[np.ndarray] = []
        alt_major_index = random.choice([idx for idx in range(COL_COUNT) if idx != odd_columns[row]])
        for col in range(COL_COUNT):
            if col == odd_columns[row]:
                cells.append(_make_cell(cell_h, cell_w, outer_color, odd_geom, odd_inner))
            else:
                inner_color = inner_alt if col == alt_major_index else inner_major
                cells.append(_make_cell(cell_h, cell_w, outer_color, majority_geom, inner_color))
        rows.append(cells)

    grid = _compose_grid(rows, GREY)
    if _validate_semantics(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
