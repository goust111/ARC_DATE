from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

OUT_MIN = 20
OUT_MAX = 25
CELL_MIN = 2
CELL_MAX = 4
LINE_WIDTH = 1
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _all_grey_lines(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _cell_spans(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(lines, lines[1:]):
        assert b > a + LINE_WIDTH, "adjacent lattice lines leave no cell interior"
        spans.append((a + LINE_WIDTH, b))
    return spans


def _fill_cell(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0:r1, c0:c1] = color


def _domain_partition(diag_cols: list[int], rows: int, cols: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    diag = {(r, diag_cols[r]) for r in range(rows)}
    upper = {(r, c) for r in range(rows) for c in range(diag_cols[r])}
    lower = {(r, c) for r in range(rows) for c in range(diag_cols[r] + 1, cols)}
    return diag, upper, lower


def _edge_cells(domain: set[tuple[int, int]], rows: int, cols: int) -> set[tuple[int, int]]:
    edges = set()
    for r, c in domain:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) not in domain:
                edges.add((r, c))
                break
    return edges


def _extract_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(np.unique(g)).issubset({BLACK, GREEN, GREY}), "unexpected input colors"

    grey_rows = _all_grey_lines(g, axis=0)
    grey_cols = _all_grey_lines(g, axis=1)
    assert len(grey_rows) >= 5 and len(grey_cols) >= 5, "expected a complete lattice"
    assert grey_rows[0] == 0 and grey_rows[-1] == g.shape[0] - 1, "lattice must include top/bottom border"
    assert grey_cols[0] == 0 and grey_cols[-1] == g.shape[1] - 1, "lattice must include left/right border"

    row_spans = _cell_spans(grey_rows)
    col_spans = _cell_spans(grey_cols)
    rows, cols = len(row_spans), len(col_spans)
    assert rows >= 4 and cols >= 4, "expected at least a 4x4 cell array"

    diag_cols: list[int] = []
    for rr, row_span in enumerate(row_spans):
        marked_cols = []
        for cc, col_span in enumerate(col_spans):
            r0, r1 = row_span
            c0, c1 = col_span
            cell = g[r0:r1, c0:c1]
            assert set(np.unique(cell)).issubset({BLACK, GREEN}), "cell interiors may contain only black and marker green"
            green_count = int(np.sum(cell == GREEN))
            assert green_count in (0, 1), "each cell may contain at most one marker"
            if green_count == 1:
                marked_cols.append(cc)
        assert len(marked_cols) == 1, f"expected exactly one marked cell in cell-row {rr}"
        diag_cols.append(marked_cols[0])

    start = diag_cols[0]
    assert all(diag_cols[r] == start + r for r in range(rows)), "marked cells must follow a unit-step diagonal"
    assert start == 0 or start == 1 or start > 0
    assert diag_cols[-1] < cols, "diagonal must stay inside the lattice"
    return row_spans, col_spans, diag_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor lattice cell interiors by splitting the array with the green-marked diagonal.

    Input structure:
    - A grey 1-pixel lattice fills the outer border and divides the grid into a complete rectangular array of black cell interiors.
    - Some cells contain exactly one green marker pixel; these marked cells form a top-to-bottom diagonal with one marked cell per cell-row.
    - Non-marked cell interiors are black, and lattice-line pixels remain grey.

    Rule:
    - Recover the lattice cell array and the diagonal cell in each cell-row from the green markers.
    - Treat marked cells as the diagonal boundary, cells strictly left of it as the upper domain, and cells strictly right of it as the lower domain.
    - Within each domain, classify a cell as an edge cell iff one of its 4-neighbor cell slots leaves the domain, then fill upper-domain edges red, upper-domain interiors blue, lower-domain edges yellow, lower-domain interiors orange, and every diagonal cell grey.

    Key insight:
    - The green markers do not mark cells to preserve; they redefine the local perimeter of two subdomains inside one lattice, so edge-vs-interior must be computed after the diagonal split rather than on the whole array.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, diag_cols = _extract_structure(g)
    rows, cols = len(row_spans), len(col_spans)
    diag, upper, lower = _domain_partition(diag_cols, rows, cols)
    assert upper and lower, "the diagonal must split the lattice into two non-empty domains"

    upper_edge = _edge_cells(upper, rows, cols)
    lower_edge = _edge_cells(lower, rows, cols)

    out = g.copy()
    for r, row_span in enumerate(row_spans):
        for c, col_span in enumerate(col_spans):
            if (r, c) in diag:
                color = GREY
            elif (r, c) in upper_edge:
                color = RED
            elif (r, c) in upper:
                color = BLUE
            elif (r, c) in lower_edge:
                color = YELLOW
            else:
                assert (r, c) in lower
                color = ORANGE
            _fill_cell(out, row_span, col_span, color)

    return out


def _explicit_expected(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    start_col: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, set[tuple[int, int]]]]:
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    inp = np.full((height, width), GREY, dtype=int)

    row_spans = []
    r = 1
    for _ in range(rows):
        row_spans.append((r, r + cell_h))
        r += cell_h + 1

    col_spans = []
    c = 1
    for _ in range(cols):
        col_spans.append((c, c + cell_w))
        c += cell_w + 1

    for row_span in row_spans:
        for col_span in col_spans:
            _fill_cell(inp, row_span, col_span, BLACK)

    diag_cols = [start_col + r for r in range(rows)]
    for r_idx, c_idx in enumerate(diag_cols):
        rr = random.randrange(*row_spans[r_idx])
        cc = random.randrange(*col_spans[c_idx])
        inp[rr, cc] = GREEN

    out = inp.copy()
    diag, upper, lower = _domain_partition(diag_cols, rows, cols)
    upper_edge = _edge_cells(upper, rows, cols)
    lower_edge = _edge_cells(lower, rows, cols)

    for r_idx, row_span in enumerate(row_spans):
        for c_idx, col_span in enumerate(col_spans):
            if (r_idx, c_idx) in diag:
                color = GREY
            elif (r_idx, c_idx) in upper_edge:
                color = RED
            elif (r_idx, c_idx) in upper:
                color = BLUE
            elif (r_idx, c_idx) in lower_edge:
                color = YELLOW
            else:
                color = ORANGE
            _fill_cell(out, row_span, col_span, color)

    meta = {
        "diag": diag,
        "upper": upper,
        "lower": lower,
        "upper_edge": upper_edge,
        "lower_edge": lower_edge,
    }
    return inp, out, meta


def _is_outer(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = cell
    return r in (0, rows - 1) or c in (0, cols - 1)


def _valid_layout(rows: int, cols: int, cell_h: int, cell_w: int, start_col: int) -> bool:
    if not (1 <= start_col <= cols - rows - 1):
        return False

    inp, out, meta = _explicit_expected(rows, cols, cell_h, cell_w, start_col)
    if not (OUT_MIN <= inp.shape[0] <= OUT_MAX and OUT_MIN <= inp.shape[1] <= OUT_MAX):
        return False

    upper = meta["upper"]
    lower = meta["lower"]
    upper_edge = meta["upper_edge"]
    lower_edge = meta["lower_edge"]
    upper_interior = upper - upper_edge
    lower_interior = lower - lower_edge

    if not upper_interior or not lower_interior:
        return False
    if not any(not _is_outer(cell, rows, cols) for cell in upper_edge):
        return False
    if not any(not _is_outer(cell, rows, cols) for cell in lower_edge):
        return False

    colors = set(np.unique(out))
    if colors != {GREY, RED, BLUE, YELLOW, ORANGE}:
        return False
    return True


VALID_LAYOUTS = [
    (rows, cols, cell_h, cell_w, start_col)
    for rows in range(4, 7)
    for cols in range(rows + 2, 9)
    for cell_h in range(CELL_MIN, CELL_MAX + 1)
    for cell_w in range(CELL_MIN, CELL_MAX + 1)
    for start_col in range(1, max(1, cols - rows))
    if _valid_layout(rows, cols, cell_h, cell_w, start_col)
]


def _semantic_check(inp: np.ndarray, expected: np.ndarray, spec: tuple[int, int, int, int, int]) -> bool:
    rows, cols, cell_h, cell_w, start_col = spec
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if set(np.unique(inp)) != {BLACK, GREEN, GREY}:
        return False
    if set(np.unique(expected)) != {GREY, RED, BLUE, YELLOW, ORANGE}:
        return False

    diag_cols = [start_col + r for r in range(rows)]
    diag, upper, lower = _domain_partition(diag_cols, rows, cols)
    upper_edge = _edge_cells(upper, rows, cols)
    lower_edge = _edge_cells(lower, rows, cols)
    upper_interior = upper - upper_edge
    lower_interior = lower - lower_edge

    if not upper_interior or not lower_interior:
        return False
    if not any(not _is_outer(cell, rows, cols) for cell in upper_edge):
        return False
    if not any(not _is_outer(cell, rows, cols) for cell in lower_edge):
        return False

    row_spans = [(1 + r * (cell_h + 1), 1 + r * (cell_h + 1) + cell_h) for r in range(rows)]
    col_spans = [(1 + c * (cell_w + 1), 1 + c * (cell_w + 1) + cell_w) for c in range(cols)]
    for r_idx, row_span in enumerate(row_spans):
        green_cells = 0
        for c_idx, col_span in enumerate(col_spans):
            cell = inp[row_span[0] : row_span[1], col_span[0] : col_span[1]]
            green_count = int(np.sum(cell == GREEN))
            if green_count == 1:
                green_cells += 1
                if c_idx != diag_cols[r_idx]:
                    return False
            elif green_count != 0:
                return False
        if green_cells != 1:
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    assert VALID_LAYOUTS, "no valid generator layouts found"
    for _ in range(MAX_GEN_ATTEMPTS):
        spec = random.choice(VALID_LAYOUTS)
        inp, expected, _ = _explicit_expected(*spec)
        if _semantic_check(inp, expected, spec):
            return inp
    raise RuntimeError("failed to generate a valid diagonal-domain lattice input")
