from __future__ import annotations

import numpy as np


GRID_LIMIT = 30
SEPARATOR_WIDTH = 1
GEN_RETRY_LIMIT = 300
HEADER_PALETTE_SIZE_CHOICES = (2, 3)
LAYOUT_CHOICES = [
    (n_row_cells, n_col_cells, cell_h, cell_w)
    for n_row_cells in range(3, 7)
    for n_col_cells in range(3, 7)
    for cell_h in range(2, 6)
    for cell_w in range(2, 6)
    if n_row_cells * cell_h + (n_row_cells - 1) * SEPARATOR_WIDTH <= GRID_LIMIT
    and n_col_cells * cell_w + (n_col_cells - 1) * SEPARATOR_WIDTH <= GRID_LIMIT
]


def _solid_color(block: np.ndarray) -> int:
    colors = np.unique(block)
    assert len(colors) == 1, "expected a solid-colored cell"
    return int(colors[0])


def _spans_from_separators(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    seps = [int(x) for x in separators]
    assert seps, "missing separator lines"
    assert seps[0] > 0 and seps[-1] < length - 1, "separator lines cannot touch the border"
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        assert start < sep, "separator lines must leave non-empty cells"
        spans.append((start, sep))
        start = sep + 1
    assert start < length, "separator lines must leave a final cell"
    spans.append((start, length))
    return spans


def _extract_headers(
    g: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> tuple[int, list[int], list[int]]:
    arb = _solid_color(g[row_spans[0][0] : row_spans[0][1], col_spans[0][0] : col_spans[0][1]])
    col_headers = [
        _solid_color(g[row_spans[0][0] : row_spans[0][1], c0:c1])
        for c0, c1 in col_spans[1:]
    ]
    row_headers = [
        _solid_color(g[r0:r1, col_spans[0][0] : col_spans[0][1]])
        for r0, r1 in row_spans[1:]
    ]
    assert arb not in row_headers and arb not in col_headers, "arbiter must be unique to the corner"
    return arb, row_headers, col_headers


def _find_layout(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    h, w = g.shape
    for color in np.unique(g):
        sep_rows = np.flatnonzero(np.all(g == color, axis=1))
        sep_cols = np.flatnonzero(np.all(g == color, axis=0))
        if len(sep_rows) == 0 or len(sep_cols) == 0:
            continue
        try:
            row_spans = _spans_from_separators(h, sep_rows)
            col_spans = _spans_from_separators(w, sep_cols)
            assert len(row_spans) >= 2 and len(col_spans) >= 2, "need header row and header column"
            assert len({r1 - r0 for r0, r1 in row_spans}) == 1, "row cells must have equal height"
            assert len({c1 - c0 for c0, c1 in col_spans}) == 1, "column cells must have equal width"
            _extract_headers(g, row_spans, col_spans)
        except AssertionError:
            continue
        candidates.append((int(color), row_spans, col_spans))
    assert len(candidates) == 1, "expected exactly one separator lattice"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each content cell of the separator lattice from the row-header and column-header color comparison.

    Input structure:
    - The grid contains full separator rows and columns of one shared color, dividing the canvas into equal rectangular cells.
    - The first cell row is a header row of solid-colored cells, and the first cell column is a header column of solid-colored cells.
    - The top-left corner header cell is a unique arbiter color that does not appear in the other headers.
    - All non-header cells start as background-colored content cells.

    Rule:
    - Locate the separator lattice and read the solid color in each top header cell and each left header cell.
    - For every content cell, compare its row-header color with its column-header color.
    - If the two header colors match, fill the entire cell with that shared color.
    - Otherwise fill the cell with the unique top-left arbiter color, while keeping separators and headers unchanged.

    Key insight:
    - The top-left corner color is the mismatch result, so solving the task requires both reading the header grid and using the corner as the arbitration color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep_color, row_spans, col_spans = _find_layout(g)
    arb, row_headers, col_headers = _extract_headers(g, row_spans, col_spans)

    out = g.copy()
    for i, (r0, r1) in enumerate(row_spans[1:], start=1):
        row_color = row_headers[i - 1]
        for j, (c0, c1) in enumerate(col_spans[1:], start=1):
            col_color = col_headers[j - 1]
            out[r0:r1, c0:c1] = row_color if row_color == col_color else arb
    return out


def _build_spans(count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    start = 0
    for idx in range(count):
        end = start + cell_size
        spans.append((start, end))
        if idx != count - 1:
            separators.append(end)
            start = end + SEPARATOR_WIDTH
    return spans, separators


def _balanced_headers(length: int, colors: list[int]) -> np.ndarray:
    base = np.array([colors[idx % len(colors)] for idx in range(length)], dtype=int)
    return base[np.random.permutation(length)]


def _sample_headers(n_rows: int, n_cols: int, palette: list[int]) -> tuple[np.ndarray, np.ndarray]:
    row_headers = _balanced_headers(n_rows, palette[:2])
    col_headers = _balanced_headers(n_cols, palette[:2])
    if len(palette) >= 3 and n_rows >= 4 and n_cols >= 4 and np.random.rand() < 0.5:
        row_headers[np.random.randint(n_rows)] = int(palette[2])
        col_headers[np.random.randint(n_cols)] = int(palette[2])
    match_count = sum(int(r == c) for r in row_headers for c in col_headers)
    total = n_rows * n_cols
    assert 0 < match_count < total, "generator must include both match and mismatch cells"
    return row_headers, col_headers


def _paint_input(
    n_row_cells: int,
    n_col_cells: int,
    cell_h: int,
    cell_w: int,
    separator_color: int,
    background_color: int,
    arb_color: int,
    row_headers: np.ndarray,
    col_headers: np.ndarray,
) -> np.ndarray:
    row_spans, sep_rows = _build_spans(n_row_cells, cell_h)
    col_spans, sep_cols = _build_spans(n_col_cells, cell_w)
    height = row_spans[-1][1]
    width = col_spans[-1][1]

    grid = np.full((height, width), background_color, dtype=int)
    for r in sep_rows:
        grid[r, :] = separator_color
    for c in sep_cols:
        grid[:, c] = separator_color

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if i == 0 and j == 0:
                color = arb_color
            elif i == 0:
                color = int(col_headers[j - 1])
            elif j == 0:
                color = int(row_headers[i - 1])
            else:
                color = background_color
            grid[r0:r1, c0:c1] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_row_cells, n_col_cells, cell_h, cell_w = LAYOUT_CHOICES[np.random.randint(len(LAYOUT_CHOICES))]
    palette_size = int(np.random.choice(HEADER_PALETTE_SIZE_CHOICES))
    colors = np.random.choice(10, size=palette_size + 3, replace=False)
    separator_color = int(colors[0])
    background_color = int(colors[1])
    arb_color = int(colors[2])
    header_palette = [int(x) for x in colors[3:]]

    row_headers, col_headers = _sample_headers(n_row_cells - 1, n_col_cells - 1, header_palette)
    inp = _paint_input(
        n_row_cells,
        n_col_cells,
        cell_h,
        cell_w,
        separator_color,
        background_color,
        arb_color,
        row_headers,
        col_headers,
    )

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape or max(inp.shape) > GRID_LIMIT:
        return None
    if not np.any(out != inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-mix ARC input")
