import random
from collections import Counter

import numpy as np


BG = 0
GEN_SEPARATOR = 5
GEN_PALETTE = tuple(color for color in range(10) if color not in {BG, GEN_SEPARATOR})
MAX_SIDE = 30
GEN_RETRIES = 300
LAYOUT_CHOICES = [
    (n, cell_h, cell_w, line_w)
    for n in range(2, 7)
    for cell_h in range(2, 6)
    for cell_w in range(2, 6)
    for line_w in (1, 2)
    if n * cell_h + (n + 1) * line_w <= MAX_SIDE
    and n * cell_w + (n + 1) * line_w <= MAX_SIDE
]


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev))
        start = prev = idx
    bands.append((start, prev))
    return bands


def _interior_spans(bands: list[tuple[int, int]], limit: int) -> list[tuple[int, int]] | None:
    if len(bands) < 2 or bands[0][0] != 0 or bands[-1][1] != limit - 1:
        return None
    spans: list[tuple[int, int]] = []
    for i in range(len(bands) - 1):
        start = bands[i][1] + 1
        stop = bands[i + 1][0]
        if start >= stop:
            return None
        spans.append((start, stop))
    return spans


def _grid_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        row_bands = _bands(np.where(np.all(g == color, axis=1))[0])
        col_bands = _bands(np.where(np.all(g == color, axis=0))[0])
        if len(row_bands) < 3 or len(row_bands) != len(col_bands):
            continue

        row_spans = _interior_spans(row_bands, h)
        col_spans = _interior_spans(col_bands, w)
        if row_spans is None or col_spans is None or len(row_spans) != len(col_spans):
            continue

        row_band_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
        col_band_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
        row_cell_heights = {r1 - r0 for r0, r1 in row_spans}
        col_cell_widths = {c1 - c0 for c0, c1 in col_spans}
        if (
            len(row_band_heights) != 1
            or len(col_band_widths) != 1
            or len(row_cell_heights) != 1
            or len(col_cell_widths) != 1
        ):
            continue

        candidates.append((len(row_spans), color, row_spans, col_spans))

    assert candidates, "expected a regular separator lattice"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_n = candidates[0][0]
    best = [item for item in candidates if item[0] == best_n]
    assert len(best) == 1, "separator lattice must be unique"
    _, sep, row_spans, col_spans = best[0]
    return sep, row_spans, col_spans


def _sample_offset(
    grid: np.ndarray,
    sep: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    counts: Counter[tuple[int, int]] = Counter()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1]
            pts = np.argwhere((patch != BG) & (patch != sep))
            assert pts.shape[0] <= 1, "each cell must contain at most one colored point"
            if pts.shape[0] == 1:
                counts[(int(pts[0, 0]), int(pts[0, 1]))] += 1

    assert counts, "need at least one colored point to infer the sampling offset"
    most_common = counts.most_common()
    assert len(most_common) == 1 or most_common[0][1] > most_common[1][1], "sampling offset must be unique"
    sample_r, sample_c = most_common[0][0]

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1]
            pts = np.argwhere((patch != BG) & (patch != sep))
            if pts.shape[0] == 1:
                assert (int(pts[0, 0]), int(pts[0, 1])) == (sample_r, sample_c)

    return sample_r, sample_c


def _expand(matrix: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(matrix, dtype=int), cell_h, axis=0), cell_w, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the hidden color matrix from a separator grid, then expand each sampled
    color into a solid block of the cell's interior size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep, row_spans, col_spans = _grid_structure(g)
    sample_r, sample_c = _sample_offset(g, sep, row_spans, col_spans)

    cell_h = row_spans[0][1] - row_spans[0][0]
    cell_w = col_spans[0][1] - col_spans[0][0]
    n = len(row_spans)
    matrix = np.zeros((n, n), dtype=int)
    for r, (r0, _) in enumerate(row_spans):
        for c, (c0, _) in enumerate(col_spans):
            matrix[r, c] = int(g[r0 + sample_r, c0 + sample_c])

    return _expand(matrix, cell_h, cell_w)


def _render_input(matrix: np.ndarray, cell_h: int, cell_w: int, line_w: int, sample_r: int, sample_c: int) -> np.ndarray:
    n = int(matrix.shape[0])
    h = n * cell_h + (n + 1) * line_w
    w = n * cell_w + (n + 1) * line_w
    grid = np.full((h, w), GEN_SEPARATOR, dtype=int)

    for r in range(n):
        for c in range(n):
            r0 = line_w + r * (cell_h + line_w)
            c0 = line_w + c * (cell_w + line_w)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG
            color = int(matrix[r, c])
            if color != BG:
                grid[r0 + sample_r, c0 + sample_c] = color

    return grid


def _random_matrix(n: int) -> np.ndarray:
    total = n * n
    min_active = max(2, total // 4)
    max_active = total - 1
    active_count = random.randint(min_active, max_active)
    active_cells = random.sample(range(total), active_count)

    color_count = 2 if active_count < 3 else random.choice((2, 3))
    colors = random.sample(GEN_PALETTE, color_count)
    matrix = np.zeros((n, n), dtype=int)

    seeded_colors = [colors[0], colors[1]]
    for idx, cell in enumerate(active_cells):
        color = seeded_colors[idx] if idx < len(seeded_colors) else random.choice(colors)
        r, c = divmod(cell, n)
        matrix[r, c] = color

    assert 0 < int(np.count_nonzero(matrix)) < total
    assert len({int(v) for v in matrix.ravel() if int(v) != BG}) >= 2
    return matrix


def _gen_attempt() -> np.ndarray | None:
    n, cell_h, cell_w, line_w = random.choice(LAYOUT_CHOICES)
    sample_r = random.randrange(cell_h)
    sample_c = random.randrange(cell_w)
    matrix = _random_matrix(n)

    inp = _render_input(matrix, cell_h, cell_w, line_w, sample_r, sample_c)
    out = _expand(matrix, cell_h, cell_w)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
