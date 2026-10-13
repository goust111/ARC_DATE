import random
from collections import Counter

import numpy as np


def _spans_from_separator_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(mask.tolist()):
        if is_sep:
            if start is not None:
                spans.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {
        int(grid[r, 0])
        for r in range(h)
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(w)
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected a unique separator color"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the compact color table encoded by solid cells inside a separator lattice.

    Input structure:
    - One separator color fills complete rows and complete columns, partitioning the grid into rectangular cells.
    - Most cells contain repeated multicolor patterns.
    - A contiguous rectangular block of cells is filled solidly, one color per cell.

    Rule:
    - Split the grid at separator rows and columns.
    - Keep only cells whose full interior is a single color.
    - Crop to the occupied rectangle of those solid cells and replace each cell by its color.

    Key insight:
    - Periodic patterned cells are noise; only completely solid cells encode the reduced output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_color(g)
    row_mask = np.all(g == sep, axis=1)
    col_mask = np.all(g == sep, axis=0)
    assert np.any(row_mask) and np.any(col_mask), "separator lattice must include rows and columns"

    row_spans = _spans_from_separator_mask(row_mask)
    col_spans = _spans_from_separator_mask(col_mask)
    assert row_spans and col_spans, "separator lattice must define non-empty cells"

    solid_cells: dict[tuple[int, int], int] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = g[r0:r1, c0:c1]
            if np.all(cell == cell[0, 0]):
                solid_cells[(ri, ci)] = int(cell[0, 0])
    assert solid_cells, "expected at least one solid cell"

    solid_rows = [ri for ri, _ in solid_cells]
    solid_cols = [ci for _, ci in solid_cells]
    out_r0, out_r1 = min(solid_rows), max(solid_rows)
    out_c0, out_c1 = min(solid_cols), max(solid_cols)

    out = np.empty((out_r1 - out_r0 + 1, out_c1 - out_c0 + 1), dtype=int)
    for ri in range(out_r0, out_r1 + 1):
        for ci in range(out_c0, out_c1 + 1):
            assert (ri, ci) in solid_cells, "solid cells must form one rectangle"
            out[ri - out_r0, ci - out_c0] = solid_cells[(ri, ci)]
    return out


# --------------- generator ---------------

COLOR_COUNT = 10
ROW_COUNT_RANGE = (3, 6)
COL_COUNT_RANGE = (3, 7)
CELL_HEIGHT_RANGE = (2, 6)
CELL_WIDTH_RANGE = (2, 6)
SEPARATOR_WIDTH_OPTIONS = (1, 1, 1, 2, 2, 3)
OUTER_BORDER_PROB = 0.5
RETRY_LIMIT = 400


def _tile_motif(motif: np.ndarray, height: int, width: int, rng: random.Random) -> np.ndarray:
    block = np.tile(
        motif,
        (
            (height + motif.shape[0] - 1) // motif.shape[0],
            (width + motif.shape[1] - 1) // motif.shape[1],
        ),
    )[:height, :width]
    if height > 1:
        block = np.roll(block, rng.randrange(height), axis=0)
    if width > 1:
        block = np.roll(block, rng.randrange(width), axis=1)
    return block


def _pattern_motif(bg: int, a1: int, a2: int, rng: random.Random) -> np.ndarray:
    motifs = [
        np.array([[bg, a1], [a2, bg]], dtype=int),
        np.array([[bg, bg, a1], [a2, bg, bg]], dtype=int),
        np.array([[bg, a1, bg], [bg, bg, a2], [a1, bg, bg]], dtype=int),
        np.array([[a1, bg, bg], [bg, bg, a2]], dtype=int),
        np.array([[bg, bg, a1], [bg, a2, bg], [a1, bg, bg]], dtype=int),
    ]
    motif = motifs[rng.randrange(len(motifs))].copy()
    for _ in range(rng.randrange(4)):
        motif = np.rot90(motif)
    if rng.random() < 0.5:
        motif = np.fliplr(motif)
    if rng.random() < 0.5:
        motif = np.flipud(motif)
    return motif


def _noise_cell(height: int, width: int, bg: int, accents: list[int], rng: random.Random) -> np.ndarray:
    a1 = rng.choice(accents)
    a2 = rng.choice(accents)
    motif = _pattern_motif(bg, a1, a2, rng)
    cell = _tile_motif(motif, height, width, rng)
    if len(np.unique(cell)) < 2:
        cell[0, 0] = a1
        cell[-1, -1] = a2
    return cell


def _sample_band_sizes(count: int, low: int, high: int, rng: random.Random) -> list[int]:
    return [rng.randint(low, high) for _ in range(count)]


def _layout_spans(
    sizes: list[int], separator_width: int, outer_border: bool
) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []

    pos = 0
    if outer_border:
        separators.extend(range(pos, pos + separator_width))
        pos += separator_width

    for index, size in enumerate(sizes):
        spans.append((pos, pos + size))
        pos += size
        if index != len(sizes) - 1 or outer_border:
            separators.extend(range(pos, pos + separator_width))
            pos += separator_width

    return spans, separators, pos


def _sample_solid_block(rows: int, cols: int, rng: random.Random) -> tuple[int, int, int, int]:
    while True:
        block_h = rng.randint(1, rows)
        block_w = rng.randint(1, cols)
        if block_h * block_w < rows * cols:
            break
    block_r0 = rng.randint(0, rows - block_h)
    block_c0 = rng.randint(0, cols - block_w)
    return block_r0, block_c0, block_h, block_w


def _sample_output_colors(
    height: int, width: int, bg: int, sep: int, rng: random.Random
) -> np.ndarray:
    palette = [c for c in range(COLOR_COUNT) if c != sep]
    rng.shuffle(palette)
    take = rng.randint(2, min(6, len(palette)))
    chosen = palette[:take]
    if bg not in chosen and rng.random() < 0.75:
        chosen[-1] = bg

    flat = [rng.choice(chosen) for _ in range(height * width)]
    if height * width > 1 and len(set(flat)) == 1:
        alternatives = [c for c in chosen if c != flat[0]]
        flat[rng.randrange(len(flat))] = rng.choice(alternatives)
    return np.asarray(flat, dtype=int).reshape(height, width)


def _render_grid(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    total_height: int,
    total_width: int,
    separator: int,
    bg: int,
    block: tuple[int, int, int, int],
    out: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((total_height, total_width), separator, dtype=int)
    accent_pool = [c for c in range(COLOR_COUNT) if c not in {separator, bg}]

    block_r0, block_c0, block_h, block_w = block
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            height = r1 - r0
            width = c1 - c0
            if block_r0 <= ri < block_r0 + block_h and block_c0 <= ci < block_c0 + block_w:
                color = int(out[ri - block_r0, ci - block_c0])
                cell = np.full((height, width), color, dtype=int)
            else:
                cell = _noise_cell(height, width, bg, accent_pool, rng)
            grid[r0:r1, c0:c1] = cell
    return grid


def _dominant_color_is_unique(grid: np.ndarray, color: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    return all(counts[color] > count for other, count in counts.items() if other != color)


def _valid_grid(
    grid: np.ndarray,
    separator: int,
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_separators: list[int],
    col_separators: list[int],
    block: tuple[int, int, int, int],
    out: np.ndarray,
) -> bool:
    h, w = grid.shape
    row_mask = np.all(grid == separator, axis=1)
    col_mask = np.all(grid == separator, axis=0)
    if [i for i, ok in enumerate(row_mask.tolist()) if ok] != row_separators:
        return False
    if [i for i, ok in enumerate(col_mask.tolist()) if ok] != col_separators:
        return False
    if not _dominant_color_is_unique(grid, bg):
        return False

    block_r0, block_c0, block_h, block_w = block
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = grid[r0:r1, c0:c1]
            unique = np.unique(cell)
            if separator in unique:
                return False
            inside = block_r0 <= ri < block_r0 + block_h and block_c0 <= ci < block_c0 + block_w
            if inside:
                if unique.size != 1:
                    return False
                if int(unique[0]) != int(out[ri - block_r0, ci - block_c0]):
                    return False
            elif unique.size < 2:
                return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    row_count = rng.randint(*ROW_COUNT_RANGE)
    col_count = rng.randint(*COL_COUNT_RANGE)
    row_sizes = _sample_band_sizes(row_count, *CELL_HEIGHT_RANGE, rng)
    col_sizes = _sample_band_sizes(col_count, *CELL_WIDTH_RANGE, rng)
    separator_width = rng.choice(SEPARATOR_WIDTH_OPTIONS)
    outer_border = rng.random() < OUTER_BORDER_PROB

    row_spans, row_separators, total_height = _layout_spans(row_sizes, separator_width, outer_border)
    col_spans, col_separators, total_width = _layout_spans(col_sizes, separator_width, outer_border)

    separator = rng.randrange(COLOR_COUNT)
    bg_choices = [c for c in range(COLOR_COUNT) if c != separator]
    bg = rng.choice(bg_choices)

    block = _sample_solid_block(row_count, col_count, rng)
    out = _sample_output_colors(block[2], block[3], bg, separator, rng)
    grid = _render_grid(
        row_spans,
        col_spans,
        total_height,
        total_width,
        separator,
        bg,
        block,
        out,
        rng,
    )

    if not _valid_grid(
        grid,
        separator,
        bg,
        row_spans,
        col_spans,
        row_separators,
        col_separators,
        block,
        out,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
