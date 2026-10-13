from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
INPUT_COLORS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _solid_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    assert axis in (0, 1)
    lines: list[tuple[int, int]] = []
    limit = int(grid.shape[axis])
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            lines.append((idx, color))
    return lines


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*sorted(separators), int(limit)]:
        assert start < sep, "separator lines must leave non-empty content bands"
        bands.append((start, sep))
        start = sep + 1
    return bands


def _extract_blocks(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    solid_rows = _solid_lines(grid, axis=0)
    solid_cols = _solid_lines(grid, axis=1)
    assert solid_rows and solid_cols, "expected separator rows and columns"

    row_colors = {color for _idx, color in solid_rows}
    col_colors = {color for _idx, color in solid_cols}
    assert len(row_colors) == 1 and row_colors == col_colors, "separator lines must share one color"
    separator = next(iter(row_colors))

    row_bands = _content_bands([idx for idx, _color in solid_rows], grid.shape[0])
    col_bands = _content_bands([idx for idx, _color in solid_cols], grid.shape[1])
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a grid of repeated blocks"

    block_heights = {r1 - r0 for r0, r1 in row_bands}
    block_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(block_heights) == 1 and len(block_widths) == 1, "all blocks must have equal size"
    block_h = block_heights.pop()
    block_w = block_widths.pop()

    blocks = np.empty((len(row_bands), len(col_bands), block_h, block_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            blocks[br, bc] = grid[r0:r1, c0:c1]

    assert not np.any(blocks == separator), "separator color must stay on separator lines"
    return separator, row_bands, col_bands, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the repeated template from tiled copies and mark the missing cells from the damaged copy in blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert not np.any(g == BLUE), "blue is reserved for the output"

    _separator, row_bands, col_bands, blocks = _extract_blocks(g)
    block_rows, block_cols, block_h, block_w = map(int, blocks.shape)
    assert block_rows * block_cols >= 4, "expected enough copies for majority voting"

    motif = np.empty((block_h, block_w), dtype=int)
    for r in range(block_h):
        for c in range(block_w):
            motif[r, c] = _unique_mode(blocks[:, :, r, c])

    diffs = blocks != motif[None, None, :, :]
    diff_counts = diffs.reshape(block_rows, block_cols, block_h * block_w).sum(axis=2)
    damaged_blocks = np.argwhere(diff_counts > 0)
    assert damaged_blocks.shape[0] == 1, "expected exactly one damaged block"

    damaged_r, damaged_c = map(int, damaged_blocks[0])
    missing_mask = diffs[damaged_r, damaged_c]
    assert np.any(missing_mask), "damaged block must differ from the consensus"

    out = motif.copy()
    out[missing_mask] = BLUE
    return out


# --------------- generator ---------------

MIN_TILE_H = 3
MAX_TILE_H = 7
MIN_TILE_W = 3
MAX_TILE_W = 7
MIN_BLOCK_ROWS = 2
MAX_BLOCK_ROWS = 4
MIN_BLOCK_COLS = 2
MAX_BLOCK_COLS = 5
SEPARATOR_WIDTH = 1
MAX_GRID_SIDE = 30

MIN_NON_BG_CELLS = 3
MAX_MOTIF_COLORS = 4
MIN_DENSITY = 0.2
MAX_DENSITY = 0.48
MAX_DAMAGE_CELLS = 6
RETRY_LIMIT = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _partition(total: int, parts: int) -> list[int]:
    counts = [1] * parts
    for _ in range(total - parts):
        counts[random.randrange(parts)] += 1
    random.shuffle(counts)
    return counts


def _grow_connected_subset(
    available: set[tuple[int, int]], size: int, height: int, width: int, attempts: int = 80
) -> set[tuple[int, int]] | None:
    if size <= 0:
        return set()
    if len(available) < size:
        return None

    pool = tuple(available)
    for _ in range(attempts):
        seed = random.choice(pool)
        blob = {seed}
        frontier = {cell for cell in _neighbors4(seed[0], seed[1], height, width) if cell in available}
        while len(blob) < size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            blob.add(cell)
            for nb in _neighbors4(cell[0], cell[1], height, width):
                if nb in available and nb not in blob:
                    frontier.add(nb)
        if len(blob) == size:
            return blob
    return None


def _sample_tile_size() -> tuple[int, int]:
    tile_h = random.randint(MIN_TILE_H, MAX_TILE_H)
    tile_w = random.randint(MIN_TILE_W, MAX_TILE_W)
    return tile_h, tile_w


def _sample_block_counts(tile_h: int, tile_w: int) -> tuple[int, int] | None:
    max_rows = min(MAX_BLOCK_ROWS, (MAX_GRID_SIDE + SEPARATOR_WIDTH) // (tile_h + SEPARATOR_WIDTH))
    max_cols = min(MAX_BLOCK_COLS, (MAX_GRID_SIDE + SEPARATOR_WIDTH) // (tile_w + SEPARATOR_WIDTH))
    if max_rows < MIN_BLOCK_ROWS or max_cols < MIN_BLOCK_COLS:
        return None
    return random.randint(MIN_BLOCK_ROWS, max_rows), random.randint(MIN_BLOCK_COLS, max_cols)


def _sample_non_bg_count(area: int) -> int:
    low = max(MIN_NON_BG_CELLS, int(np.ceil(area * MIN_DENSITY)))
    high = min(area - 2, int(np.floor(area * MAX_DENSITY)))
    if high < low:
        low = high = max(MIN_NON_BG_CELLS, min(area - 2, area // 2))
    assert high >= low >= 1
    return random.randint(low, high)


def _build_scatter_motif(height: int, width: int, bg: int, colors: list[int], non_bg_count: int) -> np.ndarray:
    motif = np.full((height, width), bg, dtype=int)
    counts = _partition(non_bg_count, len(colors))
    cells = random.sample(range(height * width), non_bg_count)
    values: list[int] = []
    for color, count in zip(colors, counts):
        values.extend([int(color)] * count)
    random.shuffle(values)
    for flat, value in zip(cells, values):
        motif[flat // width, flat % width] = value
    return motif


def _build_blob_motif(height: int, width: int, bg: int, colors: list[int], non_bg_count: int) -> np.ndarray | None:
    motif = np.full((height, width), bg, dtype=int)
    counts = _partition(non_bg_count, len(colors))
    available = {(r, c) for r in range(height) for c in range(width)}
    order = sorted(range(len(colors)), key=lambda idx: (-counts[idx], random.random()))
    for idx in order:
        blob = _grow_connected_subset(available, counts[idx], height, width)
        if blob is None:
            return None
        for r, c in blob:
            motif[r, c] = colors[idx]
        available.difference_update(blob)
    return motif


def _build_motif(height: int, width: int, bg: int, colors: list[int], non_bg_count: int) -> np.ndarray | None:
    motif_style = random.choice(("scatter", "blob"))
    if motif_style == "scatter":
        return _build_scatter_motif(height, width, bg, colors, non_bg_count)
    return _build_blob_motif(height, width, bg, colors, non_bg_count)


def _pick_damage_cells(motif: np.ndarray, bg: int) -> set[tuple[int, int]] | None:
    non_bg_cells = {tuple(map(int, rc)) for rc in np.argwhere(motif != bg)}
    max_damage = min(MAX_DAMAGE_CELLS, len(non_bg_cells) - 1)
    if max_damage < 1:
        return None

    damage_style = random.choice(("scatter", "connected"))
    if damage_style == "connected" and max_damage >= 2:
        size = random.randint(2, max_damage)
        chosen = _grow_connected_subset(non_bg_cells, size, motif.shape[0], motif.shape[1])
        if chosen is not None:
            return chosen

    size = random.randint(1, max_damage)
    return set(random.sample(tuple(non_bg_cells), size))


def _tile_grid(motif: np.ndarray, block_rows: int, block_cols: int, separator: int) -> np.ndarray:
    tile_h, tile_w = map(int, motif.shape)
    height = block_rows * tile_h + (block_rows - 1) * SEPARATOR_WIDTH
    width = block_cols * tile_w + (block_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), separator, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (tile_h + SEPARATOR_WIDTH)
            c0 = bc * (tile_w + SEPARATOR_WIDTH)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = motif
    return grid


def _apply_damage(
    grid: np.ndarray,
    motif: np.ndarray,
    block_rows: int,
    block_cols: int,
    bg: int,
    damage_cells: set[tuple[int, int]],
) -> tuple[int, int]:
    tile_h, tile_w = map(int, motif.shape)
    damaged_br = random.randrange(block_rows)
    damaged_bc = random.randrange(block_cols)
    r0 = damaged_br * (tile_h + SEPARATOR_WIDTH)
    c0 = damaged_bc * (tile_w + SEPARATOR_WIDTH)
    for r, c in damage_cells:
        grid[r0 + r, c0 + c] = bg
    return damaged_br, damaged_bc


def _gen_attempt() -> np.ndarray | None:
    tile_h, tile_w = _sample_tile_size()
    block_counts = _sample_block_counts(tile_h, tile_w)
    if block_counts is None:
        return None
    block_rows, block_cols = block_counts

    area = tile_h * tile_w
    non_bg_count = _sample_non_bg_count(area)
    max_color_count = min(MAX_MOTIF_COLORS, non_bg_count)
    color_count = random.randint(1, max_color_count)

    bg = random.choice(INPUT_COLORS)
    motif_palette = random.sample([color for color in INPUT_COLORS if color != bg], color_count)
    separator_choices = [color for color in INPUT_COLORS if color not in {bg, *motif_palette}]
    if not separator_choices:
        return None
    separator = random.choice(separator_choices)

    motif = _build_motif(tile_h, tile_w, bg, motif_palette, non_bg_count)
    if motif is None:
        return None

    damage_cells = _pick_damage_cells(motif, bg)
    if not damage_cells:
        return None

    out = motif.copy()
    for r, c in damage_cells:
        out[r, c] = BLUE

    inp = _tile_grid(motif, block_rows, block_cols, separator)
    _apply_damage(inp, motif, block_rows, block_cols, bg, damage_cells)

    if np.any(inp == BLUE):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {RETRY_LIMIT} attempts")
