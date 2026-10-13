from __future__ import annotations

import random

import numpy as np


BLUE = 1
INPUT_PALETTE = [0, 2, 3, 4, 5, 6, 7, 8, 9]

SEPARATOR_WIDTH = 1
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

BLOCK_H_RANGE = (3, 7)
BLOCK_W_RANGE = (3, 7)
BLOCK_ROWS_MAX = 4
BLOCK_COLS_MAX = 5

MOTIF_COLOR_COUNT_RANGE = (1, 3)
FOREGROUND_FRACTION_RANGE = (0.18, 0.42)
DAMAGE_COUNT_CAP = 5

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _unique_mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values.reshape(-1), return_counts=True)
    order = np.argsort(-counts)
    assert len(order) == 1 or int(counts[order[0]]) > int(counts[order[1]])
    return int(colors[order[0]])


def _separator_structure(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(g)):
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color with full rows and columns"
    return candidates[0]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(separators)
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*separators, length]:
        assert sep > start, "separator lines must leave nonempty blocks"
        bands.append((start, sep))
        start = sep + 1
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

    Input structure:
    - The grid is partitioned by one-cell solid separator rows and columns into equal blocks.
    - Every block is a copy of one motif except a single damaged block, where some motif pixels were overwritten
      with the block background color.
    - Blue does not appear in the input.

    Rule:
    - Recover the consensus motif by taking the unique majority color at each local block coordinate.
    - Find the unique block that differs from that consensus.
    - Return the consensus motif with the differing coordinates recolored blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert not np.any(g == BLUE), "blue must be absent from the input"

    _separator, row_separators, col_separators = _separator_structure(g)
    row_bands = _bands(g.shape[0], row_separators)
    col_bands = _bands(g.shape[1], col_separators)

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all blocks must have equal size"

    block_h = heights.pop()
    block_w = widths.pop()
    assert block_h > 0 and block_w > 0

    blocks: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            blocks.append(g[r0:r1, c0:c1].copy())
    assert len(blocks) >= 3, "need at least three copies for majority consensus"

    stack = np.stack(blocks, axis=0)
    motif = np.empty((block_h, block_w), dtype=int)
    for r in range(block_h):
        for c in range(block_w):
            motif[r, c] = _unique_mode(stack[:, r, c])

    diffs = stack != motif
    diff_counts = diffs.reshape(len(blocks), -1).sum(axis=1)
    damaged = np.flatnonzero(diff_counts > 0)
    assert len(damaged) == 1, "expected exactly one damaged block"

    missing_mask = diffs[int(damaged[0])]
    assert np.any(missing_mask), "damaged block must differ from the consensus"

    out = motif.copy()
    out[missing_mask] = BLUE
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    counts = [1] * parts
    for _ in range(total - parts):
        counts[random.randrange(parts)] += 1
    random.shuffle(counts)
    return counts


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _grow_blob(canvas: np.ndarray, color: int, size: int) -> bool:
    height, width = map(int, canvas.shape)
    empty = np.argwhere(canvas < 0)
    if len(empty) < size:
        return False

    seed_r, seed_c = map(int, empty[random.randrange(len(empty))])
    cells = {(seed_r, seed_c)}
    frontier = set(_neighbors4(seed_r, seed_c, height, width))

    while len(cells) < size:
        choices = [(r, c) for r, c in frontier if int(canvas[r, c]) < 0]
        if not choices:
            return False
        r, c = random.choice(choices)
        cells.add((r, c))
        frontier.discard((r, c))
        for rr, cc in _neighbors4(r, c, height, width):
            if int(canvas[rr, cc]) < 0 and (rr, cc) not in cells:
                frontier.add((rr, cc))

    for r, c in cells:
        canvas[r, c] = color
    return True


def _build_motif(height: int, width: int, bg: int, colors: list[int]) -> np.ndarray | None:
    area = height * width
    min_fg = max(len(colors), int(np.ceil(area * FOREGROUND_FRACTION_RANGE[0])))
    max_fg = min(area - 1, int(np.floor(area * FOREGROUND_FRACTION_RANGE[1])))
    if min_fg > max_fg:
        return None

    fg_count = random.randint(min_fg, max_fg)
    blob_sizes = _random_partition(fg_count, len(colors))
    motif = np.full((height, width), -1, dtype=int)

    for color, size in sorted(zip(colors, blob_sizes), key=lambda item: item[1], reverse=True):
        if not _grow_blob(motif, int(color), int(size)):
            return None

    motif[motif < 0] = bg
    return motif


def _choose_block_count(tile_size: int, hard_cap: int) -> int | None:
    max_count = min(hard_cap, (MAX_GRID_SIDE + SEPARATOR_WIDTH) // (tile_size + SEPARATOR_WIDTH))
    if max_count < 2:
        return None
    return random.randint(2, max_count)


def _render_lattice(motif: np.ndarray, block_rows: int, block_cols: int, separator: int) -> np.ndarray:
    tile_h, tile_w = map(int, motif.shape)
    height = block_rows * tile_h + (block_rows - 1) * SEPARATOR_WIDTH
    width = block_cols * tile_w + (block_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), separator, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (tile_h + SEPARATOR_WIDTH)
            c0 = bc * (tile_w + SEPARATOR_WIDTH)
            grid[r0:r0 + tile_h, c0:c0 + tile_w] = motif
    return grid


def _damage_block(inp: np.ndarray, motif: np.ndarray, block_rows: int, block_cols: int, bg: int) -> np.ndarray:
    non_bg = np.argwhere(motif != bg)
    assert len(non_bg) >= 2
    max_damage = min(DAMAGE_COUNT_CAP, len(non_bg) - 1)
    damage_count = random.randint(1, max_damage)
    damage_cells = non_bg[np.random.choice(len(non_bg), size=damage_count, replace=False)]

    tile_h, tile_w = map(int, motif.shape)
    damaged_br = random.randrange(block_rows)
    damaged_bc = random.randrange(block_cols)
    base_r = damaged_br * (tile_h + SEPARATOR_WIDTH)
    base_c = damaged_bc * (tile_w + SEPARATOR_WIDTH)

    out = motif.copy()
    out[damage_cells[:, 0], damage_cells[:, 1]] = BLUE

    inp[
        base_r + damage_cells[:, 0],
        base_c + damage_cells[:, 1],
    ] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(*BLOCK_H_RANGE)
    tile_w = random.randint(*BLOCK_W_RANGE)
    block_rows = _choose_block_count(tile_h, BLOCK_ROWS_MAX)
    block_cols = _choose_block_count(tile_w, BLOCK_COLS_MAX)
    if block_rows is None or block_cols is None:
        return None

    color_count = random.randint(*MOTIF_COLOR_COUNT_RANGE)
    chosen = random.sample(INPUT_PALETTE, 2 + color_count)
    bg, separator = chosen[:2]
    motif_colors = chosen[2:]

    motif = _build_motif(tile_h, tile_w, bg, motif_colors)
    if motif is None:
        return None
    if np.any(motif == separator):
        return None

    grid = _render_lattice(motif, block_rows, block_cols, separator)
    expected = _damage_block(grid, motif, block_rows, block_cols, bg)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
