from collections import Counter

import numpy as np

from arc_utils import Color, get_bg


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty set of copied cells"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique consensus color"
    return int(ranked[0][0])


def _solid_lines(g: np.ndarray, axis: int) -> list[tuple[int, int]]:
    assert axis in (0, 1)
    lines: list[tuple[int, int]] = []
    limit = g.shape[axis]
    for idx in range(limit):
        line = g[idx, :] if axis == 0 else g[:, idx]
        color = int(line[0])
        if np.all(line == color):
            lines.append((idx, color))
    return lines


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*separators, limit]:
        assert sep > start, "separator lines must leave nonempty content bands"
        bands.append((start, sep))
        start = sep + 1
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

    Input structure:
    - The input is a lattice of equal-size blocks separated by one-cell solid rows
      and columns of the same separator color.
    - All blocks are copies of one motif except exactly one block, where some
      motif pixels have been overwritten by the motif background color.
    - Blue is absent from the input and is reserved as the output marker color.

    Rule:
    - Recover the motif by taking the unique majority color at each local block
      coordinate across all block copies.
    - Find the single block that differs from this consensus motif.
    - Return the consensus motif, recoloring the differing local coordinates blue.

    Key insight:
    - The repeated copies provide redundancy: the damaged positions are exactly
      the coordinates where the lone nonconforming copy has background instead of
      the consensus motif color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.BLUE), "blue must be absent from the input"

    solid_rows = _solid_lines(g, axis=0)
    solid_cols = _solid_lines(g, axis=1)
    assert solid_rows and solid_cols, "expected solid separator rows and columns"

    separator_colors = {color for _, color in solid_rows} | {color for _, color in solid_cols}
    assert len(separator_colors) == 1, "all separator lines must have the same color"
    separator = next(iter(separator_colors))

    row_bands = _content_bands([idx for idx, _ in solid_rows], g.shape[0])
    col_bands = _content_bands([idx for idx, _ in solid_cols], g.shape[1])
    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "content blocks must have equal size"

    block_h = heights.pop()
    block_w = widths.pop()
    block_rows = len(row_bands)
    block_cols = len(col_bands)
    assert block_rows * block_cols >= 3, "expected enough repeated blocks for consensus"

    blocks = np.empty((block_rows, block_cols, block_h, block_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            blocks[br, bc] = g[r0:r1, c0:c1]

    assert not np.any(blocks == separator), "separator color should not appear inside content blocks"
    block_bg = get_bg(blocks)

    motif = np.empty((block_h, block_w), dtype=int)
    for rr in range(block_h):
        for cc in range(block_w):
            motif[rr, cc] = _unique_mode(blocks[:, :, rr, cc])

    diffs = blocks != motif
    diff_counts = diffs.reshape(block_rows, block_cols, block_h * block_w).sum(axis=2)
    damaged_blocks = np.argwhere(diff_counts > 0)
    assert damaged_blocks.shape[0] == 1, "expected exactly one damaged copy of the motif"

    damaged_r, damaged_c = map(int, damaged_blocks[0])
    missing_mask = diffs[damaged_r, damaged_c]
    assert np.any(missing_mask), "damaged copy must contain at least one missing pixel"
    assert np.all(blocks[damaged_r, damaged_c][missing_mask] == block_bg), "damage must be overwritten with background"
    assert np.all(motif[missing_mask] != block_bg), "only non-background motif pixels should be marked"

    out = motif.copy()
    out[missing_mask] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

GEN_BG_COLOR = 8
GEN_PALETTE = (0, 2, 3, 4, 5, 6, 7, 9)

# Motif size, color count, and density bounds.
GEN_TILE_SIZE_MIN, GEN_TILE_SIZE_MAX = 3, 7
GEN_MOTIF_COLOR_MIN, GEN_MOTIF_COLOR_MAX = 2, 4
GEN_NON_BG_MIN, GEN_NON_BG_MAX = 6, 10

# Damage and lattice constraints.
GEN_DAMAGE_MIN, GEN_DAMAGE_MAX = 2, 4
GEN_MIN_INTACT_NON_BG = 1
GEN_BLOCK_COUNT_MIN, GEN_BLOCK_COUNT_MAX = 3, 6
GEN_SEPARATOR_WIDTH = 1
GEN_MAX_GRID_SIDE = 30

GEN_RETRY_LIMIT = 300


def _gen_valid_non_bg_counts(size: int, n_colors: int) -> list[int]:
    area = size * size
    valid = []
    for count in range(GEN_NON_BG_MIN, min(GEN_NON_BG_MAX, area - GEN_MIN_INTACT_NON_BG) + 1):
        bg_count = area - count
        if count >= n_colors and (count + n_colors - 1) // n_colors <= bg_count:
            valid.append(count)
    return valid


def _gen_balanced_counts(total: int, n_parts: int, cap: int) -> list[int]:
    counts = [1] * n_parts
    remaining = total - n_parts
    while remaining:
        choices = [i for i, count in enumerate(counts) if count < cap]
        idx = int(np.random.choice(choices))
        counts[idx] += 1
        remaining -= 1
    return counts


def _gen_build_motif(size: int) -> np.ndarray:
    n_colors = int(np.random.randint(GEN_MOTIF_COLOR_MIN, GEN_MOTIF_COLOR_MAX + 1))
    counts_by_size = _gen_valid_non_bg_counts(size, n_colors)
    non_bg_count = int(np.random.choice(counts_by_size))
    motif_colors = list(np.random.choice(GEN_PALETTE, size=n_colors, replace=False))
    motif = np.full((size, size), GEN_BG_COLOR, dtype=int)
    flat_cells = np.random.choice(size * size, size=non_bg_count, replace=False)
    color_counts = _gen_balanced_counts(non_bg_count, n_colors, size * size - non_bg_count)
    values = []
    for color, count in zip(motif_colors, color_counts):
        values.extend([int(color)] * count)
    np.random.shuffle(values)
    rows, cols = np.divmod(flat_cells, size)
    motif[rows, cols] = values
    return motif


def _gen_separator(motif: np.ndarray) -> int:
    motif_colors = set(map(int, motif[motif != GEN_BG_COLOR]))
    choices = [color for color in GEN_PALETTE if color not in motif_colors]
    return int(np.random.choice(choices))


def _gen_damage_cells(motif: np.ndarray) -> np.ndarray:
    non_bg_cells = np.argwhere(motif != GEN_BG_COLOR)
    max_damage = min(GEN_DAMAGE_MAX, len(non_bg_cells) - GEN_MIN_INTACT_NON_BG)
    damage_count = int(np.random.randint(GEN_DAMAGE_MIN, max_damage + 1))
    damage_ix = np.random.choice(len(non_bg_cells), size=damage_count, replace=False)
    return non_bg_cells[damage_ix]


def _gen_block_count(size: int) -> int:
    max_count = min(GEN_BLOCK_COUNT_MAX, GEN_MAX_GRID_SIDE // (size + GEN_SEPARATOR_WIDTH))
    return int(np.random.randint(GEN_BLOCK_COUNT_MIN, max_count + 1))


def _gen_tile_lattice(motif: np.ndarray, block_rows: int, block_cols: int, separator: int) -> np.ndarray:
    tile_h, tile_w = motif.shape
    rows = block_rows * tile_h + (block_rows - 1) * GEN_SEPARATOR_WIDTH
    cols = block_cols * tile_w + (block_cols - 1) * GEN_SEPARATOR_WIDTH
    grid = np.full((rows, cols), separator, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (tile_h + GEN_SEPARATOR_WIDTH)
            c0 = bc * (tile_w + GEN_SEPARATOR_WIDTH)
            grid[r0:r0 + tile_h, c0:c0 + tile_w] = motif
    return grid


def _gen_apply_damage(inp: np.ndarray, size: int, block_rows: int, block_cols: int, damage_cells: np.ndarray) -> None:
    damaged_br = int(np.random.randint(block_rows))
    damaged_bc = int(np.random.randint(block_cols))
    base_r = damaged_br * (size + GEN_SEPARATOR_WIDTH)
    base_c = damaged_bc * (size + GEN_SEPARATOR_WIDTH)
    inp[base_r + damage_cells[:, 0], base_c + damage_cells[:, 1]] = GEN_BG_COLOR


def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.randint(GEN_TILE_SIZE_MIN, GEN_TILE_SIZE_MAX + 1))
    motif = _gen_build_motif(size)
    separator = _gen_separator(motif)
    damage_cells = _gen_damage_cells(motif)
    out = motif.copy()
    out[damage_cells[:, 0], damage_cells[:, 1]] = Color.BLUE
    block_rows = _gen_block_count(size)
    block_cols = _gen_block_count(size)
    inp = _gen_tile_lattice(motif, block_rows, block_cols, separator)
    _gen_apply_damage(inp, size, block_rows, block_cols, damage_cells)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
