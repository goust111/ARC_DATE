import numpy as np


BLUE = 1
PALETTE = tuple(range(10))
GEN_RETRIES = 300


def _solid_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    length = grid.shape[1 - axis]
    for idx in range(grid.shape[axis]):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color) and len(np.unique(line)) == 1 and line.size == length:
            lines.append((int(idx), color))
    return lines


def _content_bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    seps = sorted(set(int(x) for x in separators))
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in seps:
        if start < idx:
            bands.append((start, idx))
        start = idx + 1
    if start < size:
        bands.append((start, size))
    return bands


def _majority_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    assert len(order) >= 1
    if len(order) > 1:
        assert counts[order[0]] > counts[order[1]], "expected a unique majority color"
    return int(colors[order[0]])


def _infer_background(blocks: np.ndarray) -> int:
    return _majority_color(blocks)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == BLUE), "blue must be absent from the input"

    solid_rows = _solid_lines(g, axis=0)
    solid_cols = _solid_lines(g, axis=1)
    assert solid_rows and solid_cols, "expected solid separator rows and columns"

    sep_colors = {color for _, color in solid_rows} | {color for _, color in solid_cols}
    assert len(sep_colors) == 1, "separator rows and columns must share one color"
    separator = next(iter(sep_colors))

    row_bands = _content_bands([idx for idx, _ in solid_rows], g.shape[0])
    col_bands = _content_bands([idx for idx, _ in solid_cols], g.shape[1])
    assert len(row_bands) >= 1 and len(col_bands) >= 1

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all blocks must have equal size"

    block_h = heights.pop()
    block_w = widths.pop()
    block_rows = len(row_bands)
    block_cols = len(col_bands)
    block_count = block_rows * block_cols
    assert block_count >= 3, "need at least three blocks for consensus"

    blocks = np.empty((block_rows, block_cols, block_h, block_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            assert block.shape == (block_h, block_w)
            assert not np.any(block == separator), "separator color cannot appear inside blocks"
            blocks[br, bc] = block

    motif = np.empty((block_h, block_w), dtype=int)
    for r in range(block_h):
        for c in range(block_w):
            motif[r, c] = _majority_color(blocks[:, :, r, c])

    bg = _infer_background(blocks)
    diff_masks = blocks != motif
    diff_counts = diff_masks.reshape(block_rows, block_cols, -1).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape[0] == 1, "expected exactly one damaged block"
    dr, dc = map(int, damaged[0])

    missing_mask = diff_masks[dr, dc]
    assert np.any(missing_mask), "damaged block must differ from consensus"
    damaged_block = blocks[dr, dc]
    assert np.all(damaged_block[missing_mask] == bg), "damage must overwrite pixels with background"
    assert np.all(motif[missing_mask] != bg), "only missing motif pixels should be marked"

    out = motif.copy()
    out[missing_mask] = BLUE
    return out


def _random_palette(n: int) -> list[int]:
    colors = [c for c in PALETTE if c != BLUE]
    perm = np.random.permutation(colors)
    return [int(x) for x in perm[:n]]


def _sample_motif(height: int, width: int, bg: int, fg_colors: list[int]) -> np.ndarray | None:
    motif = np.full((height, width), bg, dtype=int)
    cell_count = height * width
    min_non_bg = max(4, cell_count // 4)
    max_non_bg = max(min_non_bg, cell_count - 2)
    if min_non_bg > max_non_bg:
        return None
    target = int(np.random.randint(min_non_bg, max_non_bg + 1))
    positions = np.random.permutation(cell_count)[:target]
    splits = np.random.multinomial(target, [1.0 / len(fg_colors)] * len(fg_colors))
    if np.any(splits == 0):
        return None

    pos_idx = 0
    for color, count in zip(fg_colors, splits):
        for flat in positions[pos_idx : pos_idx + count]:
            r = int(flat // width)
            c = int(flat % width)
            motif[r, c] = int(color)
        pos_idx += int(count)

    rows_with_fg = np.any(motif != bg, axis=1).sum()
    cols_with_fg = np.any(motif != bg, axis=0).sum()
    if rows_with_fg < 2 or cols_with_fg < 2:
        return None
    return motif


def _build_grid(blocks: np.ndarray, separator: int) -> np.ndarray:
    block_rows, block_cols, block_h, block_w = blocks.shape
    height = block_rows * block_h + (block_rows + 1)
    width = block_cols * block_w + (block_cols + 1)
    grid = np.full((height, width), separator, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = 1 + br * (block_h + 1)
            c0 = 1 + bc * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = blocks[br, bc]
    return grid


def _gen_attempt() -> np.ndarray | None:
    block_h = int(np.random.randint(3, 8))
    block_w = int(np.random.randint(3, 8))
    block_rows = int(np.random.randint(1, 5))
    block_cols = int(np.random.randint(2, 5))
    if block_rows * block_cols < 3:
        return None

    palette = _random_palette(int(np.random.randint(4, 7)))
    separator = palette[0]
    bg = palette[1]
    fg_count = int(np.random.randint(2, min(4, len(palette) - 2) + 1))
    fg_colors = palette[2 : 2 + fg_count]

    motif = _sample_motif(block_h, block_w, bg, fg_colors)
    if motif is None:
        return None

    blocks = np.repeat(motif[None, None, :, :], block_rows, axis=0)
    blocks = np.repeat(blocks, block_cols, axis=1)

    fg_positions = np.argwhere(motif != bg)
    if fg_positions.shape[0] < 2:
        return None

    damaged_index = int(np.random.randint(block_rows * block_cols))
    dr = damaged_index // block_cols
    dc = damaged_index % block_cols

    max_damage = min(5, fg_positions.shape[0] - 1)
    if max_damage < 1:
        return None
    damage_count = int(np.random.randint(1, max_damage + 1))
    chosen = fg_positions[np.random.permutation(len(fg_positions))[:damage_count]]
    for r, c in chosen:
        blocks[dr, dc, int(r), int(c)] = bg

    grid = _build_grid(blocks, separator)
    expected = motif.copy()
    missing_mask = np.zeros_like(motif, dtype=bool)
    missing_mask[chosen[:, 0], chosen[:, 1]] = True
    expected[missing_mask] = BLUE

    if np.any(grid == BLUE):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
