import random
from collections import Counter

import numpy as np


BLUE = 1
GREY = 5
TEAL = 8
PALETTE = (0, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bands_from_separators(separators: list[int], size: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must be single-pixel gaps between content bands"
        bands.append((start, sep))
        start = sep + 1
    assert start < size, "grid must end with a content band"
    bands.append((start, size))
    return bands


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.ravel()))
    top = counts.most_common()
    assert len(top) >= 1
    if len(top) >= 2:
        assert top[0][1] > top[1][1], "per-cell consensus must be unique"
    return int(top[0][0])


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == BLUE), "blue is reserved for output markers"

    teal_rows = [r for r in range(g.shape[0]) if np.all(g[r] == TEAL)]
    teal_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == TEAL)]
    assert len(teal_rows) == 1 and len(teal_cols) == 1, "expected exactly one teal cross row and column"
    cross_row = teal_rows[0]
    cross_col = teal_cols[0]

    row_separators = sorted(
        [r for r in range(g.shape[0]) if r != cross_row and np.all(g[r, np.arange(g.shape[1]) != cross_col] == GREY)]
        + [cross_row]
    )
    col_separators = sorted(
        [c for c in range(g.shape[1]) if c != cross_col and np.all(g[np.arange(g.shape[0]) != cross_row, c] == GREY)]
        + [cross_col]
    )

    row_bands = _bands_from_separators(row_separators, g.shape[0])
    col_bands = _bands_from_separators(col_separators, g.shape[1])
    block_hs = {r1 - r0 for r0, r1 in row_bands}
    block_ws = {c1 - c0 for c0, c1 in col_bands}
    assert len(block_hs) == 1 and len(block_ws) == 1, "all blocks must share the same size"
    block_size = block_hs.pop()
    assert block_size == block_ws.pop(), "blocks must be square"

    block_rows = len(row_bands)
    block_cols = len(col_bands)
    assert block_rows >= 3 and block_cols >= 3, "the cross must split a block lattice into four quadrants"
    assert block_rows * block_cols >= 6, "need enough copies for majority voting"

    blocks = np.empty((block_rows, block_cols, block_size, block_size), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            assert block.shape == (block_size, block_size)
            assert not np.any(block == GREY), "separator color must stay out of blocks"
            assert not np.any(block == TEAL), "cross color must stay on separator lines"
            blocks[br, bc] = block

    cross_row_sep_index = row_separators.index(cross_row)
    cross_col_sep_index = col_separators.index(cross_col)
    return {
        "grid": g,
        "cross_row": cross_row,
        "cross_col": cross_col,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "block_size": block_size,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "blocks": blocks,
        "cross_row_sep_index": cross_row_sep_index,
        "cross_col_sep_index": cross_col_sep_index,
    }


def _consensus_and_damage(parsed: dict) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    blocks = parsed["blocks"]
    block_rows, block_cols, block_size, _ = blocks.shape
    motif = np.empty((block_size, block_size), dtype=int)
    for rr in range(block_size):
        for cc in range(block_size):
            motif[rr, cc] = _unique_mode(blocks[:, :, rr, cc])

    diffs = blocks != motif
    diff_counts = diffs.reshape(block_rows, block_cols, block_size * block_size).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape == (1, 2), "expected exactly one damaged block"
    damaged_block = tuple(map(int, damaged[0]))
    damage_mask = diffs[damaged_block]

    bg = get_bg(motif)
    assert np.any(damage_mask), "damage set must be non-empty"
    damaged_tile = blocks[damaged_block]
    assert np.all(damaged_tile[damage_mask] == bg), "damaged pixels must be overwritten by motif background"
    assert np.all(motif[damage_mask] != bg), "only non-background motif pixels may be damaged"
    assert np.any((motif == bg) & ~damage_mask), "motif should keep some genuine background pixels"
    assert np.any(damaged_tile != bg), "damaged copy must retain intact evidence of the motif"
    return motif, damaged_block, damage_mask


def _expected_output(parsed: dict, motif: np.ndarray, damaged_block: tuple[int, int], damage_mask: np.ndarray) -> np.ndarray:
    out = parsed["grid"].copy()
    br, bc = damaged_block
    r0, r1 = parsed["row_bands"][br]
    c0, c1 = parsed["col_bands"][bc]
    out[r0:r1, c0:c1] = motif

    for rr, cc in np.argwhere(damage_mask):
        src_r = r0 + int(rr)
        src_c = c0 + int(cc)
        dst_r = 2 * parsed["cross_row"] - src_r
        dst_c = 2 * parsed["cross_col"] - src_c
        if 0 <= dst_r < out.shape[0] and 0 <= dst_c < out.shape[1]:
            out[dst_r, dst_c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the lone damaged tile by majority vote and mark the point-reflected damage coordinates in blue.

    Input structure:
    - The grid is a lattice of equal square blocks separated by one-cell grey rows and columns.
    - Exactly one separator row and one separator column are replaced by a full teal cross, which acts as a coordinate axis.
    - Every block is a copy of one hidden motif except for a single damaged block where some non-background motif pixels were overwritten by the motif background.

    Rule:
    - Recover the shared motif by taking the unique majority color at each local block coordinate across all block copies.
    - Locate the single damaged block and the local coordinates where it differs from the recovered motif, then repair that block in place.
    - For each damaged absolute pixel coordinate, reflect it through the teal cross intersection with r' = 2*cross_row - r and c' = 2*cross_col - c, and color the reflected cell blue when it stays inside the grid.

    Key insight:
    - The repeated tiles diagnose which pixels are missing, while the teal cross is used only as a global point-reflection reference for where the blue markers must be placed.
    """
    parsed = _parse_instance(grid)
    motif, damaged_block, damage_mask = _consensus_and_damage(parsed)
    return _expected_output(parsed, motif, damaged_block, damage_mask)


def _random_dimensions() -> tuple[int, int, int]:
    candidates = []
    for block_size in range(3, 6):
        for block_rows in range(3, 6):
            for block_cols in range(3, 6):
                height = block_rows * block_size + (block_rows - 1)
                width = block_cols * block_size + (block_cols - 1)
                if height <= 30 and width <= 30:
                    candidates.append((block_size, block_rows, block_cols))
    return random.choice(candidates)


def _random_motif(block_size: int) -> np.ndarray | None:
    bg = random.choice(PALETTE)
    color_pool = [c for c in PALETTE if c != bg]
    color_count = random.randint(2, min(4, len(color_pool)))
    motif_colors = random.sample(color_pool, color_count)

    cell_count = block_size * block_size
    min_non_bg = max(5, block_size + 1, color_count + 1)
    max_non_bg = min(cell_count - 2, max(min_non_bg, int(round(cell_count * 0.58))))
    if min_non_bg > max_non_bg:
        return None

    non_bg_count = random.randint(min_non_bg, max_non_bg)
    coords = random.sample([(r, c) for r in range(block_size) for c in range(block_size)], non_bg_count)

    motif = np.full((block_size, block_size), bg, dtype=int)
    random.shuffle(coords)
    for color, coord in zip(motif_colors, coords[:color_count]):
        motif[coord] = color
    for coord in coords[color_count:]:
        motif[coord] = random.choice(motif_colors)

    if len(set(map(int, motif.ravel())) - {bg}) < 2:
        return None
    if not np.any(motif == bg):
        return None
    return motif


def _mirrored_block_index(index: int, sep_index: int) -> int:
    return 2 * sep_index + 1 - index


def _choose_damage_cells(
    motif: np.ndarray,
    mirrored_pref: str,
) -> list[tuple[int, int]] | None:
    bg = get_bg(motif)
    candidates = [tuple(map(int, p)) for p in np.argwhere(motif != bg)]
    max_damage = min(4, len(candidates) - 1)
    if max_damage < 1:
        return None

    for _ in range(80):
        damage_count = random.randint(1, max_damage)
        cells = random.sample(candidates, damage_count)
        reflected_values = [int(motif[motif.shape[0] - 1 - r, motif.shape[1] - 1 - c]) for r, c in cells]
        if mirrored_pref == "mixed" and not ({bg} & set(reflected_values) and (set(reflected_values) - {bg})):
            continue
        return cells
    if mirrored_pref == "mixed":
        return _choose_damage_cells(motif, "any")
    return None


def _build_grid(
    motif: np.ndarray,
    block_rows: int,
    block_cols: int,
    cross_row_sep_index: int,
    cross_col_sep_index: int,
    damaged_block: tuple[int, int],
    damage_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, dict]:
    block_size = motif.shape[0]
    height = block_rows * block_size + (block_rows - 1)
    width = block_cols * block_size + (block_cols - 1)
    bg = get_bg(motif)

    grid = np.full((height, width), bg, dtype=int)
    row_separators = [k * (block_size + 1) - 1 for k in range(1, block_rows)]
    col_separators = [k * (block_size + 1) - 1 for k in range(1, block_cols)]
    for r in row_separators:
        grid[r, :] = GREY
    for c in col_separators:
        grid[:, c] = GREY

    cross_row = row_separators[cross_row_sep_index]
    cross_col = col_separators[cross_col_sep_index]
    grid[cross_row, :] = TEAL
    grid[:, cross_col] = TEAL

    row_bands = _bands_from_separators(row_separators, height)
    col_bands = _bands_from_separators(col_separators, width)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            grid[r0:r1, c0:c1] = motif

    dbr, dbc = damaged_block
    r0, r1 = row_bands[dbr]
    c0, c1 = col_bands[dbc]
    damaged_tile = motif.copy()
    for rr, cc in damage_cells:
        damaged_tile[rr, cc] = bg
    grid[r0:r1, c0:c1] = damaged_tile

    truth = {
        "motif": motif,
        "damaged_block": damaged_block,
        "damage_cells": tuple(sorted(damage_cells)),
        "cross_row": cross_row,
        "cross_col": cross_col,
        "row_bands": row_bands,
        "col_bands": col_bands,
    }
    return grid, truth


def _validate_semantics(grid: np.ndarray, truth: dict) -> bool:
    try:
        parsed = _parse_instance(grid)
        motif, damaged_block, damage_mask = _consensus_and_damage(parsed)

        if damaged_block != truth["damaged_block"]:
            return False
        if not np.array_equal(motif, truth["motif"]):
            return False
        truth_mask = np.zeros_like(motif, dtype=bool)
        for rr, cc in truth["damage_cells"]:
            truth_mask[rr, cc] = True
        if not np.array_equal(damage_mask, truth_mask):
            return False

        src_br, src_bc = damaged_block
        dst_br = _mirrored_block_index(src_br, parsed["cross_row_sep_index"])
        dst_bc = _mirrored_block_index(src_bc, parsed["cross_col_sep_index"])
        if not (0 <= dst_br < parsed["block_rows"] and 0 <= dst_bc < parsed["block_cols"]):
            return False

        src_r0, _ = parsed["row_bands"][src_br]
        src_c0, _ = parsed["col_bands"][src_bc]
        blue_targets = []
        for rr, cc in np.argwhere(damage_mask):
            abs_r = src_r0 + int(rr)
            abs_c = src_c0 + int(cc)
            ref_r = 2 * parsed["cross_row"] - abs_r
            ref_c = 2 * parsed["cross_col"] - abs_c
            if not (0 <= ref_r < grid.shape[0] and 0 <= ref_c < grid.shape[1]):
                return False
            if ref_r == parsed["cross_row"] or ref_c == parsed["cross_col"]:
                return False
            if abs_r == parsed["cross_row"] or abs_c == parsed["cross_col"]:
                return False
            if not ((abs_r < parsed["cross_row"]) ^ (ref_r < parsed["cross_row"])):
                return False
            if not ((abs_c < parsed["cross_col"]) ^ (ref_c < parsed["cross_col"])):
                return False
            dst_br = next((i for i, (r0, r1) in enumerate(parsed["row_bands"]) if r0 <= ref_r < r1), None)
            dst_bc = next((i for i, (c0, c1) in enumerate(parsed["col_bands"]) if c0 <= ref_c < c1), None)
            if dst_br is None or dst_bc is None:
                return False
            if (dst_br, dst_bc) != (_mirrored_block_index(src_br, parsed["cross_row_sep_index"]), _mirrored_block_index(src_bc, parsed["cross_col_sep_index"])):
                return False
            blue_targets.append((ref_r, ref_c))

        if len(set(blue_targets)) != len(blue_targets):
            return False

        expected = _expected_output(parsed, motif, damaged_block, damage_mask)
        if not np.array_equal(main(grid), expected):
            return False
        if np.count_nonzero(expected == BLUE) != len(truth["damage_cells"]):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    block_size, block_rows, block_cols = _random_dimensions()
    motif = _random_motif(block_size)
    if motif is None:
        return None

    cross_row_sep_index = random.randrange(block_rows - 1)
    cross_col_sep_index = random.randrange(block_cols - 1)
    valid_blocks = []
    for br in range(block_rows):
        for bc in range(block_cols):
            ref_br = _mirrored_block_index(br, cross_row_sep_index)
            ref_bc = _mirrored_block_index(bc, cross_col_sep_index)
            if 0 <= ref_br < block_rows and 0 <= ref_bc < block_cols:
                valid_blocks.append((br, bc))
    if not valid_blocks:
        return None

    damaged_block = random.choice(valid_blocks)
    mirrored_pref = random.choice(["any", "mixed", "any"])
    damage_cells = _choose_damage_cells(motif, mirrored_pref)
    if damage_cells is None:
        return None

    grid, truth = _build_grid(
        motif,
        block_rows,
        block_cols,
        cross_row_sep_index,
        cross_col_sep_index,
        damaged_block,
        damage_cells,
    )
    if not _validate_semantics(grid, truth):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
