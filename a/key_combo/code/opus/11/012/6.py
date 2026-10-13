from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

RETRY_LIMIT = 500


def _separator_lines(grid: np.ndarray, axis: int) -> list[int]:
    assert axis in (0, 1)
    indices: list[int] = []
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, 1:-1] if axis == 0 else grid[1:-1, idx]
        if line.size > 0 and np.all(line == GREY):
            indices.append(idx)
    return indices


def _content_bands(separators: list[int]) -> list[tuple[int, int]]:
    return [(separators[i] + 1, separators[i + 1]) for i in range(len(separators) - 1)]


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _extract_blocks(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    sep_rows = _separator_lines(grid, axis=0)
    sep_cols = _separator_lines(grid, axis=1)
    assert len(sep_rows) == 4 and len(sep_cols) == 4, "expected a 3x3 lattice with outer separator lines"

    row_bands = _content_bands(sep_rows)
    col_bands = _content_bands(sep_cols)
    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all tile bands must have equal size"

    tile_h = heights.pop()
    tile_w = widths.pop()
    blocks = np.empty((3, 3, tile_h, tile_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            blocks[br, bc] = grid[r0:r1, c0:c1]
    return blocks, row_bands, col_bands, sep_rows, sep_cols


def _consensus_block(blocks: np.ndarray) -> np.ndarray:
    tile_h, tile_w = blocks.shape[2:]
    motif = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            motif[r, c] = _mode(blocks[:, :, r, c])
    return motif


def _indicator_side(
    indicator: tuple[int, int],
    damaged_block: tuple[int, int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> str:
    ir, ic = indicator
    damaged_br, damaged_bc = damaged_block

    if ir < sep_rows[0]:
        c0, c1 = col_bands[damaged_bc]
        assert damaged_br == 0 and c0 <= ic < c1
        return "top"
    if ir > sep_rows[-1]:
        c0, c1 = col_bands[damaged_bc]
        assert damaged_br == 2 and c0 <= ic < c1
        return "bottom"
    if ic < sep_cols[0]:
        r0, r1 = row_bands[damaged_br]
        assert damaged_bc == 0 and r0 <= ir < r1
        return "left"
    if ic > sep_cols[-1]:
        r0, r1 = row_bands[damaged_br]
        assert damaged_bc == 2 and r0 <= ir < r1
        return "right"
    raise AssertionError("indicator must sit outside the separator frame")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the repeated motif, mirror it by the indicated axis, and use that mirror to refill only the damaged cells.

    Input structure:
    - The input is a 3 by 3 lattice of equal-sized tiles, bounded and separated by
      one-cell grey lines on a black outer frame.
    - Eight tiles are identical red/blue-on-black motif copies, while one edge tile
      is damaged by replacing some motif pixels with black.
    - A single green indicator pixel sits on the outer frame next to the damaged
      tile, above/below/left/right of the separator border around that tile.

    Rule:
    - Recover the consensus motif by taking the unique majority color at each local
      coordinate across the 9 tiles.
    - Locate the unique damaged tile and record the coordinates where it differs
      from the consensus motif.
    - If the green indicator is above or below the damaged tile, vertically flip
      the consensus motif; if it is left or right of the damaged tile, horizontally
      flip the consensus motif.
    - Return the consensus motif, except at the damaged coordinates where the colors
      are replaced by the corresponding colors from the mirrored motif.

    Key insight:
    - The repeated copies reveal both the undamaged consensus tile and the damaged
      coordinates, but the actual refill colors come from a mirror chosen by the
      indicator's relative position rather than from the consensus tile directly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    indicator_cells = np.argwhere(g == GREEN)
    assert indicator_cells.shape[0] == 1, "expected a single green indicator"
    indicator = tuple(map(int, indicator_cells[0]))

    blocks, row_bands, col_bands, sep_rows, sep_cols = _extract_blocks(g)
    motif = _consensus_block(blocks)

    diffs = blocks != motif
    diff_counts = diffs.reshape(3, 3, -1).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape[0] == 1, "expected exactly one damaged tile"
    damaged_br, damaged_bc = map(int, damaged[0])
    damaged_mask = diffs[damaged_br, damaged_bc]
    assert damaged_mask.any(), "damaged tile must differ from the consensus"

    side = _indicator_side(indicator, (damaged_br, damaged_bc), row_bands, col_bands, sep_rows, sep_cols)
    mirrored = np.flipud(motif) if side in ("top", "bottom") else np.fliplr(motif)

    out = motif.copy()
    out[damaged_mask] = mirrored[damaged_mask]
    return out


def _separator_positions(tile_size: int) -> list[int]:
    return [1, tile_size + 2, 2 * tile_size + 3, 3 * tile_size + 4]


def _make_canvas(tile_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    side = 3 * tile_size + 6
    grid = np.full((side, side), BLACK, dtype=int)
    sep_rows = _separator_positions(tile_size)
    sep_cols = _separator_positions(tile_size)
    grid[sep_rows, 1:-1] = GREY
    grid[1:-1, sep_cols] = GREY
    row_bands = _content_bands(sep_rows)
    col_bands = _content_bands(sep_cols)
    return grid, row_bands, col_bands


def _random_motif(tile_size: int) -> np.ndarray:
    area = tile_size * tile_size
    min_fg = max(4, area // 4)
    max_fg = min(area - 2, max(min_fg, area // 2))
    fg_count = random.randint(min_fg, max_fg)
    motif = np.full((tile_size, tile_size), BLACK, dtype=int)

    chosen = random.sample(range(area), fg_count)
    colors = [RED, BLUE]
    values = [RED, BLUE]
    for _ in range(fg_count - 2):
        values.append(random.choice(colors))
    random.shuffle(values)

    for flat_idx, color in zip(chosen, values):
        r, c = divmod(flat_idx, tile_size)
        motif[r, c] = color
    return motif


def _repair_candidates(motif: np.ndarray, axis: str) -> list[tuple[int, int]]:
    mirrored = np.flipud(motif) if axis == "ud" else np.fliplr(motif)
    coords: list[tuple[int, int]] = []
    for r in range(motif.shape[0]):
        for c in range(motif.shape[1]):
            if motif[r, c] != BLACK and mirrored[r, c] != BLACK and mirrored[r, c] != motif[r, c]:
                coords.append((r, c))
    return coords


def _valid_motif(motif: np.ndarray) -> bool:
    if set(np.unique(motif)) != {BLACK, BLUE, RED}:
        return False
    if np.array_equal(motif, np.flipud(motif)) or np.array_equal(motif, np.fliplr(motif)):
        return False
    if len(_repair_candidates(motif, "ud")) < 2:
        return False
    if len(_repair_candidates(motif, "lr")) < 2:
        return False
    return True


def _generate_motif(tile_size: int) -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        motif = _random_motif(tile_size)
        if _valid_motif(motif):
            return motif
    raise RuntimeError("failed to generate a non-degenerate motif")


def _expected_output(motif: np.ndarray, damage_cells: list[tuple[int, int]], side: str) -> np.ndarray:
    mirrored = np.flipud(motif) if side in ("top", "bottom") else np.fliplr(motif)
    out = motif.copy()
    for r, c in damage_cells:
        out[r, c] = mirrored[r, c]
    return out


def _place_indicator(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    damaged_block: tuple[int, int],
    side: str,
) -> None:
    damaged_br, damaged_bc = damaged_block
    if side == "top":
        c0, c1 = col_bands[damaged_bc]
        grid[0, random.randrange(c0, c1)] = GREEN
    elif side == "bottom":
        c0, c1 = col_bands[damaged_bc]
        grid[-1, random.randrange(c0, c1)] = GREEN
    elif side == "left":
        r0, r1 = row_bands[damaged_br]
        grid[random.randrange(r0, r1), 0] = GREEN
    else:
        r0, r1 = row_bands[damaged_br]
        grid[random.randrange(r0, r1), -1] = GREEN


def _attempt_generate() -> np.ndarray | None:
    tile_size = random.randint(3, 7)
    motif = _generate_motif(tile_size)
    side = random.choice(["top", "bottom", "left", "right"])
    axis = "ud" if side in ("top", "bottom") else "lr"
    candidates = _repair_candidates(motif, axis)

    non_black_count = int(np.count_nonzero(motif))
    max_damage = min(4, len(candidates), non_black_count - 1)
    if max_damage < 2:
        return None
    damage_count = random.randint(2, max_damage)
    damage_cells = random.sample(candidates, damage_count)

    grid, row_bands, col_bands = _make_canvas(tile_size)
    if side == "top":
        damaged_block = (0, random.randrange(3))
    elif side == "bottom":
        damaged_block = (2, random.randrange(3))
    elif side == "left":
        damaged_block = (random.randrange(3), 0)
    else:
        damaged_block = (random.randrange(3), 2)

    expected = _expected_output(motif, damage_cells, side)
    if np.array_equal(expected, motif):
        return None

    for br in range(3):
        for bc in range(3):
            r0, r1 = row_bands[br]
            c0, c1 = col_bands[bc]
            tile = motif.copy()
            if (br, bc) == damaged_block:
                for rr, cc in damage_cells:
                    tile[rr, cc] = BLACK
            grid[r0:r1, c0:c1] = tile

    _place_indicator(grid, row_bands, col_bands, damaged_block, side)

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated retries")
