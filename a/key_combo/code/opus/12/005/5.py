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

PALETTE_POOL = [BLUE, RED, GREEN, YELLOW]
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_problem(grid: np.ndarray) -> tuple[list[int], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert separator_cols.size == 1, "expected a single full-height grey separator"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0

    bar_rows = np.flatnonzero(np.any((left != BLACK), axis=1))
    assert bar_rows.size in (3, 4), "expected 3 or 4 palette bars"
    assert np.all(np.diff(bar_rows) == 1), "palette bars must be vertically contiguous"

    palette: list[int] = []
    bar_width: int | None = None
    bar_slice: tuple[int, int] | None = None
    for r in bar_rows:
        cols = np.flatnonzero(left[r] != BLACK)
        assert cols.size > 0
        assert np.all(np.diff(cols) == 1), "each palette bar must be a solid run"
        colors = sorted(set(map(int, left[r, cols])))
        assert len(colors) == 1 and colors[0] != GREY
        palette.append(colors[0])
        if bar_width is None:
            bar_width = int(cols.size)
            bar_slice = (int(cols[0]), int(cols[-1]) + 1)
        else:
            assert int(cols.size) == bar_width
            assert bar_slice == (int(cols[0]), int(cols[-1]) + 1)

    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert bar_width is not None

    fg_bbox = _bbox(right == ORANGE)
    assert fg_bbox is not None, "expected an orange mask on the right"
    r0, c0, r1, c1 = fg_bbox
    n = len(palette)
    assert r1 - r0 + 1 == n and c1 - c0 + 1 == n, "mask bbox must be N x N"

    mask_patch = right[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((mask_patch == BLACK) | (mask_patch == ORANGE)), "mask must be binary"
    mask = mask_patch == ORANGE
    assert np.all(mask.any(axis=1)) and np.all(mask.any(axis=0)), "mask bbox must be tight"
    assert bar_width == n, "self-embedded mask blocks require bar width == mask size"
    return palette, mask, bar_width


def _render_output(palette: list[int], mask: np.ndarray, block_size: int) -> np.ndarray:
    n = len(palette)
    assert mask.shape == (n, n)
    assert block_size == n

    out = np.zeros((n * block_size, n * block_size), dtype=int)
    reversed_palette = list(reversed(palette))
    for i in range(n):
        color = reversed_palette[i]
        pattern_block = np.where(mask, color, BLACK).astype(int)
        for j in range(n):
            rr = slice(i * block_size, (i + 1) * block_size)
            cc = slice(j * block_size, (j + 1) * block_size)
            if mask[i, j]:
                out[rr, cc] = color
            else:
                out[rr, cc] = pattern_block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a binary mask into colored blocks using the reversed palette from the stacked bars.

    Input structure:
    - The grid has black background and one full-height grey separator column.
    - On the left of the separator, 3 or 4 one-row horizontal color bars are stacked contiguously top-to-bottom.
    - On the right of the separator, an orange-on-black N x N binary mask appears, where N is the number of bars.

    Rule:
    - Read the left bars from top to bottom to get palette P, then reverse it to P'.
    - For each mask cell (i, j), create an N x N output block colored by row index i.
    - If the mask cell is orange, the whole block becomes the solid color P'[i]; if it is black, the block becomes a self-copy of the mask, with orange positions recolored to P'[i] and black positions kept black.

    Key insight:
    - The separator splits two independent clues: the left stack gives the row-color order, and the right mask controls both the macro block layout and the micro pattern inside every background block.
    """
    palette, mask, block_size = _extract_problem(grid)
    return _render_output(palette, mask, block_size)


def _mask_is_rich(mask: np.ndarray) -> bool:
    n = mask.shape[0]
    if mask.shape != (n, n):
        return False
    fg = int(mask.sum())
    if fg < n or fg >= n * n:
        return False
    if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
        return False
    if not np.all((~mask).any(axis=1)) or not np.all((~mask).any(axis=0)):
        return False
    for k in (1, 2, 3):
        if np.array_equal(mask, np.rot90(mask, k)):
            return False
    return True


def _sample_mask(n: int) -> np.ndarray:
    for _ in range(200):
        mask = np.random.randint(0, 2, size=(n, n), dtype=int).astype(bool)
        if _mask_is_rich(mask):
            return mask
    raise RuntimeError("failed to sample a non-degenerate mask")


def _build_input(
    palette: list[int],
    mask: np.ndarray,
    bar_top: int,
    bar_left: int,
    mask_top: int,
    mask_left: int,
    height: int,
    width: int,
    sep_col: int,
) -> np.ndarray:
    n = len(palette)
    grid = np.zeros((height, width), dtype=int)
    grid[:, sep_col] = GREY
    for i, color in enumerate(palette):
        grid[bar_top + i, bar_left : bar_left + n] = color
    grid[mask_top : mask_top + n, mask_left : mask_left + n] = np.where(mask, ORANGE, BLACK)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    palette = random.sample(PALETTE_POOL, n)
    mask = _sample_mask(n)

    bar_top = random.randint(1, 3)
    mask_top = random.randint(1, 3)
    bottom_margin = random.randint(1, 2)
    height = max(bar_top + n, mask_top + n) + bottom_margin

    bar_left = random.randint(1, 2)
    gap_before_sep = random.randint(0, 2)
    sep_col = bar_left + n + gap_before_sep
    gap_after_sep = random.randint(0, 2)
    mask_left = sep_col + 1 + gap_after_sep
    right_margin = random.randint(1, 2)
    width = mask_left + n + right_margin

    grid = _build_input(
        palette=palette,
        mask=mask,
        bar_top=bar_top,
        bar_left=bar_left,
        mask_top=mask_top,
        mask_left=mask_left,
        height=height,
        width=width,
        sep_col=sep_col,
    )

    parsed_palette, parsed_mask, parsed_block = _extract_problem(grid)
    if parsed_palette != palette:
        return None
    if parsed_block != n or not np.array_equal(parsed_mask, mask):
        return None

    out = _render_output(palette, mask, n)
    if out.shape != (n * n, n * n):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    row_colors = list(reversed(palette))
    for i, color in enumerate(row_colors):
        if not np.any(out[i * n : (i + 1) * n] == color):
            return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
