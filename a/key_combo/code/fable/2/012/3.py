from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
SIZE_CHOICES = (3, 4, 5)
FILL_COLOR_COUNT = 3
PATTERN_COEFFS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
MAX_GENERATION_ATTEMPTS = 300


def _find_guides(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square tile"
    assert h * w <= 30, "input tile must respect ARC size bounds"

    row_guides = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r] == g[r, 0])]
    col_guides = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c])]

    assert len(row_guides) == 1, "expected exactly one uniform guide row"
    assert len(col_guides) == 1, "expected exactly one uniform guide column"

    guide_row, row_color = row_guides[0]
    guide_col, col_color = col_guides[0]
    assert row_color == col_color, "guide row and guide column must share one color"
    return int(guide_row), int(guide_col), int(row_color)


def _row_block(tile: np.ndarray, anchor_col: int, block_col: int, mode: str) -> np.ndarray:
    if mode == "mirror":
        mirrored = abs(int(block_col) - int(anchor_col)) % 2 == 1
        return np.fliplr(tile) if mirrored else tile
    if mode == "copy":
        return tile
    raise ValueError(f"unsupported horizontal mode: {mode}")


def _col_block(tile: np.ndarray, anchor_row: int, block_row: int, mode: str) -> np.ndarray:
    if mode == "mirror":
        mirrored = abs(int(block_row) - int(anchor_row)) % 2 == 1
        return np.flipud(tile) if mirrored else tile
    if mode == "copy":
        return tile
    raise ValueError(f"unsupported vertical mode: {mode}")


def _expand_cross(
    tile: np.ndarray,
    block_row: int,
    block_col: int,
    *,
    horizontal_mode: str = "mirror",
    vertical_mode: str = "mirror",
) -> np.ndarray:
    n = int(tile.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)

    for bc in range(n):
        block = _row_block(tile, block_col, bc, horizontal_mode)
        out[block_row * n : (block_row + 1) * n, bc * n : (bc + 1) * n] = block

    for br in range(n):
        block = _col_block(tile, block_row, br, vertical_mode)
        out[br * n : (br + 1) * n, block_col * n : (block_col + 1) * n] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the input tile into a cross-shaped fold-out whose anchor block is chosen by the guide-row/guide-column intersection.

    Input structure:
    - The input is an n x n tile with exactly one full row and one full column painted in the same guide color.
    - The guide row and guide column intersect at one anchor cell, while the remaining cells form a generally asymmetric motif.

    Rule:
    - Build an n x n array of blocks, each block itself having size n x n, on an otherwise black canvas.
    - If the guide row is r and the guide column is c, fill only block row r and block column c, leaving all off-cross blocks black.
    - Place the original tile in block (r, c). Moving one block left or right reflects the previous block across the shared vertical edge, so horizontal neighbors alternate between the original tile and its left-right mirror.
    - Moving one block up or down reflects the previous block across the shared horizontal edge, so vertical neighbors alternate between the original tile and its up-down mirror.

    Key insight:
    - The uniform row and column do more than mark the cross: their intersection fixes the anchor block and therefore the folding phase, so the result is a mirrored fold-out rather than plain repeated tiling.
    """
    tile = np.asarray(grid, dtype=int)
    guide_row, guide_col, _guide_color = _find_guides(tile)
    return _expand_cross(tile, guide_row, guide_col)


def _copy_cross(tile: np.ndarray, guide_row: int, guide_col: int) -> np.ndarray:
    return _expand_cross(tile, guide_row, guide_col, horizontal_mode="copy", vertical_mode="copy")


def _fill_non_guide_cells(
    grid: np.ndarray,
    guide_row: int,
    guide_col: int,
    fill_palette: list[int],
) -> None:
    coeff_r, coeff_c = random.choice(PATTERN_COEFFS)
    phase = random.randrange(len(fill_palette))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if r == guide_row or c == guide_col:
                continue
            idx = (coeff_r * r + coeff_c * c + phase) % len(fill_palette)
            grid[r, c] = int(fill_palette[idx])

    free_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if r != guide_row and c != guide_col]
    random.shuffle(free_cells)
    perturbations = random.randint(1, max(1, len(free_cells) // 2))
    for r, c in free_cells[:perturbations]:
        choices = [color for color in fill_palette if color != int(grid[r, c])]
        grid[r, c] = random.choice(choices)


def _sample_tile() -> np.ndarray:
    n = random.choice(SIZE_CHOICES)
    guide_row = random.randrange(n)
    guide_col = random.randrange(n)
    guide_color = random.choice(NON_BLACK_COLORS)
    fill_palette = random.sample([color for color in NON_BLACK_COLORS if color != guide_color], FILL_COLOR_COUNT)

    grid = np.full((n, n), guide_color, dtype=int)
    _fill_non_guide_cells(grid, guide_row, guide_col, fill_palette)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        guide_row, guide_col, _guide_color = _find_guides(grid)
        actual = main(grid)
    except AssertionError:
        return False

    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return False

    plain_copy = _copy_cross(grid, guide_row, guide_col)
    if np.array_equal(actual, plain_copy):
        return False

    n = int(grid.shape[0])
    alt_anchor = (0, 0) if (guide_row, guide_col) != (0, 0) else (n - 1, n - 1)
    alt_fold = _expand_cross(grid, alt_anchor[0], alt_anchor[1])
    if np.array_equal(actual, alt_fold):
        return False

    if actual.shape != (n * n, n * n) or max(actual.shape) > 30:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_tile()
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid cross-fold input")
