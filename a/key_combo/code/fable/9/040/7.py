from __future__ import annotations

from collections import Counter

import numpy as np


GEN_ATTEMPTS = 2000
_RNG = np.random.default_rng()


def _find_guide_row(grid: np.ndarray) -> tuple[int, int]:
    n = int(grid.shape[0])
    candidates: list[tuple[int, int]] = []
    for r in range(n):
        color = int(grid[r, 0])
        if not np.all(grid[r, :] == color):
            continue
        elsewhere = np.delete(grid, r, axis=0)
        if not np.any(elsewhere == color):
            candidates.append((r, color))
    assert len(candidates) == 1, "expected exactly one full guide row with a unique color"
    return candidates[0]


def _infer_background(grid: np.ndarray, guide_row: int) -> int:
    non_guide = np.delete(grid, guide_row, axis=0).ravel()
    counts = Counter(map(int, non_guide))
    assert counts, "expected cells outside the guide row"
    most_common = counts.most_common()
    assert len(most_common) == 1 or most_common[0][1] > most_common[1][1], "background must be uniquely dominant outside the guide row"
    return int(most_common[0][0])


def _shift_tile(grid: np.ndarray, guide_row: int, guide_color: int, bg: int, target_row: int) -> np.ndarray:
    tile = np.array(grid, copy=True)
    tile[guide_row, :] = bg
    tile[target_row, :] = guide_color
    return tile


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    guide_row, guide_color = _find_guide_row(g)
    bg = _infer_background(g, guide_row)

    out = np.full((n * n, n * n), bg, dtype=int)
    band_r0 = guide_row * n
    for block_col in range(n):
        tile = _shift_tile(g, guide_row, guide_color, bg, block_col)
        c0 = block_col * n
        out[band_r0 : band_r0 + n, c0 : c0 + n] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input across the guide-selected band, but move the guide line within each copy to the block-column row.

    Input structure:
    - The input is an n x n tile containing exactly one full single-color guide row.
    - The guide color does not appear outside that row; the remaining cells are background or scattered pattern pixels.

    Rule:
    - Build an n-by-n block canvas of n x n blocks, filled with the tile background color.
    - Only block row r is used, where r is the input guide-row index.
    - In block column k of that band, copy the tile after erasing its original guide row to background, then redraw the full guide line on row k inside that copy.
    - The moved guide line overwrites any pattern pixels on the row it passes through.

    Key insight:
    - The solid row is a cursor, not preserved content: it selects the output band globally, then relocates locally according to each block column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h >= 2, "expected a square tile of size at least 2"

    return _expected_output(g)


def _validate_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        n = int(g.shape[0])
        if g.ndim != 2 or g.shape[0] != g.shape[1] or not (3 <= n <= 5):
            return False

        guide_row, guide_color = _find_guide_row(g)
        bg = _infer_background(g, guide_row)

        non_guide_rows = [r for r in range(n) if r != guide_row]
        row_signatures = set()
        pattern_colors = set()
        for r in non_guide_rows:
            row = g[r, :]
            if np.all(row == row[0]):
                return False
            fg_cols = np.flatnonzero(row != bg)
            if len(fg_cols) == 0 or len(fg_cols) >= n:
                return False
            row_signatures.add(tuple(map(int, row)))
            pattern_colors.update(map(int, row[fg_cols]))

        if guide_color in pattern_colors or len(pattern_colors) < 2:
            return False
        if len(row_signatures) < 2:
            return False

        out = main(g)
        if out.shape != (n * n, n * n):
            return False

        diagonal_matches = 0
        for block_row in range(n):
            for block_col in range(n):
                block = out[block_row * n : (block_row + 1) * n, block_col * n : (block_col + 1) * n]
                if block_row != guide_row:
                    if not np.all(block == bg):
                        return False
                    continue

                expected_block = _shift_tile(g, guide_row, guide_color, bg, block_col)
                if not np.array_equal(block, expected_block):
                    return False
                if np.array_equal(block, g):
                    diagonal_matches += 1
                    if block_col != guide_row:
                        return False
                elif block_col == guide_row:
                    return False

        return diagonal_matches == 1
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    guide_row = int(rng.integers(0, n))
    grid = np.zeros((n, n), dtype=int)

    palette = list(range(1, 10))
    guide_color = int(rng.choice(palette))
    remaining = [color for color in palette if color != guide_color]

    rows = [r for r in range(n) if r != guide_row]
    double_row = int(rng.choice(rows)) if n >= 4 else None
    pattern_coords: list[tuple[int, int]] = []
    for r in rows:
        count = 2 if r == double_row else int(rng.integers(1, 3))
        count = min(count, n - 1)
        cols = rng.choice(n, size=count, replace=False)
        for c in cols:
            pattern_coords.append((r, int(c)))

    if len(pattern_coords) < 2:
        return None

    num_pattern_colors = 2 if len(pattern_coords) < 4 else int(rng.integers(2, 4))
    pattern_palette = list(map(int, rng.choice(remaining, size=num_pattern_colors, replace=False)))

    rng.shuffle(pattern_coords)
    for color, (r, c) in zip(pattern_palette, pattern_coords):
        grid[r, c] = color
    for r, c in pattern_coords[len(pattern_palette) :]:
        grid[r, c] = int(rng.choice(pattern_palette))

    grid[guide_row, :] = guide_color

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
