from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BLUE = Color.BLUE
PALETTE = tuple(color for color in Color.ALL_COLORS if color != BLUE)
MAX_GENERATION_ATTEMPTS = 400


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask.tolist()):
        if bool(flag) == bool(value):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _infer_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    for sep_color in sorted(map(int, np.unique(g))):
        row_mask = np.all(g == sep_color, axis=1)
        col_mask = np.all(g == sep_color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue
        if bool(row_mask[0]) or bool(row_mask[-1]) or bool(col_mask[0]) or bool(col_mask[-1]):
            continue

        row_tiles = _runs(row_mask, False)
        col_tiles = _runs(col_mask, False)
        row_seps = _runs(row_mask, True)
        col_seps = _runs(col_mask, True)
        if len(row_tiles) < 2 or len(col_tiles) < 2:
            continue
        if len(row_seps) != len(row_tiles) - 1 or len(col_seps) != len(col_tiles) - 1:
            continue

        row_sizes = {end - start for start, end in row_tiles}
        col_sizes = {end - start for start, end in col_tiles}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        tile_size = int(next(iter(row_sizes)))
        if tile_size != int(next(iter(col_sizes))) or tile_size <= 0:
            continue

        return sep_color, row_tiles, col_tiles, tile_size

    raise AssertionError("failed to infer a valid separator-grid layout")


def _aligned_tiles(
    grid: np.ndarray,
    row_tiles: list[tuple[int, int]],
    col_tiles: list[tuple[int, int]],
) -> np.ndarray:
    aligned: list[np.ndarray] = []
    for row_index, (r0, r1) in enumerate(row_tiles):
        for col_index, (c0, c1) in enumerate(col_tiles):
            tile = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            aligned.append(np.roll(tile, shift=(-row_index, -col_index), axis=(0, 1)))
    return np.stack(aligned, axis=0)


def _majority_vote(aligned: np.ndarray) -> np.ndarray:
    tile_count, tile_size, _ = map(int, aligned.shape)
    consensus = np.empty((tile_size, tile_size), dtype=int)
    for r in range(tile_size):
        for c in range(tile_size):
            colors, counts = np.unique(aligned[:, r, c], return_counts=True)
            best = int(np.argmax(counts))
            assert int(counts[best]) > tile_count // 2, "aligned votes must have a strict majority"
            consensus[r, c] = int(colors[best])
    return consensus


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert not np.any(g == BLUE), "blue is reserved for output markers"

    _, row_tiles, col_tiles, _ = _infer_layout(g)
    aligned = _aligned_tiles(g, row_tiles, col_tiles)
    consensus = _majority_vote(aligned)

    mismatches = aligned != consensus
    mismatch_counts = mismatches.reshape(mismatches.shape[0], -1).sum(axis=1)
    damaged = np.flatnonzero(mismatch_counts > 0)
    assert len(damaged) == 1, "expected exactly one damaged tile after alignment"
    return consensus, mismatches[int(damaged[0])]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Undo lattice-index cyclic shifts, recover the consensus base tile, and mark the damaged pixels blue.

    Input structure:
    - Whole separator rows and whole separator columns of one color partition the grid into an `R x C` array of equal square tiles.
    - Tile `(i, j)` is the same hidden base tile rolled down by `i` and right by `j`.
    - Exactly one tile is additionally damaged by replacing some foreground pixels with the base tile's background color.
    - Blue does not appear in the input and is reserved for the output marker.

    Rule:
    - Reverse-roll each tile by `(-i, -j)` into the common base coordinate frame.
    - Take a per-cell majority vote over the aligned tiles to reconstruct the hidden base tile.
    - Find the unique aligned tile that disagrees with that consensus.
    - Output the consensus tile, but repaint the disagreement cells of the damaged tile blue.

    Key insight:
    - The lattice position is itself the shift code. Once every tile is unrolled by its row/column index, all redundancy becomes visible and the single damaged tile is the only outlier.
    """
    consensus, damaged_mask = _solve_details(input_grid)
    out = consensus.copy()
    out[damaged_mask] = BLUE
    return out


def _roll_for_lattice(base: np.ndarray, row_index: int, col_index: int) -> np.ndarray:
    return np.roll(base, shift=(row_index, col_index), axis=(0, 1))


def _choose_distinct(colors: list[int], count: int) -> list[int]:
    return random.sample(colors, k=count)


def _paint_random_shape(base: np.ndarray, color: int) -> None:
    size = int(base.shape[0])
    kind = random.choice(["rect", "hline", "vline", "point", "diag", "plus"])

    if kind == "rect":
        r0 = random.randrange(size)
        r1 = random.randrange(r0, size)
        c0 = random.randrange(size)
        c1 = random.randrange(c0, size)
        base[r0 : r1 + 1, c0 : c1 + 1] = color
    elif kind == "hline":
        r = random.randrange(size)
        c0 = random.randrange(size)
        c1 = random.randrange(c0, size)
        base[r, c0 : c1 + 1] = color
    elif kind == "vline":
        c = random.randrange(size)
        r0 = random.randrange(size)
        r1 = random.randrange(r0, size)
        base[r0 : r1 + 1, c] = color
    elif kind == "diag":
        length = random.randint(1, size)
        r = random.randrange(size)
        c = random.randrange(size)
        dr, dc = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        for _ in range(length):
            if 0 <= r < size and 0 <= c < size:
                base[r, c] = color
            r += dr
            c += dc
    elif kind == "plus":
        r = random.randrange(size)
        c = random.randrange(size)
        arm = random.randint(0, max(1, size // 2))
        for delta in range(-arm, arm + 1):
            if 0 <= r + delta < size:
                base[r + delta, c] = color
            if 0 <= c + delta < size:
                base[r, c + delta] = color
    else:
        r = random.randrange(size)
        c = random.randrange(size)
        base[r, c] = color


def _generate_base(tile_size: int, bg: int, fg_colors: list[int]) -> np.ndarray | None:
    base = np.full((tile_size, tile_size), bg, dtype=int)
    shape_count = random.randint(3, 6)
    for _ in range(shape_count):
        _paint_random_shape(base, random.choice(fg_colors))

    non_bg = np.argwhere(base != bg)
    if len(non_bg) < max(2, tile_size):
        return None
    if len(non_bg) >= tile_size * tile_size:
        return None

    rolled_variants = {
        _roll_for_lattice(base, row_shift, col_shift).tobytes()
        for row_shift in range(min(3, tile_size))
        for col_shift in range(min(3, tile_size))
    }
    if len(rolled_variants) < 2:
        return None
    return base


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    if rows * cols < 4:
        return None

    max_tile_by_rows = (30 - (rows - 1)) // rows
    max_tile_by_cols = (30 - (cols - 1)) // cols
    max_tile = min(max_tile_by_rows, max_tile_by_cols, 7)
    if max_tile < 3:
        return None
    tile_size = random.randint(3, max_tile)

    bg, sep_color = _choose_distinct(list(PALETTE), 2)
    fg_pool = [color for color in PALETTE if color not in {bg, sep_color}]
    fg_count = random.randint(1, min(4, len(fg_pool)))
    fg_colors = _choose_distinct(fg_pool, fg_count)

    base = _generate_base(tile_size, bg, fg_colors)
    if base is None:
        return None

    fg_cells = np.argwhere(base != bg)
    if len(fg_cells) == 0:
        return None
    damage_count = random.randint(1, max(1, min(len(fg_cells), tile_size)))
    chosen = random.sample(range(len(fg_cells)), k=damage_count)
    damage_mask = np.zeros_like(base, dtype=bool)
    for index in chosen:
        r, c = map(int, fg_cells[index])
        damage_mask[r, c] = True

    damaged_tile = (random.randrange(rows), random.randrange(cols))
    damaged_base = base.copy()
    damaged_base[damage_mask] = bg

    height = rows * tile_size + (rows - 1)
    width = cols * tile_size + (cols - 1)
    grid = np.full((height, width), sep_color, dtype=int)
    for row_index in range(rows):
        for col_index in range(cols):
            tile = damaged_base if (row_index, col_index) == damaged_tile else base
            rolled = _roll_for_lattice(tile, row_index, col_index)
            r0 = row_index * (tile_size + 1)
            c0 = col_index * (tile_size + 1)
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = rolled

    expected = base.copy()
    expected[damage_mask] = BLUE

    if np.any(grid == BLUE):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid cyclic-consensus repair input")
