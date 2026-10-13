from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PALETTE = (3, 4, 5, 6, 7, 8, 9)

GRID_SIZE_RANGE = (13, 21)
SEED_COUNT_CHOICES = (2, 3)
BAND_SIZE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400


def _bands_from_seeds(size: int, seeds: Iterable[int]) -> list[tuple[int, int]]:
    seeds = [int(x) for x in sorted(seeds)]
    assert seeds
    assert len(seeds) == len(set(seeds))
    assert 2 <= seeds[0] and seeds[-1] < size
    edges = [1, *seeds, size]
    bands = [(a, b) for a, b in zip(edges, edges[1:], strict=False)]
    assert all(a < b for a, b in bands)
    return bands


def _seed_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row0 = np.asarray(grid[0, :], dtype=int)
    col0 = np.asarray(grid[:, 0], dtype=int)
    red_cols = [int(c) for c in np.where(row0 == RED)[0] if int(c) != 0]
    blue_rows = [int(r) for r in np.where(col0 == BLUE)[0] if int(r) != 0]
    assert len(red_cols) >= 2 and len(blue_rows) >= 2
    return blue_rows, red_cols


def _cell_fg_colors(cell: np.ndarray) -> set[int]:
    return set(map(int, np.unique(cell))) - {BLACK}


def _find_target_cell(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    blue_rows, red_cols = _seed_rows_cols(grid)
    row_bands = _bands_from_seeds(grid.shape[0], blue_rows)
    col_bands = _bands_from_seeds(grid.shape[1], red_cols)

    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            cell = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            if len(_cell_fg_colors(cell)) == 1:
                candidates.append(((br, bc), cell.copy()))
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Use edge seed coordinates as hidden partition lines and crop the unique monochrome cell.

    Input structure:
    - The grid has a black background with red seed pixels only on the top row and blue seed
      pixels only on the left column.
    - The seed columns and seed rows implicitly partition the interior region
      `(rows 1..H-1, cols 1..W-1)` into rectangular cells; no separator lines are drawn.
    - Interior cells contain scattered non-black pixels, with exactly one cell whose non-black
      pixels all share a single color.

    Rule:
    - Read top-row red seeds as vertical partition coordinates and left-column blue seeds as
      horizontal partition coordinates.
    - Split the interior region using those coordinates into rectangular cells.
    - Find the unique cell whose non-black pixels use exactly one color, and return that whole
      cell cropped out exactly, preserving both its black pixels and its foreground color.

    Key insight:
    - The partition grid is hidden: the edge seeds act like undrawn rays that define cell
      boundaries, and only after reconstructing those boundaries can the anomalous cell be found.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5
    _idx, cell = _find_target_cell(g)
    return cell


def _sample_band_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(x) for x in rng.integers(BAND_SIZE_RANGE[0], BAND_SIZE_RANGE[1] + 1, size=count)]


def _seed_positions_from_band_sizes(sizes: list[int]) -> tuple[int, list[int]]:
    pos = 1
    seeds: list[int] = []
    for size in sizes[:-1]:
        pos += size
        seeds.append(pos)
    total = sum(sizes) + 1
    return total, seeds


def _make_target_cell(rng: np.random.Generator, h: int, w: int, color: int) -> np.ndarray:
    area = h * w
    min_fg = max(2, math.ceil(area * 0.25))
    max_fg = max(min_fg, math.floor(area * 0.65))
    for _ in range(200):
        fg_count = int(rng.integers(min_fg, max_fg + 1))
        cell = np.zeros((h, w), dtype=int)
        idx = rng.choice(area, size=fg_count, replace=False)
        cell.flat[idx] = color
        rows, cols = np.where(cell == color)
        assert rows.size > 0
        touches_all_borders = (
            rows.min() == 0
            and rows.max() == h - 1
            and cols.min() == 0
            and cols.max() == w - 1
        )
        if np.any(cell == 0) and not touches_all_borders:
            return cell
    raise RuntimeError("failed to build a non-degenerate target cell")


def _fill_distractor_cell(
    rng: np.random.Generator,
    h: int,
    w: int,
    palette: tuple[int, ...],
    required_colors: Iterable[int] = (),
) -> np.ndarray:
    area = h * w
    required = list(dict.fromkeys(int(c) for c in required_colors))
    max_colors = min(4, area, len(palette))
    min_colors = max(2, len(required))
    color_count = int(rng.integers(min_colors, max_colors + 1))
    extras = [c for c in palette if c not in required]
    rng.shuffle(extras)
    colors = required + extras[: max(0, color_count - len(required))]
    fg_min = max(len(colors), math.ceil(area * 0.35))
    fg_max = max(fg_min, math.floor(area * 0.8))
    fg_count = int(rng.integers(fg_min, fg_max + 1))

    cell = np.zeros((h, w), dtype=int)
    idx = rng.choice(area, size=fg_count, replace=False)
    for k, color in enumerate(colors):
        cell.flat[int(idx[k])] = int(color)
    if fg_count > len(colors):
        cell.flat[idx[len(colors) :]] = rng.choice(colors, size=fg_count - len(colors), replace=True)
    assert len(_cell_fg_colors(cell)) >= 2
    return cell


def _constructed_bands_from_sizes(row_sizes: list[int], col_sizes: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []

    r = 1
    for size in row_sizes:
        row_bands.append((r, r + size))
        r += size
    c = 1
    for size in col_sizes:
        col_bands.append((c, c + size))
        c += size
    return row_bands, col_bands


def _validate_generated(
    grid: np.ndarray,
    target_index: tuple[int, int],
    target_cell: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
    target_color: int,
) -> bool:
    row_bands, col_bands = _constructed_bands_from_sizes(row_sizes, col_sizes)
    if tuple(grid.shape) != (sum(row_sizes) + 1, sum(col_sizes) + 1):
        return False
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        return False
    if np.count_nonzero(grid[0, 1:]) < 2 or np.count_nonzero(grid[1:, 0]) < 2:
        return False
    if np.any(grid[0, 1:] == BLUE) or np.any(grid[1:, 0] == RED):
        return False
    if np.any(grid[1:, 1:] == RED) or np.any(grid[1:, 1:] == BLUE):
        return False

    mono_cells: list[tuple[int, int]] = []
    target_color_seen_elsewhere = False
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            cell = grid[r0:r1, c0:c1]
            colors = _cell_fg_colors(cell)
            if (br, bc) == target_index:
                if not np.array_equal(cell, target_cell):
                    return False
                if colors != {target_color}:
                    return False
                if not np.any(cell == BLACK):
                    return False
            else:
                if len(colors) < 2:
                    return False
                if target_color in colors:
                    target_color_seen_elsewhere = True
            if len(colors) == 1:
                mono_cells.append((br, bc))
    if mono_cells != [target_index]:
        return False
    if not target_color_seen_elsewhere:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, target_cell)


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    row_seed_count = int(rng.choice(SEED_COUNT_CHOICES))
    col_seed_count = int(rng.choice(SEED_COUNT_CHOICES))
    row_sizes = _sample_band_sizes(rng, row_seed_count + 1)
    col_sizes = _sample_band_sizes(rng, col_seed_count + 1)

    h, blue_rows = _seed_positions_from_band_sizes(row_sizes)
    w, red_cols = _seed_positions_from_band_sizes(col_sizes)
    if not (GRID_SIZE_RANGE[0] <= h <= GRID_SIZE_RANGE[1] and GRID_SIZE_RANGE[0] <= w <= GRID_SIZE_RANGE[1]):
        return None
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        return None

    row_bands, col_bands = _constructed_bands_from_sizes(row_sizes, col_sizes)
    grid = np.zeros((h, w), dtype=int)
    grid[0, red_cols] = RED
    grid[blue_rows, 0] = BLUE

    target_index = (
        int(rng.integers(0, len(row_bands))),
        int(rng.integers(0, len(col_bands))),
    )
    target_color = int(rng.choice(PALETTE))
    target_r0, target_r1 = row_bands[target_index[0]]
    target_c0, target_c1 = col_bands[target_index[1]]
    target_cell = _make_target_cell(
        rng,
        target_r1 - target_r0,
        target_c1 - target_c0,
        target_color,
    )

    distractors = [
        (br, bc)
        for br in range(len(row_bands))
        for bc in range(len(col_bands))
        if (br, bc) != target_index
    ]
    same_axis = [
        cell
        for cell in distractors
        if cell[0] == target_index[0] or cell[1] == target_index[1]
    ]
    if not same_axis:
        return None
    must_include_target = {same_axis[int(rng.integers(0, len(same_axis)))]}
    extra_count = int(rng.integers(0, min(3, len(distractors)) + 1))
    rng.shuffle(distractors)
    for cell in distractors[:extra_count]:
        must_include_target.add(cell)

    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            if (br, bc) == target_index:
                grid[r0:r1, c0:c1] = target_cell
                continue
            required = (target_color,) if (br, bc) in must_include_target else ()
            cell = _fill_distractor_cell(rng, r1 - r0, c1 - c0, PALETTE, required)
            grid[r0:r1, c0:c1] = cell

    if not _validate_generated(grid, target_index, target_cell, row_sizes, col_sizes, target_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
