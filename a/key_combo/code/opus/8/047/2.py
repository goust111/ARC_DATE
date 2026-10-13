from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
LATTICE_SIZE = 5
OUTPUT_SIZE = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _clipped_spiral_edges(center: tuple[int, int], size: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    def inside(r: int, c: int) -> bool:
        return 0 <= r < size and 0 <= c < size

    r, c = center
    assert inside(r, c)
    visited = {(r, c)}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    step_len = 1
    direction_index = 0

    while len(visited) < size * size:
        for _ in range(2):
            dr, dc = directions[direction_index % 4]
            for _ in range(step_len):
                nr, nc = r + dr, c + dc
                if inside(r, c) and inside(nr, nc):
                    edges.append(((r, c), (nr, nc)))
                r, c = nr, nc
                if inside(r, c):
                    visited.add((r, c))
            direction_index += 1
        step_len += 1
        assert step_len <= size * 2

    return edges


def _separator_slices(indices: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(indices) == 2
    cuts = [-1, *indices, limit]
    slices = [(cuts[i] + 1, cuts[i + 1]) for i in range(3)]
    lengths = [end - start for start, end in slices]
    assert all(length > 0 for length in lengths)
    assert len(set(lengths)) == 1
    return slices


def _compress_regions(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r] == SEPARATOR)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2

    row_slices = _separator_slices(sep_rows, h)
    col_slices = _separator_slices(sep_cols, w)

    compact = np.zeros((3, 3), dtype=int)
    for r, (r0, r1) in enumerate(row_slices):
        for c, (c0, c1) in enumerate(col_slices):
            block = g[r0:r1, c0:c1]
            color = int(block[0, 0])
            assert color != SEPARATOR
            assert np.all(block == color)
            compact[r, c] = color
    return compact


def _paint_spiral(out: np.ndarray, color: int, center: tuple[int, int]) -> None:
    out[::2, ::2] = color
    for (r1, c1), (r2, c2) in _clipped_spiral_edges(center, LATTICE_SIZE):
        out[r1 + r2, c1 + c2] = color


def _render_output(compact: np.ndarray) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BACKGROUND, dtype=int)
    for r in range(3):
        for c in range(3):
            color = int(compact[r, c])
            if color == BACKGROUND:
                continue
            _paint_spiral(out, color, (2 * r, 2 * c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a separator-partitioned 3x3 region map into overlapping lattice spirals.

    Input structure:
    - The input is a 2D grid split by exactly two full grey separator rows and two full grey
      separator columns, creating a uniform 3x3 array of rectangular regions.
    - Each region is filled with a single color: black (0) for background or a non-grey
      foreground color reused across one or more regions.
    - The separator lines are one cell thick and are discarded by the transformation.

    Rule:
    - Identify the grey separator rows and columns, then compress each of the nine regions to
      one pixel to form a 3x3 blueprint.
    - Create a 9x9 black output interpreted as a doubled 5x5 lattice whose even-even cells are
      nodes and whose odd parity cells are edge midpoints.
    - For every non-black compressed cell in row-major order, paint all lattice nodes in that
      cell's color, then draw the clipped square spiral centered at lattice node (2r, 2c) using
      step lengths 1,1,2,2,... in the direction cycle up, right, down, left.
    - When spirals overlap, later compressed cells in row-major order overwrite earlier colors.

    Key insight:
    - The separator grid is first reduced to a coarse 3x3 map, and those compressed coordinates
      act as colored spiral centers on a shared lattice where overwrite order is determined by
      reading order in the compressed map.
    """
    compact = _compress_regions(grid)
    return _render_output(compact)


def _sample_compact() -> np.ndarray:
    compact = np.zeros((3, 3), dtype=int)

    nonzero_count = random.randint(2, 4)
    positions = random.sample(range(9), nonzero_count)
    distinct_colors = random.randint(1, min(3, nonzero_count))
    colors = random.sample(PALETTE, distinct_colors)

    assignments = colors[:]
    while len(assignments) < nonzero_count:
        assignments.append(random.choice(colors))
    random.shuffle(assignments)

    for pos, color in zip(positions, assignments):
        compact[pos // 3, pos % 3] = color

    return compact


def _expand_compact(compact: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    h = 3 * cell_h + 2
    w = 3 * cell_w + 2
    grid = np.full((h, w), SEPARATOR, dtype=int)

    for r in range(3):
        for c in range(3):
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            grid[r0:r0 + cell_h, c0:c0 + cell_w] = compact[r, c]

    return grid


def _valid_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    compact = _compress_regions(grid)
    if not np.array_equal(main(grid), _render_output(compact)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        compact = _sample_compact()
        cell_h = random.choice((3, 4))
        cell_w = random.choice((3, 4))
        grid = _expand_compact(compact, cell_h, cell_w)
        if _valid_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid separator-spiral input")
