from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
CELL_SIZE = 3
ALL_POSITIONS = tuple((r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE))
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_ATTEMPTS = 300


def _interior_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), int(length)]
    intervals = [
        (bounds[i] + 1, bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i] + 1 < bounds[i + 1]
    ]
    assert intervals, "expected at least one cell interval"
    sizes = {end - start for start, end in intervals}
    assert sizes == {CELL_SIZE}, "every cell interior must be 3x3"
    return intervals


def _find_cells(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == ORANGE)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == ORANGE)]
    row_intervals = _interior_intervals(row_seps, g.shape[0])
    col_intervals = _interior_intervals(col_seps, g.shape[1])
    assert len(row_intervals) * len(col_intervals) > 1, "expected multiple cells"
    return row_intervals, col_intervals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every separated 3x3 cell by merging the colored positions seen across all cells.

    Input structure:
    - A black grid is divided by full orange separator rows and columns into equal 3x3 square cells.
    - Each cell uses the same 3x3 internal position layout, but only one to four positions are colored and the remaining positions are black.
    - Whenever the same relative position is colored in different cells, it has the same color.

    Rule:
    - For each relative position inside the 3x3 cell, collect the non-black color visible at that position in any cell.
    - Replace a black internal position in every cell with the collected color for that same relative position.
    - Keep orange separator lines unchanged, so every cell becomes the same fully colored 3x3 pattern.

    Key insight:
    - No single cell contains the full pattern; the original 3x3 image is recovered by comparing matching positions across incomplete copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_intervals, col_intervals = _find_cells(g)

    pattern = np.full((CELL_SIZE, CELL_SIZE), BLACK, dtype=int)
    for rr, cc in ALL_POSITIONS:
        seen = set()
        for r0, _ in row_intervals:
            for c0, _ in col_intervals:
                color = int(g[r0 + rr, c0 + cc])
                if color != BLACK:
                    seen.add(color)
        assert len(seen) == 1, "each relative position must have one recoverable color"
        pattern[rr, cc] = seen.pop()

    out = g.copy()
    for r0, _ in row_intervals:
        for c0, _ in col_intervals:
            for rr, cc in ALL_POSITIONS:
                r, c = r0 + rr, c0 + cc
                assert out[r, c] in (BLACK, pattern[rr, cc]), "visible cell colors must agree"
                if out[r, c] == BLACK:
                    out[r, c] = pattern[rr, cc]
    return out


# --------------- generator ---------------


def _grid_shape(cell_rows: int, cell_cols: int) -> tuple[int, int]:
    return cell_rows * CELL_SIZE + cell_rows - 1, cell_cols * CELL_SIZE + cell_cols - 1


def _cell_origin(br: int, bc: int) -> tuple[int, int]:
    return br * (CELL_SIZE + 1), bc * (CELL_SIZE + 1)


def _make_pattern() -> np.ndarray:
    palette_size = random.randint(4, len(PALETTE))
    colors = random.sample(PALETTE, palette_size)
    flat = colors[:]
    while len(flat) < CELL_SIZE * CELL_SIZE:
        flat.append(random.choice(colors))
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(CELL_SIZE, CELL_SIZE)


def _make_visibility_masks(cell_count: int) -> list[set[tuple[int, int]]]:
    masks: list[set[tuple[int, int]]] = [set() for _ in range(cell_count)]

    for pos in random.sample(list(ALL_POSITIONS), len(ALL_POSITIONS)):
        candidates = [i for i, mask in enumerate(masks) if len(mask) < 4]
        masks[random.choice(candidates)].add(pos)

    for mask in masks:
        if not mask:
            mask.add(random.choice(ALL_POSITIONS))

    for mask in masks:
        target = random.randint(max(1, len(mask)), 4)
        while len(mask) < target:
            mask.add(random.choice(ALL_POSITIONS))

    assert all(1 <= len(mask) <= 4 for mask in masks)
    covered = set().union(*masks)
    assert covered == set(ALL_POSITIONS)
    return masks


def _gen_attempt() -> np.ndarray | None:
    cell_rows = random.randint(2, 7)
    cell_cols = random.randint(2, 7)
    h, w = _grid_shape(cell_rows, cell_cols)
    if max(h, w) > 30:
        return None

    pattern = _make_pattern()
    masks = _make_visibility_masks(cell_rows * cell_cols)

    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(CELL_SIZE, h, CELL_SIZE + 1):
        grid[r, :] = ORANGE
    for c in range(CELL_SIZE, w, CELL_SIZE + 1):
        grid[:, c] = ORANGE

    idx = 0
    for br in range(cell_rows):
        for bc in range(cell_cols):
            r0, c0 = _cell_origin(br, bc)
            for rr, cc in masks[idx]:
                grid[r0 + rr, c0 + cc] = pattern[rr, cc]
            idx += 1

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if np.any(out == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
