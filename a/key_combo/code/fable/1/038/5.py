from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == YELLOW, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full yellow separator row"
    return int(rows[0])


def _extract_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator_row(grid)
    top = np.asarray(grid[:sep, :], dtype=int)
    bottom = np.asarray(grid[sep + 1 :, :], dtype=int)
    assert top.shape == bottom.shape, "top and bottom masks must have the same shape"
    n_rows, n_cols = map(int, top.shape)
    assert n_rows == n_cols, "each half must be an n x n mask"
    return top != BLACK, bottom != BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the mask intersection as a block layout and the mask xor as the repeated block content.

    Input structure:
    - One full yellow row splits the grid into two same-sized square masks.
    - The upper mask uses red on black, and the lower mask uses blue on black.
    - Some cells are colored in both masks, while others are colored in exactly one mask.

    Rule:
    - Compare the two masks cellwise as binary occupancy.
    - Cells colored in both masks form an n x n intersection layout, and cells colored in exactly one mask form an n x n xor pattern.
    - Build an n^2 x n^2 black output divided into n x n blocks; for every active layout cell, paint one green copy of the xor pattern inside its corresponding block.

    Key insight:
    - The same two masks are compared twice for two different roles: intersection chooses which macro-blocks exist, while xor chooses the micro-pattern drawn inside each chosen block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top_mask, bottom_mask = _extract_masks(g)
    intersection = top_mask & bottom_mask
    xor_mask = top_mask ^ bottom_mask

    tiled = np.kron(intersection.astype(int), xor_mask.astype(int))
    return (tiled * GREEN).astype(int)


def _random_shape_mask(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    coords = [(r, c) for r in range(n) for c in range(n)]
    start = coords[int(rng.integers(len(coords)))]
    chosen = {start}
    frontier = {start}

    while len(chosen) < count:
        if frontier and rng.random() < 0.8:
            base = list(frontier)[int(rng.integers(len(frontier)))]
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
            if candidates:
                cell = candidates[int(rng.integers(len(candidates)))]
            else:
                remaining = [coord for coord in coords if coord not in chosen]
                cell = remaining[int(rng.integers(len(remaining)))]
        else:
            remaining = [coord for coord in coords if coord not in chosen]
            cell = remaining[int(rng.integers(len(remaining)))]
        chosen.add(cell)
        frontier.add(cell)

    for r, c in chosen:
        mask[r, c] = True
    return mask


def _choose_counts(rng: np.random.Generator, n: int) -> tuple[int, int]:
    cells = n * n
    max_intersection = max(1, cells - 3)
    intersection_count = int(rng.integers(1, max_intersection + 1))
    max_xor = cells - intersection_count - 1
    xor_count = int(rng.integers(2, max_xor + 1))
    return intersection_count, xor_count


def _sample_derived_masks(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    cells = n * n
    for _ in range(200):
        intersection_count, xor_count = _choose_counts(rng, n)
        intersection = _random_shape_mask(rng, n, intersection_count)

        remaining = np.argwhere(~intersection)
        xor = np.zeros((n, n), dtype=bool)
        choice = remaining[rng.choice(len(remaining), size=xor_count, replace=False)]
        xor[choice[:, 0], choice[:, 1]] = True

        if intersection.sum() == 0 or xor.sum() < 2:
            continue
        if intersection.sum() + xor.sum() >= cells:
            continue

        rows_i = np.any(intersection, axis=1)
        cols_i = np.any(intersection, axis=0)
        rows_x = np.any(xor, axis=1)
        cols_x = np.any(xor, axis=0)
        if np.array_equal(rows_i, rows_x) and np.array_equal(cols_i, cols_x):
            continue

        return intersection, xor

    raise AssertionError("failed to sample non-degenerate derived masks")


def _split_xor_between_sides(
    rng: np.random.Generator, xor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(xor)
    count = len(coords)
    assert count >= 2, "xor mask must have at least two cells"

    order = rng.permutation(count)
    split = int(rng.integers(1, count))
    top_coords = coords[order[:split]]
    bottom_coords = coords[order[split:]]

    if len(top_coords) == 0 or len(bottom_coords) == 0:
        top_coords = coords[order[:1]]
        bottom_coords = coords[order[1:]]

    top_only = np.zeros_like(xor, dtype=bool)
    bottom_only = np.zeros_like(xor, dtype=bool)
    top_only[top_coords[:, 0], top_coords[:, 1]] = True
    bottom_only[bottom_coords[:, 0], bottom_coords[:, 1]] = True
    return top_only, bottom_only


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        n = int(rng.integers(2, 6))
        intersection, xor = _sample_derived_masks(rng, n)
        top_only, bottom_only = _split_xor_between_sides(rng, xor)

        top_mask = intersection | top_only
        bottom_mask = intersection | bottom_only

        if not np.any(top_mask & ~bottom_mask):
            continue
        if not np.any(bottom_mask & ~top_mask):
            continue
        if not np.any(top_mask & bottom_mask):
            continue
        if np.all(top_mask | bottom_mask):
            continue

        grid = np.full((2 * n + 1, n), BLACK, dtype=int)
        grid[:n, :][top_mask] = RED
        grid[n + 1 :, :][bottom_mask] = BLUE
        grid[n, :] = YELLOW
        return grid

    raise AssertionError("failed to generate a valid input")
