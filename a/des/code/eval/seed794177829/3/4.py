import random
from collections import Counter

import numpy as np


BG_COLOR = 0
GRID_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    assert int(np.count_nonzero(counts == counts.max())) == 1, "background must be uniquely dominant"
    return int(values[np.argmax(counts)])


def _find_grid_structure(
    grid: np.ndarray, bg: int
) -> tuple[int, np.ndarray, np.ndarray, int, int]:
    candidates: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        row_lines = np.flatnonzero(np.all(grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(grid == color, axis=0))
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        if row_lines[0] != 0 or row_lines[-1] != grid.shape[0] - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != grid.shape[1] - 1:
            continue

        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue

        cell_h = int(row_steps[0] - 1)
        cell_w = int(col_steps[0] - 1)
        if cell_h < 1 or cell_w < 1:
            continue

        line_mask = np.zeros(grid.shape, dtype=bool)
        line_mask[row_lines, :] = True
        line_mask[:, col_lines] = True
        if np.any((grid == color) & (~line_mask)):
            continue

        candidates.append((color, row_lines, col_lines, cell_h, cell_w))

    assert len(candidates) == 1, "expected one regular grey grid"
    return candidates[0]


def _stripe_mask(shape: tuple[int, int], seed_r: int, seed_c: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    height, width = shape
    if (seed_r, seed_c) in ((0, 0), (height - 1, width - 1)):
        return ((rows - cols) - (seed_r - seed_c)) % 2 == 0
    return ((rows + cols) - (seed_r + seed_c)) % 2 == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand corner seeds into diagonal stripe patterns inside a grey cell grid.

    Input structure:
    - Black background is the dominant color.
    - One-pixel grey rows and columns form a regular rectangular grid.
    - Some cells contain exactly one non-grey, non-background seed in a corner.

    Rule:
    - Top-left and bottom-right seeds select main-diagonal stripes.
    - Top-right and bottom-left seeds select anti-diagonal stripes.
    - Color every other diagonal so the seed lies on a colored stripe.
    - Empty cells stay black and the grey grid lines remain unchanged.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(grid)
    grey, row_lines, col_lines, _cell_h, _cell_w = _find_grid_structure(grid, bg)

    out = grid.copy()
    seeded_cells = 0
    for top, bottom in zip(row_lines[:-1], row_lines[1:]):
        for left, right in zip(col_lines[:-1], col_lines[1:]):
            cell = grid[top + 1 : bottom, left + 1 : right]
            assert cell.size > 0, "cells must have positive area"

            seed_coords = np.argwhere((cell != bg) & (cell != grey))
            assert len(seed_coords) <= 1, "each cell must be empty or contain one seed"
            if len(seed_coords) == 0:
                assert np.all(cell == bg), "empty cells must contain only background"
                continue

            seeded_cells += 1
            seed_r, seed_c = map(int, seed_coords[0])
            height, width = cell.shape
            valid_corners = {
                (0, 0),
                (0, width - 1),
                (height - 1, 0),
                (height - 1, width - 1),
            }
            assert (seed_r, seed_c) in valid_corners, "seed must lie in a cell corner"

            color = int(cell[seed_r, seed_c])
            patch = np.full(cell.shape, bg, dtype=int)
            patch[_stripe_mask(cell.shape, seed_r, seed_c)] = color
            out[top + 1 : bottom, left + 1 : right] = patch

    assert seeded_cells > 0, "expected at least one seeded cell"
    return out


def _grid_lines(count: int, cell_size: int) -> np.ndarray:
    total = count * cell_size + count + 1
    return np.arange(0, total, cell_size + 1, dtype=int)


def _cell_slice(line_positions: np.ndarray, index: int) -> slice:
    return slice(int(line_positions[index] + 1), int(line_positions[index + 1]))


def _corner_position(label: str, cell_h: int, cell_w: int) -> tuple[int, int]:
    if label == "tl":
        return 0, 0
    if label == "tr":
        return 0, cell_w - 1
    if label == "bl":
        return cell_h - 1, 0
    return cell_h - 1, cell_w - 1


def _seeded_layout(
    n_rows: int, n_cols: int
) -> tuple[list[tuple[int, int]], list[str]]:
    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    max_seeds = max(2, n_rows * n_cols - 1)
    seed_count = random.randint(2, max_seeds)
    seeded_cells = random.sample(all_cells, seed_count)

    corners = [random.choice(["tl", "br"]), random.choice(["tr", "bl"])]
    corners.extend(random.choice(["tl", "tr", "bl", "br"]) for _ in range(seed_count - 2))
    random.shuffle(corners)
    return seeded_cells, corners


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(4, 6)
    cell_w = random.randint(4, 6)
    n_rows = random.randint(2, 5)
    n_cols = random.randint(2, 5)

    row_lines = _grid_lines(n_rows, cell_h)
    col_lines = _grid_lines(n_cols, cell_w)
    height = int(row_lines[-1] + 1)
    width = int(col_lines[-1] + 1)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[row_lines, :] = GRID_COLOR
    grid[:, col_lines] = GRID_COLOR

    seeded_cells, corner_labels = _seeded_layout(n_rows, n_cols)
    active_palette = random.sample(SEED_COLORS, random.randint(2, min(5, len(SEED_COLORS))))
    for (cell_r, cell_c), corner_label in zip(seeded_cells, corner_labels):
        row_slice = _cell_slice(row_lines, cell_r)
        col_slice = _cell_slice(col_lines, cell_c)
        seed_r, seed_c = _corner_position(corner_label, cell_h, cell_w)
        color = random.choice(active_palette)
        grid[row_slice.start + seed_r, col_slice.start + seed_c] = color

    counts = Counter(map(int, grid.ravel()))
    if counts[BG_COLOR] <= counts[GRID_COLOR]:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
