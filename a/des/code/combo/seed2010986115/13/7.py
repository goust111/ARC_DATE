import random
from typing import Iterable

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _priority_colors(grid: np.ndarray, bg: int) -> list[int]:
    colors = [int(value) for value in grid[:, -1] if int(value) != bg]
    assert colors, "rightmost column must contain a non-background priority legend"
    assert len(colors) == len(set(colors)), "priority colors must be unique in the legend"
    return colors


def _tower_coords(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid[:, :-1] != bg)
    assert coords.size > 0, "expected at least one signal tower"
    return coords


def _proposal_sets(
    grid: np.ndarray, bg: int, towers: np.ndarray
) -> tuple[list[set[int]], list[set[int]]]:
    h, w = grid.shape
    row_colors = [set() for _ in range(h)]
    col_colors = [set() for _ in range(w - 1)]
    for r, c in towers:
        color = int(grid[r, c])
        row_colors[int(r)].add(color)
        col_colors[int(c)].add(color)
    return row_colors, col_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast each tower's color across its row and column, and resolve overlaps
    by the top-to-bottom priority order encoded in the rightmost column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid must be at least 2x2"

    bg = _most_common_color(g)
    priority = _priority_colors(g, bg)
    rank = {color: index for index, color in enumerate(priority)}

    towers = _tower_coords(g, bg)
    tower_mask = np.zeros_like(g, dtype=bool)
    tower_mask[towers[:, 0], towers[:, 1]] = True

    tower_colors = {int(g[r, c]) for r, c in towers}
    assert tower_colors <= set(priority), "every tower color must appear in the priority legend"

    row_colors, col_colors = _proposal_sets(g, bg, towers)

    out = g.copy()
    for r in range(h):
        for c in range(w - 1):
            if tower_mask[r, c]:
                continue
            proposals = row_colors[r] | col_colors[c]
            if not proposals:
                continue
            out[r, c] = min(proposals, key=rank.__getitem__)

    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
GRID_H_RANGE = (8, 14)
GRID_W_RANGE = (9, 16)
PRIORITY_COUNT_RANGE = (2, 5)
MAX_TOWERS = 6
EXTRA_TOWERS_MAX = 2
LEGEND_ROW_RETRIES = 80
GENERATION_RETRIES = 400


def _sample_legend_rows(height: int, count: int) -> list[int] | None:
    for _ in range(LEGEND_ROW_RETRIES):
        rows = sorted(random.sample(range(height), count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _has_priority_conflict(
    row_colors: list[set[int]], col_colors: list[set[int]], towers: np.ndarray, shape: tuple[int, int]
) -> bool:
    h, w = shape
    tower_mask = np.zeros(shape, dtype=bool)
    tower_mask[towers[:, 0], towers[:, 1]] = True
    for r in range(h):
        for c in range(w - 1):
            if tower_mask[r, c]:
                continue
            if row_colors[r] and col_colors[c] and len(row_colors[r] | col_colors[c]) >= 2:
                return True
    return False


def _count_changed_cells(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero(inp != out))


def _random_choices(pool: Iterable[int], count: int) -> list[int]:
    return random.sample(list(pool), count)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    bg = random.choice(PALETTE)
    available = [color for color in PALETTE if color != bg]

    max_towers = min(MAX_TOWERS, h - 1, w - 2)
    if max_towers < PRIORITY_COUNT_RANGE[0]:
        return None

    max_priority = min(PRIORITY_COUNT_RANGE[1], max_towers, (h + 1) // 2, len(available))
    if max_priority < PRIORITY_COUNT_RANGE[0]:
        return None
    priority_count = random.randint(PRIORITY_COUNT_RANGE[0], max_priority)

    legend_rows = _sample_legend_rows(h, priority_count)
    if legend_rows is None:
        return None

    priority = _random_choices(available, priority_count)
    grid = np.full((h, w), bg, dtype=int)
    for row, color in zip(legend_rows, priority):
        grid[row, -1] = color

    extra_towers = random.randint(0, min(EXTRA_TOWERS_MAX, max_towers - priority_count))
    tower_count = priority_count + extra_towers

    tower_rows = random.sample(range(h), tower_count)
    tower_cols = random.sample(range(w - 1), tower_count)

    tower_colors = priority.copy()
    while len(tower_colors) < tower_count:
        tower_colors.append(random.choice(priority))
    random.shuffle(tower_colors)

    for row, col, color in zip(tower_rows, tower_cols, tower_colors):
        grid[row, col] = color

    towers = _tower_coords(grid, bg)
    row_colors, col_colors = _proposal_sets(grid, bg, towers)
    if not _has_priority_conflict(row_colors, col_colors, towers, grid.shape):
        return None

    out = main(grid)
    if _count_changed_cells(grid, out) < tower_count + 4:
        return None

    has_row_only = any(row_colors[r] and not col_colors[c] and grid[r, c] == bg for r in range(h) for c in range(w - 1))
    has_col_only = any(col_colors[c] and not row_colors[r] and grid[r, c] == bg for r in range(h) for c in range(w - 1))
    if not (has_row_only and has_col_only):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
