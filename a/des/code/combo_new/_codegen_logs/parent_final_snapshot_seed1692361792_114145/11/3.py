import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GRID_SIZE = 30
MAX_COLOR_CELLS = 5
GEN_RETRIES = 300
PLACEMENT_RETRIES = 50


def _cell_spans(length: int, cut_indices: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(cut_indices) + [length]
    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _partition_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_cuts = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    col_cuts = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _cell_spans(int(grid.shape[0]), row_cuts)
    col_spans = _cell_spans(int(grid.shape[1]), col_cuts)
    assert row_spans and col_spans, "expected at least one cell span on each axis"
    return row_spans, col_spans


def _count_filled_cells(grid: np.ndarray) -> Counter[int]:
    counts: Counter[int] = Counter()
    row_spans, col_spans = _partition_spans(grid)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = grid[r0:r1, c0:c1]
            values = np.unique(cell)
            assert GREY not in values, "cell interior cannot contain divider color"
            assert values.size == 1, "each cell must be uniformly colored"
            color = int(values[0])
            if color != BLACK:
                counts[color] += 1
    assert counts, "expected at least one colored cell"
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort partitioned color regions by how many grid cells they occupy, then rewrite them as compact columns.

    Input structure:
    - Black background with one-pixel-thick full grey divider rows and columns.
    - The grey dividers partition the canvas into rectangular cells.
    - Every cell interior is uniform: either all black or all one non-black, non-grey color.
    - The same foreground color may occupy multiple separated cells, and different colors occupy distinct cell counts.

    Rule:
    - Detect the rectangular cells induced by the full grey divider rows and columns.
    - Count, for each non-black and non-grey color, how many cells are filled with that color.
    - Sort the colors by that cell count from fewest to most.
    - Return a grid with the same height as the input and one output column per color; column k is filled from the top with the ranked color for exactly its cell count rows, and all remaining pixels are black.

    Key insight:
    - The grey lines define the counting units: a color's rank depends on how many divided cells it occupies, not on pixel area or on raw connected components.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = _count_filled_cells(g)
    ordered = sorted(counts.items(), key=lambda item: item[1])
    rank_counts = [count for _color, count in ordered]
    assert len(set(rank_counts)) == len(rank_counts), "foreground cell counts must be unique"
    assert rank_counts[-1] <= g.shape[0], "output column height must fit within the grid height"

    out = np.full((g.shape[0], len(ordered)), BLACK, dtype=int)
    for col, (color, count) in enumerate(ordered):
        out[:count, col] = int(color)
    return out


def _sample_counts(rng: random.Random) -> list[int]:
    n_colors = rng.randint(1, min(len(COLOR_POOL), MAX_COLOR_CELLS))
    counts = rng.sample(range(1, MAX_COLOR_CELLS + 1), n_colors)
    rng.shuffle(counts)
    return counts


def _candidate_cell_shapes(required_cells: int, max_count: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for n_rows in range(2, 6):
        for n_cols in range(2, 7):
            total = n_rows * n_cols
            if total < required_cells:
                continue
            if (total + 1) // 2 < max_count:
                continue
            shapes.append((n_rows, n_cols))
    return shapes


def _sample_band_sizes(
    rng: random.Random,
    n_bands: int,
    max_band: int,
    min_full_size: int = 0,
) -> list[int] | None:
    for _ in range(200):
        sizes = [rng.randint(1, max_band) for _ in range(n_bands)]
        full_size = sum(sizes) + n_bands - 1
        if min_full_size <= full_size <= MAX_GRID_SIZE:
            return sizes
    return None


def _neighbors(pos: tuple[int, int], n_rows: int, n_cols: int) -> set[tuple[int, int]]:
    r, c = pos
    out: set[tuple[int, int]] = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            out.add((nr, nc))
    return out


def _independent_choices(
    candidates: list[tuple[int, int]],
    neighbors: dict[tuple[int, int], set[tuple[int, int]]],
    need: int,
    start: int = 0,
    blocked: set[tuple[int, int]] | None = None,
    chosen: list[tuple[int, int]] | None = None,
):
    if blocked is None:
        blocked = set()
    if chosen is None:
        chosen = []
    if len(chosen) == need:
        yield tuple(chosen)
        return
    if len(candidates) - start < need - len(chosen):
        return

    for idx in range(start, len(candidates)):
        pos = candidates[idx]
        if pos in blocked:
            continue
        chosen.append(pos)
        next_blocked = blocked | {pos} | neighbors[pos]
        yield from _independent_choices(candidates, neighbors, need, idx + 1, next_blocked, chosen)
        chosen.pop()


def _place_color_cells(
    rng: random.Random, n_rows: int, n_cols: int, counts: list[int]
) -> list[list[tuple[int, int]]] | None:
    positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    neighbors = {pos: _neighbors(pos, n_rows, n_cols) for pos in positions}
    order = sorted(range(len(counts)), key=lambda idx: -counts[idx])
    placements: list[list[tuple[int, int]] | None] = [None] * len(counts)
    used: set[tuple[int, int]] = set()

    def backtrack(step: int) -> bool:
        if step == len(order):
            return True

        color_idx = order[step]
        need = counts[color_idx]
        available = [pos for pos in positions if pos not in used]
        if len(available) < sum(counts[idx] for idx in order[step:]):
            return False
        rng.shuffle(available)

        tried = 0
        for choice in _independent_choices(available, neighbors, need):
            tried += 1
            for pos in choice:
                used.add(pos)
            placements[color_idx] = list(choice)
            if backtrack(step + 1):
                return True
            placements[color_idx] = None
            for pos in choice:
                used.remove(pos)
            if tried >= PLACEMENT_RETRIES:
                break
        return False

    if backtrack(0):
        return [cells for cells in placements if cells is not None]
    return None


def _paint_partition_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    color_cells: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_spans.append((cursor, cursor + size))
        cursor += size
        if idx + 1 < len(row_sizes):
            grid[cursor, :] = GREY
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_spans.append((cursor, cursor + size))
        cursor += size
        if idx + 1 < len(col_sizes):
            grid[:, cursor] = GREY
            cursor += 1

    for color, cells in zip(colors, color_cells):
        for r_idx, c_idx in cells:
            r0, r1 = row_spans[r_idx]
            c0, c1 = col_spans[c_idx]
            grid[r0:r1, c0:c1] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts = _sample_counts(rng)
    required_cells = sum(counts)
    shapes = _candidate_cell_shapes(required_cells, max(counts))
    if not shapes:
        return None

    n_rows, n_cols = rng.choice(shapes)
    max_col_band = max(1, min(5, (MAX_GRID_SIZE - (n_cols - 1)) // n_cols))
    row_sizes = _sample_band_sizes(rng, n_rows, 5, min_full_size=max(counts))
    col_sizes = _sample_band_sizes(rng, n_cols, max_col_band)
    if row_sizes is None or col_sizes is None:
        return None

    color_cells = _place_color_cells(rng, n_rows, n_cols, counts)
    if color_cells is None:
        return None

    colors = rng.sample(COLOR_POOL, len(counts))
    rng.shuffle(colors)
    grid = _paint_partition_grid(row_sizes, col_sizes, color_cells, colors)
    observed = sorted(_count_filled_cells(grid).values())
    if observed != sorted(counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) <= MAX_GRID_SIZE and max(out.shape) <= MAX_GRID_SIZE:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid grid")
