import random
from collections import Counter

import numpy as np


LINE_COLOR = 4
BG_COLOR = 0
CELL_SIDE_OPTIONS = (4, 5, 6)
CELL_COUNT_OPTIONS = (2, 3, 4)
SHAPE_SIZE_MIN = 3
SHAPE_SIZE_MAX = 8
FOREGROUND_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GENERATION_RETRIES = 300


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in separators) + [int(length)]
    bands: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            bands.append((a + 1, b))
    return bands


def _partition_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == LINE_COLOR)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == LINE_COLOR)]
    assert row_separators and col_separators, "expected full separator rows and columns"

    row_bands = _bands(grid.shape[0], row_separators)
    col_bands = _bands(grid.shape[1], col_separators)
    assert row_bands and col_bands, "expected non-empty partition cells"

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "expected equal partitions"
    assert next(iter(heights)) == next(iter(widths)), "expected square cells"
    return row_bands, col_bands


def _count_foreground(cell: np.ndarray) -> int:
    return int(np.count_nonzero((cell != BG_COLOR) & (cell != LINE_COLOR)))


def _unique_mode(values: list[int]) -> int:
    freq = Counter(values)
    mode_value, mode_count = max(freq.items(), key=lambda item: item[1])
    assert sum(count == mode_count for count in freq.values()) == 1, "expected strict mode"
    return int(mode_value)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every partition cell whose foreground pixel count equals the cross-cell mode.

    Input structure:
    - A square black grid is split by full yellow separator rows and columns into equal square cells.
    - Each separator-free cell contains one small single-color 4-connected foreground shape.
    - Foreground colors vary by cell, but the separators stay yellow and the rest of the cell is black.

    Rule:
    - Detect the full yellow separator lines and recover the equal square cells between them.
    - Count, for each cell, how many pixels are foreground (neither black nor yellow).
    - Find the unique mode of those per-cell counts across the whole partition.
    - In every cell whose count equals that mode, erase its foreground pixels to black while keeping separator lines unchanged.

    Key insight:
    - The erase target is decided only after comparing foreground counts across all partition cells, not from any one cell's color, shape, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    row_bands, col_bands = _partition_cells(g)
    counts: list[int] = []
    cells: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for rb in row_bands:
        for cb in col_bands:
            r0, r1 = rb
            c0, c1 = cb
            cell = g[r0:r1, c0:c1]
            counts.append(_count_foreground(cell))
            cells.append((rb, cb))

    mode_count = _unique_mode(counts)
    out = g.copy()
    for (rb, cb), count in zip(cells, counts):
        if count != mode_count:
            continue
        r0, r1 = rb
        c0, c1 = cb
        cell = out[r0:r1, c0:c1]
        cell[(cell != BG_COLOR) & (cell != LINE_COLOR)] = BG_COLOR
    return out


def _build_connected_shape(side: int, size: int) -> np.ndarray:
    cells = {(random.randrange(side), random.randrange(side))}
    while len(cells) < size:
        base_r, base_c = random.choice(tuple(cells))
        neighbors = [
            (base_r - 1, base_c),
            (base_r + 1, base_c),
            (base_r, base_c - 1),
            (base_r, base_c + 1),
        ]
        candidates = [
            (r, c)
            for r, c in neighbors
            if 0 <= r < side and 0 <= c < side and (r, c) not in cells
        ]
        if candidates:
            cells.add(random.choice(candidates))

    patch = np.zeros((side, side), dtype=int)
    for r, c in cells:
        patch[r, c] = 1
    return patch


def _make_count_multiset(total_cells: int, max_size: int) -> list[int] | None:
    if total_cells < 4:
        return None

    size_values = list(range(SHAPE_SIZE_MIN, max_size + 1))
    random.shuffle(size_values)
    mode_value = random.choice(size_values)

    candidates = []
    for mode_freq in range(2, total_cells):
        remainder = total_cells - mode_freq
        if remainder == 0:
            continue
        if mode_freq > remainder:
            candidates.append(mode_freq)
        elif remainder >= 2:
            candidates.append(mode_freq)
    if not candidates:
        return None
    mode_freq = random.choice(candidates)

    counts = [mode_value] * mode_freq
    remaining = total_cells - mode_freq
    other_freq: Counter[int] = Counter()
    other_values = [value for value in size_values if value != mode_value]
    random.shuffle(other_values)

    while remaining > 0:
        valid_values = [value for value in other_values if other_freq[value] < mode_freq - 1]
        if not valid_values:
            return None
        value = random.choice(valid_values)
        max_block = min(mode_freq - 1 - other_freq[value], remaining)
        count_here = random.randint(1, max_block)
        other_freq[value] += count_here
        counts.extend([value] * count_here)
        remaining -= count_here

    if len(set(counts)) < 2:
        return None
    freq = Counter(counts)
    if freq[mode_value] != max(freq.values()):
        return None
    if sum(v == freq[mode_value] for v in freq.values()) != 1:
        return None

    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    cell_count = random.choice(CELL_COUNT_OPTIONS)
    cell_side = random.choice(CELL_SIDE_OPTIONS)
    grid_side = cell_count * cell_side + (cell_count - 1)
    if grid_side > 30:
        return None

    total_cells = cell_count * cell_count
    max_shape_size = min(SHAPE_SIZE_MAX, cell_side * cell_side - 1)
    if max_shape_size < SHAPE_SIZE_MIN:
        return None

    counts = _make_count_multiset(total_cells, max_shape_size)
    if counts is None:
        return None

    grid = np.zeros((grid_side, grid_side), dtype=int)
    for i in range(1, cell_count):
        sep = i * cell_side + (i - 1)
        grid[sep, :] = LINE_COLOR
        grid[:, sep] = LINE_COLOR

    colors = list(FOREGROUND_COLORS)
    if total_cells > len(colors):
        colors = [random.choice(FOREGROUND_COLORS) for _ in range(total_cells)]
    else:
        random.shuffle(colors)
        colors = colors[:total_cells]

    index = 0
    for gr in range(cell_count):
        for gc in range(cell_count):
            r0 = gr * (cell_side + 1)
            c0 = gc * (cell_side + 1)
            color = int(colors[index])
            patch = _build_connected_shape(cell_side, counts[index]) * color
            grid[r0 : r0 + cell_side, c0 : c0 + cell_side] = patch
            index += 1

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
