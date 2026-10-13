import random
from collections import Counter

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
PALETTE = (BLUE, RED, GREEN)
MAX_ATTEMPTS = 300

ROW_PERMUTATIONS = {
    1: {RED: BLUE, BLUE: GREEN, GREEN: RED},
    2: {RED: GREEN, BLUE: RED, GREEN: BLUE},
    3: {RED: BLUE, BLUE: RED, GREEN: GREEN},
}


def _count_segments(row: np.ndarray) -> int:
    """Count contiguous non-background runs in one row."""
    row = np.asarray(row, dtype=int)
    if row.size == 0:
        return 0
    starts = (row != BACKGROUND) & np.concatenate(([True], row[:-1] == BACKGROUND))
    return int(np.count_nonzero(starts))


def _apply_row_permutation(row: np.ndarray, segment_count: int) -> np.ndarray:
    out = np.array(row, dtype=int, copy=True)
    mapping = ROW_PERMUTATIONS.get(int(segment_count))
    if mapping is None:
        return out
    for src, dst in mapping.items():
        out[row == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each row according to how many foreground segments it contains.

    Input structure:
    - The grid is a small black-background canvas with foreground colors blue, red, and green.
    - Each non-empty row contains one to three contiguous non-black segments.
    - A segment is any maximal horizontal run of non-black cells, even if colors change inside the run.

    Rule:
    - Scan each row independently and count its non-black segments.
    - If a row has one segment, apply the permutation red->blue, blue->green, green->red to every foreground cell in that row.
    - If a row has two segments, apply red->green, blue->red, green->blue.
    - If a row has three segments, swap red and blue while leaving green unchanged; background always stays black.

    Key insight:
    - The operative color mapping is not global: each row chooses its own permutation from its geometric segment count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r in range(g.shape[0]):
        count = _count_segments(g[r])
        out[r] = _apply_row_permutation(g[r], count)
    return out


def _build_row(width: int, segment_count: int, required_colors: list[int]) -> np.ndarray:
    assert 1 <= segment_count <= 3

    usable = width - (segment_count - 1)
    min_cells = max(segment_count, len(required_colors), 3)
    max_cells = min(usable, segment_count * 5)
    total_cells = random.randint(min_cells, max_cells)

    lengths = [1] * segment_count
    remaining = total_cells - segment_count
    capacities = [4] * segment_count
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacities) if cap > 0]
        idx = random.choice(choices)
        lengths[idx] += 1
        capacities[idx] -= 1
        remaining -= 1

    extra_spaces = width - total_cells - (segment_count - 1)
    spaces = [0] * (segment_count + 1)
    for _ in range(extra_spaces):
        spaces[random.randrange(segment_count + 1)] += 1

    row = np.zeros(width, dtype=int)
    spans: list[tuple[int, int]] = []
    foreground_positions: list[int] = []

    pos = spaces[0]
    for i, length in enumerate(lengths):
        start = pos
        end = pos + length
        spans.append((start, end))
        foreground_positions.extend(range(start, end))
        pos = end
        if i < segment_count - 1:
            pos += 1 + spaces[i + 1]

    colors = [random.choice(PALETTE) for _ in foreground_positions]
    if required_colors:
        forced = random.sample(range(len(colors)), len(required_colors))
        for idx, color in zip(forced, required_colors):
            colors[idx] = color

    multicolor_segments = [i for i, length in enumerate(lengths) if length >= 2]
    if multicolor_segments and random.random() < 0.75:
        seg_idx = random.choice(multicolor_segments)
        seg_offset = sum(lengths[:seg_idx])
        first, second = random.sample(PALETTE, 2)
        colors[seg_offset] = first
        colors[seg_offset + 1] = second

    for pos, color in zip(foreground_positions, colors):
        row[pos] = color
    return row


def _generate_candidate() -> np.ndarray:
    height = random.randint(8, 14)
    width = random.randint(10, 20)

    row_specs = [
        (1, [BLUE, RED]),
        (1, [GREEN]),
        (2, [BLUE, RED]),
        (2, [GREEN]),
        (3, [BLUE, RED]),
        (3, [GREEN]),
    ]

    extras = height - len(row_specs)
    extra_counts = random.choices([0, 1, 2, 3], weights=[2, 3, 3, 3], k=extras)
    row_specs.extend((count, []) for count in extra_counts)
    random.shuffle(row_specs)

    grid = np.zeros((height, width), dtype=int)
    for r, (segment_count, required_colors) in enumerate(row_specs):
        if segment_count == 0:
            continue
        grid[r] = _build_row(width, segment_count, required_colors)
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(np.unique(grid)).issubset({BACKGROUND, BLUE, RED, GREEN}):
        return False

    segment_freq = Counter()
    color_coverage = {1: set(), 2: set(), 3: set()}

    for row in grid:
        segments = _count_segments(row)
        if segments not in (0, 1, 2, 3):
            return False
        segment_freq[segments] += 1
        if segments in color_coverage:
            color_coverage[segments].update(int(v) for v in row if v != BACKGROUND)

    if any(segment_freq[count] < 2 for count in (1, 2, 3)):
        return False
    if any(color_coverage[count] != set(PALETTE) for count in (1, 2, 3)):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_candidate()
        if _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
