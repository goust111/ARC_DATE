from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_SIZE = 8
MAX_SIZE = 12
MIN_COLORS = 3
MAX_COLORS = 5
MAX_GENERATION_ATTEMPTS = 400
_RNG = random.Random()


def _unique_active_color(interior: np.ndarray) -> int:
    values = [int(v) for v in interior.flat if int(v) != BACKGROUND]
    assert values, "expected at least one non-background interior pixel"
    counts = Counter(values)
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique most frequent interior color"
    return int(ranked[0][0])


def _compute_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 3 <= height <= 30 and 3 <= width <= 30, "grid must fit ARC bounds"

    interior = g[:-1, :-1]
    row_labels = g[:-1, -1]
    col_labels = g[-1, :-1]
    corner = int(g[-1, -1])

    assert corner == BACKGROUND, "expected black bottom-right corner"
    assert not np.any(g == GREY), "grey is reserved for the output-only fallback color"
    assert np.all(row_labels != BACKGROUND), "expected non-background row labels"
    assert np.all(col_labels != BACKGROUND), "expected non-background column labels"
    assert len(set(map(int, row_labels))) >= 2, "row labels should use at least two colors"
    assert len(set(map(int, col_labels))) >= 2, "column labels should use at least two colors"

    label_colors = set(map(int, row_labels)) | set(map(int, col_labels))
    assert len(label_colors) >= 3, "expected at least three candidate label colors"
    assert set(int(v) for v in interior.flat if int(v) != BACKGROUND) <= label_colors, (
        "interior scatter colors must come from the label palette"
    )

    active = _unique_active_color(interior)

    out = g.copy()
    match = row_labels[:, None] == col_labels[None, :]
    out[:-1, :-1] = BACKGROUND
    out[:-1, :-1][match] = GREY
    out[:-1, :-1][(row_labels[:, None] == active) & match] = active
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely most frequent interior scatter color to choose which matching row/column labels stay colored.

    Input structure:
    - Black background fills the grid except for a right-edge row-label column, a bottom column-label row, and sparse interior scatter pixels.
    - Every interior row has one non-black label in the rightmost column, and every interior column has one non-black label in the bottom row.
    - The interior scatter pixels use the same candidate colors as the border labels, while the bottom-right corner stays black.

    Rule:
    - Count non-black pixels only inside the interior rectangle, excluding the bottom row and rightmost column.
    - Let A be the unique most frequent interior color. For each interior cell, compare its row label RL and column label CL.
    - If RL = CL = A, fill that cell with A; if RL = CL but that shared color is not A, fill it gray; if RL != CL, leave it black. Keep the label row and label column unchanged.

    Key insight:
    - The interior scatter is not decoration: it elects a single active color, and that statistical winner decides which label-matching cells stay fully colored and which matching cells collapse to gray.
    """
    return _compute_output(grid)


def _sample_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(_RNG.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _make_label_sequence(length: int, colors: list[int]) -> np.ndarray:
    order = colors[:]
    _RNG.shuffle(order)
    runs = _sample_partition(length, len(order))
    seq: list[int] = []
    for color, run in zip(order, runs):
        seq.extend([int(color)] * int(run))
    return np.array(seq, dtype=int)


def _sample_counts(colors: list[int], active: int, capacity: int) -> dict[int, int] | None:
    others = [c for c in colors if c != active]
    other_counts = {c: _RNG.randint(1, 4) for c in others}
    second = max(other_counts.values())
    active_count = second + _RNG.randint(1, 4)
    counts = {active: active_count, **other_counts}
    total = sum(counts.values())
    if total > min(capacity // 2, 18):
        return None
    return counts


def _cell_groups(row_labels: np.ndarray, col_labels: np.ndarray, active: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    active_match: list[tuple[int, int]] = []
    inactive_match: list[tuple[int, int]] = []
    mismatch: list[tuple[int, int]] = []
    for r, rl in enumerate(row_labels):
        for c, cl in enumerate(col_labels):
            rl_i = int(rl)
            cl_i = int(cl)
            if rl_i != cl_i:
                mismatch.append((r, c))
            elif rl_i == active:
                active_match.append((r, c))
            else:
                inactive_match.append((r, c))
    return active_match, inactive_match, mismatch


def _place_scatter(
    grid: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    counts: dict[int, int],
    active: int,
) -> bool:
    active_match, inactive_match, mismatch = _cell_groups(row_labels, col_labels, active)
    if not active_match or not inactive_match or len(mismatch) < 2:
        return False

    used: set[tuple[int, int]] = set()

    def take(pool: list[tuple[int, int]]) -> tuple[int, int]:
        options = [pos for pos in pool if pos not in used]
        if not options:
            raise ValueError("no free positions")
        pos = _RNG.choice(options)
        used.add(pos)
        return pos

    try:
        ar, ac = take(mismatch)
        grid[ar, ac] = active

        non_active_colors = [c for c in counts if c != active]
        seeded_other = _RNG.choice(non_active_colors)
        br, bc = take(mismatch)
        grid[br, bc] = seeded_other

        remaining: list[int] = []
        for color, amount in counts.items():
            if color == active:
                amount -= 1
            if color == seeded_other:
                amount -= 1
            remaining.extend([int(color)] * int(amount))

        all_cells = [(r, c) for r in range(grid.shape[0] - 1) for c in range(grid.shape[1] - 1)]
        free_cells = [pos for pos in all_cells if pos not in used]
        if len(free_cells) < len(remaining):
            return False
        chosen = _RNG.sample(free_cells, len(remaining))
        _RNG.shuffle(remaining)
        for (r, c), color in zip(chosen, remaining):
            grid[r, c] = int(color)
    except ValueError:
        return False

    return True


def _validate_grid(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    interior = grid[:-1, :-1]
    row_labels = grid[:-1, -1]
    col_labels = grid[-1, :-1]
    active = _unique_active_color(interior)

    row_counts = Counter(map(int, row_labels))
    col_counts = Counter(map(int, col_labels))
    eq_areas = {color: row_counts[color] * col_counts[color] for color in set(row_counts) | set(col_counts)}
    non_active_areas = [area for color, area in eq_areas.items() if color != active]
    if not non_active_areas or max(non_active_areas) < eq_areas[active]:
        return False

    active_match, inactive_match, mismatch = _cell_groups(row_labels, col_labels, active)
    if not active_match or not inactive_match or not mismatch:
        return False

    out_interior = out[:-1, :-1]
    if GREY not in set(map(int, out_interior.flat)):
        return False
    if active not in set(map(int, out_interior.flat)):
        return False
    if BACKGROUND not in set(map(int, out_interior.flat)):
        return False

    if np.array_equal(interior, out_interior):
        return False

    mismatched_active = [
        (r, c)
        for r in range(interior.shape[0])
        for c in range(interior.shape[1])
        if int(interior[r, c]) == active and int(row_labels[r]) != int(col_labels[c])
    ]
    if not mismatched_active:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(MIN_SIZE, MAX_SIZE)
    width = _RNG.randint(MIN_SIZE, MAX_SIZE)
    interior_h = height - 1
    interior_w = width - 1
    color_count = _RNG.randint(MIN_COLORS, min(MAX_COLORS, interior_h, interior_w, len(PALETTE)) + 1)
    colors = _RNG.sample(list(PALETTE), color_count)
    active = _RNG.choice(colors)

    row_labels = _make_label_sequence(interior_h, colors)
    col_labels = _make_label_sequence(interior_w, colors)

    counts = _sample_counts(colors, active, interior_h * interior_w)
    if counts is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:-1, -1] = row_labels
    grid[-1, :-1] = col_labels

    if not _place_scatter(grid, row_labels, col_labels, counts, active):
        return None

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
