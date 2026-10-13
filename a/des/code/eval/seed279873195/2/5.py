import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
VOTE_COLORS = (1, 2, 3)  # blue, red, green
GEN_RETRIES = 300


def _infer_separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = [int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    shared = [color for color in set(row_colors) & set(col_colors) if color != BG]
    assert shared, "expected full separator rows and columns"
    return max(shared, key=lambda color: (row_colors.count(color) + col_colors.count(color), color))


def _cell_intervals(full_separator_mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(full_separator_mask)
    assert idx.size >= 2, "expected at least two separator lines"

    groups: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        cur = int(raw)
        if cur == prev + 1:
            prev = cur
            continue
        groups.append((start, prev))
        start = prev = cur
    groups.append((start, prev))

    assert groups[0][0] == 0, "expected an outer separator border"
    assert groups[-1][1] == full_separator_mask.size - 1, "expected an outer separator border"

    intervals: list[tuple[int, int]] = []
    for (_, left_end), (right_start, _) in zip(groups, groups[1:]):
        assert right_start > left_end + 1, "separators must enclose non-empty cells"
        intervals.append((left_end + 1, right_start))

    sizes = {end - start for start, end in intervals}
    assert len(sizes) == 1, "all cells must have the same size"
    return intervals


def _winner_for_cell(cell: np.ndarray, separator: int) -> int:
    counts: dict[int, int] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for r in range(cell.shape[0]):
        for c in range(cell.shape[1]):
            color = int(cell[r, c])
            if color in (BG, separator):
                continue
            counts[color] = counts.get(color, 0) + 1
            first_seen.setdefault(color, (r, c))

    assert counts, "each cell must contain at least one vote pixel"
    best = max(counts.values())
    candidates = [color for color, count in counts.items() if count == best]
    return min(candidates, key=lambda color: first_seen[color])


def _winner_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    separator = _infer_separator_color(grid)
    row_intervals = _cell_intervals(np.all(grid == separator, axis=1))
    col_intervals = _cell_intervals(np.all(grid == separator, axis=0))

    cell_heights = {r1 - r0 for r0, r1 in row_intervals}
    cell_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "uniform cells required"

    cell_h = next(iter(cell_heights))
    cell_w = next(iter(cell_widths))
    assert cell_h == cell_w, "cells must be square"

    winners = np.zeros((len(row_intervals), len(col_intervals)), dtype=int)
    for rr, (r0, r1) in enumerate(row_intervals):
        for cc, (c0, c1) in enumerate(col_intervals):
            winners[rr, cc] = _winner_for_cell(grid[r0:r1, c0:c1], separator)
    return winners, cell_h, cell_w


def _render_filled(winners: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    return np.repeat(np.repeat(winners, cell_h, axis=0), cell_w, axis=1)


def _outline_regions(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    out = np.full_like(filled, BG)
    for r in range(h):
        for c in range(w):
            color = int(filled[r, c])
            if color == BG:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = color
                continue
            if (
                int(filled[r - 1, c]) != color
                or int(filled[r + 1, c]) != color
                or int(filled[r, c - 1]) != color
                or int(filled[r, c + 1]) != color
            ):
                out[r, c] = color
    return out


def _has_same_color_adjacency(winners: np.ndarray) -> bool:
    rows, cols = winners.shape
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and int(winners[r, c]) == int(winners[r + 1, c]):
                return True
            if c + 1 < cols and int(winners[r, c]) == int(winners[r, c + 1]):
                return True
    return False


def _row_major_positions(cell_size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(cell_size) for c in range(cell_size)]


def _sample_positions(cell_size: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    return rng.sample(_row_major_positions(cell_size), count)


def _make_vote_pixels(winner: int, cell_size: int, tie_mode: bool, rng: random.Random) -> list[tuple[int, int, int]]:
    others = [color for color in VOTE_COLORS if color != winner]
    if tie_mode:
        loser = rng.choice(others)
        copies = rng.choice((1, 2))
        positions = sorted(_sample_positions(cell_size, 2 * copies, rng))
        colors = [winner, loser]
        if copies == 2:
            tail = [winner, loser]
            rng.shuffle(tail)
            colors.extend(tail)
        return [(r, c, color) for (r, c), color in zip(positions, colors)]

    if rng.random() < 0.25:
        total = rng.randint(2, 5)
        positions = _sample_positions(cell_size, total, rng)
        return [(r, c, winner) for r, c in positions]

    loser = rng.choice(others)
    total = rng.randint(3, 5)
    winner_count = rng.randint(total // 2 + 1, total - 1)
    loser_count = total - winner_count
    positions = _sample_positions(cell_size, total, rng)
    colors = [winner] * winner_count + [loser] * loser_count
    rng.shuffle(colors)
    return [(r, c, color) for (r, c), color in zip(positions, colors)]


def _choose_winners(rows: int, cols: int, rng: random.Random) -> np.ndarray | None:
    palette = rng.sample(VOTE_COLORS, rng.randint(2, 3))
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    seed_count = rng.randint(len(palette), min(rows * cols, len(palette) + 3))
    seed_positions = rng.sample(all_cells, seed_count)

    seed_colors = palette[:]
    while len(seed_colors) < seed_count:
        seed_colors.append(rng.choice(palette))
    rng.shuffle(seed_colors)

    winners = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            best_idx = min(
                range(seed_count),
                key=lambda idx: (abs(r - seed_positions[idx][0]) + abs(c - seed_positions[idx][1]), idx),
            )
            winners[r, c] = seed_colors[best_idx]

    for _ in range(rng.randint(0, max(1, rows * cols // 6))):
        r = rng.randrange(rows)
        c = rng.randrange(cols)
        neighbors = []
        if r > 0:
            neighbors.append((r - 1, c))
        if r + 1 < rows:
            neighbors.append((r + 1, c))
        if c > 0:
            neighbors.append((r, c - 1))
        if c + 1 < cols:
            neighbors.append((r, c + 1))
        if neighbors and rng.random() < 0.7:
            nr, nc = rng.choice(neighbors)
            winners[r, c] = winners[nr, nc]
        else:
            winners[r, c] = rng.choice(palette)

    if np.unique(winners).size < 2:
        return None
    if not _has_same_color_adjacency(winners):
        return None
    return winners


def _render_input(
    winners: np.ndarray,
    cell_size: int,
    tie_cells: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    rows, cols = winners.shape
    h = rows * cell_size + rows + 1
    w = cols * cell_size + cols + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[:: cell_size + 1, :] = GREY
    grid[:, :: cell_size + 1] = GREY

    for r in range(rows):
        for c in range(cols):
            votes = _make_vote_pixels(int(winners[r, c]), cell_size, (r, c) in tie_cells, rng)
            base_r = 1 + r * (cell_size + 1)
            base_c = 1 + c * (cell_size + 1)
            for vr, vc, color in votes:
                grid[base_r + vr, base_c + vc] = color
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 6)
    cols = rng.randint(3, 6)
    cell_size = rng.randint(4, 6)

    winners = _choose_winners(rows, cols, rng)
    if winners is None:
        return None

    require_tie = rng.random() < 0.5
    tie_count = 0
    if require_tie:
        max_ties = max(1, min(rows * cols - 1, rows * cols // 3))
        tie_count = rng.randint(1, max_ties)

    tie_cells = set(rng.sample([(r, c) for r in range(rows) for c in range(cols)], tie_count)) if tie_count else set()
    grid = _render_input(winners, cell_size, tie_cells, rng)

    expected = _outline_regions(_render_filled(winners, cell_size, cell_size))
    if not np.array_equal(main(grid), expected):
        return None

    if require_tie and not tie_cells:
        return None
    if np.count_nonzero(expected) == 0:
        return None
    if np.unique(expected[expected != BG]).size < 2:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a partitioned board, each cell votes for a fill color, equal-color cells merge,
    and only the merged regions' outlines remain after removing the separator lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    winners, cell_h, cell_w = _winner_grid(g)
    filled = _render_filled(winners, cell_h, cell_w)
    return _outline_regions(filled)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
