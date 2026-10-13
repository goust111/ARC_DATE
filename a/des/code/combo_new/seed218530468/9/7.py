from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
VOTE_COLORS = (BLUE, RED, GREEN)

MAX_GRID_SIDE = 30
MIN_PARTITIONS = 4
MAX_PARTITIONS = 6
MIN_CELL_SIZE = 2
MAX_CELL_SIZE = 5
GENERATION_ATTEMPTS = 400
BOUNDARY_MISMATCH_PROB = 0.72
MULTICOLOR_REGION_PROB = 0.65


def _uniform_color(line: np.ndarray) -> int | None:
    if line.size == 0:
        return None
    color = int(line[0])
    return color if bool(np.all(line == color)) else None


def _region_intervals(separator_lines: list[int], size: int) -> list[tuple[int, int]]:
    marks = [-1] + sorted(set(int(x) for x in separator_lines)) + [int(size)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(marks, marks[1:]):
        if int(right - left) > 1:
            intervals.append((int(left + 1), int(right)))
    return intervals


def _parse_partition_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    full_rows: dict[int, list[int]] = {}
    for r in range(height):
        color = _uniform_color(g[r, :])
        if color is not None:
            full_rows.setdefault(color, []).append(r)

    full_cols: dict[int, list[int]] = {}
    for c in range(width):
        color = _uniform_color(g[:, c])
        if color is not None:
            full_cols.setdefault(color, []).append(c)

    candidates = []
    for color in set(full_rows) & set(full_cols):
        row_regions = _region_intervals(full_rows[color], height)
        col_regions = _region_intervals(full_cols[color], width)
        if len(row_regions) < 3 or len(col_regions) < 3:
            continue
        row_sizes = {int(r1 - r0) for r0, r1 in row_regions}
        col_sizes = {int(c1 - c0) for c0, c1 in col_regions}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        candidates.append(
            (
                len(full_rows[color]) + len(full_cols[color]),
                len(row_regions) * len(col_regions),
                int(color),
                row_regions,
                col_regions,
            )
        )

    assert candidates, "failed to recover the separator grid"
    _score, _cells, separator, row_regions, col_regions = max(candidates)
    return separator, row_regions, col_regions


def _vote_winner(region: np.ndarray) -> int:
    counts = [int(np.count_nonzero(region == color)) for color in VOTE_COLORS]
    best = max(counts)
    assert best > 0, "each partition must contain at least one vote pixel"
    assert counts.count(best) == 1, "partition vote must have a unique winner"
    return int(VOTE_COLORS[counts.index(best)])


def _boundary_sources(i: int, j: int, rows: int, cols: int) -> list[tuple[int, int]]:
    source_i = 1 if i == 0 else rows - 2 if i == rows - 1 else i
    source_j = 1 if j == 0 else cols - 2 if j == cols - 1 else j
    return [(int(source_i), int(source_j))]


def _resolved_partition_colors(core_winners: np.ndarray) -> np.ndarray:
    winners = np.asarray(core_winners, dtype=int).copy()
    rows, cols = map(int, winners.shape)
    for i in range(rows):
        for j in range(cols):
            if 0 < i < rows - 1 and 0 < j < cols - 1:
                continue
            src_colors = [int(winners[r, c]) for r, c in _boundary_sources(i, j, rows, cols)]
            counts = Counter(src_colors)
            best_count = counts.most_common(1)[0][1]
            assert list(counts.values()).count(best_count) == 1, "boundary inheritance must be unambiguous"
            winners[i, j] = int(counts.most_common(1)[0][0])
    return winners


def _solve_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    _separator, row_regions, col_regions = _parse_partition_grid(g)
    rows, cols = len(row_regions), len(col_regions)

    core_winners = np.zeros((rows, cols), dtype=int)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            r0, r1 = row_regions[i]
            c0, c1 = col_regions[j]
            core_winners[i, j] = _vote_winner(g[r0:r1, c0:c1])

    return row_regions, col_regions, _resolved_partition_colors(core_winners)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill table cells from core-majority votes and let boundary cells inherit inward core winners.

    Input structure:
    - Grey full rows and columns partition the canvas into an m-by-n table of equal rectangular cells, with m,n >= 3.
    - Inside each cell, black is the empty background and red/blue/green single-pixel votes are scattered.
    - Cells on the outer ring are boundary cells; the remaining cells are core cells.

    Rule:
    - For each core cell, count its red, blue, and green pixels and take the unique majority color.
    - Repaint the entire core cell interior to that winner color.
    - Boundary cells ignore their own vote pixels and instead copy the winner color from the inward neighboring core cell(s); in the regular table layout here this is the core cell directly behind that side, or the diagonally inward core cell at a corner.
    - Grey separator lines stay unchanged.

    Key insight:
    - The first pattern is a within-cell majority vote, but the real twist is that edge cells are not self-determined: they are colored by winners that radiate outward from the core.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_regions, col_regions, fill_colors = _solve_layout(g)
    out = g.copy()
    for i, (r0, r1) in enumerate(row_regions):
        for j, (c0, c1) in enumerate(col_regions):
            out[r0:r1, c0:c1] = int(fill_colors[i, j])
    return out


def _grid_size(partitions: int, cell_size: int) -> int:
    return int(partitions * cell_size + partitions + 1)


def _sample_counts(
    area: int,
    winner: int,
    rng: np.random.Generator,
    prefer_multicolor: bool,
) -> dict[int, int]:
    others = [color for color in VOTE_COLORS if color != winner]
    max_dots = min(int(area), max(1, int(area - area // 4)))

    for _ in range(120):
        min_winner = 2 if prefer_multicolor and area >= 3 else 1
        max_winner = max(1, max_dots - 1) if prefer_multicolor and area >= 3 else max_dots
        winner_count = int(rng.integers(min_winner, max_winner + 1))
        budget = max_dots - winner_count
        limit = max(0, winner_count - 1)
        first = int(rng.integers(0, min(limit, budget) + 1))
        budget -= first
        second = int(rng.integers(0, min(limit, budget) + 1))
        if prefer_multicolor and area >= 3 and first + second == 0:
            continue
        counts = {int(winner): winner_count, int(others[0]): first, int(others[1]): second}
        if counts[winner] > max(first, second):
            return counts

    return {int(winner): 1, int(others[0]): 0, int(others[1]): 0}


def _paint_region(
    shape: tuple[int, int],
    counts: dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    region = np.full(shape, BLACK, dtype=int)
    total = int(sum(counts.values()))
    assert 0 < total <= region.size
    picks = rng.choice(region.size, size=total, replace=False)
    colors: list[int] = []
    for color in VOTE_COLORS:
        colors.extend([int(color)] * int(counts.get(color, 0)))
    rng.shuffle(colors)
    for idx, color in zip(picks.tolist(), colors):
        region.flat[int(idx)] = int(color)
    return region


def _target_partition_colors(core_colors: np.ndarray) -> np.ndarray:
    full = np.zeros((core_colors.shape[0] + 2, core_colors.shape[1] + 2), dtype=int)
    full[1:-1, 1:-1] = np.asarray(core_colors, dtype=int)
    return _resolved_partition_colors(full)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(MIN_PARTITIONS, MAX_PARTITIONS + 1))
    cols = int(rng.integers(MIN_PARTITIONS, MAX_PARTITIONS + 1))

    max_cell_h = min(MAX_CELL_SIZE, (MAX_GRID_SIDE - rows - 1) // rows)
    max_cell_w = min(MAX_CELL_SIZE, (MAX_GRID_SIDE - cols - 1) // cols)
    if max_cell_h < MIN_CELL_SIZE or max_cell_w < MIN_CELL_SIZE:
        return None

    cell_h = int(rng.integers(MIN_CELL_SIZE, max_cell_h + 1))
    cell_w = int(rng.integers(MIN_CELL_SIZE, max_cell_w + 1))
    height = _grid_size(rows, cell_h)
    width = _grid_size(cols, cell_w)
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    inner_rows, inner_cols = rows - 2, cols - 2
    core_colors = rng.choice(VOTE_COLORS, size=(inner_rows, inner_cols))
    if inner_rows * inner_cols >= 2 and len(set(map(int, core_colors.flat))) < 2:
        flat_index = int(rng.integers(0, core_colors.size))
        choices = [color for color in VOTE_COLORS if color != int(core_colors.flat[flat_index])]
        core_colors.flat[flat_index] = int(rng.choice(choices))
    if inner_rows * inner_cols >= 3 and bool(rng.random() < 0.55):
        forced_cells = rng.choice(core_colors.size, size=3, replace=False)
        for idx, color in zip(forced_cells.tolist(), VOTE_COLORS):
            core_colors.flat[int(idx)] = int(color)

    fill_colors = _target_partition_colors(core_colors)
    own_major = fill_colors.copy()

    boundary_cells = [(i, j) for i in range(rows) for j in range(cols) if i in (0, rows - 1) or j in (0, cols - 1)]
    corner_cells = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    side_cells = [cell for cell in boundary_cells if cell not in corner_cells]
    forced_mismatch = {corner_cells[int(rng.integers(0, len(corner_cells)))]}
    if side_cells:
        forced_mismatch.add(side_cells[int(rng.integers(0, len(side_cells)))])

    for i, j in boundary_cells:
        target = int(fill_colors[i, j])
        should_mismatch = (i, j) in forced_mismatch or bool(rng.random() < BOUNDARY_MISMATCH_PROB)
        if should_mismatch:
            choices = [color for color in VOTE_COLORS if color != target]
            own_major[i, j] = int(rng.choice(choices))
        else:
            own_major[i, j] = target

    grid = np.full((height, width), GREY, dtype=int)
    for i in range(rows):
        r0 = 1 + i * (cell_h + 1)
        r1 = r0 + cell_h
        for j in range(cols):
            c0 = 1 + j * (cell_w + 1)
            c1 = c0 + cell_w
            area = cell_h * cell_w
            prefer_multicolor = bool(rng.random() < MULTICOLOR_REGION_PROB)
            counts = _sample_counts(area, int(own_major[i, j]), rng, prefer_multicolor)
            grid[r0:r1, c0:c1] = _paint_region((cell_h, cell_w), counts, rng)

    try:
        row_regions, col_regions, solved_colors = _solve_layout(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved_colors, fill_colors):
        return None
    if len(row_regions) != rows or len(col_regions) != cols:
        return None
    if len(set(map(int, np.unique(grid))) & set(VOTE_COLORS)) < 3:
        return None
    if len(set(map(int, fill_colors.flat))) < 2:
        return None

    corner_mismatches = 0
    side_mismatches = 0
    for i, (r0, r1) in enumerate(row_regions):
        for j, (c0, c1) in enumerate(col_regions):
            region_winner = _vote_winner(grid[r0:r1, c0:c1])
            if i in (0, rows - 1) or j in (0, cols - 1):
                if region_winner != int(fill_colors[i, j]):
                    if (i, j) in corner_cells:
                        corner_mismatches += 1
                    else:
                        side_mismatches += 1

    if corner_mismatches < 1 or side_mismatches < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
