import random

import numpy as np


BLACK = 0
RED = 2
NON_BLACK_COLORS = tuple(range(1, 10))
MODES = ("edge_sparse", "overlap", "saturated")
MAX_GENERATION_ATTEMPTS = 400


def _read_legend(grid: np.ndarray) -> list[int]:
    bottom = grid[-1]
    legend_len = 0
    while legend_len < grid.shape[1] and int(bottom[legend_len]) != BLACK:
        legend_len += 1
    legend = [int(v) for v in bottom[:legend_len]]
    assert legend, "bottom-row legend must be non-empty"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    assert bool((bottom[legend_len:] == BLACK).all()), "legend must be contiguous from the left edge"
    return legend


def _find_towers(grid: np.ndarray) -> list[tuple[int, int]]:
    towers = [tuple(map(int, pos)) for pos in np.argwhere(grid[:-1, :] == RED)]
    assert towers, "at least one red signal tower is required"
    return towers


def _coverage_counts(height: int, width: int, towers: list[tuple[int, int]]) -> np.ndarray:
    counts = np.zeros((height - 1, width), dtype=int)
    for r, c in towers:
        r0 = max(0, int(r - 2))
        r1 = min(height - 2, int(r + 2))
        c0 = max(0, int(c - 2))
        c1 = min(width - 1, int(c + 2))
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each non-legend cell by the number of 5x5 tower coverages that hit it.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert height >= 2 and width >= 1

    legend = _read_legend(g)
    towers = _find_towers(g)
    counts = _coverage_counts(height, width, towers)

    out = g.copy()
    palette = np.array([BLACK] + legend, dtype=int)
    out[:-1, :] = palette[np.clip(counts, 0, len(legend))]
    return out


def _all_tower_cells(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height - 1) for c in range(width)]


def _coverage_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) <= 4 and abs(int(a[1]) - int(b[1])) <= 4


def _is_edge_band(r: int, c: int, height: int, width: int) -> bool:
    return int(r) < 2 or int(c) < 2 or int(r) > height - 4 or int(c) > width - 3


def _choose_legend(width: int, mode: str) -> list[int]:
    if mode == "saturated":
        legend_len = random.randint(2, min(4, width))
    elif mode == "overlap":
        legend_len = random.randint(3, min(6, width))
    else:
        legend_len = random.randint(2, min(5, width))

    include_red = random.random() < 0.5
    if include_red:
        others = [c for c in NON_BLACK_COLORS if c != RED]
        legend = random.sample(others, legend_len - 1) + [RED]
        random.shuffle(legend)
        return legend

    pool = [c for c in NON_BLACK_COLORS if c != RED]
    return random.sample(pool, legend_len)


def _choose_interior_target(height: int, width: int) -> tuple[int, int]:
    row_lo, row_hi = 2, height - 4
    col_lo, col_hi = 2, width - 3
    if row_lo > row_hi or col_lo > col_hi:
        return random.randrange(height - 1), random.randrange(width)
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _cells_around(target: tuple[int, int], height: int, width: int) -> list[tuple[int, int]]:
    tr, tc = target
    cells = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = int(tr + dr)
            cc = int(tc + dc)
            if 0 <= rr < height - 1 and 0 <= cc < width:
                cells.append((rr, cc))
    random.shuffle(cells)
    return cells


def _gen_edge_sparse(height: int, width: int) -> list[tuple[int, int]] | None:
    all_cells = _all_tower_cells(height, width)
    edge_cells = [cell for cell in all_cells if _is_edge_band(cell[0], cell[1], height, width)]
    if not edge_cells:
        return None

    tower_count = random.randint(2, 4)
    for _ in range(80):
        first = random.choice(edge_cells)
        positions = [first]
        candidates = all_cells[:]
        random.shuffle(candidates)
        for pos in candidates:
            if pos == first:
                continue
            if any(_coverage_overlaps(pos, prev) for prev in positions):
                continue
            positions.append(pos)
            if len(positions) == tower_count:
                return positions
    return None


def _gen_overlap(height: int, width: int) -> list[tuple[int, int]] | None:
    target = _choose_interior_target(height, width)
    cluster_size = random.choice((2, 3))
    total_towers = random.randint(cluster_size + 1, cluster_size + 3)

    cluster = _cells_around(target, height, width)
    if len(cluster) < cluster_size:
        return None
    positions = cluster[:cluster_size]

    all_cells = _all_tower_cells(height, width)
    far_candidates = [
        cell
        for cell in all_cells
        if cell not in positions and all(not _coverage_overlaps(cell, prev) for prev in positions)
    ]
    random.shuffle(far_candidates)
    positions.extend(far_candidates[: total_towers - cluster_size])
    if len(positions) != total_towers:
        return None
    return positions


def _gen_saturated(height: int, width: int, legend_len: int) -> list[tuple[int, int]] | None:
    target = _choose_interior_target(height, width)
    max_towers = min(7, len(_cells_around(target, height, width)))
    min_towers = legend_len + 1
    if min_towers > max_towers:
        return None

    tower_count = random.randint(min_towers, max_towers)
    cluster = _cells_around(target, height, width)
    return cluster[:tower_count]


def _build_grid(height: int, width: int, legend: list[int], towers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for idx, color in enumerate(legend):
        grid[height - 1, idx] = int(color)
    for r, c in towers:
        grid[r, c] = RED
    return grid


def _valid_input(grid: np.ndarray, mode: str, legend: list[int], towers: list[tuple[int, int]]) -> bool:
    height, width = grid.shape
    if height < 2 or width < len(legend):
        return False
    if len(set(legend)) != len(legend):
        return False
    if not np.array_equal(grid[-1, len(legend) :], np.full(width - len(legend), BLACK, dtype=int)):
        return False
    if any(r >= height - 1 for r, _ in towers):
        return False
    if len(towers) != len(set(towers)):
        return False

    parsed_legend = _read_legend(grid)
    if parsed_legend != legend:
        return False

    counts = _coverage_counts(height, width, towers)
    covered = int(np.count_nonzero(counts > 0))
    uncovered = int(np.count_nonzero(counts == 0))
    if covered < 12 or uncovered < max(4, counts.size // 10):
        return False

    max_count = int(counts.max())
    if max_count < 1:
        return False
    if mode == "edge_sparse":
        if max_count != 1:
            return False
        if not any(_is_edge_band(r, c, height, width) for r, c in towers):
            return False
    elif mode == "overlap":
        if max_count < 2 or max_count > len(legend):
            return False
        if int(np.count_nonzero(counts >= 2)) < 3:
            return False
    elif mode == "saturated":
        if max_count <= len(legend):
            return False
        if int(np.count_nonzero(counts > len(legend))) < 1:
            return False
    else:
        return False

    out = main(grid)
    if not np.array_equal(out[-1], grid[-1]):
        return False
    if np.array_equal(out[:-1], grid[:-1]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    if mode == "edge_sparse":
        height = random.randint(9, 15)
        width = random.randint(9, 16)
    elif mode == "overlap":
        height = random.randint(10, 16)
        width = random.randint(10, 16)
    else:
        height = random.randint(11, 16)
        width = random.randint(11, 16)

    legend = _choose_legend(width, mode)

    if mode == "edge_sparse":
        towers = _gen_edge_sparse(height, width)
    elif mode == "overlap":
        towers = _gen_overlap(height, width)
    else:
        towers = _gen_saturated(height, width, len(legend))

    if towers is None:
        return None

    grid = _build_grid(height, width, legend, towers)
    if not _valid_input(grid, mode, legend, towers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
