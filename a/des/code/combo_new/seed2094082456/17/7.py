import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATE_ATTEMPTS = 400


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        indices = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == BLACK))]
        edge_last = grid.shape[0] - 1
    else:
        indices = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == BLACK))]
        edge_last = grid.shape[1] - 1

    if not indices or indices[0] != 0:
        indices = [0] + indices
    if indices[-1] != edge_last:
        indices = indices + [edge_last]
    return indices


def _region_spans(separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for left, right in zip(separators, separators[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _neighbor_colors(region_colors: np.ndarray, r: int, c: int) -> list[int]:
    rows, cols = region_colors.shape
    neighbors: list[int] = []
    if r > 0:
        neighbors.append(int(region_colors[r - 1, c]))
    if r + 1 < rows:
        neighbors.append(int(region_colors[r + 1, c]))
    if c > 0:
        neighbors.append(int(region_colors[r, c - 1]))
    if c + 1 < cols:
        neighbors.append(int(region_colors[r, c + 1]))
    return neighbors


def _solve_region_colors(region_colors: np.ndarray) -> np.ndarray:
    out = np.array(region_colors, dtype=int, copy=True)
    rows, cols = out.shape
    for r in range(rows):
        for c in range(cols):
            neighbors = _neighbor_colors(region_colors, r, c)
            if neighbors and len(set(neighbors)) == 1:
                out[r, c] = neighbors[0]
    return out


def _extract_region_colors(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    row_spans = _region_spans(_separator_indices(grid, axis=0))
    col_spans = _region_spans(_separator_indices(grid, axis=1))
    region_colors = np.zeros((len(row_spans), len(col_spans)), dtype=int)

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = np.unique(block)
            colors = colors[colors != BLACK]
            assert len(colors) == 1, "each region must contain exactly one non-black color"
            region_colors[r, c] = int(colors[0])

    return row_spans, col_spans, region_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor any whole window cell whose existing neighbors all share one color.

    Input structure:
    - Black full rows and columns form a 1-pixel-thick outer frame and interior separators.
    - These separators partition the grid into rectangular regions.
    - Each region is uniformly filled with one non-black color, and colors may repeat across regions.

    Rule:
    - Keep every black separator row and column unchanged.
    - For each rectangular region, inspect the colors of the regions directly above, below, left, and right across one black separator line.
    - Ignore directions that do not exist because the region sits on the outer border.
    - If all existing neighbor colors are identical, repaint the entire region to that unanimous neighbor color.
    - Otherwise leave the region unchanged.

    Key insight:
    - The transformation is controlled by strict neighbor unanimity, not by majority voting or local pixel diffusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans, region_colors = _extract_region_colors(g)
    if region_colors.size == 0:
        return g.copy()

    solved_colors = _solve_region_colors(region_colors)
    out = g.copy()

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(solved_colors[r, c])

    return out


def _sample_sizes(count: int, limit: int = 30, min_size: int = 2, max_size: int = 6) -> list[int]:
    total_min = count * min_size
    total_max = min(count * max_size, limit - (count + 1))
    total = random.randint(total_min, total_max)

    sizes: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        lo = max(min_size, remaining - left * max_size)
        hi = min(max_size, remaining - left * min_size)
        sizes.append(random.randint(lo, hi))
        remaining -= sizes[-1]

    random.shuffle(sizes)
    return sizes


def _sample_region_colors(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    colors = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            candidates = [random.choice(palette)]
            if r > 0 and random.random() < 0.7:
                candidates.append(int(colors[r - 1, c]))
            if c > 0 and random.random() < 0.7:
                candidates.append(int(colors[r, c - 1]))
            if r > 0 and c > 0 and random.random() < 0.3:
                candidates.append(int(colors[r - 1, c - 1]))
            colors[r, c] = random.choice(candidates)

    noise_count = random.randint(1, max(1, rows * cols // 4))
    for _ in range(noise_count):
        rr = random.randrange(rows)
        cc = random.randrange(cols)
        colors[rr, cc] = random.choice(palette)

    return colors


def _render_grid(row_sizes: list[int], col_sizes: list[int], region_colors: np.ndarray) -> np.ndarray:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.zeros((height, width), dtype=int)

    r0 = 1
    for r, region_h in enumerate(row_sizes):
        c0 = 1
        for c, region_w in enumerate(col_sizes):
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = int(region_colors[r, c])
            c0 += region_w + 1
        r0 += region_h + 1

    return grid


def _example_is_good(region_colors: np.ndarray) -> bool:
    distinct_colors = set(map(int, region_colors.flatten()))
    if len(distinct_colors) < 2:
        return False

    solved = _solve_region_colors(region_colors)
    if not np.any(solved != region_colors):
        return False

    consensus_changes = 0
    conflict_keeps = 0
    edge_changes = 0
    rows, cols = region_colors.shape

    for r in range(rows):
        for c in range(cols):
            neighbors = _neighbor_colors(region_colors, r, c)
            unique = set(neighbors)
            if len(unique) == 1 and int(solved[r, c]) != int(region_colors[r, c]):
                consensus_changes += 1
                if r in (0, rows - 1) or c in (0, cols - 1):
                    edge_changes += 1
            elif len(unique) >= 2:
                conflict_keeps += 1

    if consensus_changes < 1 or conflict_keeps < 1:
        return False
    if edge_changes < 1:
        return False
    if len(set(map(int, solved.flatten()))) < 2:
        return False

    change_ratio = consensus_changes / float(region_colors.size)
    return 0.08 <= change_ratio <= 0.75


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        region_rows = random.randint(2, 6)
        region_cols = random.randint(2, 6)
        row_sizes = _sample_sizes(region_rows)
        col_sizes = _sample_sizes(region_cols)

        palette_size = random.choice((2, 3))
        palette = random.sample(COLOR_POOL, palette_size)
        region_colors = _sample_region_colors(region_rows, region_cols, palette)

        if not _example_is_good(region_colors):
            continue

        grid = _render_grid(row_sizes, col_sizes, region_colors)
        solved_colors = _solve_region_colors(region_colors)
        expected = _render_grid(row_sizes, col_sizes, solved_colors)

        if not np.array_equal(main(grid), expected):
            continue

        return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
