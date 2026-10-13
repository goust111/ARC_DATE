from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300
PALETTE_LEN = 5


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _palette_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if w >= PALETTE_LEN:
        mask[0, :PALETTE_LEN] = True
    return mask


def _find_seed_and_wall(
    grid: np.ndarray,
    bg: int,
    palette_colors: list[int],
    palette_mask: np.ndarray,
) -> tuple[int, int]:
    colors = {
        int(v)
        for v in np.unique(grid)
        if int(v) != bg and int(v) not in set(palette_colors)
    }
    assert len(colors) == 2, "expected exactly seed and wall colors besides palette"

    summaries: list[tuple[int, bool, int, int]] = []
    for color in colors:
        comps = _components((grid == color) & ~palette_mask)
        sizes = [len(comp) for comp in comps]
        max_size = max(sizes, default=0)
        is_seed_like = max_size == 1
        summaries.append((color, is_seed_like, max_size, int(sum(sizes))))

    seed_like = [item for item in summaries if item[1]]
    if len(seed_like) == 1:
        seed = int(seed_like[0][0])
    else:
        seed = min(summaries, key=lambda item: (item[2], item[3]))[0]
    wall = next(color for color in colors if color != seed)
    return int(seed), int(wall)


def _closed_regions(grid: np.ndarray, bg: int, seed: int) -> list[list[tuple[int, int]]]:
    palette_mask = _palette_mask(grid.shape)
    passable = ((grid == bg) | (grid == seed)) & ~palette_mask
    h, w = grid.shape
    regions = [comp for comp in _components(passable) if not _touches_border(comp, h, w)]
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the blue seed pixels inside each enclosed region, use that count as an
    index into the top-left 1x5 palette strip, and fill the region accordingly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= PALETTE_LEN

    bg = _most_common_color(g)
    palette = [int(v) for v in g[0, :PALETTE_LEN]]
    assert len(set(palette)) == PALETTE_LEN

    palette_mask = _palette_mask(g.shape)
    seed, wall = _find_seed_and_wall(g, bg, palette, palette_mask)
    assert seed != wall

    out = g.copy()
    for comp in _closed_regions(g, bg, seed):
        k = sum(int(g[r, c] == seed) for r, c in comp)
        assert 1 <= k <= PALETTE_LEN
        fill_color = int(palette[k - 1])
        for r, c in comp:
            if int(g[r, c]) == seed:
                out[r, c] = bg
            elif int(g[r, c]) == bg:
                out[r, c] = fill_color

    return out


def _partition_labels(n_rows: int, n_cols: int, n_regions: int) -> np.ndarray:
    labels = -np.ones((n_rows, n_cols), dtype=int)
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for label, (r, c) in enumerate(random.sample(cells, n_regions)):
        labels[r, c] = label

    unassigned = {cell for cell in cells if labels[cell] == -1}
    while unassigned:
        growable: list[tuple[int, int, list[int]]] = []
        for r, c in unassigned:
            neighbors = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and labels[nr, nc] != -1:
                    neighbors.append(int(labels[nr, nc]))
            if neighbors:
                growable.append((r, c, neighbors))
        assert growable
        r, c, neighbors = random.choice(growable)
        labels[r, c] = random.choice(neighbors)
        unassigned.remove((r, c))
    return labels


def _choose_spread_points(coords: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    coord_set = set(coords)
    preferred = [
        (r, c)
        for r, c in coords
        if all((r + dr, c + dc) in coord_set for dr, dc in DIR4)
    ]
    pool = preferred if len(preferred) >= k else list(coords)

    chosen = [random.choice(pool)]
    remaining = [p for p in pool if p != chosen[0]]
    while len(chosen) < k:
        best = None
        best_score = -1
        random.shuffle(remaining)
        for point in remaining:
            score = min(abs(point[0] - r) + abs(point[1] - c) for r, c in chosen)
            if score > best_score:
                best_score = score
                best = point
        assert best is not None
        chosen.append(best)
        remaining = [p for p in remaining if p != best]
    return chosen


def _draw_regions(
    labels: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    top: int,
    left: int,
    bottom_margin: int,
    right_margin: int,
    wall: int,
    bg: int,
    palette: list[int],
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    n_rows, n_cols = labels.shape

    row_ranges: list[tuple[int, int]] = []
    pos = top + 1
    for height in row_heights:
        row_ranges.append((pos, pos + height - 1))
        pos += height + 1
    bottom_wall = pos - 1

    col_ranges: list[tuple[int, int]] = []
    pos = left + 1
    for width in col_widths:
        col_ranges.append((pos, pos + width - 1))
        pos += width + 1
    right_wall = pos - 1

    h = bottom_wall + 1 + bottom_margin
    w = right_wall + 1 + right_margin
    if w < PALETTE_LEN:
        return np.empty((0, 0), dtype=int), []

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :PALETTE_LEN] = np.array(palette, dtype=int)

    top_wall = top
    left_wall = left
    grid[top_wall, left_wall : right_wall + 1] = wall
    grid[bottom_wall, left_wall : right_wall + 1] = wall
    grid[top_wall : bottom_wall + 1, left_wall] = wall
    grid[top_wall : bottom_wall + 1, right_wall] = wall

    for r in range(n_rows - 1):
        sep_r = row_ranges[r][1] + 1
        for c in range(n_cols):
            if int(labels[r, c]) != int(labels[r + 1, c]):
                c0, c1 = col_ranges[c]
                grid[sep_r, c0 : c1 + 1] = wall

    for r in range(n_rows):
        r0, r1 = row_ranges[r]
        for c in range(n_cols - 1):
            if int(labels[r, c]) != int(labels[r, c + 1]):
                sep_c = col_ranges[c][1] + 1
                grid[r0 : r1 + 1, sep_c] = wall

    for r in range(n_rows - 1):
        sep_r = row_ranges[r][1] + 1
        for c in range(n_cols - 1):
            sep_c = col_ranges[c][1] + 1
            around = {
                int(labels[r, c]),
                int(labels[r + 1, c]),
                int(labels[r, c + 1]),
                int(labels[r + 1, c + 1]),
            }
            if len(around) != 1:
                grid[sep_r, sep_c] = wall

    regions = _closed_regions(grid, bg, seed=bg + 1_000_000)
    return grid, regions


def _expected_output(
    grid: np.ndarray,
    regions: list[list[tuple[int, int]]],
    seed_counts: list[int],
    palette: list[int],
    seed: int,
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for comp, count in zip(regions, seed_counts):
        fill = int(palette[count - 1])
        for r, c in comp:
            if int(grid[r, c]) == seed:
                out[r, c] = bg
            elif int(grid[r, c]) == bg:
                out[r, c] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    colors = random.sample(list(range(1, 10)), 7)
    wall, seed = colors[0], colors[1]
    palette = colors[2:]

    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 5)
    max_regions = min(5, n_rows * n_cols)
    if max_regions < 3:
        return None
    n_regions = random.randint(3, max_regions)

    labels = _partition_labels(n_rows, n_cols, n_regions)
    row_heights = [random.randint(3, 6) for _ in range(n_rows)]
    col_widths = [random.randint(3, 6) for _ in range(n_cols)]
    top = random.randint(2, 5)
    left = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    grid, regions = _draw_regions(
        labels,
        row_heights,
        col_widths,
        top,
        left,
        bottom_margin,
        right_margin,
        wall,
        bg,
        palette,
    )
    if grid.size == 0 or len(regions) != n_regions:
        return None

    seed_counts = random.sample(list(range(1, PALETTE_LEN + 1)), n_regions)
    for comp, count in zip(regions, seed_counts):
        if len(comp) <= count:
            return None

    for comp, count in zip(regions, seed_counts):
        points = _choose_spread_points(comp, count)
        for r, c in points:
            grid[r, c] = seed

    expected = _expected_output(grid, regions, seed_counts, palette, seed, bg)

    if _most_common_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
