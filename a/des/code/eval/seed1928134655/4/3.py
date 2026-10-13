import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    h = grid.shape[0]
    mapping: dict[int, int] = {}
    r = 0
    while r < h:
        if int(grid[r, 0]) == bg:
            r += 1
            continue
        assert r + 1 < h, "legend entries must be 2 pixels tall"
        seed = int(grid[r, 0])
        grow = int(grid[r + 1, 0])
        assert seed != bg and grow != bg, "legend cells must be non-background"
        assert seed != grow, "legend seed and growth colors must differ"
        assert seed not in mapping, "legend seed colors must be unique"
        mapping[seed] = grow
        r += 2
    assert mapping, "expected at least one legend entry"
    return mapping


def _detect_fence(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for color in np.unique(grid):
        c = int(color)
        if c == bg:
            continue
        cols = [col for col in range(1, w) if np.all(grid[:, col] == c)]
        if len(cols) >= 2:
            candidates.append((c, cols))
    assert candidates, "failed to find fence columns"

    fence, fence_cols = max(candidates, key=lambda item: (len(item[1]), int((grid == item[0]).sum())))
    first_col = min(fence_cols)
    fence_rows = [row for row in range(h) if np.all(grid[row, first_col:] == fence)]
    assert len(fence_rows) >= 2, "failed to find fence rows"
    return fence, sorted(fence_rows), sorted(fence_cols)


def _spans(separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(separators, separators[1:]):
        if a + 1 <= b - 1:
            spans.append((a + 1, b - 1))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left-edge legend, then grow each seed as a cross inside its fenced
    rectangle and fill the remaining corner cells with the mapped growth color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BACKGROUND
    assert np.any(g == bg), "expected black background pixels"
    legend = _parse_legend(g, bg)
    fence, fence_rows, fence_cols = _detect_fence(g, bg)

    out = g.copy()
    for r0, r1 in _spans(fence_rows):
        for c0, c1 in _spans(fence_cols):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            seeds = np.argwhere((block != bg) & (block != fence))
            assert len(seeds) == 1, "each fenced region must contain exactly one seed"

            sr, sc = map(int, seeds[0])
            seed_color = int(block[sr, sc])
            assert seed_color in legend, "every seed color must appear in the legend"
            grow_color = legend[seed_color]

            out[r0 : r1 + 1, c0 : c1 + 1] = grow_color
            out[r0 + sr, c0 : c1 + 1] = seed_color
            out[r0 : r1 + 1, c0 + sc] = seed_color

    return out


# --------------- generator ---------------

BACKGROUND = 0
FENCE = 7
USABLE_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
ROW_COUNT_RANGE = (1, 3)
COL_COUNT_RANGE = (1, 4)
CELL_SIZE_RANGE = (3, 8)
LEFT_GUTTER_CHOICES = (1, 2)
LEGEND_ENTRY_RANGE = (2, 4)
GEN_RETRIES = 300


def _rand_partition(count: int) -> list[int]:
    parts = [random.randint(*CELL_SIZE_RANGE) for _ in range(count)]
    if count > 1 and len(set(parts)) == 1:
        idx = random.randrange(count)
        delta = 1 if parts[idx] < CELL_SIZE_RANGE[1] else -1
        parts[idx] += delta
    return parts


def _build_fence_positions(sizes: list[int], start: int = 0) -> list[int]:
    positions = [start]
    cur = start
    for size in sizes:
        cur += size + 1
        positions.append(cur)
    return positions


def _region_spans(separators: list[int]) -> list[tuple[int, int]]:
    return [(a + 1, b - 1) for a, b in zip(separators, separators[1:])]


def _sample_assignments(seed_colors: list[int], region_count: int) -> list[int]:
    assignments = seed_colors.copy()
    while len(assignments) < region_count:
        assignments.append(random.choice(seed_colors))
    random.shuffle(assignments)
    return assignments


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*ROW_COUNT_RANGE)
    n_cols = random.randint(*COL_COUNT_RANGE)
    if n_rows * n_cols < 2:
        return None

    row_sizes = _rand_partition(n_rows)
    col_sizes = _rand_partition(n_cols)
    left_gutter = random.choice(LEFT_GUTTER_CHOICES)

    fence_rows = _build_fence_positions(row_sizes, start=0)
    fence_cols = _build_fence_positions(col_sizes, start=left_gutter)
    height = fence_rows[-1] + 1
    width = fence_cols[-1] + 1

    max_entries = min(LEGEND_ENTRY_RANGE[1], len(USABLE_COLORS) // 2, n_rows * n_cols, height // 2)
    if max_entries < LEGEND_ENTRY_RANGE[0]:
        return None
    legend_entries = random.randint(LEGEND_ENTRY_RANGE[0], max_entries)

    colors = USABLE_COLORS[:]
    random.shuffle(colors)
    seed_colors = colors[:legend_entries]
    grow_colors = colors[legend_entries : 2 * legend_entries]
    mapping = dict(zip(seed_colors, grow_colors))

    grid = np.full((height, width), BACKGROUND, dtype=int)

    top_pad = random.randint(0, height - 2 * legend_entries)
    for i, seed in enumerate(seed_colors):
        grid[top_pad + 2 * i, 0] = seed
        grid[top_pad + 2 * i + 1, 0] = mapping[seed]

    for col in fence_cols:
        grid[:, col] = FENCE
    for row in fence_rows:
        grid[row, fence_cols[0] :] = FENCE

    assignments = _sample_assignments(seed_colors, n_rows * n_cols)
    idx = 0
    off_center = False
    for r0, r1 in _region_spans(fence_rows):
        for c0, c1 in _region_spans(fence_cols):
            seed = assignments[idx]
            idx += 1

            sr = random.randint(r0, r1)
            sc = random.randint(c0, c1)
            center_r = (r0 + r1) // 2
            center_c = (c0 + c1) // 2
            if sr != center_r or sc != center_c:
                off_center = True
            grid[sr, sc] = seed

    if not off_center:
        r0, r1 = _region_spans(fence_rows)[0]
        c0, c1 = _region_spans(fence_cols)[0]
        grid[r0, c0] = grid[r0, c0]
        grid[(r0 + r1) // 2, (c0 + c1) // 2] = BACKGROUND
        grid[r0, c0] = assignments[0]

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
