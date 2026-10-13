from collections import Counter

import numpy as np


_RNG = np.random.default_rng()

GRID_H_RANGE = (6, 10)
GRID_W_RANGE = (8, 15)
N_FOREGROUND_RANGE = (2, 5)
GENERATION_RETRIES = 400
COUNT_SAMPLE_RETRIES = 200
ALLOCATION_RETRIES = 120


def _get_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _global_color_order(grid: np.ndarray, bg: int) -> list[int]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    height, width = grid.shape

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg:
                continue
            counts[color] += 1
            first_seen.setdefault(color, r * width + c)

    return sorted(counts, key=lambda color: (-counts[color], first_seen[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each row's foreground pixels using the global foreground color frequencies.

    The background is the dominant color. Every non-background color is ranked by its
    total count across the whole grid. Each row keeps the same number of pixels of each
    color, but those pixels are packed from the left in descending global-frequency order.
    Background pixels occupy the remaining cells on the right.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")

    bg = _get_background(g)
    order = _global_color_order(g, bg)
    if not order:
        return g.copy()

    out = np.full_like(g, bg)
    for r in range(g.shape[0]):
        row_counts = Counter(int(color) for color in g[r] if int(color) != bg)
        write_col = 0
        for color in order:
            count = row_counts.get(color, 0)
            if count == 0:
                continue
            out[r, write_col : write_col + count] = color
            write_col += count

    return out


def _sample_counts(rng: np.random.Generator, area: int, height: int, n_colors: int) -> list[int] | None:
    min_total = max(n_colors * 3, height + 3)
    max_total = min(int(area * 0.48), area // 2)
    max_count = max(n_colors + 3, area // 3)

    if max_total < min_total or max_count < n_colors + 1:
        return None

    pool = np.arange(2, max_count + 1)
    if pool.size < n_colors:
        return None

    for _ in range(COUNT_SAMPLE_RETRIES):
        counts = np.sort(rng.choice(pool, size=n_colors, replace=False))[::-1]
        total = int(counts.sum())
        if total < min_total or total > max_total:
            continue
        if area - total <= int(counts[0]):
            continue
        return [int(v) for v in counts]

    return None


def _allocate_rows(
    rng: np.random.Generator,
    counts: list[int],
    row_limits: np.ndarray,
) -> np.ndarray | None:
    height = int(row_limits.size)
    n_colors = len(counts)
    alloc = np.zeros((height, n_colors), dtype=int)
    row_used = np.zeros(height, dtype=int)

    for color_index, count in enumerate(counts):
        remaining_cap = row_limits - row_used
        available = np.flatnonzero(remaining_cap > 0)
        if available.size == 0:
            return None

        min_rows = 1
        if count >= 2 and available.size >= 2 and (color_index < min(2, n_colors) or rng.random() < 0.65):
            min_rows = 2

        max_rows = min(int(count), int(available.size))
        if color_index == 0 and available.size > 1:
            max_rows = min(max_rows, int(available.size) - 1)
        if max_rows < min_rows:
            return None

        chosen = None
        weights = remaining_cap[available].astype(float)
        weights /= weights.sum()
        for _ in range(ALLOCATION_RETRIES):
            n_rows = int(rng.integers(min_rows, max_rows + 1))
            rows = np.array(rng.choice(available, size=n_rows, replace=False, p=weights), dtype=int)
            if int(remaining_cap[rows].sum()) >= int(count):
                chosen = rows
                break
        if chosen is None:
            return None

        alloc[chosen, color_index] += 1
        row_used[chosen] += 1
        remaining = int(count - chosen.size)

        while remaining > 0:
            cap = row_limits[chosen] - row_used[chosen]
            viable = chosen[cap > 0]
            if viable.size == 0:
                return None
            probs = (row_limits[viable] - row_used[viable]).astype(float)
            probs /= probs.sum()
            row = int(rng.choice(viable, p=probs))
            alloc[row, color_index] += 1
            row_used[row] += 1
            remaining -= 1

    nonempty_rows = alloc.sum(axis=1) > 0
    if not np.any(nonempty_rows & (alloc[:, 0] == 0)):
        return None
    if not np.any((alloc > 0).sum(axis=1) >= 2):
        return None

    return alloc


def _render_grid(
    rng: np.random.Generator,
    height: int,
    width: int,
    bg: int,
    colors: list[int],
    alloc: np.ndarray,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)

    for r in range(height):
        row_total = int(alloc[r].sum())
        if row_total == 0:
            continue

        cols = np.array(rng.choice(width, size=row_total, replace=False), dtype=int)
        tokens = np.concatenate(
            [np.full(int(alloc[r, idx]), int(color), dtype=int) for idx, color in enumerate(colors) if int(alloc[r, idx]) > 0]
        )
        sorted_tokens = np.concatenate(
            [np.full(int(alloc[r, idx]), int(color), dtype=int) for idx, color in enumerate(colors) if int(alloc[r, idx]) > 0]
        )

        if tokens.size > 1:
            for _ in range(20):
                rng.shuffle(tokens)
                if not np.array_equal(tokens, sorted_tokens):
                    break

        for c, color in zip(cols, tokens):
            grid[r, int(c)] = int(color)

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    area = height * width

    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]

    max_colors = min(N_FOREGROUND_RANGE[1], len(palette), width - 1)
    n_colors = int(rng.integers(N_FOREGROUND_RANGE[0], max_colors + 1))
    colors = [int(color) for color in rng.choice(palette, size=n_colors, replace=False)]

    counts = _sample_counts(rng, area, height, n_colors)
    if counts is None:
        return None

    total_fg = int(sum(counts))
    min_active_rows = max(2, (total_fg + (width - 2)) // (width - 1))
    if min_active_rows > height:
        return None

    allow_empty_rows = bool(rng.random() < 0.5) and min_active_rows < height
    if allow_empty_rows:
        active_rows = int(rng.integers(min_active_rows, height))
    else:
        active_rows = height

    row_limits = np.zeros(height, dtype=int)
    active = np.array(rng.choice(height, size=active_rows, replace=False), dtype=int)
    row_limits[active] = width - 1

    alloc = _allocate_rows(rng, counts, row_limits)
    if alloc is None:
        return None

    grid = _render_grid(rng, height, width, bg, colors, alloc)

    values, actual_counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, actual_counts)}
    if count_map.get(bg, 0) <= max(counts):
        return None
    if sorted(count_map.get(color, 0) for color in colors) != sorted(counts):
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    if int(np.count_nonzero(out != grid)) < max(3, height // 2):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATION_RETRIES} attempts")
