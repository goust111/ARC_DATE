import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_baseline(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(candidates) == 1, "expected one full grey baseline"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each one-column tower below the baseline with height equal to its color count.
    """
    g = np.asarray(grid, dtype=int)
    base = _find_baseline(g)
    out = g.copy()
    h, w = g.shape

    for c in range(w):
        bottom_to_top = []
        r = base - 1
        while r >= 0 and int(g[r, c]) not in (BLACK, GREY):
            bottom_to_top.append(int(g[r, c]))
            r -= 1
        if not bottom_to_top:
            continue
        k = len(set(bottom_to_top))
        assert base + k < h, "not enough space below baseline"
        for i in range(k):
            out[base + 1 + i, c] = bottom_to_top[k - 1 - i]
    return out


def _spaced_columns(width: int, count: int) -> list[int] | None:
    usable = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(usable, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        width = random.randint(15, 25)
        tower_count = random.randint(3, 7)
        cols = _spaced_columns(width, tower_count)
        if cols is None:
            continue

        heights = [random.randint(3, 8) for _ in range(tower_count)]
        max_h = max(heights)
        base = random.randint(max_h + 1, max_h + 5)
        below = max(4, min(max_h, len(COLORS))) + random.randint(2, 5)
        height = base + 1 + below
        grid = np.full((height, width), BLACK, dtype=int)
        grid[base, :] = GREY

        for c, tower_h in zip(cols, heights):
            distinct = random.randint(2, min(tower_h, 5, len(COLORS)))
            palette = random.sample(COLORS, distinct)
            bottom = palette[:]
            while len(bottom) < tower_h:
                bottom.append(random.choice(palette))
            random.shuffle(bottom[distinct:])
            # Preserve the chosen distinct colors near the bottom often enough to make K visible.
            for i, color in enumerate(bottom):
                grid[base - 1 - i, c] = color

        result = main(grid)
        assert isinstance(result, np.ndarray)
        return grid
    raise RuntimeError("failed to generate tower reflection input")
