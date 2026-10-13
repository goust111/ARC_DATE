import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    heights = []
    first_green = []
    for c in range(w):
        height = 0
        for r in range(h - 1, -1, -1):
            if grid[r, c] == 3:
                height += 1
            else:
                break
        heights.append(height)
        first_green.append(h - height)

    marker = tuple(np.argwhere(grid == 2)[0])
    exclusion = tuple(np.argwhere(grid == 4)[0])
    phase = marker[1] % 2
    cols = [c for c in range(w) if c % 2 == phase and c != exclusion[1] and first_green[c] > 0]
    peak = max(cols, key=lambda c: heights[c])

    for c in cols:
        out[:first_green[c], c] = 2 if c == peak else 4
    out[marker] = 0
    out[exclusion] = 0
    return out


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(14, 17)
        w = random.randint(13, 17)
        grid = np.zeros((h, w), dtype=int)
        heights = [random.randint(2, min(8, h - 4)) for _ in range(w)]
        phase = random.randint(0, 1)
        ex_col = random.randrange(1 - phase, w, 2) if random.random() < 0.5 else random.randrange(phase, w, 2)
        candidates = [c for c in range(w) if c % 2 == phase and c != ex_col]
        peak = random.choice(candidates)
        max_other = max(heights[c] for c in candidates if c != peak)
        heights[peak] = min(min(8, h - 4), max_other + random.randint(1, 3))
        if sum(1 for c in candidates if heights[c] == heights[peak]) != 1:
            continue
        for c, ht in enumerate(heights):
            grid[h - ht:, c] = 3
        mark_col = random.randrange(phase, w, 2)
        if mark_col == ex_col:
            continue
        mr = random.randint(0, h - heights[mark_col] - 1)
        er = random.randint(0, h - heights[ex_col] - 1)
        if mr == er:
            er = (er + 1) % max(1, h - heights[ex_col])
        grid[mr, mark_col] = 2
        grid[er, ex_col] = 4
        return grid
    raise RuntimeError("failed to generate input")
