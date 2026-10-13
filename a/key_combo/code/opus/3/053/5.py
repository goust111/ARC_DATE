import random

import numpy as np


def _shift_right(row, amount):
    out = np.zeros_like(row)
    if amount == 0:
        return row.copy()
    out[amount:] = row[:-amount]
    return out


def _infer_period(rows):
    n = len(rows)
    for p in (1, 2, 3):
        for s in (0, 1, 2, 3):
            ok = False
            good = True
            for i in range(n - p):
                ok = True
                if not np.array_equal(rows[i + p], _shift_right(rows[i], s)):
                    good = False
                    break
            if ok and good:
                return p, s
    return 1, 0


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the shifted row pattern to 12 rows and mask its columns by the gray sprite."""
    gray = np.argwhere(grid == 5)
    r0, c0 = gray.min(axis=0)
    r1, c1 = gray.max(axis=0)
    sprite = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    open_cols = np.any(sprite == 5, axis=0)
    tile_w = sprite.shape[1]

    pattern_mask = (grid != 0) & (grid != 5)
    pattern_rows = [int(r) for r in np.where(pattern_mask.any(axis=1))[0]]
    color = int(grid[pattern_mask][0])
    rows = [(grid[r, :12] == color) for r in pattern_rows]
    p, s = _infer_period(rows)
    while len(rows) < 12:
        rows.append(_shift_right(rows[-p], s))

    out = np.zeros((12, 12), dtype=grid.dtype)
    for r, row in enumerate(rows[:12]):
        out[r, row] = color
    for c in range(12):
        if not open_cols[c % tile_w]:
            out[:, c] = 0
    return out


def generate_input() -> np.ndarray:
    height = random.randint(8, 10)
    grid = np.zeros((height, 12), dtype=int)
    sh = random.choice([2, 3])
    sw = random.choice([2, 3, 4, 6])
    sprite = np.zeros((sh, sw), dtype=int)
    open_cols = random.sample(range(sw), random.randint(1, sw - 1))
    if len(open_cols) == sw:
        open_cols.pop()
    for c in open_cols:
        for r in random.sample(range(sh), random.randint(1, sh)):
            sprite[r, c] = 5
    if not any(np.all(sprite[:, c] == 0) for c in range(sw)):
        sprite[:, 0] = 0
    if not any(np.all(sprite[:, c] == 5) for c in range(sw)):
        sprite[:, -1] = 5
    grid[:sh, :sw] = sprite

    start = sh + 1
    nrows = min(random.randint(4, 5), height - start)
    p = random.choice([2, 3])
    s = random.choice([1, 2])
    color = random.choice([1, 2, 3, 4, 6, 8, 9])
    seeds = []
    for i in range(p):
        row = np.zeros(12, dtype=bool)
        cols = random.sample(range(0, 8), random.randint(3, 5))
        row[cols] = True
        seeds.append(row)
    rows = []
    for i in range(nrows):
        if i < p:
            row = seeds[i].copy()
        else:
            row = _shift_right(rows[i - p], s)
        rows.append(row)
        grid[start + i, row] = color
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert result.shape == (12, 12)
