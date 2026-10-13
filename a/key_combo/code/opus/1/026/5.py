import random
import numpy as np


def _tile_cavity(out: np.ndarray, src: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    sub = src[r0:r1, c0:c1]
    pts = np.argwhere((sub != 0) & (sub != 2))
    if len(pts) == 0:
        return
    sr0, sr1 = int(pts[:, 0].min()), int(pts[:, 0].max())
    sc0, sc1 = int(pts[:, 1].min()), int(pts[:, 1].max())
    tile = sub[sr0 : sr1 + 1, sc0 : sc1 + 1]
    th, tw = tile.shape
    for rr in range(r1 - r0):
        for cc in range(c1 - c0):
            val = int(tile[(rr - sr0) % th, (cc - sc0) % tw])
            out[r0 + rr, c0 + cc] = val if val != 2 else 0


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    red_rows = [r for r in range(h) if np.all(grid[r] == 2)]
    red_cols = [c for c in range(w) if np.all(grid[:, c] == 2)]
    out = np.array(grid, copy=True)

    row_bounds = [-1] + red_rows + [h]
    col_bounds = [-1] + red_cols + [w]
    for a, b in zip(row_bounds, row_bounds[1:]):
        for x, y in zip(col_bounds, col_bounds[1:]):
            if b - a > 1 and y - x > 1:
                _tile_cavity(out, grid, a + 1, b, x + 1, y)

    crossings = {(r, c) for r in red_rows for c in red_cols}
    for r in red_rows:
        for c in range(w):
            if (r, c) in crossings:
                out[r, c] = 2
            elif 0 < r < h - 1 and out[r - 1, c] == out[r + 1, c] and out[r - 1, c] != 0:
                out[r, c] = out[r - 1, c]
            else:
                out[r, c] = 2
    for c in red_cols:
        for r in range(h):
            if (r, c) in crossings:
                out[r, c] = 2
            elif 0 < c < w - 1 and out[r, c - 1] == out[r, c + 1] and out[r, c - 1] != 0:
                out[r, c] = out[r, c - 1]
            else:
                out[r, c] = 2
    return out


def _put_seed(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    patterns = [
        [(0, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]
    pat = random.choice(patterns)
    ph = max(r for r, _ in pat) + 1
    pw = max(c for _, c in pat) + 1
    sr = random.randint(r0, r1 - ph)
    sc = random.randint(c0, c1 - pw)
    for dr, dc in pat:
        grid[sr + dr, sc + dc] = color


def generate_input() -> np.ndarray:
    colors = [1, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = random.randint(14, 17)
        w = random.randint(14, 17)
        grid = np.zeros((h, w), dtype=int)
        red_rows = [random.randint(5, h - 6)]
        red_cols = [random.randint(5, w - 6)]
        for r in red_rows:
            grid[r, :] = 2
        for c in red_cols:
            grid[:, c] = 2

        row_bounds = [-1] + red_rows + [h]
        col_bounds = [-1] + red_cols + [w]
        base_color = random.choice(colors)
        for i, (a, b) in enumerate(zip(row_bounds, row_bounds[1:])):
            for j, (x, y) in enumerate(zip(col_bounds, col_bounds[1:])):
                color = base_color if (i + j) % 2 == 0 else random.choice(colors)
                _put_seed(grid, a + 1, b, x + 1, y, color)
        out = main(grid)
        red_total = int(np.sum(grid == 2))
        red_after = int(np.sum(out == 2))
        if 0 < red_after < red_total:
            return grid
    return grid
