import random

import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    line_color = None
    axis = None
    idx = None
    for r in range(h):
        vals = [v for v in np.unique(grid[r]) if v != BG]
        if len(vals) == 1 and np.count_nonzero(grid[r] == vals[0]) == w:
            line_color = int(vals[0])
            axis = "H"
            idx = r
            break
    if axis is None:
        for c in range(w):
            vals = [v for v in np.unique(grid[:, c]) if v != BG]
            if len(vals) == 1 and np.count_nonzero(grid[:, c] == vals[0]) == h:
                line_color = int(vals[0])
                axis = "V"
                idx = c
                break
    out = np.zeros_like(grid)
    if axis == "H":
        out[idx, :] = line_color
        top = np.count_nonzero(grid[:idx] != BG)
        bottom = np.count_nonzero(grid[idx + 1 :] != BG)
        sparse = grid[:idx] if top <= bottom else grid[idx + 1 :]
        dense = grid[idx + 1 :] if top <= bottom else grid[:idx]
        sparse_top = top <= bottom
        if sparse_top:
            out[:idx] = sparse
        else:
            out[idx + 1 :] = sparse
        dense_coords = np.argwhere(dense != BG)
        for r, c in dense_coords:
            if sparse_top:
                nr = idx - 1 - r
            else:
                nr = idx + 1 + (dense.shape[0] - 1 - r)
            if 0 <= nr < h:
                if out[nr, c] not in (BG, line_color):
                    out[nr, c] = line_color
                elif out[nr, c] == BG:
                    out[nr, c] = dense[r, c]
    else:
        out[:, idx] = line_color
        left = np.count_nonzero(grid[:, :idx] != BG)
        right = np.count_nonzero(grid[:, idx + 1 :] != BG)
        sparse = grid[:, :idx] if left <= right else grid[:, idx + 1 :]
        dense = grid[:, idx + 1 :] if left <= right else grid[:, :idx]
        sparse_left = left <= right
        if sparse_left:
            out[:, :idx] = sparse
        else:
            out[:, idx + 1 :] = sparse
        dense_coords = np.argwhere(dense != BG)
        for r, c in dense_coords:
            if sparse_left:
                nc = idx - 1 - c
            else:
                nc = idx + 1 + (dense.shape[1] - 1 - c)
            if 0 <= nc < w:
                if out[r, nc] not in (BG, line_color):
                    out[r, nc] = line_color
                elif out[r, nc] == BG:
                    out[r, nc] = dense[r, c]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    if random.choice([True, False]):
        r = random.randint(8, 13)
        grid[r, :] = color
        dense_side = random.choice(["T", "B"])
        for _ in range(random.randint(6, 10)):
            rr = random.randint(1, r - 2) if dense_side == "T" else random.randint(r + 2, 20)
            cc = random.randint(1, 20)
            grid[rr, cc] = random.choice([v for v in range(1, 10) if v != color])
        other = "B" if dense_side == "T" else "T"
        for _ in range(random.randint(2, 4)):
            rr = random.randint(1, r - 2) if other == "T" else random.randint(r + 2, 20)
            cc = random.randint(1, 20)
            grid[rr, cc] = random.choice([v for v in range(1, 10) if v != color])
    else:
        c = random.randint(8, 13)
        grid[:, c] = color
        dense_side = random.choice(["L", "R"])
        for _ in range(random.randint(6, 10)):
            rr = random.randint(1, 20)
            cc = random.randint(1, c - 2) if dense_side == "L" else random.randint(c + 2, 20)
            grid[rr, cc] = random.choice([v for v in range(1, 10) if v != color])
        other = "R" if dense_side == "L" else "L"
        for _ in range(random.randint(2, 4)):
            rr = random.randint(1, 20)
            cc = random.randint(1, c - 2) if other == "L" else random.randint(c + 2, 20)
            grid[rr, cc] = random.choice([v for v in range(1, 10) if v != color])
    return grid
