import random

import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    colors = [c for c in np.unique(grid) if c != BG]
    for color in colors:
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 - r0 != c1 - c0 or (r1 - r0 + 1) % 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        n = sub.shape[0] // 2
        quads = {
            "tl": sub[:n, :n],
            "tr": sub[:n, n:],
            "bl": sub[n:, :n],
            "br": sub[n:, n:],
        }
        missing = [k for k, v in quads.items() if np.all(v == BG)]
        if not missing:
            continue
        miss = missing[0]
        if miss == "tl":
            fill = np.rot90(quads["tr"], 1)
            out[r0 : r0 + n, c0 : c0 + n] = np.where(fill != BG, fill, out[r0 : r0 + n, c0 : c0 + n])
        elif miss == "tr":
            fill = np.rot90(quads["br"], 1)
            out[r0 : r0 + n, c0 + n : c1 + 1] = np.where(fill != BG, fill, out[r0 : r0 + n, c0 + n : c1 + 1])
        elif miss == "bl":
            fill = np.rot90(quads["tl"], -1)
            out[r0 + n : r1 + 1, c0 : c0 + n] = np.where(fill != BG, fill, out[r0 + n : r1 + 1, c0 : c0 + n])
        else:
            fill = np.rot90(quads["bl"], 1)
            out[r0 + n : r1 + 1, c0 + n : c1 + 1] = np.where(fill != BG, fill, out[r0 + n : r1 + 1, c0 + n : c1 + 1])
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        n = random.choice([2, 3])
        size = 2 * n
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        base = np.random.randint(0, 2, size=(n, n)) * color
        if not np.any(base == color):
            base[0, 0] = color
        quads = [base, np.rot90(base, -1), np.rot90(base, 2), np.rot90(base, 1)]
        miss = random.randrange(4)
        names = ["tl", "tr", "br", "bl"]
        for idx, name in enumerate(names):
            if idx == miss:
                continue
            rr = top + (n if name in ("bl", "br") else 0)
            cc = left + (n if name in ("tr", "br") else 0)
            grid[rr : rr + n, cc : cc + n] = quads[idx]
    return grid
