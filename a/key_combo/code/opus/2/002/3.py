import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    divs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    cuts = [-1] + divs + [grid.shape[1]]
    panels = [grid[:, cuts[i] + 1 : cuts[i + 1]] for i in range(4)]
    inner = panels[0] != 0
    outer = panels[1] != 0
    fg = int(np.bincount(panels[2].ravel()).argmax())
    bg = int(np.bincount(panels[3].ravel()).argmax())
    th, tw = inner.shape
    out = np.full((outer.shape[0] * th, outer.shape[1] * tw), bg, dtype=int)

    k = 0
    for r in range(outer.shape[0]):
        for c in range(outer.shape[1]):
            rr, cc = r * th, c * tw
            if outer[r, c]:
                mask = np.rot90(inner, -k)
                out[rr : rr + th, cc : cc + tw] = np.where(mask, fg, bg)
                k += 1
    return out


def _asym_mask(n):
    while True:
        mask = np.zeros((n, n), dtype=bool)
        cells = {(0, 0), (1, 0), (1, 1), (2, 1)}
        while len(cells) < random.randint(4, n * n - 2):
            r, c = random.choice(tuple(cells))
            for nr, nc in random.sample([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)], 4):
                if 0 <= nr < n and 0 <= nc < n:
                    cells.add((nr, nc))
                    break
        for r, c in cells:
            mask[r, c] = True
        if any(not np.array_equal(mask, np.rot90(mask, k)) for k in [1, 2, 3]):
            return mask


def generate_input() -> np.ndarray:
    n = random.choice([3, 4])
    h = w = n
    grid = np.zeros((h, 4 * w + 3), dtype=int)
    for c in [w, 2 * w + 1, 3 * w + 2]:
        grid[:, c] = 5

    inner = _asym_mask(n)
    outer = np.zeros((n, n), dtype=bool)
    coords = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(coords)
    for r, c in coords[: random.randint(4, n * n - 1)]:
        outer[r, c] = True
    if np.count_nonzero(outer) < 4:
        outer.ravel()[:4] = True

    grid[:, :w] = np.where(inner, random.choice([1, 2, 3, 4, 6, 7, 8, 9]), 0)
    grid[:, w + 1 : 2 * w + 1] = np.where(outer, random.choice([1, 2, 3, 4, 6, 7, 8, 9]), 0)
    fg, bg = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    grid[:, 2 * w + 2 : 3 * w + 2] = fg
    grid[:, 3 * w + 3 :] = bg
    return grid
