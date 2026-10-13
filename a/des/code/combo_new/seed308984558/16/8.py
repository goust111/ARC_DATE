import random
from collections import deque

import numpy as np


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _transform(patch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "id":
        return patch
    if mode == "h":
        return np.fliplr(patch)
    if mode == "v":
        return np.flipud(patch)
    return np.rot90(patch, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """Apply each compass-encoded transform to the matching colored object in its quadrant."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    cr, cc = h // 2, w // 2
    mapping = {
        int(g[cr - 1, cc]): "id" if np.count_nonzero(g[cr - 2 : cr, cc] == g[cr - 1, cc]) == 1 else "rot",
        int(g[cr + 1, cc]): "h" if g[cr + 1, cc + 1] == 0 else "v",
        int(g[cr, cc - 1]): "v" if g[cr + 1, cc - 1] == 0 else "h",
        int(g[cr, cc + 1]): "rot",
    }
    for color in sorted(set(map(int, g.ravel())) - {0, 5, *mapping.keys()}):
        pass
    for color, mode in mapping.items():
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color)) if abs(int(r) - cr) > 2 or abs(int(c) - cc) > 2]
        if not pts:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        patch = (g[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int) * color
        out[r0 : r1 + 1, c0 : c1 + 1] = _transform(patch, mode if mode != "rot" else "rot")
    return out


def generate_input() -> np.ndarray:
    n = random.choice([17, 19, 21])
    g = np.zeros((n, n), dtype=int)
    cr = cc = n // 2
    g[cr, cc] = 5
    colors = random.sample([1, 2, 3, 4], 4)
    g[cr - 1, cc] = colors[0]
    g[cr + 1, cc] = colors[1]
    g[cr, cc - 1] = colors[2]
    g[cr, cc + 1] = colors[3]
    g[cr - 2, cc] = colors[0]
    g[cr + 1, cc + 1] = colors[1]
    g[cr - 1, cc - 1] = colors[2]
    g[cr, cc + 2] = colors[3]
    base = [(0, 0), (1, 0), (1, 1)]
    spots = [(2, 2), (2, n - 6), (n - 6, 2), (n - 6, n - 6)]
    modes = ["id", "h", "v", "rot"]
    for color, (r0, c0), mode in zip(colors, spots, modes):
        patch = np.zeros((3, 3), dtype=int)
        for r, c in base:
            patch[r, c] = color
        patch = _transform(patch, mode if mode != "rot" else "rot")
        g[r0 : r0 + 3, c0 : c0 + 3] = np.maximum(g[r0 : r0 + 3, c0 : c0 + 3], patch)
    return g
