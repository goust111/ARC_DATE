import random
from collections import deque

import numpy as np


BG = 7
PALETTE = [3, 4, 0, 5, 2, 8, 9, 6, 1]


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the gray sprite mask from the marker and color visible cells by Manhattan rings."""
    h, w = grid.shape
    marker_candidates = [(int(r), int(c)) for r, c in np.argwhere((grid != BG) & (grid != 5))]
    if not marker_candidates:
        comps = _components(grid == 5)
        singleton = [comp[0] for comp in comps if len(comp) == 1]
        marker_candidates = singleton
    mr, mc = marker_candidates[0]
    marker_color = int(grid[mr, mc])
    sprite = max(_components(grid == 5), key=len)
    rows = [r for r, _ in sprite]
    cols = [c for _, c in sprite]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    tile = grid[r0 : r1 + 1, c0 : c1 + 1] == 5
    th, tw = tile.shape
    start = PALETTE.index(marker_color)

    out = np.full_like(grid, BG)
    for r in range(h):
        for c in range(w):
            if tile[(r - mr) % th, (c - mc) % tw]:
                d = abs(r - mr) + abs(c - mc)
                out[r, c] = PALETTE[(start + d) % len(PALETTE)]
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([14, 15, 16])
    grid = np.full((h, w), BG, dtype=int)
    th, tw = random.choice([(3, 4), (4, 3)])
    tile = np.ones((th, tw), dtype=bool)
    holes = random.sample([(r, c) for r in range(th) for c in range(tw)], random.randint(2, 4))
    for r, c in holes:
        tile[r, c] = False
    if not tile.any():
        tile[0, 0] = True
    sr = random.choice([0, h - th])
    sc = random.choice([0, w - tw])
    grid[sr : sr + th, sc : sc + tw][tile] = 5
    while True:
        mr = random.randint(3, h - 4)
        mc = random.randint(3, w - 4)
        if not (sr - 1 <= mr <= sr + th and sc - 1 <= mc <= sc + tw):
            break
    grid[mr, mc] = random.choice([1, 2, 3, 4, 6, 8, 9])
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
