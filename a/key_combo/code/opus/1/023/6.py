import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    stamp_comps = []
    for color in (4, 7):
        for comp in _components(grid == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                stamp_comps.append((color, r0, r1, c0, c1, set(comp)))

    region_color = np.zeros_like(grid)
    for comp in _components(grid != 5):
        seeds = [(r, c, int(grid[r, c])) for r, c in comp if grid[r, c] in (1, 2, 3)]
        if not seeds:
            continue
        color = seeds[0][2]
        for r, c in comp:
            if grid[r, c] == 0 or grid[r, c] == color:
                out[r, c] = color
                region_color[r, c] = color

    for color, r0, r1, c0, c1, original in stamp_comps:
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        for dc_sign in (-1, 1):
            step = 1
            while True:
                nr0 = r0 + step * sh
                nc0 = c0 + step * dc_sign * sw
                any_inside = False
                for dr in range(sh):
                    for dc in range(sw):
                        r, c = nr0 + dr, nc0 + dc
                        if 0 <= r < h and 0 <= c < w:
                            any_inside = True
                            out[r, c] = int(region_color[r, c]) if region_color[r, c] != 0 else color
                if not any_inside:
                    break
                step += 1
        for r, c in original:
            out[r, c] = color
    return out


def _fill_rect(grid, r0, r1, c0, c1, color):
    r = random.randint(r0 + 1, r1 - 1)
    c = random.randint(c0 + 1, c1 - 1)
    grid[r, c] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    rsep1 = random.randint(5, 6)
    rsep2 = random.randint(11, 12)
    csep = random.randint(8, 10)
    grid[rsep1, :] = 5
    grid[rsep2, :] = 5
    grid[:, csep] = 5
    regions = [
        (0, rsep1 - 1, 0, csep - 1),
        (rsep1 + 1, rsep2 - 1, csep + 1, 17),
        (rsep2 + 1, 17, 0, csep - 1),
    ]
    for box, color in zip(regions, [1, 2, 3]):
        _fill_rect(grid, *box, color)
    sh, sw = random.choice([(2, 2), (2, 3)])
    sr = random.randint(1, rsep1 - sh - 1)
    sc = random.randint(1, csep - sw - 1)
    grid[sr:sr + sh, sc:sc + sw] = random.choice([4, 7])
    if random.random() < 0.4:
        sr2 = random.randint(rsep1 + 2, rsep2 - sh - 1)
        sc2 = random.randint(csep + 2, 17 - sw)
        grid[sr2:sr2 + sh, sc2:sc2 + sw] = 7 if grid[sr, sc] == 4 else 4
    return grid
