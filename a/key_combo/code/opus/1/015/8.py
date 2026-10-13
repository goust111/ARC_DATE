import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _rot_coords(coords, k):
    coords = [(r, c, color) for r, c, color in coords]
    maxr = max(r for r, _, _ in coords)
    maxc = max(c for _, c, _ in coords)
    for _ in range(k % 4):
        coords = [(c, maxr - r, color) for r, c, color in coords]
        maxr, maxc = maxc, maxr
    mr = min(r for r, _, _ in coords)
    mc = min(c for _, c, _ in coords)
    return sorted((r - mr, c - mc, color) for r, c, color in coords)


def _rot_arr(arr, k):
    return np.rot90(arr, -(k % 4))


def _legend(grid):
    result = {}
    for r in range(grid.shape[0] - 1):
        block = grid[r:r + 2, 0:2]
        if np.all(block[:, 0] != 0) and np.all(block[:, 1] != 0):
            if len(set(block[:, 0].tolist())) == 1 and len(set(block[:, 1].tolist())) == 1:
                result[int(block[0, 0])] = int(block[0, 1])
    return result


def main(grid: np.ndarray) -> np.ndarray:
    legend = _legend(grid)
    nonlegend = np.ones(grid.shape, dtype=bool)
    nonlegend[:, :2] = False
    comps = _components((grid != 0) & nonlegend)
    rect = max(comps, key=len)
    rows = [r for r, _ in rect]
    cols = [c for _, c in rect]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    crop = grid[r0:r1 + 1, c0:c1 + 1]
    outside = grid.copy()
    outside[r0:r1 + 1, c0:c1 + 1] = 0
    outside[:, :2] = 0
    inside_counts = {int(v): int(np.sum(crop == v)) for v in set(crop.ravel()) if v != 0}
    marker_colors = {color for color in inside_counts if np.any(outside == color)}
    internal = []
    external = []
    for color in marker_colors:
        for rr, cc in np.argwhere(crop == color):
            internal.append((int(rr), int(cc), color))
        for rr, cc in np.argwhere(outside == color):
            external.append((int(rr), int(cc), color))
    er0 = min(r for r, _, _ in external)
    ec0 = min(c for _, c, _ in external)
    external_norm = sorted((r - er0, c - ec0, color) for r, c, color in external)
    rotation = 0
    for k in range(4):
        if _rot_coords(internal, k) == external_norm:
            rotation = k
            break
    rotated = _rot_arr(crop, rotation)
    counts = {int(v): int(np.sum(rotated == v)) for v in set(rotated.ravel()) if v != 0 and v not in marker_colors}
    main_color = max(counts, key=counts.get)
    halo = legend.get(main_color, 5)
    out = np.full((rotated.shape[0] + 2, rotated.shape[1] + 2), halo, dtype=int)
    out[1:-1, 1:-1] = rotated
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    mapping = {1: 4, 3: 9, 8: 2}
    for i, (left, right) in enumerate(mapping.items()):
        r = 1 + i * 3
        grid[r:r + 2, 0] = left
        grid[r:r + 2, 1] = right
    main_color = random.choice(list(mapping))
    rh, rw = random.choice([(5, 6), (6, 5)])
    r0, c0 = random.randint(3, 7), random.randint(5, 8)
    grid[r0:r0 + rh, c0:c0 + rw] = main_color
    marker_colors = random.sample([c for c in [1, 2, 3, 4, 6, 8, 9] if c != main_color], 2)
    markers = [(1, 1, marker_colors[0]), (rh - 2, rw - 2, marker_colors[1])]
    for dr, dc, color in markers:
        grid[r0 + dr, c0 + dc] = color
    grid[r0 + rh // 2, c0 + rw // 2] = 0
    k = random.randint(1, 3)
    rel = _rot_coords(markers, k)
    for _ in range(100):
        er, ec = random.randint(1, 13), random.randint(3, 13)
        cells = [(er + dr, ec + dc, color) for dr, dc, color in rel]
        if all(0 <= r < 16 and 0 <= c < 16 and grid[r, c] == 0 for r, c, _ in cells):
            if all(r < r0 - 1 or r > r0 + rh or c < c0 - 1 or c > c0 + rw for r, c, _ in cells):
                for r, c, color in cells:
                    grid[r, c] = color
                return grid
    return grid
