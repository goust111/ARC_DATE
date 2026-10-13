import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _dilate(points, shape):
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    mask[nr, nc] = True
    return mask


def _legend(grid):
    mask = np.isin(grid[:5, :6], [1, 2, 3, 4])
    items = []
    for comp in _components(mask):
        pts = [(r, c) for r, c in comp]
        color = int(grid[pts[0][0], pts[0][1]])
        if all(int(grid[r, c]) == color for r, c in pts):
            items.append((color, len(pts)))
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """Create gap-channel segments from the intersection of two dilated orange paths and recolor ranked segments by the legend."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    orange = _components(g == 7)
    if len(orange) < 2:
        return out
    orange = sorted(orange, key=len, reverse=True)[:2]
    channel = _dilate(orange[0], g.shape) & _dilate(orange[1], g.shape) & (g != 7) & (g != 9)
    segments = _components(channel)
    segments.sort(key=lambda comp: (-len(comp), min(comp)))
    for color, rank in _legend(g):
        if 1 <= rank <= len(segments):
            for r, c in segments[rank - 1]:
                if out[r, c] == 0:
                    out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    grid[8, 5:17] = 7
    grid[10, 5:17] = 7
    for c in random.choice([(7, 11), (8, 12)]):
        grid[9, c] = 9
    colors = random.sample([1, 2, 3, 4], 3)
    heights = random.sample([1, 2, 3], 3)
    for i, (color, height) in enumerate(zip(colors, heights)):
        grid[:height, i * 2] = color
    return grid
