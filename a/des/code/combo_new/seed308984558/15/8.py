import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect path segments in legend order by snapping each unmarked endpoint onto the previous grey marker."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    order = [int(v) for v in g[:6, 0] if int(v) != 0]
    placed_marker = None
    for color in order:
        comps = _components(g == color)
        pts = max(comps, key=len)
        s = set(pts)
        ends = [(r, c) for r, c in pts if sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 1]
        marked = None
        marker_pos = None
        for r, c in ends:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1] and g[r + dr, c + dc] == 5:
                    marked = (r, c)
                    marker_pos = (r + dr, c + dc)
        free_ends = [e for e in ends if e != marked]
        if not free_ends:
            continue
        free = free_ends[0]
        if placed_marker is None:
            placed_marker = marker_pos
            continue
        dr = placed_marker[0] - free[0]
        dc = placed_marker[1] - free[1]
        out[placed_marker] = color
        for r, c in pts:
            out[r + dr, c + dc] = color
        if marker_pos:
            placed_marker = (marker_pos[0] + dr, marker_pos[1] + dc)
            out[marker_pos[0] + dr, marker_pos[1] + dc] = 5
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for i, color in enumerate(colors):
        g[i, 0] = color
        r0 = random.randint(5, g.shape[0] - 5)
        c0 = random.randint(4, g.shape[1] - 5)
        if random.choice([True, False]):
            g[r0, c0 : c0 + 4] = color
            g[r0 - 1, c0] = 5
        else:
            g[r0 : r0 + 3, c0] = color
            g[r0, c0 + 1] = 5
    return g
