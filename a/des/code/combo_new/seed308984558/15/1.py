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


def _holes(patch: np.ndarray, color: int) -> int:
    bg = patch != color
    seen = np.zeros_like(bg, dtype=bool)
    count = 0
    for r, c in zip(*np.where(bg)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        edge = False
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            edge |= rr in (0, bg.shape[0] - 1) or cc in (0, bg.shape[1] - 1)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < bg.shape[0] and 0 <= nc < bg.shape[1] and bg[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        if not edge:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each inner sprite by hole count, with direction determined by the tagged frame corner."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    rs, cs = np.where(g == 5)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    side_map = {}
    for color in sorted(set(map(int, g.ravel())) - {0, 5}):
        sides = set()
        if np.any(g[r0, :] == color):
            sides.add("top")
        if np.any(g[r1, :] == color):
            sides.add("bottom")
        if np.any(g[:, c0] == color):
            sides.add("left")
        if np.any(g[:, c1] == color):
            sides.add("right")
        if sides:
            side_map[color] = sides
    for color in side_map:
        for pts in _components((g == color) & (np.indices(g.shape)[0] > r0) & (np.indices(g.shape)[0] < r1) & (np.indices(g.shape)[1] > c0) & (np.indices(g.shape)[1] < c1)):
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            a0, a1, b0, b1 = min(rs), max(rs), min(cs), max(cs)
            patch = out[a0 : a1 + 1, b0 : b1 + 1].copy()
            k = _holes(patch, color)
            cw = side_map[color] in ({"top", "right"}, {"bottom", "left"})
            rot = np.rot90(patch, -k if cw else k)
            out[a0 : a1 + 1, b0 : b1 + 1] = rot
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    r0, c0 = 2, 2
    r1, c1 = g.shape[0] - 3, g.shape[1] - 3
    g[r0 : r1 + 1, c0] = 5
    g[r0 : r1 + 1, c1] = 5
    g[r0, c0 : c1 + 1] = 5
    g[r1, c0 : c1 + 1] = 5
    shapes = [
        np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]]),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]]),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    ]
    corners = [({"top", "right"}), ({"top", "left"}), ({"bottom", "right"}), ({"bottom", "left"})]
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        sides = random.choice(corners)
        if "top" in sides:
            g[r0, random.randint(c0 + 1, c1 - 1)] = color
        if "bottom" in sides:
            g[r1, random.randint(c0 + 1, c1 - 1)] = color
        if "left" in sides:
            g[random.randint(r0 + 1, r1 - 1), c0] = color
        if "right" in sides:
            g[random.randint(r0 + 1, r1 - 1), c1] = color
        patch = random.choice(shapes) * color
        for _ in range(100):
            rr = random.randint(r0 + 1, r1 - 3)
            cc = random.randint(c0 + 1, c1 - 3)
            if np.all(g[rr - 1 : rr + 4, cc - 1 : cc + 4] == 0):
                g[rr : rr + 3, cc : cc + 3] = np.maximum(g[rr : rr + 3, cc : cc + 3], patch)
                break
    return g
