import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
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
            yield pts


def _perimeter(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll each square ring from its gap into a straight colored segment.

    Input structure:
    - Black background with several distinct-color square one-pixel rings.
    - Each ring has exactly one missing perimeter pixel, which appears as a black gap.
    - Ring sizes are small square frames and colors are unique.

    Rule:
    - Find the missing perimeter cell of each ring and use its side as the outward direction.
    - Clear the original ring.
    - Draw a straight segment from the gap cell outward whose length equals the number of colored ring pixels.

    Key insight:
    - The gap gives both the start point and direction; the ring perimeter count gives the line length.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        pts_all = np.argwhere(g == color)
        if pts_all.size == 0:
            continue
        for pts in _components(g == color):
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != c1 - c0 or r1 - r0 < 2:
                continue
            peri = _perimeter(r0, c0, r1, c1)
            missing = list(peri - set(pts))
            if len(missing) != 1:
                continue
            gap = missing[0]
            if gap[0] == r0:
                direction = (-1, 0)
            elif gap[0] == r1:
                direction = (1, 0)
            elif gap[1] == c0:
                direction = (0, -1)
            else:
                direction = (0, 1)
            for k in range(len(pts)):
                r = gap[0] + direction[0] * k
                c = gap[1] + direction[1] * k
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
        g = np.zeros((h, w), dtype=int)
        n = rng.randint(2, 5)
        occupied = set()
        ok = True
        for color in rng.sample(colors, n):
            placed = False
            for _try in range(200):
                size = rng.randint(3, 5)
                side = rng.choice(["top", "bottom", "left", "right"])
                length = 4 * size - 5
                margin_top = length - 1 if side == "top" else 0
                margin_bottom = length - 1 if side == "bottom" else 0
                margin_left = length - 1 if side == "left" else 0
                margin_right = length - 1 if side == "right" else 0
                if margin_top > h - size - margin_bottom or margin_left > w - size - margin_right:
                    continue
                r0 = rng.randint(margin_top, h - size - margin_bottom)
                c0 = rng.randint(margin_left, w - size - margin_right)
                r1, c1 = r0 + size - 1, c0 + size - 1
                if side == "top":
                    gap = (r0, rng.randint(c0 + 1, c1 - 1))
                    direction = (-1, 0)
                elif side == "bottom":
                    gap = (r1, rng.randint(c0 + 1, c1 - 1))
                    direction = (1, 0)
                elif side == "left":
                    gap = (rng.randint(r0 + 1, r1 - 1), c0)
                    direction = (0, -1)
                else:
                    gap = (rng.randint(r0 + 1, r1 - 1), c1)
                    direction = (0, 1)
                ring = _perimeter(r0, c0, r1, c1) - {gap}
                line = {(gap[0] + direction[0] * k, gap[1] + direction[1] * k) for k in range(length)}
                footprint = ring | line
                padded = {(r + dr, c + dc) for r, c in footprint for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if padded & occupied:
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                    continue
                occupied |= padded
                for r, c in ring:
                    g[r, c] = color
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
