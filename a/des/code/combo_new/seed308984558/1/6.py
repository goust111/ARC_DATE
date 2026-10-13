import random
from collections import Counter, deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _holes(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touch = False
            while q:
                x, y = q.popleft()
                touch |= x in (0, h - 1) or y in (0, w - 1)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and bg[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if not touch:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each colored sprite under its matching horizontal band and sort by hole count.

    Input structure:
    - A black background contains several full-width one-row color bands.
    - Below the bands are small monochrome sprites whose colors match one of the bands.
    - Sprites can be solid or contain enclosed background holes.

    Rule:
    - Remove the original sprites while leaving the bands unchanged.
    - Group sprites by band color and sort each group by enclosed-hole count from fewest to most.
    - Place the sorted sprites left-to-right immediately below the matching band with one blank column between bounding boxes.

    Key insight:
    - Color picks the destination band, but the ordering comes from counting topological holes inside the sprite shape.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bands = []
    for r in range(g.shape[0]):
        vals = [v for v in np.unique(g[r]) if v != BG]
        if len(vals) == 1 and np.all(g[r] == vals[0]):
            bands.append((r, int(vals[0])))
    band_rows = {r for r, _ in bands}
    sprites = {}
    for color in [v for v in np.unique(g) if v != BG]:
        for cells in _components(g == int(color)):
            if any(r in band_rows for r, _ in cells):
                continue
            sprites.setdefault(int(color), []).append(cells)
            for r, c in cells:
                out[r, c] = BG
    for r, color in bands:
        items = sorted(sprites.get(color, []), key=lambda cells: (_holes(cells), len(cells)))
        cur = 0
        for cells in items:
            r0, c0, r1, c1 = _bbox(cells)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for x, y in cells:
                mask[x - r0, y - c0] = True
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if mask[x, y]:
                        out[r + 1 + x, cur + y] = color
            cur += mask.shape[1] + 1
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
]


def generate_input() -> np.ndarray:
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    w = random.randint(16, 24)
    rows = []
    base = 1
    for color in colors:
        rows.append((base, color))
        base += random.randint(6, 8)
    if base + 8 > 30:
        return generate_input()
    h = max(random.randint(18, 28), base + 8)
    g = np.zeros((h, w), dtype=int)
    for r, color in rows:
        g[r] = color
    for r, color in rows:
        for _ in range(random.randint(2, 3)):
            shape = random.choice(SHAPES)
            sh, sw = shape.shape
            for _ in range(200):
                r0 = random.randint(r + 3, h - sh - 1)
                c0 = random.randint(0, w - sw)
                if np.any(g[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]):
                    continue
                for x in range(sh):
                    for y in range(sw):
                        if shape[x, y]:
                            g[r0 + x, c0 + y] = color
                break
    return g
