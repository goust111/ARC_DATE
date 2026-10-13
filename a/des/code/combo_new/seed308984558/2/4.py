import random
from collections import deque

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the motif from the smallest filled container into every empty container using the target border color.

    Input structure:
    - A black background contains several colored hollow rectangles of different sizes.
    - Exactly one smallest container already holds a motif in a non-border color.
    - All other containers are empty inside.

    Rule:
    - Identify the smallest container that contains a motif and use that motif as the source.
    - Paste the motif into every empty container, aligned to the interior top-left corner and clipped if needed.
    - Recolor pasted pixels to the destination border color while leaving the source motif unchanged.

    Key insight:
    - The source is determined by container size, and the copied shape changes color to match each destination border.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = []
    for color in [v for v in np.unique(g) if v != BG]:
        for cells in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 - 1):
                inner = g[r0 + 1 : r1, c0 + 1 : c1]
                frames.append((int(color), r0, c0, r1, c1, inner.copy()))
    source = min([f for f in frames if np.any((f[5] != BG))], key=lambda f: f[5].shape[0] * f[5].shape[1])
    src_mask = source[5] != BG
    for color, r0, c0, r1, c1, inner in frames:
        if np.any(inner != BG):
            continue
        h = min(src_mask.shape[0], inner.shape[0])
        w = min(src_mask.shape[1], inner.shape[1])
        patch = src_mask[:h, :w]
        out[r0 + 1 : r0 + 1 + h, c0 + 1 : c0 + 1 + w][patch] = color
    return out


MOTIF = np.array([[1, 1, 0], [0, 1, 1]])


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    sizes = sorted(random.sample([(5, 5), (5, 7), (6, 8), (7, 9)], len(colors)))
    source_done = False
    for color, (fh, fw) in zip(colors, sizes):
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = color
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = color
            if not source_done:
                ph, pw = MOTIF.shape
                g[r0 + 1 : r0 + 1 + ph, c0 + 1 : c0 + 1 + pw][MOTIF == 1] = random.choice([v for v in colors if v != color])
                source_done = True
            break
    return g
