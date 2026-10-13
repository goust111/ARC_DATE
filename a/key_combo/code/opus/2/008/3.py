import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    corners = np.argwhere(g == 2)
    rows = sorted(set(int(r) for r, _ in corners))
    cols = sorted(set(int(c) for _, c in corners))
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    work = g[r0 + 1:r1, c0 + 1:c1]
    h, w = work.shape
    markers = []
    for r in range(h):
        for c in range(w):
            color = int(work[r, c])
            if color not in (0, 5):
                d = min(r, c, h - 1 - r, w - 1 - c)
                markers.append((d, color))
    palette = [2] + [color for _, color in sorted(markers)]
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            d = min(r, c, h - 1 - r, w - 1 - c)
            out[r, c] = palette[d % len(palette)]
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(17, 20)
    g = np.zeros((h, w), dtype=int)
    inner_h = random.randint(8, min(11, h - 5))
    inner_w = random.randint(8, min(11, w - 5))
    r0 = random.randint(2, h - inner_h - 3)
    c0 = random.randint(2, w - inner_w - 3)
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    g[r0:r1 + 1, c0:c1 + 1] = 5
    for rr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        g[rr, cc] = 2
    colors = random.sample([1, 3, 4], random.randint(2, 3))
    max_d = max(1, min(inner_h, inner_w) // 2 - 1)
    depths = random.sample(range(1, max_d + 1), min(len(colors), max_d))
    colors = colors[:len(depths)]
    for color, d in zip(colors, depths):
        candidates = []
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if min(rr - (r0 + 1), cc - (c0 + 1), (r1 - 1) - rr, (c1 - 1) - cc) == d:
                    candidates.append((rr, cc))
        rr, cc = random.choice(candidates)
        g[rr, cc] = color
    for _ in range(random.randint(0, 3)):
        rr = random.randrange(h)
        cc = random.randrange(w)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            g[rr, cc] = random.choice([6, 7, 8, 9])
    return g
