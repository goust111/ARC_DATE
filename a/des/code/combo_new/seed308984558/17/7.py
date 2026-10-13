import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp template interiors from the left panel onto right-panel markers by border color."""
    g = np.array(grid, dtype=int)
    sep = int(np.argmax(np.all(g == 5, axis=0)))
    out = g.copy()
    templates = {}
    left = g[:, :sep]
    for color in sorted(set(map(int, left.ravel())) - {0}):
        rs, cs = np.where(left == color)
        if len(rs) < 12:
            continue
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        border = patch == color
        if border[0].all() and border[-1].all() and border[:, 0].all() and border[:, -1].all():
            templates[color] = patch[1:-1, 1:-1].copy()
    marks = sorted([(int(r), int(c), int(g[r, c])) for r, c in zip(*np.where((g[:, sep + 1 :] != 0)))], key=lambda x: (x[0], x[1]))
    marks = [(r, c + sep + 1, color) for r, c, color in marks]
    for r, c, color in marks:
        tpl = templates.get(color)
        if tpl is None:
            continue
        th, tw = tpl.shape
        r0 = r - th // 2
        c0 = c - tw // 2
        for rr in range(th):
            for cc in range(tw):
                if 0 <= r0 + rr < g.shape[0] and 0 <= c0 + cc < g.shape[1]:
                    out[r0 + rr, c0 + cc] = tpl[rr, cc]
    return out


def generate_input() -> np.ndarray:
    th = tw = 3
    n = random.randint(2, 3)
    left_w = n * (tw + 3) + 1
    h = random.randint(14, 20)
    w = min(30, left_w + 1 + random.randint(6, 10))
    g = np.zeros((h, w), dtype=int)
    sep = left_w
    g[:, sep] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    patterns = []
    for i, color in enumerate(colors):
        border = color
        inner_color = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != color])
        inner = np.zeros((th, tw), dtype=int)
        inner[random.randint(0, 2), :] = inner_color
        inner[:, random.randint(0, 2)] = inner_color
        patterns.append((color, inner))
        r0, c0 = 2 + i * 4, 1
        g[r0 : r0 + th + 2, c0 : c0 + tw + 2] = border
        g[r0 + 1 : r0 + th + 1, c0 + 1 : c0 + tw + 1] = inner
    for _ in range(random.randint(3, 6)):
        color, _ = random.choice(patterns)
        rr = random.randint(1, h - 2)
        cc = random.randint(sep + 2, w - 2)
        g[rr, cc] = color
    return g
