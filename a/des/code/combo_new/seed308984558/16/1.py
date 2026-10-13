import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey square frame by matching its inner shape class to the corner legend."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    legend = {}
    for r in range(0, 9, 3):
        patch = g[r : r + 2, :5]
        label = int(g[r, 4])
        shape = patch[:, :3]
        if np.any(shape[0] != 0) and np.all(shape[1] == 0):
            legend["h"] = label
        elif np.any(shape[:, 0] != 0) and np.all(shape[:, 1:] == 0):
            legend["v"] = label
        else:
            legend["b"] = label
    mask = g == 5
    seen = np.zeros_like(mask, dtype=bool)
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        pts = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rs, cs = zip(*pts)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 < 9 and c0 < 5:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        rows, cols = np.where(inner != 0)
        if len(rows) == 0:
            continue
        kind = "b" if len(set(rows)) > 1 and len(set(cols)) > 1 else "h" if len(set(rows)) == 1 else "v"
        for rr, cc in pts:
            out[rr, cc] = legend[kind]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    samples = [("h", 1), ("v", 2), ("b", 3)]
    labels = random.sample([4, 6, 7, 8, 9], 3)
    for i, ((kind, color), label) in enumerate(zip(samples, labels)):
        r = i * 3
        if kind == "h":
            g[r, :3] = color
        elif kind == "v":
            g[r : r + 2, 0] = color
        else:
            g[r : r + 2, :2] = color
        g[r, 4] = label
    for _ in range(random.randint(2, 4)):
        size = random.randint(4, 5)
        r0 = random.randint(10, g.shape[0] - size - 1)
        c0 = random.randint(6, g.shape[1] - size - 1)
        g[r0 : r0 + size, c0] = 5
        g[r0 : r0 + size, c0 + size - 1] = 5
        g[r0, c0 : c0 + size] = 5
        g[r0 + size - 1, c0 : c0 + size] = 5
        kind, color = random.choice(samples)
        if kind == "h":
            g[r0 + size // 2, c0 + 1 : c0 + size - 1] = color
        elif kind == "v":
            g[r0 + 1 : r0 + size - 1, c0 + size // 2] = color
        else:
            g[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = color
    return g
