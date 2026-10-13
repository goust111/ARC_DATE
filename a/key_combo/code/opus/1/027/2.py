import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    cross_color = 5
    rows = [r for r in range(h) if np.all(grid[r, :] == cross_color)]
    cols = [c for c in range(w) if np.all(grid[:, c] == cross_color)]
    cr = rows[0] if rows else h // 2
    cc = cols[0] if cols else w // 2
    quads = {
        "tl": (range(0, cr), range(0, cc)),
        "tr": (range(0, cr), range(cc + 1, w)),
        "bl": (range(cr + 1, h), range(0, cc)),
        "br": (range(cr + 1, h), range(cc + 1, w)),
    }
    counts = {}
    for name, (rs, cs) in quads.items():
        counts[name] = sum(1 for r in rs for c in cs if grid[r, c] not in (0, cross_color))
    source = next(name for name, n in counts.items() if list(counts.values()).count(n) == 1)
    target = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}[source]

    out = grid.copy()
    for name, (rs, cs) in quads.items():
        if name != source:
            for r in rs:
                for c in cs:
                    out[r, c] = 0
    for r in range(h):
        out[r, cc] = cross_color
    for c in range(w):
        out[cr, c] = cross_color

    rs, cs = quads[source]
    for r in rs:
        for c in cs:
            if grid[r, c] not in (0, cross_color):
                rr = cr - (r - cr)
                cc2 = cc - (c - cc)
                out[rr, cc2] = cross_color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    half = rng.choice([4, 5, 6, 7])
    n = 2 * half + 1
    grid = np.zeros((n, n), dtype=int)
    grid[half, :] = 5
    grid[:, half] = 5
    quads = {
        "tl": [(r, c) for r in range(half) for c in range(half)],
        "tr": [(r, c) for r in range(half) for c in range(half + 1, n)],
        "bl": [(r, c) for r in range(half + 1, n) for c in range(half)],
        "br": [(r, c) for r in range(half + 1, n) for c in range(half + 1, n)],
    }
    source = rng.choice(list(quads))
    base = rng.randint(2, min(8, half * half - 2))
    unique = base + rng.choice([-1, 1, 2])
    unique = max(1, min(half * half - 1, unique))
    if unique == base:
        unique = base + 1
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for name, cells in quads.items():
        k = unique if name == source else base
        for r, c in rng.sample(cells, k):
            grid[r, c] = rng.choice(colors)
    return grid
