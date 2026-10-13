import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    controls = {r for r in range(h) if np.any(grid[r] == 2)}
    sources = [c for c in range(w) if grid[0, c] == 3]
    row_paths = {}
    path_points = set()
    for idx, start_c in enumerate(sorted(sources)):
        c = start_c
        for r in range(h):
            if r > 0 and r in controls:
                c += 1
            if not (0 <= c < w):
                break
            row_paths.setdefault(r, []).append((idx, c))
            path_points.add((r, c))
            if out[r, c] == 0:
                out[r, c] = 3

    seeds = set()
    for r, vals in row_paths.items():
        vals = sorted(vals, key=lambda x: x[1])
        for (i1, c1), (i2, c2) in zip(vals, vals[1:]):
            if i1 != i2 and c2 == c1 + 1:
                for rr in range(r + 1, h):
                    if out[rr, c1] == 0:
                        seeds.add((rr, c1))
                        break
    for r, c in seeds:
        out[r, c] = 7

    current = set(seeds)
    generation = 2
    while current:
        active = set(current)
        for r, c in path_points:
            if any(sr == r for sr, _ in current):
                active.add((r, c))
        children = set()
        by_row = {}
        for r, c in active:
            by_row.setdefault(r, set()).add(c)
        for r, cols in by_row.items():
            if r + 2 >= h:
                continue
            cols = sorted(cols)
            for c1, c2 in zip(cols, cols[1:]):
                if c2 == c1 + 2 and out[r + 2, c1 + 1] == 0:
                    children.add((r + 2, c1 + 1))
        color = 6 if generation % 2 == 0 else 7
        for r, c in children:
            out[r, c] = color
        current = children
        generation += 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 20)
    w = random.randint(13, 16)
    g = np.zeros((h, w), dtype=int)
    starts = random.sample(range(2, w - 3), random.randint(3, 5))
    starts.sort()
    if not any(b == a + 1 for a, b in zip(starts, starts[1:])):
        starts[1] = starts[0] + 1
    for c in sorted(set(starts)):
        g[0, c] = 3
    for r in random.sample(range(2, h - 2), random.randint(3, 6)):
        g[r, random.randint(1, 3)] = 2
    return g
