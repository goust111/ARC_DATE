import random
import numpy as np

BLACK = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    out = np.zeros_like(grid)
    used = set()
    tracks = []
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] != BLACK:
                d = c
                while d < w and grid[r, d] != BLACK:
                    d += 1
                if d - c >= 5:
                    tracks.append(("h", r, c, d - 1))
                    for x in range(c, d):
                        used.add((r, x))
                c = d
            else:
                c += 1
    for c in range(w):
        r = 0
        while r < h:
            if grid[r, c] != BLACK and (r, c) not in used:
                d = r
                while d < h and grid[d, c] != BLACK:
                    d += 1
                if d - r >= 5:
                    tracks.append(("v", c, r, d - 1))
                r = d
            else:
                r += 1
    for kind, a, b, d in tracks:
        if kind == "h":
            seeds = [(c, grid[a, c]) for c in range(b, d + 1) if grid[a, c] not in (BLACK, GREY)]
            seeds.sort()
            for idx, (c, color) in enumerate(seeds, start=1):
                out[a, c] = color
                for step in range(1, idx + 1):
                    out[a - step, c] = color
        else:
            seeds = [(r, grid[r, a]) for r in range(b, d + 1) if grid[r, a] not in (BLACK, GREY)]
            seeds.sort()
            for idx, (r, color) in enumerate(seeds, start=1):
                out[r, a] = color
                for step in range(1, idx + 1):
                    out[r, a - step] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 22), random.randint(14, 22)
    grid = np.zeros((h, w), dtype=int)
    tracks = random.randint(2, 4)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(tracks):
        if random.random() < 0.5:
            seed_count = random.randint(2, 4)
            length = random.randint(max(5, seed_count + 2), min(10, w - 2))
            r = random.randint(seed_count + 1, h - 2)
            c0 = random.randint(1, w - length - 1)
            if np.any(grid[r, c0 : c0 + length] != 0):
                continue
            grid[r, c0 : c0 + length] = GREY
            spots = sorted(random.sample(range(c0, c0 + length), seed_count))
            colors = random.sample(palette, seed_count)
            for c in spots:
                grid[r, c] = colors.pop()
        else:
            seed_count = random.randint(2, 4)
            length = random.randint(max(5, seed_count + 2), min(10, h - 2))
            c = random.randint(seed_count + 1, w - 2)
            r0 = random.randint(1, h - length - 1)
            if np.any(grid[r0 : r0 + length, c] != 0):
                continue
            grid[r0 : r0 + length, c] = GREY
            spots = sorted(random.sample(range(r0, r0 + length), seed_count))
            colors = random.sample(palette, seed_count)
            for r in spots:
                grid[r, c] = colors.pop()
    return grid
