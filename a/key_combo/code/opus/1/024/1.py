import random
import numpy as np


def _frames(grid):
    h, w = grid.shape
    out = []
    for color in [c for c in np.unique(grid) if c != 0]:
        runs = []
        for r in range(h):
            c = 0
            while c < w:
                if grid[r, c] != color:
                    c += 1
                    continue
                start = c
                while c < w and grid[r, c] == color:
                    c += 1
                if c - start >= 4:
                    runs.append((r, start, c - 1))
        for i, (r1, c1, c2) in enumerate(runs):
            for r2, d1, d2 in runs[i + 1:]:
                if r2 - r1 < 3 or (c1, c2) != (d1, d2):
                    continue
                if np.all(grid[r1:r2 + 1, c1] == color) and np.all(grid[r1:r2 + 1, c2] == color):
                    interior = grid[r1 + 1:r2, c1 + 1:c2]
                    if interior.size and not np.any(interior == color):
                        out.append((r1, c1, r2, c2, int(color)))
    out.sort(key=lambda x: -((x[2] - x[0] + 1) * (x[3] - x[1] + 1)))
    kept = []
    for f in out:
        cells = {(r, c) for r in range(f[0], f[2] + 1) for c in range(f[1], f[3] + 1)
                 if r in (f[0], f[2]) or c in (f[1], f[3])}
        if not any(cells & k for _, k in kept):
            kept.append((f, cells))
    return [f for f, _ in kept]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    res = grid.copy()
    frames = _frames(grid)
    border_cells = set()
    for r1, c1, r2, c2, _ in frames:
        border_cells |= {(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
                         if r in (r1, r2) or c in (c1, c2)}
    for r, c in zip(*np.nonzero(grid)):
        if (r, c) not in border_cells:
            res[r, c] = 0
    for r1, c1, r2, c2, color in frames:
        border = {(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
                  if r in (r1, r2) or c in (c1, c2)}
        markers = [(r, c) for r, c in zip(*np.where(grid == color)) if (r, c) not in border_cells]
        n = 0
        for mr, mc in markers:
            if any((br, bc) in border for br in range(mr - 1, mr + 2) for bc in range(mc - 1, mc + 2)):
                n += 1
        rows = list(range(r1 + 1 + ((r2 - r1 - 1) - 1) // 2, r2))
        cols = list(range(c2 - 1, c1, -2))
        placed = 0
        for rr in rows:
            for cc in cols:
                if placed < n:
                    res[rr, cc] = color
                    placed += 1
    return res


def _draw_frame(g, r, c, hh, ww, color):
    g[r, c:c + ww] = color
    g[r + hh - 1, c:c + ww] = color
    g[r:r + hh, c] = color
    g[r:r + hh, c + ww - 1] = color


def generate_input() -> np.ndarray:
    h = w = random.choice([17, 18, 19, 20])
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], random.randint(3, 4))
    slots = [(1, 1), (1, w - 7), (h - 7, 1), (h - 7, w - 7)]
    random.shuffle(slots)
    frames = []
    for color, (sr, sc) in zip(colors, slots):
        hh = random.randint(4, 6)
        ww = random.randint(4, 6)
        r = min(max(1, sr + random.randint(0, 1)), h - hh - 1)
        c = min(max(1, sc + random.randint(0, 1)), w - ww - 1)
        _draw_frame(g, r, c, hh, ww, color)
        frames.append((r, c, r + hh - 1, c + ww - 1, color))
    used = set(zip(*np.nonzero(g)))
    for r1, c1, r2, c2, color in frames:
        border = [(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
                  if r in (r1, r2) or c in (c1, c2)]
        touch_count = random.randint(1, 3)
        total = random.randint(touch_count, 4)
        marks = []
        random.shuffle(border)
        for br, bc in border:
            opts = [(br + dr, bc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if 0 <= br + dr < h and 0 <= bc + dc < w and (br + dr, bc + dc) not in used]
            random.shuffle(opts)
            for p in opts:
                if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 2 for q in marks):
                    marks.append(p)
                    used.add(p)
                    break
            if len(marks) == touch_count:
                break
        attempts = 0
        while len(marks) < total and attempts < 500:
            attempts += 1
            p = (random.randrange(h), random.randrange(w))
            if p in used:
                continue
            dist = min(max(abs(p[0] - br), abs(p[1] - bc)) for br, bc in border)
            if dist > 1 and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 2 for q in marks):
                marks.append(p)
                used.add(p)
        for mr, mc in marks:
            g[mr, mc] = color
    return g
