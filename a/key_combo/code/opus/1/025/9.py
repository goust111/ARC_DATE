import random
import numpy as np


def _components(mask, diag=False):
    h, w = mask.shape
    ds = ((1, 0), (-1, 0), (0, 1), (0, -1)) if not diag else tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0))
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ds:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _largest_rect(mask):
    h, w = mask.shape
    best = (0, 0, 0, -1, -1)
    for r0 in range(h):
        good = np.ones(w, dtype=bool)
        for r1 in range(r0, h):
            good &= mask[r1]
            c = 0
            while c < w:
                if not good[c]:
                    c += 1
                    continue
                s = c
                while c < w and good[c]:
                    c += 1
                area = (r1 - r0 + 1) * (c - s)
                if r1 > r0 and c - s >= 2 and area > best[0]:
                    best = (area, r0, r1, s, c - 1)
    return best[1:]


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    r0, r1, c0, c1 = _largest_rect(grid == 1)
    out[grid == 1] = 0
    if r1 >= r0 and c1 >= c0:
        out[r0:r1 + 1, c0:c1 + 1] = 1
    h, w = grid.shape
    green = grid == 3
    clean_green = green.copy()
    for r, c in np.argwhere(green):
        black_n = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == 0:
                    black_n += 1
        if black_n >= 6:
            clean_green[r, c] = False
            out[r, c] = 0
    for comp in _components((out == 0), diag=False):
        if len(comp) > 2:
            continue
        ok = True
        for r, c in comp:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or ((nr, nc) not in comp and not clean_green[nr, nc]):
                        ok = False
        if ok:
            for r, c in comp:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0 and (nr, nc) not in comp:
                            out[nr, nc] = 7
    out[clean_green] = 3
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(14, 18)
    g = np.zeros((h, w), dtype=int)
    rh, rw = random.randint(3, 5), random.randint(3, 5)
    r0, c0 = random.randint(1, 3), random.randint(1, 3)
    g[r0:r0 + rh, c0:c0 + rw] = 1
    if random.random() < 0.6:
        g[r0 + random.randrange(rh), c0 + rw] = 1
    for _ in range(random.randint(3, 5)):
        rr, cc = random.randint(0, h - 1), random.randint(0, w - 1)
        if g[rr, cc] == 0:
            g[rr, cc] = 1
    gr, gc = random.randint(7, h - 6), random.randint(7, w - 6)
    g[gr:gr + 5, gc:gc + 5] = 3
    for p in random.sample([(gr + 1, gc + 1), (gr + 2, gc + 2), (gr + 3, gc + 3)], random.randint(1, 2)):
        g[p] = 0
    g[gr - 1, gc + 2] = 3
    g[gr + 2, gc + 5] = 3
    return g
