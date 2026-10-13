import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    pat_pts = np.argwhere((grid == 1) | (grid == 2) | (grid == 3))
    pr0, pc0 = pat_pts.min(axis=0)
    pr1, pc1 = pat_pts.max(axis=0)
    for r, c in zip(*np.where(grid[pr0 : pr1 + 1, pc0 : pc1 + 1] == 3)):
        rr, cc = int(pr0 + r), int(pc0 + c)
        mr, mc = int(pr0 + pr1 - rr), int(pc0 + pc1 - cc)
        if out[mr, mc] in (1, 2):
            out[rr, cc] = out[mr, mc]
    rects = _components(grid == 5)
    center_r = (pr0 + pr1) / 2
    center_c = (pc0 + pc1) / 2
    for comp in rects:
        pts = np.array(comp)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        rc = (r0 + r1) / 2
        cc = (c0 + c1) / 2
        if rc < center_r:
            edge_r = pr0
            edge_c = int(np.floor(np.clip(cc, pc0, pc1)))
        elif rc > center_r:
            edge_r = pr1
            edge_c = int(np.floor(np.clip(cc, pc0, pc1)))
        elif cc < center_c:
            edge_r = int(np.floor(np.clip(rc, pr0, pr1)))
            edge_c = pc0
        else:
            edge_r = int(np.floor(np.clip(rc, pr0, pr1)))
            edge_c = pc1
        fill = int(out[edge_r, edge_c])
        if fill in (1, 2):
            out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    pr0 = pc0 = 5
    n = 5
    pat = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            if random.random() < 0.7:
                color = random.choice([1, 2])
                pat[r, c] = color
                pat[n - 1 - r, n - 1 - c] = color
    for r in range(n):
        for c in range(n):
            if pat[r, c] == 0:
                pat[r, c] = random.choice([1, 2])
                pat[n - 1 - r, n - 1 - c] = pat[r, c]
    for r, c in random.sample(list(zip(*np.where(pat != 0))), random.randint(3, 5)):
        pat[r, c] = 3
    edge_choices = [(0, random.randrange(n)), (n - 1, random.randrange(n)), (random.randrange(n), 0), (random.randrange(n), n - 1)]
    r, c = random.choice(edge_choices)
    pat[r, c] = 3
    grid[pr0 : pr0 + n, pc0 : pc0 + n] = pat
    rects = [(1, 3, 5, 9), (11, 13, 5, 9), (5, 9, 1, 3), (5, 9, 11, 13)]
    for r0, r1, c0, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = 5
    return grid
