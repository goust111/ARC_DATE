import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    seeds = np.argwhere(out == 1)
    if len(seeds) == 0:
        return out
    sr, sc = map(int, seeds[0])
    obstacle_boxes = [_bbox(comp) for comp in _components(out == 2)]
    box_at = {}
    for box in obstacle_boxes:
        r1, c1, r2, c2 = box
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if out[r, c] == 2:
                    box_at[(r, c)] = box

    for dr0, dc0 in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        r, c = sr, sc
        dr, dc = dr0, dc0
        seen_states = set()
        for _ in range(h * w * 8):
            state = (r, c, dr, dc)
            if state in seen_states:
                break
            seen_states.add(state)
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if out[nr, nc] == 3:
                break
            if out[nr, nc] == 2:
                r1, c1, r2, c2 = box_at[(nr, nc)]
                enters_row = not (r1 <= r <= r2) and (r1 <= nr <= r2)
                enters_col = not (c1 <= c <= c2) and (c1 <= nc <= c2)
                if enters_row:
                    dr = -dr
                if enters_col:
                    dc = -dc
                if not enters_row and not enters_col:
                    dr, dc = -dr, -dc
                continue
            r, c = nr, nc
            if out[r, c] == 0:
                out[r, c] = 3

    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if out[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if out[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0 and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    out[(out == 0) & (~outside)] = 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    sr = random.randint(5, h - 6)
    sc = random.randint(5, w - 6)
    grid[sr, sc] = 1

    # Two rectangular bumpers near opposite diagonal quadrants produce real reflections.
    rects = []
    for rr, cc in ((sr - random.randint(3, 5), sc + random.randint(3, 5)),
                   (sr + random.randint(3, 5), sc - random.randint(3, 5))):
        rh = random.randint(2, 3)
        rw = random.randint(3, 5)
        r1 = max(1, min(h - rh - 2, rr))
        c1 = max(1, min(w - rw - 2, cc))
        rects.append((r1, c1, r1 + rh - 1, c1 + rw - 1))
    for r1, c1, r2, c2 in rects:
        grid[r1:r2 + 1, c1:c2 + 1] = 2
    if any(abs(r - sr) + abs(c - sc) <= 1 for r, c in np.argwhere(grid == 2)):
        return generate_input()
    return grid
