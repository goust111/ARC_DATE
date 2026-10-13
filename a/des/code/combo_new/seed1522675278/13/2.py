import random
from collections import deque

import numpy as np


BG = 0


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] != BG and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _arrow_info(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    pts = {(r - r0, c - c0) for r, c in comp}
    if h == 3 and w >= 4:
        mid = 1
        if all((mid, x) in pts for x in range(w)) and pts == {(0, w - 1), (2, w - 1)} | {(1, x) for x in range(w)}:
            return "R", (r0 + 1, c0), (r0 + 1, c1)
        if all((mid, x) in pts for x in range(w)) and pts == {(0, 0), (2, 0)} | {(1, x) for x in range(w)}:
            return "L", (r0 + 1, c1), (r0 + 1, c0)
    if w == 3 and h >= 4:
        mid = 1
        if all((x, mid) in pts for x in range(h)) and pts == {(h - 1, 0), (h - 1, 2)} | {(x, 1) for x in range(h)}:
            return "D", (r0, c0 + 1), (r1, c0 + 1)
        if all((x, mid) in pts for x in range(h)) and pts == {(0, 0), (0, 2)} | {(x, 1) for x in range(h)}:
            return "U", (r1, c0 + 1), (r0, c0 + 1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    comps = _components(grid)
    arrows = []
    blocks = []
    for comp in comps:
        info = _arrow_info(comp)
        if info:
            arrows.append((comp, info))
        else:
            blocks.append(comp)
    out = grid.copy()
    for comp, (direction, tail, tip) in arrows:
        for r, c in comp:
            out[r, c] = BG
        target = None
        for block in blocks:
            rows = [r for r, _ in block]
            cols = [c for _, c in block]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            tr, tc = tail
            if direction == "R" and c1 == tc - 1 and r0 <= tr <= r1:
                target = block
            elif direction == "L" and c0 == tc + 1 and r0 <= tr <= r1:
                target = block
            elif direction == "D" and r1 == tr - 1 and c0 <= tc <= c1:
                target = block
            elif direction == "U" and r0 == tr + 1 and c0 <= tc <= c1:
                target = block
        if target is None:
            continue
        rows = [r for r, _ in target]
        cols = [c for _, c in target]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if direction == "R":
            nr, nc = r0, tip[1] + 2
        elif direction == "L":
            nr, nc = r0, tip[1] - 1 - patch.shape[1]
        elif direction == "D":
            nr, nc = tip[0] + 2, c0
        else:
            nr, nc = tip[0] - 1 - patch.shape[0], c0
        h, w = patch.shape
        out[nr : nr + h, nc : nc + w] = np.where(patch != BG, patch, out[nr : nr + h, nc : nc + w])
    return out


def _arrow_patch(color, direction, length):
    if direction in ("R", "L"):
        arr = np.zeros((3, length + 1), dtype=int)
        arr[1, :] = color
        if direction == "R":
            arr[0, -1] = color
            arr[2, -1] = color
        else:
            arr = arr[:, ::-1]
            arr[0, 0] = color
            arr[2, 0] = color
        return arr
    arr = np.zeros((length + 1, 3), dtype=int)
    arr[:, 1] = color
    if direction == "D":
        arr[-1, 0] = color
        arr[-1, 2] = color
    else:
        arr = arr[::-1, :]
        arr[0, 0] = color
        arr[0, 2] = color
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    for color in colors:
        direction = random.choice(["R", "L", "U", "D"])
        length = random.randint(3, 5)
        arrow = _arrow_patch(color, direction, length)
        ah, aw = arrow.shape
        ph, pw = random.randint(2, 4), random.randint(2, 4)
        pattern = np.zeros((ph, pw), dtype=int)
        vals = random.sample([v for v in range(1, 10) if v != color], random.randint(1, min(4, ph * pw)))
        for r in range(ph):
            for c in range(pw):
                pattern[r, c] = random.choice(vals)
        for _ in range(300):
            if direction == "R":
                r = random.randint(3, 20 - ah)
                c = random.randint(pw + 1, 24 - aw - pw - 3)
                pr, pc = r, c - pw
            elif direction == "L":
                r = random.randint(3, 20 - ah)
                c = random.randint(pw + 3, 24 - aw - 1)
                pr, pc = r, c + aw
            elif direction == "D":
                r = random.randint(ph + 1, 24 - ah - ph - 3)
                c = random.randint(3, 20 - aw)
                pr, pc = r - ph, c
            else:
                r = random.randint(ph + 3, 24 - ah - 1)
                c = random.randint(3, 20 - aw)
                pr, pc = r + ah, c
            if not (0 <= pr and 0 <= pc and pr + ph <= 24 and pc + pw <= 24):
                continue
            if np.any(used[r : r + ah, c : c + aw] & (arrow != BG)):
                continue
            if np.any(used[pr : pr + ph, pc : pc + pw] & (pattern != BG)):
                continue
            nr, nc = pr, pc
            if direction == "R":
                nc = c + aw + 1
            elif direction == "L":
                nc = c - pw - 2
            elif direction == "D":
                nr = r + ah + 1
            else:
                nr = r - ph - 2
            if not (0 <= nr and 0 <= nc and nr + ph <= 24 and nc + pw <= 24):
                continue
            if np.any(used[nr : nr + ph, nc : nc + pw]):
                continue
            grid[r : r + ah, c : c + aw] = np.where(arrow != BG, arrow, grid[r : r + ah, c : c + aw])
            grid[pr : pr + ph, pc : pc + pw] = pattern
            used[r : r + ah, c : c + aw] |= arrow != BG
            used[pr : pr + ph, pc : pc + pw] |= pattern != BG
            break
    return grid
