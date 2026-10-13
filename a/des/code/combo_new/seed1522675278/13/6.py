import random
from collections import deque

import numpy as np


BG = 0
AZURE = 8


def _color_components(grid, color):
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def _neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) in cells]


def _tail_info(comp):
    cells = set(comp)
    deg = {cell: len(_neighbors(cell, cells)) for cell in cells}
    tip = next(cell for cell, d in deg.items() if d == 1)
    tail = [tip]
    prev = None
    cur = tip
    while True:
        nxts = [n for n in _neighbors(cur, cells) if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if deg[nxt] >= 3:
            break
        tail.append(nxt)
        prev, cur = cur, nxt
    direction = (tail[0][0] - tail[1][0], tail[0][1] - tail[1][1])
    body = cells - set(tail)
    return direction, tail[0], body


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    colors = [c for c in np.unique(grid) if c not in (BG, AZURE)]
    border = {}
    objects = []
    for color in colors:
        for comp in _color_components(grid, color):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                border[color] = comp
            else:
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                objects.append((min(rows), min(cols), color, comp))
    out = np.zeros_like(grid)
    for color, comp in border.items():
        for r, c in comp:
            out[r, c] = color
    for _, _, color, comp in sorted(objects):
        direction, tip, body = _tail_info(comp)
        dr, dc = direction
        if dr == -1:
            shift = tip[0] - 1
        elif dr == 1:
            shift = (h - 2) - tip[0]
        elif dc == -1:
            shift = tip[1] - 1
        else:
            shift = (w - 2) - tip[1]
        for step in range(shift):
            for r, c in body:
                nr, nc = r + dr * step, c + dc * step
                out[nr, nc] = AZURE
        for r, c in comp:
            nr, nc = r + dr * shift, c + dc * shift
            out[nr, nc] = color
    return out


def _object_patch(color, direction):
    bh, bw = random.randint(2, 3), random.randint(2, 3)
    body = np.full((bh, bw), color, dtype=int)
    tail_len = random.randint(2, 4)
    if direction == "U":
        arr = np.zeros((bh + tail_len, bw), dtype=int)
        arr[tail_len:, :] = body
        arr[:tail_len, bw // 2] = color
    elif direction == "D":
        arr = np.zeros((bh + tail_len, bw), dtype=int)
        arr[:bh, :] = body
        arr[bh:, bw // 2] = color
    elif direction == "L":
        arr = np.zeros((bh, bw + tail_len), dtype=int)
        arr[:, tail_len:] = body
        arr[bh // 2, :tail_len] = color
    else:
        arr = np.zeros((bh, bw + tail_len), dtype=int)
        arr[:, :bw] = body
        arr[bh // 2, bw:] = color
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 9], random.randint(2, 4))
    for color in colors:
        direction = random.choice(["U", "D", "L", "R"])
        patch = _object_patch(color, direction)
        ph, pw = patch.shape
        if direction == "U":
            seg = random.randint(3, 6)
            c0 = random.randint(2, 24 - seg - 2)
            grid[0, c0 : c0 + seg] = color
            target_col = random.randint(c0, c0 + seg - 1)
            for _ in range(200):
                r = random.randint(6, 20 - ph)
                c = target_col - pw // 2
                if c < 1 or c + pw > 23:
                    continue
                if np.any(used[r : r + ph, c : c + pw] & (patch != BG)):
                    continue
                grid[r : r + ph, c : c + pw] = np.where(patch != BG, patch, grid[r : r + ph, c : c + pw])
                used[r : r + ph, c : c + pw] |= patch != BG
                break
        elif direction == "D":
            seg = random.randint(3, 6)
            c0 = random.randint(2, 24 - seg - 2)
            grid[-1, c0 : c0 + seg] = color
            target_col = random.randint(c0, c0 + seg - 1)
            for _ in range(200):
                r = random.randint(1, 14 - ph)
                c = target_col - pw // 2
                if c < 1 or c + pw > 23:
                    continue
                if np.any(used[r : r + ph, c : c + pw] & (patch != BG)):
                    continue
                grid[r : r + ph, c : c + pw] = np.where(patch != BG, patch, grid[r : r + ph, c : c + pw])
                used[r : r + ph, c : c + pw] |= patch != BG
                break
        elif direction == "L":
            seg = random.randint(3, 6)
            r0 = random.randint(2, 24 - seg - 2)
            grid[r0 : r0 + seg, 0] = color
            target_row = random.randint(r0, r0 + seg - 1)
            for _ in range(200):
                r = target_row - ph // 2
                c = random.randint(6, 20 - pw)
                if r < 1 or r + ph > 23:
                    continue
                if np.any(used[r : r + ph, c : c + pw] & (patch != BG)):
                    continue
                grid[r : r + ph, c : c + pw] = np.where(patch != BG, patch, grid[r : r + ph, c : c + pw])
                used[r : r + ph, c : c + pw] |= patch != BG
                break
        else:
            seg = random.randint(3, 6)
            r0 = random.randint(2, 24 - seg - 2)
            grid[r0 : r0 + seg, -1] = color
            target_row = random.randint(r0, r0 + seg - 1)
            for _ in range(200):
                r = target_row - ph // 2
                c = random.randint(1, 14 - pw)
                if r < 1 or r + ph > 23:
                    continue
                if np.any(used[r : r + ph, c : c + pw] & (patch != BG)):
                    continue
                grid[r : r + ph, c : c + pw] = np.where(patch != BG, patch, grid[r : r + ph, c : c + pw])
                used[r : r + ph, c : c + pw] |= patch != BG
                break
    return grid
