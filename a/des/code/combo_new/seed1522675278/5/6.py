import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def _mask_from(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=int)
    for r, c in comp:
        mask[r - r0, c - c0] = 1
    return (r0, c0), mask


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    locks = []
    keys = []
    frame_colors = set()
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            border = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
            if len(comp) == border and r1 - r0 >= 3 and c1 - c0 >= 3:
                hole = grid[r0 + 1 : r1, c0 + 1 : c1] == BG
                if np.any(hole):
                    frame_colors.add(color)
                    locks.append((color, r0, r1, c0, c1, hole.copy()))
                    continue
            keys.append((color, comp))
    for color, comp in keys:
        for idx, (fcolor, r0, r1, c0, c1, hole) in enumerate(locks):
            (_, _), km = _mask_from(comp)
            for k in range(4):
                rot = np.rot90(km, k)
                if rot.shape == hole.shape and np.array_equal(rot, hole.astype(int)):
                    for r, c in comp:
                        out[r, c] = BG
                    for rr in range(rot.shape[0]):
                        for cc in range(rot.shape[1]):
                            if rot[rr, cc]:
                                out[r0 + 1 + rr, c0 + 1 + cc] = color
                    locks.pop(idx)
                    break
            else:
                continue
            break
    grown = out.copy()
    for color in frame_colors:
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if out[r, c] != color:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and grown[nr, nc] == BG:
                        grown[nr, nc] = color
    return grown


def generate_input() -> np.ndarray:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    grid = np.zeros((h, w), dtype=int)
    for fcolor, kcolor in zip(random.sample([1, 2, 3, 4], 2), random.sample([6, 7, 8, 9], 2)):
        hh = ww = random.randint(4, 5)
        top = random.randint(1, h - hh - 3)
        left = random.randint(1, w - ww - 3)
        grid[top : top + hh + 2, left] = fcolor
        grid[top : top + hh + 2, left + ww + 1] = fcolor
        grid[top, left : left + ww + 2] = fcolor
        grid[top + hh + 1, left : left + ww + 2] = fcolor
        hole = np.zeros((hh, ww), dtype=int)
        for _ in range(random.randint(2, 4)):
            hole[random.randint(0, hh - 1), random.randint(0, ww - 1)] = 1
        if not np.any(hole):
            hole[0, 0] = 1
        rr = random.randint(1, h - hh - 2)
        cc = random.randint(1, w - ww - 2)
        rot = np.rot90(hole, random.randint(0, 3))
        for r in range(hh):
            for c in range(ww):
                if rot[r, c]:
                    grid[rr + r, cc + c] = kcolor
    return grid
