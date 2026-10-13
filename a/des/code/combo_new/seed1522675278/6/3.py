import random
from collections import deque

import numpy as np


BG = 0
WALL = 5


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


def _boxes(grid):
    boxes = []
    for comp in _components(grid == WALL):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(comp) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
            boxes.append((r0, r1, c0, c1))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    key_patterns = {}
    locks = []
    for r0, r1, c0, c1 in _boxes(grid):
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = [c for c in np.unique(inner) if c not in (BG, WALL)]
        if colors:
            color = colors[0]
            pts = np.argwhere(inner == color)
            rr0, cc0 = pts.min(axis=0)
            rr1, cc1 = pts.max(axis=0)
            key_patterns[color] = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        else:
            border_colors = [grid[r0, c] for c in range(c0, c1 + 1)] + [grid[r1, c] for c in range(c0, c1 + 1)] + [grid[r, c0] for r in range(r0, r1 + 1)] + [grid[r, c1] for r in range(r0, r1 + 1)]
            mark = [c for c in border_colors if c not in (WALL, BG)]
            if mark:
                locks.append((r0, r1, c0, c1, mark[0]))
    for r0, r1, c0, c1, color in locks:
        if color not in key_patterns:
            continue
        pat = key_patterns[color]
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        sr = r0 + 1 + (ih - pat.shape[0]) // 2
        sc = c0 + 1 + (iw - pat.shape[1]) // 2
        for rr in range(pat.shape[0]):
            for cc in range(pat.shape[1]):
                if pat[rr, cc] == color:
                    out[sr + rr, sc + cc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    cursor = 1
    top = 2
    for color in colors:
        ih = random.randint(4, 6)
        iw = random.randint(4, 6)
        if cursor + iw + 2 >= w:
            break
        grid[top : top + ih + 2, cursor] = WALL
        grid[top : top + ih + 2, cursor + iw + 1] = WALL
        grid[top, cursor : cursor + iw + 2] = WALL
        grid[top + ih + 1, cursor : cursor + iw + 2] = WALL
        ph, pw = random.randint(2, 3), random.randint(2, 3)
        pr = top + 1 + random.randint(0, ih - ph)
        pc = cursor + 1 + random.randint(0, iw - pw)
        for rr in range(pr, pr + ph):
            for cc in range(pc, pc + pw):
                if random.random() < 0.6:
                    grid[rr, cc] = color
        if not np.any(grid[top + 1 : top + ih + 1, cursor + 1 : cursor + iw + 1] == color):
            grid[top + 1, cursor + 1] = color
        ltop = top + ih + 4
        if ltop + ih + 2 >= h:
            break
        grid[ltop : ltop + ih + 2, cursor] = WALL
        grid[ltop : ltop + ih + 2, cursor + iw + 1] = WALL
        grid[ltop, cursor : cursor + iw + 2] = WALL
        grid[ltop + ih + 1, cursor : cursor + iw + 2] = WALL
        side = random.choice(["t", "b", "l", "r"])
        if side == "t":
            grid[ltop, cursor + 1] = color
        elif side == "b":
            grid[ltop + ih + 1, cursor + iw] = color
        elif side == "l":
            grid[ltop + 1, cursor] = color
        else:
            grid[ltop + ih, cursor + iw + 1] = color
        cursor += iw + 4
    return grid
