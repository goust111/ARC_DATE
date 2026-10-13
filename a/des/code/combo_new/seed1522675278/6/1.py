import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for comp in _components(grid == FRAME):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        top_missing = np.sum(grid[r0, c0 : c1 + 1] == FRAME) < (c1 - c0 + 1)
        bot_missing = np.sum(grid[r1, c0 : c1 + 1] == FRAME) < (c1 - c0 + 1)
        left_missing = np.sum(grid[r0 : r1 + 1, c0] == FRAME) < (r1 - r0 + 1)
        right_missing = np.sum(grid[r0 : r1 + 1, c1] == FRAME) < (r1 - r0 + 1)
        if top_missing:
            dr, dc = -1, 0
            step = r1 - r0 + 1
        elif bot_missing:
            dr, dc = 1, 0
            step = r1 - r0 + 1
        elif left_missing:
            dr, dc = 0, -1
            step = c1 - c0 + 1
        else:
            dr, dc = 0, 1
            step = c1 - c0 + 1
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [(r, c) for r in range(inside.shape[0]) for c in range(inside.shape[1]) if inside[r, c] not in (BG, FRAME)]
        if not seeds:
            continue
        color = inside[seeds[0]]
        cr = (r0 + r1) / 2.0
        cc = (c0 + c1) / 2.0
        rel = [(r0 + 1 + r - cr, c0 + 1 + c - cc) for r, c in seeds]
        k = 1
        while True:
            placed = False
            for rr, cc2 in rel:
                nr = int(round(cr + rr + dr * step * k))
                nc = int(round(cc + cc2 + dc * step * k))
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    out[nr, nc] = color
                    placed = True
            if not placed:
                break
            k += 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(2, 4)):
        fh = random.randint(4, 7)
        fw = random.randint(4, 7)
        top = random.randint(2, h - fh - 3)
        left = random.randint(2, w - fw - 3)
        grid[top : top + fh, left] = FRAME
        grid[top : top + fh, left + fw - 1] = FRAME
        grid[top, left : left + fw] = FRAME
        grid[top + fh - 1, left : left + fw] = FRAME
        side = random.choice(["t", "b", "l", "r"])
        if side in "tb":
            c = random.randint(left + 1, left + fw - 2)
            grid[top if side == "t" else top + fh - 1, c] = BG
        else:
            r = random.randint(top + 1, top + fh - 2)
            grid[r, left if side == "l" else left + fw - 1] = BG
        color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        count = random.randint(1, 4)
        placed = set()
        while len(placed) < count:
            r = random.randint(top + 1, top + fh - 2)
            c = random.randint(left + 1, left + fw - 2)
            placed.add((r, c))
        for p in placed:
            grid[p] = color
    return grid
