import random
from collections import deque

import numpy as np


BG = 0
AXIS = 9


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
    horiz = np.sum(grid == AXIS, axis=1).max() == grid.shape[1]
    axis = int(np.argmax(np.sum(grid == AXIS, axis=1 if horiz else 0)))
    for color in [c for c in np.unique(grid) if c not in (BG, AXIS)]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            notch = None
            midr, midc = (r0 + r1) // 2, (c0 + c1) // 2
            if grid[r0, midc] == BG:
                notch = "U"
            elif grid[r1, midc] == BG:
                notch = "D"
            elif grid[midr, c0] == BG:
                notch = "L"
            else:
                notch = "R"
            move = (horiz and ((notch == "D" and axis > r1) or (notch == "U" and axis < r0))) or (not horiz and ((notch == "R" and axis > c1) or (notch == "L" and axis < c0)))
            if not move:
                continue
            for r, c in comp:
                out[r, c] = BG
            if horiz:
                new_comp = [(2 * axis - r, c) for r, c in comp]
            else:
                new_comp = [(r, 2 * axis - c) for r, c in comp]
            for r, c in new_comp:
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randrange(17, 29, 2)
    w = random.randrange(17, 29, 2)
    grid = np.zeros((h, w), dtype=int)
    horiz = random.choice([True, False])
    if horiz:
        axis = h // 2
        grid[axis, :] = AXIS
    else:
        axis = w // 2
        grid[:, axis] = AXIS
    for color in random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4)):
        oh = random.choice([3, 5])
        ow = random.choice([3, 5])
        top = random.randint(1, h - oh - 2)
        left = random.randint(1, w - ow - 2)
        grid[top : top + oh, left : left + ow] = color
        side = random.choice(["U", "D", "L", "R"])
        if side == "U":
            grid[top, left + ow // 2] = BG
        elif side == "D":
            grid[top + oh - 1, left + ow // 2] = BG
        elif side == "L":
            grid[top + oh // 2, left] = BG
        else:
            grid[top + oh // 2, left + ow - 1] = BG
    return grid
