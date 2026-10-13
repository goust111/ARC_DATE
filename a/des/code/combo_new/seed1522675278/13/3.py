import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


def _components(mask):
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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    comps = _components(grid == FRAME)
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        gaps = []
        complete = 0
        for side, cells in (
            ("T", [(r0, c) for c in range(c0, c1 + 1)]),
            ("B", [(r1, c) for c in range(c0, c1 + 1)]),
            ("L", [(r, c0) for r in range(r0, r1 + 1)]),
            ("R", [(r, c1) for r in range(r0, r1 + 1)]),
        ):
            missing = [cell for cell in cells if grid[cell] != FRAME]
            if missing:
                gaps.extend(missing)
            else:
                complete += 1
        mark_color = None
        for r, c in sorted(gaps):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    val = grid[nr, nc]
                    if val not in (BG, FRAME):
                        mark_color = val
                        break
            if mark_color is not None:
                break
        if mark_color is None:
            continue
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if complete == 3 and (r - r0) % 2 == 1:
                    out[r, c] = mark_color
                elif complete == 2 and (c - c0) % 2 == 1:
                    out[r, c] = mark_color
                elif complete == 1 and (r + c) % 2 == (r0 + c0) % 2:
                    out[r, c] = mark_color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    marker_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for color in marker_colors:
        gaps_n = random.choice([1, 2, 3])
        h = random.randint(4, 6)
        w = random.randint(4, 7)
        for _ in range(200):
            r0 = random.randint(1, 24 - h - 2)
            c0 = random.randint(1, 24 - w - 2)
            r1, c1 = r0 + h - 1, c0 + w - 1
            if np.any(used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                continue
            grid[r0, c0 : c1 + 1] = FRAME
            grid[r1, c0 : c1 + 1] = FRAME
            grid[r0 : r1 + 1, c0] = FRAME
            grid[r0 : r1 + 1, c1] = FRAME
            sides = random.sample(["T", "B", "L", "R"], gaps_n)
            for side in sides:
                if side == "T":
                    c = random.randint(c0 + 1, c1 - 1)
                    grid[r0, c] = BG
                    grid[r0 - 1, c] = color
                elif side == "B":
                    c = random.randint(c0 + 1, c1 - 1)
                    grid[r1, c] = BG
                    grid[r1 + 1, c] = color
                elif side == "L":
                    r = random.randint(r0 + 1, r1 - 1)
                    grid[r, c0] = BG
                    grid[r, c0 - 1] = color
                else:
                    r = random.randint(r0 + 1, r1 - 1)
                    grid[r, c1] = BG
                    grid[r, c1 + 1] = color
            used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] != BG
            break
    return grid
