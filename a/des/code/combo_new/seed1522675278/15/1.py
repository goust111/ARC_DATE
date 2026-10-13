import random
from collections import deque

import numpy as np


BG = 0


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


def _is_frame(grid, comp, color):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    exp = set()
    for c in range(c0, c1 + 1):
        exp.add((r0, c))
        exp.add((r1, c))
    for r in range(r0, r1 + 1):
        exp.add((r, c0))
        exp.add((r, c1))
    return set(comp) == exp and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    templates = {}
    frames = []
    for color in [v for v in np.unique(grid) if v != BG]:
        for comp in _components(grid == color):
            if _is_frame(grid, comp, int(color)):
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                frames.append((int(color), min(rows), max(rows), min(cols), max(cols)))
            else:
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                templates[int(color)] = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    for color, r0, r1, c0, c1 in frames:
        templ = templates[color]
        th, tw = templ.shape
        fill_color = int(np.max(grid[r0 + 1 : r1, c0 + 1 : c1]))
        for br in range(r0 + 1, r1, th):
            for bc in range(c0 + 1, c1, tw):
                tile = out[br : br + th, bc : bc + tw]
                mask = templ != BG
                tile[:, :] = np.where(mask[: tile.shape[0], : tile.shape[1]], fill_color, BG)
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3))
    templates = {}
    for idx, color in enumerate(colors):
        h = random.randint(2, 3)
        w = random.randint(2, 4)
        arr = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                if random.random() < 0.6:
                    arr[r, c] = color
        if np.count_nonzero(arr) == 0:
            arr[h // 2, w // 2] = color
        templates[color] = arr
        row = 1
        col = 2 + idx * 7
        grid[row : row + h, col : col + w] = arr
    frame_positions = [(10, 2), (10, 11), (16, 6)]
    for idx, color in enumerate(colors):
        templ = templates[color]
        th, tw = templ.shape
        mult_h = random.randint(1, 2)
        mult_w = random.randint(1, 2)
        ih, iw = th * mult_h, tw * mult_w
        r0, c0 = frame_positions[idx]
        fill = random.choice([v for v in range(1, 10) if v != color])
        grid[r0, c0 : c0 + iw + 2] = color
        grid[r0 + ih + 1, c0 : c0 + iw + 2] = color
        grid[r0 : r0 + ih + 2, c0] = color
        grid[r0 : r0 + ih + 2, c0 + iw + 1] = color
        grid[r0 + 1 : r0 + ih + 1, c0 + 1 : c0 + iw + 1] = fill
    return grid
