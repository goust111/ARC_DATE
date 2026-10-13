import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cur)
    return comps


def _frame_boxes(grid):
    boxes = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(comp) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
            boxes.append((r0, r1, c0, c1))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    frames = _frame_boxes(grid)
    top_limit = min(r0 for r0, _, _, _ in frames)
    templates = {}
    for color in [c for c in np.unique(grid[:top_limit]) if c not in (BG, GREY)]:
        pts = np.argwhere(grid[:top_limit] == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3:
            box = np.zeros((3, 3), dtype=int)
            box[: r1 - r0 + 1, : c1 - c0 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
            templates[color] = box
    for r0, r1, c0, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [c for c in np.unique(inner) if c not in (BG, GREY)]
        if not seeds:
            continue
        color = seeds[0]
        tile = templates[color]
        hh, ww = inner.shape
        filled = np.zeros_like(inner)
        for r in range(hh):
            for c in range(ww):
                filled[r, c] = tile[r % 3, c % 3]
        out[r0 + 1 : r1, c0 + 1 : c1] = filled
    return out


def generate_input() -> np.ndarray:
    h = random.randint(24, 30)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    patterns = {}
    y = 1
    for color in colors:
        pat = np.random.randint(0, 2, size=(3, 3)) * color
        if not np.any(pat == color):
            pat[1, 1] = color
        patterns[color] = pat
        x = random.randint(1, w - 4)
        grid[y : y + 3, x : x + 3] = pat
        y += 4
    boxes = random.randint(2, 3)
    cursor = 1
    start_row = y + 1
    for _ in range(boxes):
        color = random.choice(colors)
        ih = random.randint(4, 7)
        iw = random.randint(4, 7)
        if start_row + ih + 2 >= h or cursor + iw + 2 >= w:
            break
        top = start_row
        left = cursor
        grid[top : top + ih + 2, left] = GREY
        grid[top : top + ih + 2, left + iw + 1] = GREY
        grid[top, left : left + iw + 2] = GREY
        grid[top + ih + 1, left : left + iw + 2] = GREY
        sr = random.randint(top + 1, top + ih)
        sc = random.randint(left + 1, left + iw)
        grid[sr, sc] = color
        cursor += iw + 4
        start_row += random.randint(0, 2)
    return grid
