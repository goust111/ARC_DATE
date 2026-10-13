import random
from collections import deque

import numpy as np


BG = 0


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(1, h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 1 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _crop(color, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = color
    return arr


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    segments = []
    c = 0
    while c < grid.shape[1]:
        color = int(grid[0, c])
        start = c
        while c < grid.shape[1] and int(grid[0, c]) == color:
            c += 1
        if color != BG:
            segments.append((color, c - start))

    comp_map = {}
    for color, cells in _components_by_color(grid):
        comp_map[color] = _crop(color, cells)

    pieces = []
    for color, length in segments:
        piece = comp_map[color]
        rot = (length - 1) % 4
        piece = np.rot90(piece, -rot)
        pieces.append(piece)

    height = max(p.shape[0] for p in pieces)
    width = sum(p.shape[1] for p in pieces) + len(pieces) - 1
    out = np.zeros((height, width), dtype=int)
    x = 0
    for piece in pieces:
        ph, pw = piece.shape
        out[height - ph :, x : x + pw] = np.maximum(out[height - ph :, x : x + pw], piece)
        x += pw + 1
    return out


def _random_mask():
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < random.randint(2, min(6, h * w)):
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    arr = np.zeros((h, w), dtype=int)
    for r, c in cells:
        arr[r, c] = 1
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    return arr[rows][:, cols]


def generate_input() -> np.ndarray:
    count = random.randint(2, 4)
    bar_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count + random.randint(1, 2))
    keep_colors = bar_colors[:count]
    seg_lengths = [random.randint(1, 4) for _ in range(count)]
    bar = []
    for color, length in zip(keep_colors, seg_lengths):
        bar.extend([color] * length)
    w = max(len(bar) + 6, 16)
    h = random.randint(12, 20)
    grid = np.zeros((h, w), dtype=int)
    grid[0, : len(bar)] = np.array(bar, dtype=int)

    occupied = grid != BG
    placed_colors = keep_colors + bar_colors[count:]
    for idx, color in enumerate(placed_colors):
        mask = _random_mask()
        ph, pw = mask.shape
        for _ in range(200):
            top = random.randint(2, h - ph - 1)
            left = random.randint(0, w - pw)
            if occupied[max(0, top - 1) : top + ph + 1, max(0, left - 1) : left + pw + 1].any():
                continue
            grid[top : top + ph, left : left + pw][mask == 1] = color
            occupied[top : top + ph, left : left + pw] |= mask == 1
            break
    return grid
