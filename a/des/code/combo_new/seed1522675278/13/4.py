import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
    comps = _components(grid != BG)
    silhouette = max(comps, key=len)
    silhouette_set = set(silhouette)
    sil_mask = np.zeros_like(grid, dtype=bool)
    for r, c in silhouette:
        sil_mask[r, c] = True
    out = np.zeros_like(grid)
    out[sil_mask] = GREY
    anchors = {}
    for r, c in silhouette:
        if grid[r, c] not in (BG, GREY):
            anchors[int(grid[r, c])] = (r, c)
    for comp in comps:
        if set(comp) == silhouette_set:
            continue
        colors = {grid[r, c] for r, c in comp}
        if colors == {GREY}:
            continue
        if len(colors - {GREY}) != 1 or all(sil_mask[r, c] or (grid[r, c] != GREY) for r, c in comp):
            continue
        color = int(list(colors - {GREY})[0])
        if color not in anchors:
            continue
        grey_cell = next((r, c) for r, c in comp if grid[r, c] == GREY)
        ar, ac = anchors[color]
        dr, dc = ar - grey_cell[0], ac - grey_cell[1]
        for r, c in comp:
            nr, nc = r + dr, c + dc
            out[nr, nc] = GREY if grid[r, c] == GREY else color
    return out


def _rand_fragment(color):
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    arr = np.zeros((h, w), dtype=int)
    cells = [(0, 0)]
    while len(cells) < random.randint(h + 1, h * w - 1):
        r, c = random.choice(cells)
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
            cells.append((nr, nc))
    for r, c in cells:
        arr[r, c] = color
    anchor = random.choice(cells)
    arr[anchor] = GREY
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((25, 25), dtype=int)
    silhouette = np.zeros((11, 11), dtype=int)
    silhouette[1:-1, 1:-1] = GREY
    for _ in range(8):
        r = random.randint(1, 9)
        c = random.randint(1, 9)
        silhouette[r, c] = BG
    # keep 4-connected and sizeable
    while len(_components(silhouette == GREY)) != 1:
        silhouette[random.randint(1, 9), random.randint(1, 9)] = GREY
    sr, sc = 7, 7
    grid[sr : sr + 11, sc : sc + 11] = silhouette
    free = [(r, c) for r in range(sr + 1, sr + 10) for c in range(sc + 1, sc + 10) if grid[r, c] == GREY]
    random.shuffle(free)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[sr : sr + 11, sc : sc + 11] = grid[sr : sr + 11, sc : sc + 11] != BG
    for color, anchor in zip(colors, free):
        grid[anchor] = color
        frag = _rand_fragment(color)
        fh, fw = frag.shape
        for _ in range(300):
            r = random.choice(list(range(0, 5)) + list(range(18, 25 - fh + 1)))
            c = random.randint(0, 25 - fw)
            if np.any(occupied[r : r + fh, c : c + fw] & (frag != BG)):
                continue
            grid[r : r + fh, c : c + fw] = np.where(frag != BG, frag, grid[r : r + fh, c : c + fw])
            occupied[r : r + fh, c : c + fw] |= frag != BG
            break
    return grid
