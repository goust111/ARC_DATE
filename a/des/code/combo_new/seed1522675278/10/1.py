import random
import numpy as np


BG = 0
GREY = 5


def _fences(grid: np.ndarray):
    rects = []
    for r in range(grid.shape[0] - 3):
        for c in range(grid.shape[1] - 3):
            if grid[r, c] != GREY:
                continue
            for h in range(4, 8):
                for w in range(4, 8):
                    if r + h > grid.shape[0] or c + w > grid.shape[1]:
                        continue
                    if np.all(grid[r, c : c + w] == GREY) and np.all(grid[r + h - 1, c : c + w] == GREY) and np.all(grid[r : r + h, c] == GREY) and np.all(grid[r : r + h, c + w - 1] == GREY):
                        rects.append((r, r + h - 1, c, c + w - 1))
    seen = []
    for x in rects:
        if x not in seen:
            seen.append(x)
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced room by tiling the legend texture selected by the room's seed color.

    Input structure:
    - Grey closed fences contain one colored seed pixel each.
    - A legend strip on the top edge shows one small color-specific binary texture per color.

    Rule:
    - Read the color-to-texture mapping from the legend blocks.
    - For every fence, choose the texture whose color matches the interior seed.
    - Tile that texture from the interior top-left corner and overwrite the fence's black
      interior, including the seed position.
    - Keep the grey fences and the legend unchanged.

    Key insight:
    - The seed color is only a lookup key; the actual fill is the repeated texture stored
      elsewhere in the legend.
    """
    out = grid.copy()
    mapping = {}
    c = 1
    while c < grid.shape[1] - 1:
        if grid[1, c] == GREY:
            c += 1
            continue
        cell = grid[1:4, c : c + 3]
        colors = [int(x) for x in np.unique(cell) if x != BG]
        if colors:
            color = colors[0]
            mapping[color] = (cell == color).astype(int)
        c += 4
    for r0, r1, c0, c1 in _fences(grid):
        if r0 < 5:
            continue
        seeds = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BG, GREY)]
        if not seeds:
            continue
        color = int(grid[seeds[0]])
        pat = mapping[color]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = color if pat[(r - r0 - 1) % pat.shape[0], (c - c0 - 1) % pat.shape[1]] else BG
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    x = 1
    pats = {}
    for color in colors:
        pat = np.random.randint(0, 2, (3, 3))
        if pat.sum() in (0, 9):
            pat[0, 1] = 1
            pat[1, 2] = 0
        pats[color] = pat
        cell = grid[1:4, x : x + 3]
        cell[pat == 1] = color
        x += 4
    for color in colors:
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        for _ in range(200):
            r = random.randint(8, 24 - h - 1)
            c = random.randint(1, 24 - w - 1)
            if np.all(grid[r : r + h, c : c + w] == 0):
                grid[r, c : c + w] = GREY
                grid[r + h - 1, c : c + w] = GREY
                grid[r : r + h, c] = GREY
                grid[r : r + h, c + w - 1] = GREY
                sr = random.randint(r + 1, r + h - 2)
                sc = random.randint(c + 1, c + w - 2)
                grid[sr, sc] = color
                break
    return grid
