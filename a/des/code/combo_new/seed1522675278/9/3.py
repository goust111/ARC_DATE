import random
import numpy as np


BG = 0
GREY = 5


def _legend(grid: np.ndarray):
    rows = [0, 4]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[0:5, c] == GREY)]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a color-specific legend template inside each closed region, anchored at the seed pixel.

    Input structure:
    - A grey-framed legend table maps colors to small binary templates.
    - Separate grey rectangular fences each contain one colored seed pixel on black background.

    Rule:
    - Read each legend cell as a color-to-template mapping.
    - For every fenced region, find its seed color and tile that template across the
      region interior, using the seed location as the periodic origin.
    - Only black interior cells are filled; the fence and legend stay unchanged.

    Key insight:
    - Color selects which template to use, but the seed position separately defines
      the phase of the tiling inside the region.
    """
    out = grid.copy()
    rows, cols = _legend(grid)
    mapping = {}
    for i in range(len(cols) - 1):
        cell = grid[rows[0] + 1 : rows[1], cols[i] + 1 : cols[i + 1]]
        colors = [int(c) for c in np.unique(cell) if c != BG]
        if not colors:
            continue
        color = colors[0]
        mapping[color] = (cell == color).astype(int)
    for color in [int(c) for c in np.unique(grid) if c not in (BG, GREY) and c not in mapping]:
        pass
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        pts = np.argwhere(grid == GREY)
    for comp in _rectangles(grid):
        r0, r1, c0, c1 = comp
        seed = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BG, GREY)]
        if not seed:
            continue
        sr, sc = seed[0]
        color = int(grid[sr, sc])
        pat = mapping[color]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if grid[r, c] == BG and pat[(r - sr) % pat.shape[0], (c - sc) % pat.shape[1]] == 1:
                    out[r, c] = color
    return out


def _rectangles(grid: np.ndarray):
    rects = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if grid[r, c] != GREY:
                continue
            for h in range(4, 9):
                for w in range(4, 9):
                    if r + h > grid.shape[0] or c + w > grid.shape[1]:
                        continue
                    if np.all(grid[r, c : c + w] == GREY) and np.all(grid[r + h - 1, c : c + w] == GREY) and np.all(grid[r : r + h, c] == GREY) and np.all(grid[r : r + h, c + w - 1] == GREY):
                        rects.append((r, r + h - 1, c, c + w - 1))
    uniq = []
    seen = set()
    for x in rects:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return [x for x in uniq if x[0] > 4]


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    pats = []
    x = 1
    top_h = 5
    grid[0:top_h, 0] = GREY
    grid[0:top_h, 23] = BG
    rows = [0, top_h - 1]
    for color in colors:
        pat = np.random.randint(0, 2, (3, 3))
        if pat.sum() in (0, 9):
            pat[0, 0] = 1
            pat[2, 2] = 0
        pats.append((color, pat))
        grid[0:top_h, x] = GREY
        cell = grid[1:4, x + 1 : x + 4]
        cell[pat == 1] = color
        x += 4
    grid[0:top_h, x] = GREY
    grid[0, : x + 1] = GREY
    grid[top_h - 1, : x + 1] = GREY
    for color, pat in pats:
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
