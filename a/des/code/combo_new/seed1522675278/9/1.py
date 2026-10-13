import random
import numpy as np


BG = 0


def _rects(grid: np.ndarray):
    out = []
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        pts = np.argwhere(grid == color)
        if len(pts) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if np.all(grid[r0, c0 : c1 + 1] == color) and np.all(grid[r1, c0 : c1 + 1] == color) and np.all(grid[r0 : r1 + 1, c0] == color) and np.all(grid[r0 : r1 + 1, c1] == color):
            out.append((color, r0, r1, c0, c1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each legend template by the matched frame's interior size and place it inside the frame.

    Input structure:
    - Small bordered legend squares sit in the top band.
    - Larger hollow square frames below share border colors with those legend blocks.
    - Legend interiors contain black-and-color templates.

    Rule:
    - Match every large frame to the legend block with the same border color.
    - Compute the integer scale from legend interior size to frame interior size.
    - Enlarge the legend interior by nearest-neighbor block scaling and paste it into
      the matched frame interior.
    - Keep both legend band and frame borders unchanged.

    Key insight:
    - Border color provides the pairing, while the geometry of the destination frame
      determines the required scale factor for the hidden pattern.
    """
    out = grid.copy()
    legends = {}
    for color, r0, r1, c0, c1 in _rects(grid):
        if r0 <= 2:
            legends[color] = (grid[r0 + 1 : r1, c0 + 1 : c1] == color).astype(int)
    for color, r0, r1, c0, c1 in _rects(grid):
        if color not in legends or r0 <= 2:
            continue
        pat = legends[color]
        k = (r1 - r0 - 1) // pat.shape[0]
        fill = np.kron(pat, np.ones((k, k), dtype=int)) * color
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    x = 1
    legends = []
    for color in colors:
        size = random.choice([2, 3])
        pat = np.random.randint(0, 2, (size, size))
        if pat.sum() in (0, pat.size):
            pat[0, 0] = 1
            pat[-1, -1] = 0
        h = size + 2
        grid[1, x : x + h] = color
        grid[h, x : x + h] = color
        grid[1 : h + 1, x] = color
        grid[1 : h + 1, x + h - 1] = color
        block = grid[2:h, x + 1 : x + h - 1]
        block[pat == 1] = color
        legends.append((color, pat))
        x += h + 2
    for color, pat in legends:
        scale = random.randint(2, 3)
        size = pat.shape[0] * scale
        h = size + 2
        for _ in range(200):
            r = random.randint(8, grid.shape[0] - h - 1)
            c = random.randint(1, grid.shape[1] - h - 1)
            if np.all(grid[r : r + h, c : c + h] == 0):
                grid[r, c : c + h] = color
                grid[r + h - 1, c : c + h] = color
                grid[r : r + h, c] = color
                grid[r : r + h, c + h - 1] = color
                break
    return grid
