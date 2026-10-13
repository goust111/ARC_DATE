import random
import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild corner-defined rectangles and nest them at the top-left by area.

    Input structure:
    - The background is black.
    - Each non-background color appears exactly four times, at the four corners of
      an axis-aligned rectangle.
    - Different colors encode different rectangles with distinct areas.

    Rule:
    - For each color, recover the bounding rectangle from its four corner pixels and
      fill that whole rectangle solidly with the same color.
    - Sort the filled rectangles by area from largest to smallest.
    - Paint them all with their top-left corners aligned at (0, 0), so smaller
      rectangles overwrite larger ones.
    - Crop the output to the size of the largest rectangle.

    Key insight:
    - The visible pixels are only rectangle corners; the real objects are the solid
      rectangles they imply, and the final layout depends only on rectangle area.
    """
    rects = []
    for color in sorted(int(c) for c in np.unique(grid) if c != BG):
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rects.append((h * w, h, w, color))
    rects.sort(reverse=True)
    _, h0, w0, _ = rects[0]
    out = np.zeros((h0, w0), dtype=int)
    for _, h, w, color in rects:
        out[:h, :w] = color
    return out


def generate_input() -> np.ndarray:
    n = random.randint(3, 5)
    colors = random.sample(range(1, 10), n)
    dims = []
    used = set()
    while len(dims) < n:
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        area = h * w
        if area in used:
            continue
        used.add(area)
        dims.append((h, w))

    placed = []
    H = W = 24
    grid = np.zeros((H, W), dtype=int)
    for color, (h, w) in zip(colors, dims):
        for _ in range(200):
            r = random.randint(0, H - h - 1)
            c = random.randint(0, W - w - 1)
            if all(r + h + 1 <= rr or rr + hh + 1 <= r or c + w + 1 <= cc or cc + ww + 1 <= c for rr, cc, hh, ww in placed):
                placed.append((r, c, h, w))
                grid[r, c] = color
                grid[r + h - 1, c] = color
                grid[r, c + w - 1] = color
                grid[r + h - 1, c + w - 1] = color
                break
    return grid
