import random
import numpy as np


BG = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order colors into a one-row band by the reconstructed full area of their overlapped fragments.

    Input structure:
    - Distinct colored fragments overlap on a grey background, with top fragments hiding lower ones.
    - Each color's visible pixels still determine the bounding box of its full rectangle.

    Rule:
    - For each color, recover its full rectangle area from the bounding box of its visible pixels.
    - Sort colors by that reconstructed area from small to large, breaking ties by the
      smallest visible row index.
    - Output a one-row band containing those colors in order.

    Key insight:
    - The visible shape itself is incomplete, but the extremal coordinates still encode
      the hidden full extent needed for the ordering rule.
    """
    items = []
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        items.append((area, r0, color))
    items.sort()
    return np.array([[color for _, _, color in items]], dtype=int)


def generate_input() -> np.ndarray:
    grid = np.full((20, 20), BG, dtype=int)
    rects = []
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    for color in colors:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        for _ in range(200):
            r = random.randint(1, 20 - h - 1)
            c = random.randint(1, 20 - w - 1)
            if all(not (r < rr + hh and rr < r + h and c < cc + ww and cc < c + w) or (r <= rr and r + h >= rr + hh and c <= cc and c + w >= cc + ww) or (rr <= r and rr + hh >= r + h and cc <= c and cc + ww >= c + w) for rr, cc, hh, ww, _ in rects):
                rects.append((r, c, h, w, color))
                break
    rects.sort(key=lambda x: x[4])
    for r, c, h, w, color in rects:
        grid[r : r + h, c : c + w] = color
    return grid
