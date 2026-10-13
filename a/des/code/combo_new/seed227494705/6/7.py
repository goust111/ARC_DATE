import random
import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """Extract same-color bar crosses and stack them by bounding-box area.

    Input structure: Each foreground color appears as one horizontal rectangular
    strip and one vertical rectangular strip that overlap, forming a cross. No
    different-color crosses overlap.
    Rule: For each color, crop the complete cross to its minimal bounding box.
    Sort cropped crosses by bounding-box area from largest to smallest, then
    stack them top to bottom with one black separator row and left alignment.
    Key insight: Color identifies each horizontal/vertical pair, and the sorted
    output is based on the full cross crop area rather than intersection area.
    """
    crops = []
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        rs, cs = np.where(grid == col)
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        crop = np.where(grid[r0:r1 + 1, c0:c1 + 1] == col, col, BG)
        crops.append(((r1 - r0 + 1) * (c1 - c0 + 1), crop))
    crops.sort(key=lambda x: -x[0])
    if not crops:
        return np.zeros((1, 1), dtype=int)
    width = max(c.shape[1] for _, c in crops)
    height = sum(c.shape[0] for _, c in crops) + len(crops) - 1
    out = np.zeros((height, width), dtype=int)
    r = 0
    for _, crop in crops:
        out[r:r + crop.shape[0], :crop.shape[1]] = crop
        r += crop.shape[0] + 1
    return out


def generate_input() -> np.ndarray:
    h = w = 28
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    anchors = [(4, 4), (4, 16), (14, 5), (15, 17), (9, 10)]
    for col, (r, c) in zip(colors, anchors):
        hh = random.randint(1, 2)
        vw = random.randint(1, 2)
        hlen = random.randint(5, 8)
        vlen = random.randint(5, 8)
        grid[r:r + hh, c:c + hlen] = col
        grid[r - vlen // 2:r - vlen // 2 + vlen, c + hlen // 2:c + hlen // 2 + vw] = col
    return grid
