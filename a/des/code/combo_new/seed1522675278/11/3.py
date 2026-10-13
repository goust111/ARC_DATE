import random
import numpy as np


BG = 0


def factors(area: int):
    out = []
    for w in range(1, int(area**0.5) + 1):
        if area % w == 0:
            out.append((w, area // w))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered pixels into a solid rectangle and stack by area.

    Input structure:
    - The background is black.
    - Each non-background color forms scattered fragments whose total pixel count equals
      the area of one solid rectangle.

    Rule:
    - Group pixels by color and count their total area A.
    - Recover the unique rectangle dimensions (w, h) with w <= h and w*h = A.
    - Sort the rectangles by area from small to large.
    - Stack them top-to-bottom, left-aligned, on a black canvas whose width is the
      largest rectangle width and whose height is the sum of rectangle heights.

    Key insight:
    - Fragment geometry is disposable; only the per-color pixel count matters for the
      final rectangle, and only rectangle area matters for the output order.
    """
    rects = []
    for color in sorted(int(c) for c in np.unique(grid) if c != BG):
        area = int(np.sum(grid == color))
        w, h = factors(area)[0]
        rects.append((area, h, w, color))
    rects.sort()
    out = np.zeros((sum(h for _, h, _, _ in rects), max(w for _, _, w, _ in rects)), dtype=int)
    r = 0
    for _, h, w, color in rects:
        out[r : r + h, :w] = color
        r += h
    return out


def carve_fragment(area: int) -> np.ndarray:
    mask = np.zeros((area + 2, area + 2), dtype=int)
    r = c = 1
    mask[r, c] = 1
    cells = {(r, c)}
    while len(cells) < area:
        rr, cc = random.choice(list(cells))
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = rr + dr, cc + dc
        if mask[nr, nc] == 0:
            mask[nr, nc] = 1
            cells.add((nr, nc))
    pts = np.argwhere(mask == 1)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def generate_input() -> np.ndarray:
    area_pool = [2, 3, 5, 7, 11]
    while True:
        areas = random.sample(area_pool, random.randint(3, 4))
        if sum(areas) <= 28:
            break
    colors = random.sample(range(1, 10), len(areas))
    grid = np.zeros((24, 24), dtype=int)
    placed = []
    for color, area in zip(colors, areas):
        frag = carve_fragment(area)
        h, w = frag.shape
        for _ in range(200):
            r = random.randint(0, 24 - h)
            c = random.randint(0, 24 - w)
            if all(r + h + 1 <= rr or rr + hh + 1 <= r or c + w + 1 <= cc or cc + ww + 1 <= c for rr, cc, hh, ww in placed):
                placed.append((r, c, h, w))
                block = grid[r : r + h, c : c + w]
                block[frag == 1] = color
                break
    return grid
