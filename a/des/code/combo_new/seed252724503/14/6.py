import random

import numpy as np

BLACK = 0
BLUE = 1
GREY = 5
MAROON = 9


def _first_pos(coords):
    return min(coords, key=lambda p: (p[0], p[1]))


def _quadrant(p, center):
    r, c = p
    cr, cc = center
    return ("top" if r < cr else "bottom", "left" if c < cc else "right")


def _draw_segment(out: np.ndarray, a, b, color: int):
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        for c in range(ac, bc + step, step):
            if out[ar, c] != GREY:
                out[ar, c] = color
    else:
        step = 1 if br >= ar else -1
        for r in range(ar, br + step, step):
            if out[r, ac] != GREY:
                out[r, ac] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored markers with quadrant-dependent L paths and fill their bounding rectangles with maroon.

    Input structure:
    - A grey horizontal and vertical center line form a crosshair through the grid.
    - Every non-grey foreground color appears exactly twice as singleton markers.
    - Each pair lies in different crosshair-defined quadrants.

    Rule:
    - Pair markers by color and determine each marker's quadrant.
    - If the pair lies in diagonal quadrants, use the corner at (row of first, col of second); otherwise use the corner at (row of second, col of first).
    - Draw the two path legs in the pair color without overwriting grey crosshair pixels, then recolor remaining black pixels inside the pair's bounding rectangle to maroon.

    Key insight:
    - The crosshair is not decoration; it encodes whether the path must turn horizontal-first or vertical-first via the markers' quadrant relationship.
    """
    out = grid.copy()
    center = (int(np.argmax(np.sum(grid == GREY, axis=1))), int(np.argmax(np.sum(grid == GREY, axis=0))))
    colors = []
    for color in (int(v) for v in np.unique(grid) if v not in (BLACK, GREY)):
        coords = list(zip(*np.where(grid == color)))
        colors.append((_first_pos(coords), color, coords))
    colors.sort()
    for _pos, color, coords in colors:
        a, b = sorted(coords)
        qa = _quadrant(a, center)
        qb = _quadrant(b, center)
        if qa[0] != qb[0] and qa[1] != qb[1]:
            corner = (a[0], b[1])
        else:
            corner = (b[0], a[1])
        _draw_segment(out, a, corner, color)
        _draw_segment(out, corner, b, color)
        r0, r1 = sorted((a[0], b[0]))
        c0, c1 = sorted((a[1], b[1]))
        fill = out[r0:r1 + 1, c0:c1 + 1]
        fill[fill == BLACK] = MAROON
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [2, 3, 4, 6, 7, 8]
    for _ in range(400):
        h = rng.randint(15, 23)
        w = rng.randint(15, 23)
        cr, cc = h // 2, w // 2
        grid = np.zeros((h, w), dtype=int)
        grid[cr, :] = GREY
        grid[:, cc] = GREY
        occupied = {(cr, c) for c in range(w)} | {(r, cc) for r in range(h)}
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            style = rng.choice(["diag", "same_lr", "same_tb"])
            for _ in range(100):
                if style == "diag":
                    a = (rng.randint(1, cr - 1), rng.randint(1, cc - 1))
                    b = (rng.randint(cr + 1, h - 2), rng.randint(cc + 1, w - 2))
                elif style == "same_lr":
                    left = rng.choice([True, False])
                    col_range = (1, cc - 1) if left else (cc + 1, w - 2)
                    a = (rng.randint(1, cr - 1), rng.randint(*col_range))
                    b = (rng.randint(cr + 1, h - 2), rng.randint(*col_range))
                else:
                    top = rng.choice([True, False])
                    row_range = (1, cr - 1) if top else (cr + 1, h - 2)
                    a = (rng.randint(*row_range), rng.randint(1, cc - 1))
                    b = (rng.randint(*row_range), rng.randint(cc + 1, w - 2))
                if a in occupied or b in occupied or a == b:
                    continue
                grid[a] = color
                grid[b] = color
                occupied.update((a, b))
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 14/6")
