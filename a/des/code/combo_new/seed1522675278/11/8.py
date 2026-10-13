import random
import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside stamp so its indicator side points up, then center-stamp it onto the matching canvas marker.

    Input structure:
    - A large grey canvas contains uniquely colored marker pixels.
    - Outside the canvas sit small monochrome stamp shapes, each with one adjacent colored indicator pixel.

    Rule:
    - Match each indicator color to the canvas marker with the same color.
    - Determine the indicator side relative to the stamp bbox.
    - Rotate the stamp so that side becomes the top side, then center it on the matching marker.
    - Clear the outside stamps and indicators; keep stamping only inside the canvas.

    Key insight:
    - Indicator color determines the destination, while indicator position determines the rotation.
    """
    out = np.where(grid == GREY, GREY, BG)
    grey = np.argwhere(grid == GREY)
    r0, c0 = grey.min(axis=0); r1, c1 = grey.max(axis=0)
    markers = {int(grid[r, c]): (r, c) for r, c in np.argwhere((grid != BG) & (grid != GREY)) if r0 < r < r1 and c0 < c < c1}
    used = set()
    for color, (mr, mc) in markers.items():
        ind = [(r, c) for r, c in np.argwhere(grid == color) if not (r0 < r < r1 and c0 < c < c1)]
        if not ind:
            continue
        ir, ic = ind[0]
        # nearest colored component not on canvas
        stamp = [(r, c) for r, c in np.argwhere((grid != BG) & (grid != GREY) & (grid != color)) if abs(r - ir) <= 4 and abs(c - ic) <= 4]
        if not stamp:
            continue
        rs = [p[0] for p in stamp]; cs = [p[1] for p in stamp]
        sr0, sr1, sc0, sc1 = min(rs), max(rs), min(cs), max(cs)
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
        for r, c in stamp:
            mask[r - sr0, c - sc0] = 1
        if ir < sr0:
            k = 0
        elif ic > sc1:
            k = 1
        elif ir > sr1:
            k = 2
        else:
            k = 3
        mask = np.rot90(mask, k=k)
        h, w = mask.shape
        rr = min(max(r0 + 1, mr - h // 2), r1 - h)
        cc = min(max(c0 + 1, mc - w // 2), c1 - w)
        out[rr : rr + h, cc : cc + w][mask == 1] = next(int(v) for v in np.unique(grid[sr0 : sr1 + 1, sc0 : sc1 + 1]) if v not in (BG, GREY))
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    grid[7:18, 7:18] = GREY
    marker_colors = random.sample([1, 2, 3, 4], random.randint(2, 4))
    stamp_colors = random.sample([6, 7, 8, 9], len(marker_colors))
    for mcolor in marker_colors:
        grid[random.randint(9, 15), random.randint(9, 15)] = mcolor
    for mcolor, scolor in zip(marker_colors, stamp_colors):
        mask = random.choice([
            np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
            np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
        ])
        r = random.randint(1, 4); c = random.randint(1, 4)
        grid[r : r + 3, c : c + 3][mask == 1] = scolor
        side = random.choice(["u", "d", "l", "r"])
        if side == "u":
            grid[r - 1, c + 1] = mcolor
        elif side == "d":
            grid[r + 3, c + 1] = mcolor
        elif side == "l":
            grid[r + 1, c - 1] = mcolor
        else:
            grid[r + 1, c + 3] = mcolor
    return grid
