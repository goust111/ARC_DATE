import random

import numpy as np


def _anchor(grid):
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r:r + 2, c:c + 2] == 5):
                return r, c
    raise ValueError("missing 2x2 gray anchor")


def _density(grid, mask):
    area = int(mask.sum())
    if area == 0:
        return 0.0
    return float(np.sum(mask & (grid != 0) & (grid != 5))) / area


def _axis_side(grid, ar, ac):
    h, w = grid.shape
    top = np.zeros((h, w), dtype=bool)
    top[:ar, :] = True
    bottom = np.zeros((h, w), dtype=bool)
    bottom[ar + 2:, :] = True
    left = np.zeros((h, w), dtype=bool)
    left[:, :ac] = True
    right = np.zeros((h, w), dtype=bool)
    right[:, ac + 2:] = True
    td, bd = _density(grid, top), _density(grid, bottom)
    ld, rd = _density(grid, left), _density(grid, right)
    if abs(td - bd) >= abs(ld - rd):
        return "top" if td < bd else "bottom"
    return "left" if ld < rd else "right"


def _masks(shape, ar, ac, side):
    h, w = shape
    main = np.zeros((h, w), dtype=bool)
    mirror = np.zeros((h, w), dtype=bool)
    if side == "top":
        main[:ar, :] = True
        mirror[ar + 2:, :] = True
        axis = "h"
    elif side == "bottom":
        main[ar + 2:, :] = True
        mirror[:ar, :] = True
        axis = "h"
    elif side == "left":
        main[:, :ac] = True
        mirror[:, ac + 2:] = True
        axis = "v"
    else:
        main[:, ac + 2:] = True
        mirror[:, :ac] = True
        axis = "v"
    return main, mirror, axis


def _best_vector(grid, mask):
    cells = {(r, c): int(grid[r, c]) for r, c in zip(*np.where(mask & (grid != 0) & (grid != 5)))}
    best = None
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if dr == 0 and dc == 0:
                continue
            support = 0
            conflict = False
            for (r, c), color in cells.items():
                q = (r + dr, c + dc)
                if q in cells:
                    if cells[q] != color:
                        conflict = True
                        break
                    support += 1
            if not conflict and support > 0:
                key = (support, -(abs(dr) + abs(dc)), -abs(dr), -abs(dc))
                if best is None or key > best[0]:
                    best = (key, (dr, dc))
    return (1, 0) if best is None else best[1]


def _fill_side(out, mask, vec):
    h, w = out.shape
    dr, dc = vec
    visible = {(r, c): int(out[r, c]) for r, c in zip(*np.where(mask & (out != 0) & (out != 5)))}
    for r, c in zip(*np.where(mask & (out == 0))):
        found = None
        for sign in (1, -1):
            x, y = r, c
            while True:
                x += sign * dr
                y += sign * dc
                if not (0 <= x < h and 0 <= y < w) or not mask[x, y]:
                    break
                if (x, y) in visible:
                    found = visible[(x, y)]
                    break
            if found is not None:
                break
        if found is not None:
            out[r, c] = found


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    ar, ac = _anchor(out)
    side = _axis_side(out, ar, ac)
    main_mask, mirror_mask, axis = _masks(out.shape, ar, ac, side)
    vec = _best_vector(out, main_mask)
    _fill_side(out, main_mask, vec)
    mvec = (-vec[0], vec[1]) if axis == "h" else (vec[0], -vec[1])
    _fill_side(out, mirror_mask, mvec)
    out[ar:ar + 2, ac:ac + 2] = 5
    return out


def _color_for(r, c, vec, palette):
    dr, dc = vec
    return palette[(dc * r - dr * c) % len(palette)]


def generate_input() -> np.ndarray:
    h = random.choice([15, 16, 17, 18])
    w = random.choice([15, 16, 17, 18])
    g = np.zeros((h, w), dtype=int)
    ar = random.randint(5, h - 7)
    ac = random.randint(5, w - 7)
    side = random.choice(["top", "bottom", "left", "right"])
    vec = random.choice([(2, 1), (1, 2), (3, 1), (1, 3), (2, -1), (-1, 2)])
    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.choice([2, 3]))
    main_mask, mirror_mask, axis = _masks((h, w), ar, ac, side)
    mvec = (-vec[0], vec[1]) if axis == "h" else (vec[0], -vec[1])
    for r in range(h):
        for c in range(w):
            if main_mask[r, c]:
                g[r, c] = _color_for(r, c, vec, palette)
            elif mirror_mask[r, c]:
                g[r, c] = _color_for(r, c, mvec, palette)
    g[ar:ar + 2, ac:ac + 2] = 5

    # Make the side whose anchor edge is the axis visibly sparser, so the axis side is unambiguous.
    sparse = main_mask
    dense = mirror_mask
    miss_sparse = random.randint(8, 14)
    miss_dense = random.randint(2, 5)
    for mask, count in [(sparse, miss_sparse), (dense, miss_dense)]:
        cells = [(r, c) for r, c in zip(*np.where(mask)) if g[r, c] != 5]
        random.shuffle(cells)
        for r, c in cells[:count]:
            g[r, c] = 0
    return g
