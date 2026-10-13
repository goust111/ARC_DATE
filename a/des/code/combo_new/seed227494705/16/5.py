from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _binary_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _component_mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _norm(mask):
    return frozenset(map(tuple, np.argwhere(mask)))


def _flip_by_side(mask, side):
    return np.flipud(mask) if side in ("up", "down") else np.fliplr(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each outside colored piece into the matching hole of the grey template.

    Input structure: A black grid contains one large grey filled rectangle with
    unique black holes, and colored connected pieces outside the rectangle above,
    below, left, and right.  Each piece corresponds to one hole by shape.
    Rule: For each outside piece, determine whether it lies vertically or
    horizontally from the template center.  Above or below pieces are flipped
    top-to-bottom; left or right pieces are flipped left-to-right.  The flipped
    mask is matched to a black hole and painted into that hole in the piece color;
    the original outside piece is cleared.
    Key insight: Shape matching identifies the destination, but the mirror axis is
    encoded by the piece's position around the central template.
    """
    out = grid.copy()
    grey_cells = list(map(tuple, np.argwhere(grid == GREY)))
    gr0, gc0, gr1, gc1 = _bbox(grey_cells)
    center_r, center_c = (gr0 + gr1) / 2, (gc0 + gc1) / 2
    hole_comps = []
    sub = grid[gr0 : gr1 + 1, gc0 : gc1 + 1] == BLACK
    for comp in _binary_components(sub):
        cells = [(gr0 + r, gc0 + c) for r, c in comp]
        r0, c0, r1, c1 = _bbox(cells)
        if gr0 < r0 and r1 < gr1 and gc0 < c0 and c1 < gc1:
            hole_comps.append({"cells": cells, "mask": _component_mask(cells)})

    piece_mask = (grid != BLACK) & (grid != GREY)
    for comp in _binary_components(piece_mask):
        cells = comp
        colors = [int(grid[r, c]) for r, c in cells]
        color = max(set(colors), key=colors.count)
        r0, c0, r1, c1 = _bbox(cells)
        pr, pc = (r0 + r1) / 2, (c0 + c1) / 2
        side = "up" if pr < gr0 else "down" if pr > gr1 else "left" if pc < gc0 else "right"
        transformed = _flip_by_side(_component_mask(cells), side)
        for hole in hole_comps:
            if _norm(hole["mask"]) == _norm(transformed):
                for rr, cc in cells:
                    out[rr, cc] = BLACK
                for rr, cc in hole["cells"]:
                    out[rr, cc] = color
                break
    return out


def _stamp(grid, mask, top, left, color):
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((28, 28), dtype=int)
    gr0, gc0, gh, gw = 8, 8, 12, 12
    grid[gr0 : gr0 + gh, gc0 : gc0 + gw] = GREY
    shapes = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    sides = random.sample(["up", "down", "left", "right"], random.randint(2, 4))
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], len(sides))
    hole_spots = [(9, 9), (9, 14), (14, 10), (14, 15)]
    for i, side in enumerate(sides):
        shape = shapes[i]
        hole = _flip_by_side(shape, side)
        hr, hc = hole_spots[i]
        grid[hr : hr + hole.shape[0], hc : hc + hole.shape[1]][hole] = BLACK
        if side == "up":
            top, left = 3, 8 + i * 4
        elif side == "down":
            top, left = 22, 8 + i * 4
        elif side == "left":
            top, left = 9 + i * 4, 2
        else:
            top, left = 9 + i * 4, 23
        _stamp(grid, shape, top, left, colors[i])
    return grid
