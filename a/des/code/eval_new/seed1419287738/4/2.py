import random
from collections import deque

import numpy as np


BG = 0
MAGENTA = 6
ORANGE = 7
SHAPE_COLORS = [1, 2, 3, 4, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(grid, colors=None):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == BG:
                continue
            if colors is not None and color not in colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _detect_arrows(grid):
    h, w = grid.shape
    arrows = []
    seen = set()
    for color in (ORANGE, MAGENTA):
        for comp in _components(grid, {color}):
            cells = set(comp["cells"])
            if len(cells) != 3 or frozenset(cells) in seen:
                continue
            for r, c in cells:
                border_dirs = []
                if r == 0:
                    border_dirs.append((1, 0))
                if r == h - 1:
                    border_dirs.append((-1, 0))
                if c == 0:
                    border_dirs.append((0, 1))
                if c == w - 1:
                    border_dirs.append((0, -1))
                for dr, dc in border_dirs:
                    inward = (r + dr, c + dc)
                    if inward not in cells:
                        continue
                    if dr:
                        along = [cell for cell in ((r, c - 1), (r, c + 1)) if cell in cells]
                    else:
                        along = [cell for cell in ((r - 1, c), (r + 1, c)) if cell in cells]
                    if len(along) == 1:
                        arrows.append({
                            "corner": (r, c),
                            "cells": cells,
                            "dir": (dr, dc),
                            "color": color,
                        })
                        seen.add(frozenset(cells))
                        break
                if frozenset(cells) in seen:
                    break
    arrows.sort(key=lambda a: (a["corner"][0], a["corner"][1]))
    return arrows


def _component_at(grid, start, blocked_colors):
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    if color == BG or color in blocked_colors:
        return color, []
    q = deque([start])
    seen = {start}
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return color, cells


def _find_target(grid, arrow):
    h, w = grid.shape
    r, c = arrow["corner"]
    dr, dc = arrow["dir"]
    r += dr
    c += dc
    blocked = {ORANGE, MAGENTA}
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) not in blocked and int(grid[r, c]) != BG:
            return _component_at(grid, (r, c), blocked)
        r += dr
        c += dc
    return None, []


def _transform_cells(cells, mode, direction):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    if mode == ORANGE:
        dr, dc = direction
        new_mask = np.fliplr(mask) if dc else np.flipud(mask)
    else:
        dr, dc = direction
        clockwise = (dr, dc) in ((1, 0), (0, 1))
        new_mask = np.rot90(mask, -1 if clockwise else 1)

    center_r2 = r0 + r1
    center_c2 = c0 + c1
    nh, nw = new_mask.shape
    nr0 = (center_r2 - (nh - 1)) // 2
    nc0 = (center_c2 - (nw - 1)) // 2
    out = []
    for rr, cc in zip(*np.where(new_mask)):
        out.append((int(nr0 + rr), int(nc0 + cc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply border-arrow transformations to the nearest target shape.

    Input structure:
    A black grid contains compact monochrome shapes and 3-pixel L-shaped border arrows.
    Orange and magenta arrows sit on the outer edge; the arrow leg pointing inward gives
    a row or column ray toward the first target shape.

    Rule:
    Arrows are processed from top to bottom, then left to right.  The first non-arrow
    shape seen along an arrow ray is transformed in place: orange mirrors it across the
    arrow axis, while magenta rotates it 90 degrees clockwise for top/left entry arrows
    and counterclockwise for bottom/right entry arrows.  Arrow pixels are erased.

    Key insight:
    The border position chooses the target, the arrow color chooses flip versus rotation,
    and the inward direction supplies the flip axis or rotation orientation.
    """
    out = np.array(grid, copy=True)
    arrows = _detect_arrows(out)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG
    for arrow in arrows:
        color, cells = _find_target(out, arrow)
        if not cells:
            continue
        new_cells = _transform_cells(cells, arrow["color"], arrow["dir"])
        for r, c in cells:
            out[r, c] = BG
        h, w = out.shape
        for r, c in new_cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
    return out


BASE_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=int),
]


def _place_shape(grid, mask, top, left, color):
    h, w = mask.shape
    if top < 1 or left < 1 or top + h >= grid.shape[0] - 1 or left + w >= grid.shape[1] - 1:
        return False
    pts = [(top + r, left + c) for r, c in zip(*np.where(mask))]
    margin = set()
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    margin.add((rr, cc))
    if any(grid[r, c] != BG for r, c in margin):
        return False
    for r, c in pts:
        grid[r, c] = color
    return True


def _draw_arrow(grid, side, pos, color):
    h, w = grid.shape
    if side == "top":
        cells = [(0, pos), (0, pos + 1), (1, pos)]
    elif side == "bottom":
        cells = [(h - 1, pos), (h - 1, pos + 1), (h - 2, pos)]
    elif side == "left":
        cells = [(pos, 0), (pos + 1, 0), (pos, 1)]
    else:
        cells = [(pos, w - 1), (pos + 1, w - 1), (pos, w - 2)]
    if any(grid[r, c] != BG for r, c in cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _gen_attempt():
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(SHAPE_COLORS, 4)
    specs = [
        ("top", random.randint(3, w - 5), random.randint(5, h - 8), ORANGE),
        ("bottom", random.randint(3, w - 5), random.randint(5, h - 8), MAGENTA),
        ("left", random.randint(3, h - 5), random.randint(5, w - 8), ORANGE),
        ("right", random.randint(3, h - 5), random.randint(5, w - 8), MAGENTA),
    ]
    random.shuffle(specs)
    count = random.randint(2, 4)
    used_sides = set()
    placed = 0
    for i, (side, edge_pos, depth, arrow_color) in enumerate(specs[:count]):
        mask = random.choice(BASE_SHAPES)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.rot90(mask, random.randint(0, 3))
        mh, mw = mask.shape
        if side == "top":
            left = min(max(edge_pos - 1, 2), w - mw - 3)
            top = depth
            arrow_pos = left + 1
        elif side == "bottom":
            left = min(max(edge_pos - 1, 2), w - mw - 3)
            top = h - depth - mh
            arrow_pos = left + 1
        elif side == "left":
            top = min(max(edge_pos - 1, 2), h - mh - 3)
            left = depth
            arrow_pos = top + 1
        else:
            top = min(max(edge_pos - 1, 2), h - mh - 3)
            left = w - depth - mw
            arrow_pos = top + 1
        if side in used_sides:
            continue
        if not _place_shape(grid, mask, top, left, colors[i]):
            continue
        if not _draw_arrow(grid, side, arrow_pos, arrow_color):
            return None
        used_sides.add(side)
        placed += 1
    if placed < 2:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-transform puzzle")
