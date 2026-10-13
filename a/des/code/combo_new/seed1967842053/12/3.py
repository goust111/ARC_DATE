import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(grid, rows, cols):
    row_set = set(rows)
    col_set = set(cols)
    seen = set()
    comps = []
    for r in rows:
        for c in cols:
            if (r, c) in seen:
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if nx not in row_set or ny not in col_set:
                        continue
                    if (nx, ny) in seen:
                        continue
                    if int(grid[nx, ny]) == color:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comps.append({"color": color, "cells": cells})
    return comps


def _find_cross(grid):
    h, w = grid.shape
    grey_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    if grey_rows and grey_cols:
        return grey_rows[len(grey_rows) // 2], grey_cols[len(grey_cols) // 2]
    return h // 2, w // 2


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each colored object to the opposite quadrant while swapping to its same-quadrant neighbor color.

    Input structure:
    A black grid has one full grey center row and one full grey center column, splitting the odd-sized grid into four quadrants. Each quadrant contains exactly two separated 4-connected monochrome colored objects, and colored cells do not touch the grey cross.

    Rule:
    Keep the grey cross and all original objects unchanged. In each quadrant, identify its two colored objects. For each object, rotate its cell set 180 degrees around the grid center into the diagonal quadrant, but paint that copied shape with the color of the other object from the original quadrant. If a copied cell lands on an existing original colored cell or the grey cross, the original input cell remains unchanged.

    Key insight:
    The center cross defines four independent local frames: shapes move by diagonal 180-degree symmetry, while colors are supplied by the neighboring object in the source quadrant instead of traveling with the shape.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    h, w = arr.shape
    cr, cc = _find_cross(arr)
    quadrants = (
        (range(0, cr), range(0, cc)),
        (range(0, cr), range(cc + 1, w)),
        (range(cr + 1, h), range(0, cc)),
        (range(cr + 1, h), range(cc + 1, w)),
    )

    original_empty = arr == BLACK
    for rows, cols in quadrants:
        comps = _components(arr, rows, cols)
        if len(comps) != 2:
            continue
        first, second = comps
        for comp, paint in ((first, second["color"]), (second, first["color"])):
            for r, c in comp["cells"]:
                nr, nc = h - 1 - r, w - 1 - c
                if 0 <= nr < h and 0 <= nc < w and original_empty[nr, nc]:
                    out[nr, nc] = paint
    return out


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _random_shape():
    for _ in range(200):
        target = random.randint(2, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            cells.add((r + dr, c + dc))
        shape = list(_normalize(cells))
        for _ in range(random.randrange(4)):
            shape = [(c, -r) for r, c in shape]
            shape = list(_normalize(shape))
        if random.choice((False, True)):
            shape = [(r, -c) for r, c in shape]
            shape = list(_normalize(shape))
        shape = _normalize(shape)
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if height <= 4 and width <= 4:
            return shape
    return frozenset({(0, 0), (0, 1), (1, 0)})


def _allowed(qh, qw, quad):
    rows = range(qh)
    cols = range(qw)
    cells = set()
    for r in rows:
        for c in cols:
            if quad in ("TL", "TR") and r == qh - 1:
                continue
            if quad in ("BL", "BR") and r == 0:
                continue
            if quad in ("TL", "BL") and c == qw - 1:
                continue
            if quad in ("TR", "BR") and c == 0:
                continue
            cells.add((r, c))
    return cells


def _halo(cells, qh, qw):
    expanded = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < qh and 0 <= nc < qw:
                    expanded.add((nr, nc))
    return expanded


def _place_object(qh, qw, quad, occupied=(), avoid=()):
    allowed = _allowed(qh, qw, quad)
    occupied = set(occupied)
    avoid = set(avoid)
    blocked = occupied | avoid
    for _ in range(1500):
        shape = _random_shape()
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        if qh - max_r <= 0 or qw - max_c <= 0:
            continue
        off_r = random.randrange(0, qh - max_r)
        off_c = random.randrange(0, qw - max_c)
        cells = frozenset((r + off_r, c + off_c) for r, c in shape)
        if cells <= allowed and not (cells & blocked):
            return cells
    raise ValueError("could not place object")


def _mirror_local(cells, qh, qw):
    return frozenset((qh - 1 - r, qw - 1 - c) for r, c in cells)


def _place_two(qh, qw, quad, avoid=()):
    first = _place_object(qh, qw, quad, avoid=avoid)
    blocked = _halo(first, qh, qw)
    second = _place_object(qh, qw, quad, occupied=blocked, avoid=avoid)
    return [first, second]


def _make_pair(qh, qw, quad, opposite, overlap):
    source = _place_two(qh, qw, quad)
    mirrored = [_mirror_local(obj, qh, qw) for obj in source]
    target = [None, None]
    if overlap:
        idx = random.randrange(2)
        target[idx] = mirrored[idx]
        occupied = _halo(target[idx], qh, qw)
        avoid = mirrored[1 - idx]
        target[1 - idx] = _place_object(qh, qw, opposite, occupied=occupied, avoid=avoid)
    else:
        first = _place_object(qh, qw, opposite, avoid=set().union(*mirrored))
        blocked = _halo(first, qh, qw)
        second = _place_object(
            qh,
            qw,
            opposite,
            occupied=blocked,
            avoid=set().union(*mirrored),
        )
        target = [first, second]
    return source, target


def _to_global(cells, quad, qh, qw):
    cr = qh
    cc = qw
    converted = []
    for r, c in cells:
        if quad == "TL":
            converted.append((r, c))
        elif quad == "TR":
            converted.append((r, cc + 1 + c))
        elif quad == "BL":
            converted.append((cr + 1 + r, c))
        else:
            converted.append((cr + 1 + r, cc + 1 + c))
    return converted


def _paint(grid, quad, qh, qw, cells, color):
    for r, c in _to_global(cells, quad, qh, qw):
        grid[r, c] = color


def generate_input() -> np.ndarray:
    for _ in range(200):
        qh = random.randint(7, 14)
        qw = random.randint(7, 14)
        h, w = 2 * qh + 1, 2 * qw + 1
        grid = np.zeros((h, w), dtype=int)
        grid[qh, :] = GREY
        grid[:, qw] = GREY
        colors = random.sample(PALETTE, 4)
        random.shuffle(colors)
        diag_a = colors[:2]
        diag_b = colors[2:]
        try:
            tl, br = _make_pair(qh, qw, "TL", "BR", random.choice((False, True)))
            tr, bl = _make_pair(qh, qw, "TR", "BL", random.choice((False, True)))
        except ValueError:
            continue

        if random.choice((False, True)):
            tl = [tl[1], tl[0]]
        if random.choice((False, True)):
            br = [br[1], br[0]]
        if random.choice((False, True)):
            tr = [tr[1], tr[0]]
        if random.choice((False, True)):
            bl = [bl[1], bl[0]]

        for cells, color in zip(tl, diag_a):
            _paint(grid, "TL", qh, qw, cells, color)
        for cells, color in zip(br, diag_a):
            _paint(grid, "BR", qh, qw, cells, color)
        for cells, color in zip(tr, diag_b):
            _paint(grid, "TR", qh, qw, cells, color)
        for cells, color in zip(bl, diag_b):
            _paint(grid, "BL", qh, qw, cells, color)
        return grid
    raise RuntimeError("failed to generate valid input")
