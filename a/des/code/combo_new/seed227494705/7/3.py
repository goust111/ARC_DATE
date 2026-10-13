import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


ARROWS = {
    (-1, 0): {(1, 1), (0, 1), (0, 0), (0, 2)},
    (1, 0): {(1, 1), (2, 1), (2, 0), (2, 2)},
    (0, -1): {(1, 1), (1, 0), (0, 0), (2, 0)},
    (0, 1): {(1, 1), (1, 2), (0, 2), (2, 2)},
}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _rect_border_cells(r0, c0, r1, c1):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _find_fences(g: np.ndarray):
    fences = []
    for pts in _components(g == GREY):
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if set(pts) == set(_rect_border_cells(r0, c0, r1, c1)):
            fences.append((r0, c0, r1, c1))
    return fences


def _arrow_direction(pts):
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in pts}
    for direction, shape in ARROWS.items():
        sr0 = min(r for r, _c in shape)
        sc0 = min(c for _r, c in shape)
        normalized_shape = {(r - sr0, c - sc0) for r, c in shape}
        if norm == normalized_shape:
            return direction
    if len(pts) == 2:
        (r1, c1), (r2, c2) = sorted(pts)
        return (int(np.sign(r2 - r1)), int(np.sign(c2 - c1)))
    raise AssertionError("unknown arrow shape")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored arrow sprite by the count of same-color dots in its fence.

    Input structure:
    - Black background with one or more grey rectangular fences.
    - Each fence contains one colored arrow sprite and several isolated dots of the same color.
    - The grey fence border is fixed and every sprite remains inside its fence after moving.

    Rule:
    - Read the sprite's arrow direction from its 3x3 colored arrow shape.
    - Count the isolated same-color dots in the same fence.
    - Clear the dots and old sprite, then translate the sprite by that count along the arrow direction.

    Key insight:
    - Shape encodes direction while the number of matching singleton dots encodes distance.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1 in _find_fences(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(x) for x in np.unique(inner) if int(x) not in (BG, GREY)]
        for color in colors:
            comps = []
            mask = g == color
            for pts in _components(mask):
                if all(r0 < r < r1 and c0 < c < c1 for r, c in pts):
                    comps.append(pts)
            sprites = [pts for pts in comps if len(pts) > 1]
            if len(sprites) != 1:
                continue
            sprite = sprites[0]
            dots = [pts[0] for pts in comps if len(pts) == 1]
            dr, dc = _arrow_direction(sprite)
            for r, c in sprite + dots:
                out[r, c] = BG
            dist = len(dots)
            for r, c in sprite:
                out[r + dr * dist, c + dc * dist] = color
    return out


def _can_place(cells, occupied, r0, c0, r1, c1):
    for r, c in cells:
        if not (r0 < r < r1 and c0 < c < c1):
            return False
        if (r, c) in occupied:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
        g = np.zeros((h, w), dtype=int)
        n = rng.randint(1, 3)
        frame_boxes = []
        occupied_global = set()
        ok = True
        for color in rng.sample(colors, n):
            placed = False
            for _try in range(100):
                fh = rng.randint(7, 10)
                fw = rng.randint(7, 10)
                r0 = rng.randint(1, h - fh - 1)
                c0 = rng.randint(1, w - fw - 1)
                r1, c1 = r0 + fh - 1, c0 + fw - 1
                padded = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)}
                if padded & occupied_global:
                    continue
                occupied_global |= padded
                for r, c in _rect_border_cells(r0, c0, r1, c1):
                    g[r, c] = GREY
                frame_boxes.append((color, r0, c0, r1, c1))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        for color, r0, c0, r1, c1 in frame_boxes:
            direction = rng.choice(list(ARROWS))
            shape = ARROWS[direction]
            dist = rng.randint(1, 3)
            placed = False
            for _try in range(100):
                sr = rng.randint(r0 + 1, r1 - 3)
                sc = rng.randint(c0 + 1, c1 - 3)
                cells = {(sr + r, sc + c) for r, c in shape}
                moved = {(r + direction[0] * dist, c + direction[1] * dist) for r, c in cells}
                if not _can_place(cells | moved, set(), r0, c0, r1, c1):
                    continue
                occupied = set(cells)
                for r, c in cells:
                    g[r, c] = color
                dot_count = dist
                for _dot in range(dot_count):
                    dot_ok = False
                    for _dtry in range(100):
                        dr = rng.randint(r0 + 1, r1 - 1)
                        dc = rng.randint(c0 + 1, c1 - 1)
                        near = {(dr + rr, dc + cc) for rr in (-1, 0, 1) for cc in (-1, 0, 1)}
                        if near & occupied:
                            continue
                        if (dr, dc) in moved:
                            continue
                        occupied.add((dr, dc))
                        g[dr, dc] = color
                        dot_ok = True
                        break
                    if not dot_ok:
                        break
                else:
                    placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
