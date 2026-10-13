import random
from collections import deque

import numpy as np


BACKGROUND = 0
GRAY = 5


def _corner_colors(grid):
    h, w = grid.shape
    return {
        "tl": int(grid[0, 0]),
        "tr": int(grid[0, w - 1]),
        "bl": int(grid[h - 1, 0]),
        "br": int(grid[h - 1, w - 1]),
    }


def _corner_for_gap(r, c, side, h, w):
    if side == "top":
        return "tl" if abs(c - 0.5) <= abs(c - (w - 1.5)) else "tr"
    if side == "bottom":
        return "bl" if abs(c - 0.5) <= abs(c - (w - 1.5)) else "br"
    if side == "left":
        return "tl" if abs(r - 0.5) <= abs(r - (h - 1.5)) else "bl"
    return "tr" if abs(r - 0.5) <= abs(r - (h - 1.5)) else "br"


def _gray_components(grid):
    mask = grid == GRAY
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    components = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _room_from_component(grid, comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    for r, c in comp:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    missing = []
    for c in range(c0, c1 + 1):
        if grid[r0, c] != GRAY:
            missing.append((r0, c, "top"))
        if grid[r1, c] != GRAY:
            missing.append((r1, c, "bottom"))
    for r in range(r0 + 1, r1):
        if grid[r, c0] != GRAY:
            missing.append((r, c0, "left"))
        if grid[r, c1] != GRAY:
            missing.append((r, c1, "right"))

    if len(missing) != 1:
        return None
    gr, gc, side = missing[0]
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    return r0, r1, c0, c1, gr, gc, side


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each room's interior object from the corner legend selected by its border gap.

    Input structure: The grid has a black background, four 2x2 colored legend blocks
    in the corners, and one or more gray rectangular room borders in the middle. Each
    room has exactly one missing gray border pixel on one side and contains a single
    non-gray foreground object inside.
    Rule: For every gray room, detect the missing border pixel. A top or bottom gap
    chooses the nearer legend on that horizontal edge; a left or right gap chooses the
    nearer legend on that vertical edge. Replace all non-black, non-gray pixels inside
    the room with the chosen legend color while leaving borders, legends, and background
    unchanged.
    Key insight: The gap is a pointer: its side and position identify the nearest
    corner color swatch that supplies the repaint color for the enclosed object.
    """
    out = np.array(grid, copy=True)
    h, w = out.shape
    colors = _corner_colors(out)

    for comp in _gray_components(out):
        room = _room_from_component(out, comp)
        if room is None:
            continue
        r0, r1, c0, c1, gr, gc, side = room
        corner = _corner_for_gap(gr, gc, side, h, w)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        obj = (interior != BACKGROUND) & (interior != GRAY)
        interior[obj] = colors[corner]
    return out


def _gap_candidates(r0, r1, c0, c1, side, wanted_corner, h, w):
    cells = []
    if side in ("top", "bottom"):
        r = r0 if side == "top" else r1
        for c in range(c0 + 1, c1):
            if _corner_for_gap(r, c, side, h, w) == wanted_corner:
                cells.append((r, c))
    else:
        c = c0 if side == "left" else c1
        for r in range(r0 + 1, r1):
            if _corner_for_gap(r, c, side, h, w) == wanted_corner:
                cells.append((r, c))
    return cells


def _random_object_cells(inner_h, inner_w):
    kind = random.randrange(5)
    cells = set()

    if kind == 0:
        length = random.randint(2, min(inner_w, 5))
        r = random.randrange(inner_h)
        c0 = random.randint(0, inner_w - length)
        cells = {(r, c0 + i) for i in range(length)}
    elif kind == 1:
        length = random.randint(2, min(inner_h, 5))
        r0 = random.randint(0, inner_h - length)
        c = random.randrange(inner_w)
        cells = {(r0 + i, c) for i in range(length)}
    elif kind == 2:
        rh = random.randint(1, min(inner_h, 3))
        rw = random.randint(1, min(inner_w, 3))
        if rh == 1 and rw == 1:
            rw = 2 if inner_w >= 2 else 1
            rh = 2 if rw == 1 and inner_h >= 2 else rh
        r0 = random.randint(0, inner_h - rh)
        c0 = random.randint(0, inner_w - rw)
        cells = {(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)}
    elif kind == 3 and inner_h >= 2 and inner_w >= 2:
        vlen = random.randint(2, min(inner_h, 4))
        hlen = random.randint(2, min(inner_w, 4))
        r0 = random.randint(0, inner_h - vlen)
        c0 = random.randint(0, inner_w - hlen)
        pivot_r = r0 + random.randrange(vlen)
        pivot_c = c0
        cells = {(r0 + i, pivot_c) for i in range(vlen)}
        cells.update({(pivot_r, c0 + i) for i in range(hlen)})
    elif kind == 4 and inner_h >= 3 and inner_w >= 3:
        r = random.randint(1, inner_h - 2)
        c = random.randint(1, inner_w - 2)
        cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
    else:
        target = random.randint(2, min(7, inner_h * inner_w))
        cells = {(random.randrange(inner_h), random.randrange(inner_w))}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            nbrs = [
                (nr, nc)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < inner_h and 0 <= nc < inner_w
            ]
            cells.add(random.choice(nbrs))
    return cells


def _draw_room(grid, r0, c0, rh, rw, side, gap, color):
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[gap] = BACKGROUND

    inner_h = rh - 2
    inner_w = rw - 2
    for dr, dc in _random_object_cells(inner_h, inner_w):
        grid[r0 + 1 + dr, c0 + 1 + dc] = color


def generate_input() -> np.ndarray:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    grid = np.zeros((h, w), dtype=np.int8)

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(palette)
    tl, tr, bl, br = palette[:4]
    object_colors = palette[4:] or [1, 2, 3, 4]

    grid[0:2, 0:2] = tl
    grid[0:2, w - 2 : w] = tr
    grid[h - 2 : h, 0:2] = bl
    grid[h - 2 : h, w - 2 : w] = br

    occupied = np.zeros((h, w), dtype=bool)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    side_corners = {
        "top": ("tl", "tr"),
        "bottom": ("bl", "br"),
        "left": ("tl", "bl"),
        "right": ("tr", "br"),
    }
    room_count = random.randint(3, 5)
    placed = 0

    for i in range(room_count):
        side = sides[i % 4]
        wanted_corner = random.choice(side_corners[side])
        for _ in range(400):
            rh = random.randint(5, 8)
            rw = random.randint(5, 8)
            r0 = random.randint(3, h - rh - 3)
            c0 = random.randint(3, w - rw - 3)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            rr0, rr1 = max(0, r0 - 1), min(h, r1 + 2)
            cc0, cc1 = max(0, c0 - 1), min(w, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates = _gap_candidates(r0, r1, c0, c1, side, wanted_corner, h, w)
            if not candidates:
                continue
            gap = random.choice(candidates)
            color = random.choice(object_colors)
            _draw_room(grid, r0, c0, rh, rw, side, gap, color)
            occupied[rr0:rr1, cc0:cc1] = True
            placed += 1
            break

    if placed == 0:
        rh = rw = 6
        r0 = max(3, h // 2 - rh // 2)
        c0 = max(3, w // 2 - rw // 2)
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1
        side = random.choice(sides)
        candidates = _gap_candidates(
            r0, r1, c0, c1, side, random.choice(side_corners[side]), h, w
        )
        gap = random.choice(candidates) if candidates else (r0, c0 + 2)
        _draw_room(grid, r0, c0, rh, rw, side, gap, random.choice(object_colors))

    return grid
