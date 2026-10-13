import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8
MAROON = 9


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of(grid, color, rows=None):
    h, w = grid.shape
    r0, r1 = rows if rows is not None else (0, h)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(r0, r1):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (r0 <= nr < r1 and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(cells)
    return comps


def _hole_count_in_bbox(grid, cells, solid_color):
    top, left, bottom, right = _bbox(cells)
    sub = grid[top : bottom + 1, left : right + 1]
    solid = sub == solid_color
    hh, ww = sub.shape
    seen = np.zeros((hh, ww), dtype=bool)
    outside = np.zeros((hh, ww), dtype=bool)
    q = deque()
    for r in range(hh):
        for c in (0, ww - 1):
            if not solid[r, c] and not seen[r, c]:
                seen[r, c] = True
                outside[r, c] = True
                q.append((r, c))
    for c in range(ww):
        for r in (0, hh - 1):
            if not solid[r, c] and not seen[r, c]:
                seen[r, c] = True
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < hh and 0 <= nc < ww and not seen[nr, nc] and not solid[nr, nc]:
                seen[nr, nc] = True
                outside[nr, nc] = True
                q.append((nr, nc))
    holes = 0
    for r in range(hh):
        for c in range(ww):
            if solid[r, c] or outside[r, c] or seen[r, c]:
                continue
            holes += 1
            seen[r, c] = True
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hh and 0 <= nc < ww and not seen[nr, nc] and not solid[nr, nc] and not outside[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _legend(grid):
    grey = _components_of(grid, GREY)
    entries = []
    for cells in grey:
        top, left, bottom, right = _bbox(cells)
        if top > 3:
            continue
        if bottom + 1 >= grid.shape[0]:
            continue
        holes = _hole_count_in_bbox(grid, cells, GREY)
        center = (left + right) // 2
        sample = int(grid[bottom + 1, center])
        if sample not in (BLACK, BLUE, GREY, MAROON):
            entries.append((holes, sample, bottom + 1))
    if not entries:
        return {}, 0
    main_top = max(row for _, _, row in entries) + 2
    return {holes: color for holes, color, _ in entries}, main_top


def _direction_from_wall(grid, center):
    r, c = center
    h, w = grid.shape
    choices = []
    for rr in range(r - 1, -1, -1):
        if int(grid[rr, c]) == BLUE:
            choices.append((r - rr, -1, 0))
            break
    for rr in range(r + 1, h):
        if int(grid[rr, c]) == BLUE:
            choices.append((rr - r, 1, 0))
            break
    for cc in range(c - 1, -1, -1):
        if int(grid[r, cc]) == BLUE:
            choices.append((c - cc, 0, -1))
            break
    for cc in range(c + 1, w):
        if int(grid[r, cc]) == BLUE:
            choices.append((cc - c, 0, 1))
            break
    if not choices:
        return 0, 0
    _, dr, dc = min(choices)
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    hole_to_color, main_top = _legend(out)
    for cells in _components_of(out, MAROON, rows=(main_top, out.shape[0])):
        holes = _hole_count_in_bbox(out, cells, MAROON)
        color = hole_to_color.get(holes, MAROON)
        n = len(cells)
        top, left, bottom, right = _bbox(cells)
        center = ((top + bottom) // 2, (left + right) // 2)
        dr, dc = _direction_from_wall(out, center)
        for r, c in cells:
            out[r, c] = BLACK
        r, c = center
        for _ in range(n):
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                break
            if int(out[r, c]) == BLUE:
                break
            out[r, c] = color
            r += dr
            c += dc
    return out


def _draw_block_with_holes(grid, top, left, size, color, holes):
    grid[top : top + size, left : left + size] = color
    for rr, cc in holes:
        grid[top + rr, left + cc] = BLACK


def _free(grid, top, left, size):
    if top < 0 or left < 0 or top + size > grid.shape[0] or left + size > grid.shape[1]:
        return False
    return not np.any(grid[top : top + size, left : left + size] != BLACK)


def generate_input() -> np.ndarray:
    entry_count = random.randint(3, 5)
    square = 5
    gap = 2
    total_w = entry_count * square + (entry_count - 1) * gap
    h = random.randint(28, 34)
    w = random.randint(max(30, total_w + 2), max(36, total_w + 8))
    grid = np.zeros((h, w), dtype=int)
    start = random.randint(1, max(1, w - total_w - 1))
    colors = [RED, GREEN, YELLOW, AZURE, MAROON]
    random.shuffle(colors)
    hole_sites = [(1, 1), (1, 3), (3, 1), (3, 3), (2, 1)]
    for i in range(entry_count):
        left = start + i * (square + gap)
        holes = hole_sites[: i + 1]
        _draw_block_with_holes(grid, 0, left, square, GREY, holes)
        grid[square, left + square // 2] = colors[i]

    main_top = square + 2
    lanes = {
        "up": {
            "wall": (main_top, slice(4, 9)),
            "center": (random.randint(main_top + 10, h - 7), random.randint(5, 7)),
            "dir": (-1, 0),
        },
        "down": {
            "wall": (h - 1, slice(w - 9, w - 4)),
            "center": (random.randint(main_top + 4, h - 12), random.randint(w - 7, w - 5)),
            "dir": (1, 0),
        },
        "left": {
            "wall": (slice(main_top + 5, main_top + 10), 0),
            "center": (random.randint(main_top + 6, main_top + 8), random.randint(13, w - 8)),
            "dir": (0, -1),
        },
        "right": {
            "wall": (slice(h - 8, h - 3), w - 1),
            "center": (random.randint(h - 7, h - 5), random.randint(6, w - 14)),
            "dir": (0, 1),
        },
    }
    for spec in lanes.values():
        a, b = spec["wall"]
        grid[a, b] = BLUE

    directions = list(lanes)
    random.shuffle(directions)
    object_count = random.randint(3, 4)
    for idx, name in enumerate(directions[:object_count]):
        spec = lanes[name]
        cr, cc = spec["center"]
        size = random.choice([5, 6])
        top = cr - size // 2
        left = cc - size // 2
        holes_n = random.randint(1, entry_count)
        sites = [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2), (1, size // 2)]
        holes = sites[:holes_n]
        if _free(grid, top, left, size):
            _draw_block_with_holes(grid, top, left, size, MAROON, holes)
    return grid
