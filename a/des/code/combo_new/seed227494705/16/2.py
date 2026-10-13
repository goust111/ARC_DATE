from collections import deque
import random

import numpy as np

GREY = 5
DIRS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _most_common(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _same_color_components(grid, exclude=()):
    excluded = set(exclude)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _arrow_direction(cells):
    s = set(cells)
    for r, c in cells:
        for name, (dr, dc) in DIRS.items():
            if (r + dr, c + dc) in s and (r + 2 * dr, c + 2 * dc) in s:
                for pdr, pdc in ((-dc, dr), (dc, -dr)):
                    if (r + pdr, c + pdc) in s:
                        return (r + 2 * dr, c + 2 * dc), (dr, dc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Cast same-color rays from L-shaped arrow glyphs and mark ray collisions grey.

    Input structure: A black background contains separated one-color L arrow
    glyphs.  Each glyph has a corner/tail, a two-cell straight arm ending in a
    head, and a one-cell perpendicular foot; arrow colors are distinct.
    Rule: Decode every arrow direction from its corner toward the straight arm
    head.  From the head, simultaneously extend a ray of the arrow color through
    original background cells until the boundary or an original non-background
    cell.  Cells claimed by exactly one ray get that ray color; cells claimed by
    multiple rays become grey.  Original arrows remain unchanged.
    Key insight: The asymmetric L shape encodes direction before any ray is cast,
    and simultaneous ray claims produce a neutral collision color.
    """
    bg = _most_common(grid)
    out = grid.copy()
    claims = {}
    for comp in _same_color_components(grid, exclude=[bg, GREY]):
        decoded = _arrow_direction(comp["cells"])
        if decoded is None:
            continue
        (hr, hc), (dr, dc) = decoded
        r, c = hr + dr, hc + dc
        while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == bg:
            claims.setdefault((r, c), set()).add(comp["color"])
            r += dr
            c += dc

    for (r, c), cols in claims.items():
        out[r, c] = GREY if len(cols) > 1 else next(iter(cols))
    return out


def _place_arrow(grid, r, c, direction, color):
    dr, dc = DIRS[direction]
    pdr, pdc = random.choice([(-dc, dr), (dc, -dr)])
    pts = [(r, c), (r + dr, c + dc), (r + 2 * dr, c + 2 * dc), (r + pdr, c + pdc)]
    for rr, cc in pts:
        grid[rr, cc] = color
    return pts


def generate_input() -> np.ndarray:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 5))
    for color in colors:
        for _ in range(500):
            direction = random.choice(list(DIRS))
            dr, dc = DIRS[direction]
            r = random.randint(2, h - 3)
            c = random.randint(2, w - 3)
            pdr, pdc = random.choice([(-dc, dr), (dc, -dr)])
            pts = [(r, c), (r + dr, c + dc), (r + 2 * dr, c + 2 * dc), (r + pdr, c + pdc)]
            halo = [(rr + ar, cc + ac) for rr, cc in pts for ar in (-1, 0, 1) for ac in (-1, 0, 1)]
            if all(_in_bounds(grid.shape, rr, cc) and grid[rr, cc] == 0 for rr, cc in halo):
                for rr, cc in pts:
                    grid[rr, cc] = color
                break
    return grid
