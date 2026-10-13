import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
PORTAL_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col in (BG, GREEN) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": cells})
    return comps


def _portals(grid):
    portals = []
    cell_to_portal = {}
    for comp in _components(grid):
        cells = comp["cells"]
        if len(cells) != 2:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if max(rs) - min(rs) + max(cs) - min(cs) != 1:
            continue
        idx = len(portals)
        portal = {
            "color": comp["color"],
            "cells": set(cells),
            "bbox": (min(rs), min(cs), max(rs), max(cs)),
        }
        portals.append(portal)
        for cell in cells:
            cell_to_portal[cell] = idx
    by_color = {}
    for i, p in enumerate(portals):
        by_color.setdefault(p["color"], []).append(i)
    return portals, cell_to_portal, {c: v for c, v in by_color.items() if len(v) == 2}


def _green_segments(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    segs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREEN:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREEN:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            segs.append(cells)
    return segs


def _paired_portal(portal_idx, portals, pairs):
    color = portals[portal_idx]["color"]
    pair = pairs.get(color)
    if not pair:
        return None
    return pair[1] if pair[0] == portal_idx else pair[0]


def _shoot(out, portals, cell_to_portal, pairs, portal_idx, contact_cell, dr, dc, visited=None):
    if visited is None:
        visited = set()
    key = (portal_idx, contact_cell, dr, dc)
    if key in visited:
        return
    visited.add(key)
    target_idx = _paired_portal(portal_idx, portals, pairs)
    if target_idx is None:
        return
    in_portal = portals[portal_idx]
    out_portal = portals[target_idx]
    r0, c0, _, _ = in_portal["bbox"]
    off = (contact_cell[0] - r0, contact_cell[1] - c0)
    tr0, tc0, tr1, tc1 = out_portal["bbox"]
    exit_cell = (min(tr1, tr0 + off[0]), min(tc1, tc0 + off[1]))
    r, c = exit_cell[0] + dr, exit_cell[1] + dc
    color = out_portal["color"]
    seen = set()
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        if (r, c, dr, dc, color) in seen:
            return
        seen.add((r, c, dr, dc, color))
        if (r, c) in cell_to_portal:
            next_idx = cell_to_portal[(r, c)]
            _shoot(out, portals, cell_to_portal, pairs, next_idx, (r, c), dr, dc, visited)
            return
        if int(out[r, c]) == BG:
            out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    portals, cell_to_portal, pairs = _portals(grid)
    for seg in _green_segments(grid):
        endpoints = []
        s = set(seg)
        for r, c in seg:
            deg = sum((r + dr, c + dc) in s for dr, dc in DIRS)
            if deg <= 1:
                endpoints.append((r, c))
        for r, c in endpoints:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_to_portal:
                    _shoot(out, portals, cell_to_portal, pairs, cell_to_portal[(nr, nc)], (nr, nc), dr, dc)
    return out


def _place_portal(grid, color, orient):
    h, w = grid.shape
    for _ in range(200):
        r = random.randint(2, h - 3)
        c = random.randint(2, w - 3)
        cells = [(r, c), (r, c + 1)] if orient == "h" else [(r, c), (r + 1, c)]
        if all(int(grid[rr, cc]) == BG for rr, cc in cells):
            for rr, cc in cells:
                grid[rr, cc] = color
            return cells
    return None


def _clear_line(grid, start, dr, dc, length):
    cells = [(start[0] + i * dr, start[1] + i * dc) for i in range(length)]
    h, w = grid.shape
    return all(1 <= r < h - 1 and 1 <= c < w - 1 and int(grid[r, c]) == BG for r, c in cells)


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.zeros((h, w), dtype=int)
        color = random.choice(PORTAL_COLORS)
        orient = random.choice(["h", "v"])
        p1 = _place_portal(grid, color, orient)
        p2 = _place_portal(grid, color, orient)
        if p1 is None or p2 is None:
            continue
        direction = random.choice(DIRS)
        contact = random.choice(p1)
        length = random.randint(3, 7)
        start = (contact[0] - direction[0] * length, contact[1] - direction[1] * length)
        if not _clear_line(grid, start, direction[0], direction[1], length):
            continue
        for i in range(length):
            r, c = start[0] + i * direction[0], start[1] + i * direction[1]
            grid[r, c] = GREEN
        # Reject cases where the exit ray immediately leaves through the input line or another object.
        if np.array_equal(main(grid), grid):
            continue
        return grid
    raise RuntimeError("failed to generate seed138090197/3/7 input")
