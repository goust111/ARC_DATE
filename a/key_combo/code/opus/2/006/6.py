import random
from collections import deque

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in DIRS.values():
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) != 0:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def _exposed_dir(port, cell_set):
    y, x = port
    exposed = []
    for name, (dy, dx) in DIRS.items():
        if (y + dy, x + dx) not in cell_set:
            exposed.append(name)
    return exposed[0]


def _component_data(grid, cells):
    colors = {}
    for p in cells:
        colors.setdefault(int(grid[p]), []).append(p)
    body = max(colors, key=lambda color: len(colors[color]))
    ports = {color: pts[0] for color, pts in colors.items() if color != body}
    return {"cells": cells, "set": set(cells), "body": body, "ports": ports, "area": len(cells)}


def main(grid: np.ndarray) -> np.ndarray:
    comps = [_component_data(grid, cells) for cells in _components(grid)]
    anchor = max(comps, key=lambda c: c["area"])
    fragments = [c for c in comps if c is not anchor]
    port_order = sorted(anchor["ports"].items(), key=lambda item: item[1])
    canvas = {}
    for p in anchor["cells"]:
        canvas[p] = int(grid[p])
    for color, anchor_port in port_order:
        frag = next((f for f in fragments if color in f["ports"]), None)
        if frag is None:
            continue
        frag_port = frag["ports"][color]
        da = _exposed_dir(anchor_port, anchor["set"])
        dp = _exposed_dir(frag_port, frag["set"])
        rows = [p[0] for p in frag["cells"]]
        cols = [p[1] for p in frag["cells"]]
        r1, c1, r2, c2 = min(rows), min(cols), max(rows), max(cols)
        transformed = []
        for y, x in frag["cells"]:
            ly, lx = y - r1, x - c1
            if dp == da:
                if da in ("L", "R"):
                    lx = c2 - c1 - lx
                else:
                    ly = r2 - r1 - ly
            transformed.append((ly, lx, int(grid[y, x])))
        py, px = frag_port[0] - r1, frag_port[1] - c1
        if dp == da:
            if da in ("L", "R"):
                px = c2 - c1 - px
            else:
                py = r2 - r1 - py
        dy, dx = DIRS[da]
        target = (anchor_port[0] + dy, anchor_port[1] + dx)
        off_y, off_x = target[0] - py, target[1] - px
        for ly, lx, val in transformed:
            canvas[(off_y + ly, off_x + lx)] = val
    if not canvas:
        return np.zeros((0, 0), dtype=int)
    ys = [p[0] for p in canvas]
    xs = [p[1] for p in canvas]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    out = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=int)
    for (y, x), val in canvas.items():
        out[y - min_y, x - min_x] = val
    return out


def _paint_block(grid, top, left, size, port_side, color):
    grid[top : top + size, left : left + size] = 5
    mid = size // 2
    if port_side == "R":
        grid[top + mid, left + size - 1] = color
    elif port_side == "L":
        grid[top + mid, left] = color
    elif port_side == "U":
        grid[top, left + mid] = color
    else:
        grid[top + size - 1, left + mid] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((17, 17), dtype=int)
    colors = random.sample([2, 3, 4, 7], random.randint(2, 3))
    anchor_sides = random.sample(["R", "D", "L", "U"], len(colors))
    anchor_top = 6
    anchor_left = 6
    grid[anchor_top : anchor_top + 5, anchor_left : anchor_left + 5] = 5
    for color, side in zip(colors, anchor_sides):
        mid = 2
        if side == "R":
            grid[anchor_top + mid, anchor_left + 4] = color
        elif side == "L":
            grid[anchor_top + mid, anchor_left] = color
        elif side == "U":
            grid[anchor_top, anchor_left + mid] = color
        else:
            grid[anchor_top + 4, anchor_left + mid] = color
    spots = [(1, 1), (1, 12), (12, 1), (12, 12)]
    for i, (color, side) in enumerate(zip(colors, anchor_sides)):
        expose = random.choice([side, OPP[side]])
        top, left = spots[i]
        _paint_block(grid, top, left, 3, expose, color)
    return grid
