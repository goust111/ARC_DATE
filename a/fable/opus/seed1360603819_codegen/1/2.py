import random
from collections import deque

import numpy as np


BLACK = 0
EDGE_COLORS = [1, 2, 3, 4]


def _components(grid, colors=None, interior_only=True):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    r0, r1 = (1, h - 1) if interior_only else (0, h)
    c0, c1 = (1, w - 1) if interior_only else (0, w)
    for r in range(r0, r1):
        for c in range(c0, c1):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            if colors is not None and color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (r0 <= nr < r1 and c0 <= nc < c1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mask(cells):
    top, left, bottom, right = _bbox(cells)
    m = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
    for r, c in cells:
        m[r - top, c - left] = True
    return m


def _edge_info(grid):
    h, w = grid.shape
    info = {}
    edges = {
        "top": [(0, c) for c in range(1, w - 1)],
        "bottom": [(h - 1, c) for c in range(1, w - 1)],
        "left": [(r, 0) for r in range(1, h - 1)],
        "right": [(r, w - 1) for r in range(1, h - 1)],
    }
    for name, cells in edges.items():
        vals = [int(grid[r, c]) for r, c in cells if int(grid[r, c]) != BLACK]
        if not vals:
            continue
        color = max(set(vals), key=vals.count)
        gap_positions = []
        current = []
        for idx, (r, c) in enumerate(cells):
            if int(grid[r, c]) == BLACK:
                current.append(idx)
            elif current:
                gap_positions.append(current)
                current = []
        if current:
            gap_positions.append(current)
        gap = max(gap_positions, key=len) if gap_positions else []
        center = None
        if gap:
            center = gap[len(gap) // 2] + 1
        info[color] = {"edge": name, "gap_center": center}
    return info


def _allowed_axis(cells):
    m = _mask(cells)
    horizontal = np.array_equal(m, np.flipud(m))
    vertical = np.array_equal(m, np.fliplr(m))
    if horizontal and not vertical:
        return "vertical"
    if vertical and not horizontal:
        return "horizontal"
    if horizontal:
        return "vertical"
    if vertical:
        return "horizontal"
    return None


def _direction(edge):
    return {
        "top": (-1, 0, "vertical"),
        "bottom": (1, 0, "vertical"),
        "left": (0, -1, "horizontal"),
        "right": (0, 1, "horizontal"),
    }[edge]


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    edge_by_color = _edge_info(out)
    objects = []
    for comp in _components(out, set(edge_by_color), interior_only=True):
        axis = _allowed_axis(comp["cells"])
        target = edge_by_color.get(comp["color"])
        if target is None or axis is None:
            movable = False
            dr = dc = 0
            distance = 0
        else:
            dr, dc, target_axis = _direction(target["edge"])
            movable = axis == target_axis
            top, left, bottom, right = _bbox(comp["cells"])
            if target["edge"] == "top":
                distance = top - 1
            elif target["edge"] == "bottom":
                distance = h - 2 - bottom
            elif target["edge"] == "left":
                distance = left - 1
            else:
                distance = w - 2 - right
        objects.append({**comp, "movable": movable, "dr": dr, "dc": dc, "distance": distance})

    for obj in objects:
        if obj["movable"]:
            for r, c in obj["cells"]:
                out[r, c] = BLACK

    occupancy = out != BLACK
    for obj in sorted((o for o in objects if o["movable"]), key=lambda o: o["distance"]):
        cells = list(obj["cells"])
        dr, dc = obj["dr"], obj["dc"]
        while True:
            shifted = [(r + dr, c + dc) for r, c in cells]
            if any(r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1 for r, c in shifted):
                break
            if any(occupancy[r, c] for r, c in shifted):
                break
            cells = shifted
        for r, c in cells:
            out[r, c] = obj["color"]
            occupancy[r, c] = True
    return out


def _draw_mask(grid, top, left, mask, color):
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _fits(grid, top, left, mask):
    h, w = grid.shape
    if top < 1 or left < 1 or top + mask.shape[0] > h - 1 or left + mask.shape[1] > w - 1:
        return False
    region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    return not np.any(region[mask] != BLACK)


def generate_input() -> np.ndarray:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = EDGE_COLORS[:]
    random.shuffle(colors)
    edge_colors = {"top": colors[0], "bottom": colors[1], "left": colors[2], "right": colors[3]}

    grid[0, 1 : w - 1] = edge_colors["top"]
    grid[h - 1, 1 : w - 1] = edge_colors["bottom"]
    grid[1 : h - 1, 0] = edge_colors["left"]
    grid[1 : h - 1, w - 1] = edge_colors["right"]

    top_gap = random.randint(1, w - 4)
    bottom_gap = random.randint(1, w - 4)
    left_gap = random.randint(1, h - 4)
    right_gap = random.randint(1, h - 4)
    grid[0, top_gap : top_gap + 3] = BLACK
    grid[h - 1, bottom_gap : bottom_gap + 3] = BLACK
    grid[left_gap : left_gap + 3, 0] = BLACK
    grid[right_gap : right_gap + 3, w - 1] = BLACK

    horizontal_only = [
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
    ]
    vertical_only = [m.T for m in horizontal_only]
    edges = list(edge_colors)
    random.shuffle(edges)
    count = random.randint(4, 7)
    placed = 0
    attempts = 0
    while placed < count and attempts < 300:
        attempts += 1
        edge = random.choice(edges)
        moving_axis = "vertical" if edge in ("top", "bottom") else "horizontal"
        make_movable = random.random() < 0.65
        if (moving_axis == "vertical") == make_movable:
            mask = random.choice(horizontal_only)
        else:
            mask = random.choice(vertical_only)
        color = edge_colors[edge]
        mh, mw = mask.shape
        if make_movable and edge == "top":
            left = top_gap + 1 - mw // 2
            top = random.randint(4, h - mh - 2)
        elif make_movable and edge == "bottom":
            left = bottom_gap + 1 - mw // 2
            top = random.randint(1, h - mh - 5)
        elif make_movable and edge == "left":
            top = left_gap + 1 - mh // 2
            left = random.randint(4, w - mw - 2)
        elif make_movable and edge == "right":
            top = right_gap + 1 - mh // 2
            left = random.randint(1, w - mw - 5)
        else:
            top = random.randint(1, h - mh - 1)
            left = random.randint(1, w - mw - 1)
        if _fits(grid, top, left, mask):
            _draw_mask(grid, top, left, mask, color)
            placed += 1
    return grid
