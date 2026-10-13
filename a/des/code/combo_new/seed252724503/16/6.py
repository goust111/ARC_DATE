import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2
RACK = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _arrow_info(cells):
    pts = np.array(cells)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    rel = {(r - r0, c - c0) for r, c in cells}
    if r1 - r0 == 2:
        width = c1 - c0 + 1
        if {(1, c) for c in range(width)}.issubset(rel):
            if {(0, 0), (2, 0)}.issubset(rel):
                return "left", width, (r0 + 1, c0)
            if {(0, width - 1), (2, width - 1)}.issubset(rel):
                return "right", width, (r0 + 1, c0 + width - 1)
    if c1 - c0 == 2:
        height = r1 - r0 + 1
        if {(r, 1) for r in range(height)}.issubset(rel):
            if {(0, 0), (0, 2)}.issubset(rel):
                return "up", height, (r0, c0 + 1)
            if {(height - 1, 0), (height - 1, 2)}.issubset(rel):
                return "down", height, (r0 + height - 1, c0 + 1)
    raise AssertionError("unrecognized arrow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use red arrows to select colored blocks, order them by arrow length, and pack them on the bottom rack.

    Input structure:
    - The background is black and a grey horizontal rack line spans the bottom part of the grid.
    - Several solid colored rectangles are scattered elsewhere.
    - Separate red arrow objects point toward some of those rectangles; each arrow has a straight shaft plus a V-shaped one-pixel-wide head.

    Rule:
    - For each red arrow, follow its pointed direction and select the nearest colored rectangle in that ray.
    - The shaft length determines ordering: shorter arrows place their target earlier.
    - Remove every arrow and every unselected rectangle.
    - Pack the selected rectangles left-to-right with no gaps, bottom-aligned on the rack and centered as a whole strip.

    Key insight:
    - Each arrow encodes two different facts at once: its direction chooses the target block, and its length chooses the target's place in the final ordering.
    """
    g = np.asarray(grid, dtype=int)
    rack_row = int(np.where(np.all(g == RACK, axis=1))[0][0])
    out = np.full_like(g, BG)
    out[rack_row, :] = RACK
    rects = []
    rect_map = {}
    for cells in _components((g != BG) & (g != ARROW) & (g != RACK)):
        pts = np.array(cells)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        color = int(g[cells[0]])
        rect = {"cells": cells, "bbox": (r0, c0, r1, c1), "color": color}
        rects.append(rect)
        for r, c in cells:
            rect_map[(r, c)] = rect

    chosen = []
    for cells in _components(g == ARROW):
        direction, length, tip = _arrow_info(cells)
        drdc = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}[direction]
        r, c = tip
        target = None
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            if (r, c) in rect_map:
                target = rect_map[(r, c)]
                break
            r += drdc[0]
            c += drdc[1]
        if target is not None:
            chosen.append((length, target))

    seen = set()
    ordered = []
    for length, rect in sorted(chosen, key=lambda x: x[0]):
        key = rect["bbox"]
        if key not in seen:
            seen.add(key)
            ordered.append(rect)

    total_w = sum(rect["bbox"][3] - rect["bbox"][1] + 1 for rect in ordered)
    total_w += max(0, len(ordered) - 1) * 0
    start = (g.shape[1] - total_w) // 2
    cursor = start
    for rect in ordered:
        r0, c0, r1, c1 = rect["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        top = rack_row - hh
        out[top:rack_row, cursor : cursor + ww] = rect["color"]
        cursor += ww
    return out


def _draw_arrow(grid, top, left, direction, length):
    if direction == "right":
        grid[top + 1, left : left + length] = ARROW
        grid[top, left + length - 1] = ARROW
        grid[top + 2, left + length - 1] = ARROW
    elif direction == "left":
        grid[top + 1, left : left + length] = ARROW
        grid[top, left] = ARROW
        grid[top + 2, left] = ARROW
    elif direction == "down":
        grid[top : top + length, left + 1] = ARROW
        grid[top + length - 1, left] = ARROW
        grid[top + length - 1, left + 2] = ARROW
    else:
        grid[top : top + length, left + 1] = ARROW
        grid[top, left] = ARROW
        grid[top, left + 2] = ARROW


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 26), random.randint(18, 26)
        grid = np.full((h, w), BG, dtype=int)
        rack_row = h - 2
        grid[rack_row, :] = RACK
        occ = np.zeros((h, w), dtype=bool)
        occ[rack_row, :] = True
        colors = random.sample([1, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
        rects = []
        for color in colors:
            for _ in range(100):
                hh, ww = random.randint(1, 3), random.randint(2, 4)
                top = random.randint(1, rack_row - hh - 5)
                left = random.randint(1, w - ww - 2)
                if occ[top : top + hh, left : left + ww].any():
                    continue
                grid[top : top + hh, left : left + ww] = color
                occ[top : top + hh, left : left + ww] = True
                rects.append((color, top, left, hh, ww))
                break
        if len(rects) < 3:
            continue
        targets = random.sample(rects, random.randint(2, len(rects)))
        ok = True
        used_lengths = [random.choice([2, 3, 4, 5]) for _ in targets]
        for (color, top, left, hh, ww), length in zip(targets, used_lengths):
            direction = random.choice(["left", "right", "up", "down"])
            gap = random.randint(1, 3)
            if direction == "right":
                at = top + hh // 2 - 1
                al = left - gap - length
                scan = (slice(at, at + 3), slice(al, left))
            elif direction == "left":
                at = top + hh // 2 - 1
                al = left + ww + gap
                scan = (slice(at, at + 3), slice(left + ww, al + length))
            elif direction == "down":
                at = top - gap - length
                al = left + ww // 2 - 1
                scan = (slice(at, top), slice(al, al + 3))
            else:
                at = top + hh + gap
                al = left + ww // 2 - 1
                scan = (slice(top + hh, at + length), slice(al, al + 3))
            placed = False
            if 0 <= at and 0 <= al and at + 3 <= h and al + 3 <= w:
                area_r0 = max(0, at - 1)
                area_r1 = min(h, at + max(3, length) + 1)
                area_c0 = max(0, al - 1)
                area_c1 = min(w, al + max(3, length) + 1)
                if not occ[area_r0:area_r1, area_c0:area_c1].any() and not occ[scan].any():
                    _draw_arrow(grid, at, al, direction, length)
                    occ[area_r0:area_r1, area_c0:area_c1] = True
                    placed = True
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
