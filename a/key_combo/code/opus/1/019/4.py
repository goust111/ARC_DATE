import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_by_nonzero(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            val = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == val:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append((val, comp))
    return comps


def _norm_shape(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _interior_for_frame(cells):
    cell_set = set(cells)
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    seen = set()
    q = deque()
    for r in range(min_r, max_r + 1):
        for c in (min_c, max_c):
            if (r, c) not in cell_set and (r, c) not in seen:
                seen.add((r, c))
                q.append((r, c))
    for c in range(min_c, max_c + 1):
        for r in (min_r, max_r):
            if (r, c) not in cell_set and (r, c) not in seen:
                seen.add((r, c))
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if min_r <= nr <= max_r and min_c <= nc <= max_c and (nr, nc) not in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return {(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)
            if (r, c) not in cell_set and (r, c) not in seen}


def main(grid: np.ndarray) -> np.ndarray:
    """Fill closed outline frames, then use each frame shape as a recoloring template.

    Input structure: Closed single-color outline frames contain singleton color
    markers, while separate free sprites outside the frames have small connected
    shapes that may or may not match a frame outline shape.
    Rule: Each frame interior is filled with the majority marker color from that
    interior, markers are erased, and every free sprite whose normalized shape
    equals a frame outline is recolored with that frame's fill color. Nonmatching
    free sprites are erased.
    Key insight: The same outline acts first as a flood-fill boundary and then as
    the geometric template that transfers the inferred majority color to sprites.
    """
    grid = np.asarray(grid, dtype=int)
    components = _components_by_nonzero(grid)
    frame_infos = []
    marker_cells = set()
    for color, comp in components:
        if len(comp) <= 1:
            continue
        interior = _interior_for_frame(comp)
        if not interior:
            continue
        inside_nonframe = [(r, c) for r, c in interior if grid[r, c] != 0]
        if not inside_nonframe:
            continue
        counts = Counter(int(grid[r, c]) for r, c in inside_nonframe)
        max_count = max(counts.values())
        tied = {c for c, n in counts.items() if n == max_count}
        majority = None
        for r, c in sorted(inside_nonframe, key=lambda p: (p[1], p[0])):
            if int(grid[r, c]) in tied:
                majority = int(grid[r, c])
                break
        frame_infos.append({
            "color": color,
            "cells": set(comp),
            "shape": _norm_shape(comp),
            "interior": interior,
            "fill": majority,
        })
        marker_cells.update(inside_nonframe)

    frame_cells = set().union(*(f["cells"] for f in frame_infos)) if frame_infos else set()
    out = grid.copy()
    for f in frame_infos:
        for r, c in f["interior"]:
            if out[r, c] == 0:
                out[r, c] = f["fill"]
    for r, c in marker_cells:
        out[r, c] = 0

    shape_to_fill = {f["shape"]: f["fill"] for f in frame_infos}
    for color, comp in sorted(components, key=lambda x: (min(r for r, _ in x[1]), min(c for _, c in x[1]))):
        comp_set = set(comp)
        if comp_set & frame_cells or comp_set & marker_cells or len(comp) <= 1:
            continue
        shape = _norm_shape(comp)
        if shape in shape_to_fill:
            for r, c in comp:
                out[r, c] = shape_to_fill[shape]
        else:
            for r, c in comp:
                out[r, c] = 0
    return out


def _ring_cells(top, left, height, width):
    cells = set()
    for c in range(left, left + width):
        cells.add((top, c))
        cells.add((top + height - 1, c))
    for r in range(top, top + height):
        cells.add((r, left))
        cells.add((r, left + width - 1))
    return cells


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    frames = [
        (1, 1, 5, 5, 5),
        (1, 10, 5, 6, 8),
    ]
    marker_palette = [1, 2, 3, 4]
    fills = random.sample(marker_palette, 2)
    for idx, (top, left, height, width, color) in enumerate(frames):
        for r, c in _ring_cells(top, left, height, width):
            grid[r, c] = color
        interior = [(r, c) for r in range(top + 1, top + height - 1) for c in range(left + 1, left + width - 1)]
        random.shuffle(interior)
        majority = fills[idx]
        others = [c for c in marker_palette if c != majority]
        for k, (r, c) in enumerate(interior[:random.randint(4, min(6, len(interior)))]):
            grid[r, c] = majority if k < 3 else random.choice(others)

    sprite_specs = [
        (_ring_cells(12, 1, 5, 5), 6),
        (_ring_cells(11, 10, 5, 6), 7),
        ({(13, 8), (14, 7), (14, 8), (14, 9), (15, 8)}, 2),
    ]
    for cells, color in sprite_specs:
        for r, c in cells:
            grid[r, c] = color
    return grid

