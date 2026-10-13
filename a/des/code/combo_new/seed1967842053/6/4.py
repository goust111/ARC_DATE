import random
from collections import deque

import numpy as np


BG = 0
RED = 2
FRAME = 5
PATTERN_COLORS = [1, 3, 4, 6, 7, 8, 9]


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _find_nodes(grid):
    nodes = []
    for cells in _components(grid == FRAME):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAME
        if np.array_equal(sub, border):
            nodes.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "area": sub.size})
    return nodes


def _extract_pattern(grid, node):
    r0, r1, c0, c1 = node["r0"], node["r1"], node["c0"], node["c1"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != BG) & (interior != FRAME) & (interior != RED)
    if not np.any(mask):
        return np.zeros((0, 0), dtype=grid.dtype), np.zeros((0, 0), dtype=bool)
    rs, cs = np.where(mask)
    rr0, rr1 = rs.min(), rs.max()
    cc0, cc1 = cs.min(), cs.max()
    crop = interior[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    crop_mask = mask[rr0 : rr1 + 1, cc0 : cc1 + 1]
    crop[~crop_mask] = BG
    return crop, crop_mask


def _place_pattern(out, node, crop, mask):
    if crop.size == 0:
        return
    r0, r1, c0, c1 = node["r0"], node["r1"], node["c0"], node["c1"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = crop.shape
    top = r0 + 1 + (inner_h - ph) // 2
    left = c0 + 1 + (inner_w - pw) // 2
    for rr in range(ph):
        for cc in range(pw):
            if mask[rr, cc]:
                tr = top + rr
                tc = left + cc
                if r0 < tr < r1 and c0 < tc < c1:
                    out[tr, tc] = crop[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate node interior patterns through the area-sorted chain of framed rectangles.

    Input structure:
    The grid has a black background, gray one-pixel rectangular node frames, red
    single-pixel horizontal or vertical connector lines between nodes, and one
    non-black non-gray non-red pattern inside each framed node.
    Rule:
    Find every gray rectangular node, rank the nodes by the area of the full
    framed rectangle from smallest to largest, erase only the old interior
    patterns, then move each pattern to the center of the next larger node.
    The pattern from the largest node wraps around to the smallest node. Frames,
    red connectors, and background stay fixed.
    Key insight:
    The red connectors identify the objects as one chain, but the rotation order
    is determined by rectangle size rather than by physical adjacency.
    """
    out = grid.copy()
    nodes = sorted(_find_nodes(grid), key=lambda n: (n["area"], n["r0"], n["c0"]))
    if not nodes:
        return out

    patterns = [_extract_pattern(grid, node) for node in nodes]
    for node in nodes:
        r0, r1, c0, c1 = node["r0"], node["r1"], node["c0"], node["c1"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[(interior != BG) & (interior != FRAME) & (interior != RED)] = BG

    for i, (crop, mask) in enumerate(patterns):
        _place_pattern(out, nodes[(i + 1) % len(nodes)], crop, mask)
    return out


def _shape_library(max_h, max_w):
    shapes = []
    shapes.append(np.array([[1]], dtype=bool))
    if max_w >= 2:
        shapes.append(np.array([[1, 1]], dtype=bool))
    if max_h >= 2:
        shapes.append(np.array([[1], [1]], dtype=bool))
    if max_h >= 2 and max_w >= 2:
        shapes.extend(
            [
                np.array([[1, 0], [0, 1]], dtype=bool),
                np.array([[0, 1], [1, 0]], dtype=bool),
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 1], [0, 1]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[0, 1], [1, 1]], dtype=bool),
                np.array([[1, 1], [1, 1]], dtype=bool),
            ]
        )
    if max_h >= 3 and max_w >= 3:
        shapes.extend(
            [
                np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
                np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
                np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
            ]
        )
    return shapes


def _draw_frame(grid, r, c, h, w):
    grid[r, c : c + w] = FRAME
    grid[r + h - 1, c : c + w] = FRAME
    grid[r : r + h, c] = FRAME
    grid[r : r + h, c + w - 1] = FRAME


def _draw_pattern(grid, r, c, h, w, shape, color):
    inner_h = h - 2
    inner_w = w - 2
    ph, pw = shape.shape
    top = r + 1 + (inner_h - ph) // 2
    left = c + 1 + (inner_w - pw) // 2
    grid[top : top + ph, left : left + pw][shape] = color


def _choose_dims(n, horizontal):
    for _ in range(200):
        dims = []
        areas = set()
        for _i in range(n):
            h = random.randint(4, 7)
            w = random.randint(4, 7)
            if h * w in areas:
                break
            areas.add(h * w)
            dims.append((h, w))
        if len(dims) != n:
            continue
        span = sum(w if horizontal else h for h, w in dims) + (n - 1)
        if span <= 27:
            return dims
    base = [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]
    random.shuffle(base)
    return base[:n]


def generate_input() -> np.ndarray:
    for _ in range(300):
        horizontal = random.choice([True, False])
        n = random.choice([3, 4, 5])
        dims = _choose_dims(n, horizontal)
        random.shuffle(dims)
        gaps = [random.randint(1, 3) for _ in range(n - 1)]
        span = sum(w if horizontal else h for h, w in dims) + sum(gaps)
        cross = max(h if horizontal else w for h, w in dims)
        if span + 2 <= 30 and cross + 2 <= 30:
            break
    else:
        horizontal = random.choice([True, False])
        n = 4
        dims = [(4, 4), (4, 5), (5, 5), (5, 6)]
        random.shuffle(dims)
        gaps = [1, 1, 1]
        span = sum(w if horizontal else h for h, w in dims) + sum(gaps)
        cross = max(h if horizontal else w for h, w in dims)

    long_size = random.randint(span + 2, 30)
    cross_size = random.randint(cross + 2, 30)
    if horizontal:
        rows, cols = cross_size, long_size
    else:
        rows, cols = long_size, cross_size
    grid = np.zeros((rows, cols), dtype=int)

    start_long = random.randint(1, (cols if horizontal else rows) - span - 1)
    start_cross = random.randint(1, (rows if horizontal else cols) - cross - 1)
    cross_offset = random.randint(1, min((h if horizontal else w) for h, w in dims) - 2)

    positions = []
    cursor = start_long
    for i, (h, w) in enumerate(dims):
        if horizontal:
            r, c = start_cross, cursor
            cursor += w
        else:
            r, c = cursor, start_cross
            cursor += h
        positions.append((r, c, h, w))
        if i < n - 1:
            cursor += gaps[i]

    for r, c, h, w in positions:
        _draw_frame(grid, r, c, h, w)

    for i in range(n - 1):
        r, c, h, w = positions[i]
        nr, nc, nh, nw = positions[i + 1]
        if horizontal:
            rr = start_cross + cross_offset
            grid[rr, c + w : nc] = RED
        else:
            cc = start_cross + cross_offset
            grid[r + h : nr, cc] = RED

    min_inner_h = min(h - 2 for h, _w in dims)
    min_inner_w = min(w - 2 for _h, w in dims)
    shapes = _shape_library(min(3, min_inner_h), min(3, min_inner_w))
    random.shuffle(shapes)
    colors = random.sample(PATTERN_COLORS, n)
    for (r, c, h, w), shape, color in zip(positions, shapes, colors):
        _draw_pattern(grid, r, c, h, w, shape, color)

    return grid
