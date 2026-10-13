import random
from collections import deque

import numpy as np


BLACK = 0


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_border(cells):
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(cells) == expected


def _normalized_shape(cells):
    r0, c0, r1, c1 = _bbox(cells)
    rel = [(r - r0, c - c0) for r, c in cells]
    return rel, r1 - r0 + 1, c1 - c0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored fragment into the matching colored rectangular container.

    Input structure:
    The background is black. Each active color has exactly one empty one-pixel
    rectangular border container, plus several small 4-connected fragments of
    the same color scattered elsewhere. Fragment sizes are small, and they do
    not touch their matching border in the input.

    Rule:
    Keep every container border fixed and clear all original fragment cells to
    black. For each color, classify its fragments by bounding-box longest axis:
    width greater than or equal to height is horizontal, otherwise vertical.
    Horizontal fragments are copied unchanged into the upper half of that
    color's container, ordered by their original left-to-right position and
    separated by one black column. Vertical fragments are copied unchanged into
    the lower half, ordered by their original top-to-bottom position and also
    separated by one black column.

    Key insight:
    The first grouping cue is color-to-border matching, and the second cue is
    that fragment orientation selects the upper or lower packing lane inside
    the matched container.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    colors = [int(x) for x in np.unique(grid) if int(x) != BLACK]

    for color in colors:
        comps = _components(grid, color)
        borders = [comp for comp in comps if _is_rect_border(comp)]
        if not borders:
            continue
        border = max(borders, key=len)
        br0, bc0, br1, bc1 = _bbox(border)
        for r, c in border:
            out[r, c] = color

        fragments = [comp for comp in comps if comp is not border and comp not in borders]
        horizontal = []
        vertical = []
        for frag in fragments:
            rel, height, width = _normalized_shape(frag)
            r0, c0, _, _ = _bbox(frag)
            item = (r0, c0, rel, height, width)
            if width >= height:
                horizontal.append(item)
            else:
                vertical.append(item)

        inner_h = br1 - br0 - 1
        split_row = br0 + 1 + (inner_h + 1) // 2

        col = bc0 + 1
        for _, _, rel, _, width in sorted(horizontal, key=lambda x: (x[1], x[0])):
            for rr, cc in rel:
                out[br0 + 1 + rr, col + cc] = color
            col += width + 1

        col = bc0 + 1
        for _, _, rel, _, width in sorted(vertical, key=lambda x: (x[0], x[1])):
            for rr, cc in rel:
                out[split_row + rr, col + cc] = color
            col += width + 1

    return out


def _shape_bbox(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _horizontal_shape():
    choice = random.randrange(4)
    if choice == 0:
        width = random.randint(2, 4)
        return [(0, c) for c in range(width)]
    if choice == 1:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if choice == 2:
        width = random.randint(3, 4)
        nub = random.randrange(width)
        return [(0, c) for c in range(width)] + [(1, nub)]
    return [(0, 0), (0, 1), (0, 2), (1, random.choice([0, 2]))]


def _vertical_shape():
    choice = random.randrange(3)
    if choice == 0:
        height = random.randint(2, 5)
        return [(r, 0) for r in range(height)]
    if choice == 1:
        height = random.randint(3, 4)
        nub = random.randrange(height)
        return [(r, 0) for r in range(height)] + [(nub, 1)]
    return [(0, 0), (1, 0), (2, 0), (random.choice([0, 2]), 1)]


def _lane_width(shapes):
    if not shapes:
        return 0
    return sum(_shape_bbox(s)[1] for s in shapes) + len(shapes) - 1


def _mark_rect(blocked, r0, c0, r1, c1, margin=1):
    h, w = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place_shape(blocked, shape, r0, c0):
    h, w = blocked.shape
    for rr, cc in shape:
        r, c = r0 + rr, c0 + cc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    r1 = r0 + max(r for r, _ in shape)
    c1 = c0 + max(c for _, c in shape)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _place_rect(blocked, height, width):
    rows, cols = blocked.shape
    candidates = []
    for r in range(rows - height + 1):
        for c in range(cols - width + 1):
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(rows - 1, r + height)
            cc1 = min(cols - 1, c + width)
            if not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_border(grid, r0, c0, height, width, color):
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_once():
    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    grid = np.zeros((rows, cols), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    palette = [1, 2, 3, 4, 6, 7, 8]
    random.shuffle(palette)
    color_count = 2 if random.random() < 0.65 else 3
    colors = palette[:color_count]
    specs = []

    for color in colors:
        h_count = random.choice([1, 2])
        v_count = random.choice([1, 2])
        h_shapes = [_horizontal_shape() for _ in range(h_count)]
        v_shapes = [_vertical_shape() for _ in range(v_count)]
        top_w = _lane_width(h_shapes)
        bot_w = _lane_width(v_shapes)
        max_frag_h = max(_shape_bbox(s)[0] for s in h_shapes + v_shapes)
        inner_w = max(7, top_w, bot_w) + random.randint(0, 2)
        half_h = max(5, max_frag_h)
        inner_h = half_h * 2
        specs.append(
            {
                "color": color,
                "h_shapes": h_shapes,
                "v_shapes": v_shapes,
                "height": inner_h + 2,
                "width": inner_w + 2,
            }
        )

    specs.sort(key=lambda s: s["height"] * s["width"], reverse=True)
    for spec in specs:
        pos = _place_rect(blocked, spec["height"], spec["width"])
        if pos is None:
            return None
        r0, c0 = pos
        spec["pos"] = pos
        _draw_border(grid, r0, c0, spec["height"], spec["width"], spec["color"])
        _mark_rect(blocked, r0, c0, r0 + spec["height"] - 1, c0 + spec["width"] - 1)

    fragments = []
    for spec in specs:
        color = spec["color"]
        for shape in spec["h_shapes"]:
            fragments.append((color, shape))
        for shape in spec["v_shapes"]:
            fragments.append((color, shape))
    random.shuffle(fragments)

    for color, shape in fragments:
        sh, sw = _shape_bbox(shape)
        candidates = []
        for r in range(rows - sh + 1):
            for c in range(cols - sw + 1):
                if _can_place_shape(blocked, shape, r, c):
                    candidates.append((r, c))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        for rr, cc in shape:
            grid[r0 + rr, c0 + cc] = color
        _mark_rect(blocked, r0, c0, r0 + sh - 1, c0 + sw - 1)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _build_once()
        if grid is not None and max(grid.shape) <= 30:
            return grid
    return _fallback_input()


def _fallback_input():
    grid = np.zeros((26, 26), dtype=int)
    containers = [
        (1, 1, 12, 10, 2),
        (14, 13, 11, 11, 4),
    ]
    for r, c, h, w, color in containers:
        _draw_border(grid, r, c, h, w, color)
    for r, c in [(16, 2), (16, 3), (17, 3)]:
        grid[r, c] = 2
    for r, c in [(20, 2), (21, 2), (22, 2)]:
        grid[r, c] = 2
    for r, c in [(3, 18), (3, 19), (4, 18), (4, 19)]:
        grid[r, c] = 4
    for r, c in [(8, 21), (9, 21), (10, 21), (10, 22)]:
        grid[r, c] = 4
    return grid
