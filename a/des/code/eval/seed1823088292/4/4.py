import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

LINE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, MAROON]
HOLE_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _line_positions(grid):
    h, w = grid.shape
    line_set = set(LINE_COLORS)
    rows = [r for r in range(h) if sum(int(v) in line_set for v in grid[r]) >= max(3, int(0.75 * w))]
    cols = [c for c in range(w) if sum(int(v) in line_set for v in grid[:, c]) >= max(3, int(0.75 * h))]
    return rows, cols


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        cells = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_cells(cells):
    r0, c0, r1, c1 = _bbox(cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _trim(mask):
    rs, cs = np.where(mask)
    return mask[rs.min() : rs.max() + 1, cs.min() : cs.max() + 1]


def _room_bounds(grid):
    rows, cols = _line_positions(grid)
    row_edges = rows
    col_edges = cols
    rooms = []
    for i in range(len(row_edges) - 1):
        r0, r1 = row_edges[i] + 1, row_edges[i + 1]
        if r1 <= r0:
            continue
        for j in range(len(col_edges) - 1):
            c0, c1 = col_edges[j] + 1, col_edges[j + 1]
            if c1 <= c0:
                continue
            rooms.append((r0, r1, c0, c1, row_edges[i], row_edges[i + 1], col_edges[j], col_edges[j + 1]))
    return rooms


def _side_color(grid, side):
    cells = [int(grid[r, c]) for r, c in side if int(grid[r, c]) in LINE_COLORS]
    if not cells:
        return BLUE
    return Counter(cells).most_common(1)[0][0]


def _room_side_colors(grid, room):
    r0, r1, c0, c1, top, bottom, left, right = room
    return [
        _side_color(grid, [(top, c) for c in range(c0, c1)]),
        _side_color(grid, [(bottom, c) for c in range(c0, c1)]),
        _side_color(grid, [(r, left) for r in range(r0, r1)]),
        _side_color(grid, [(r, right) for r in range(r0, r1)]),
    ]


def _find_hole_and_stamp(grid, room):
    r0, r1, c0, c1, *_ = room
    sub = grid[r0:r1, c0:c1]
    azure = _components(sub == AZURE)
    grey = _components(sub == GREY)
    if not azure or not grey:
        return None
    template = max(azure, key=len)
    stamp = max(grey, key=len)
    ar0, ac0, ar1, ac1 = _bbox(template)
    box = sub[ar0 : ar1 + 1, ac0 : ac1 + 1]
    hole_mask = box == BLACK
    hole_comps = _components(hole_mask)
    if not hole_comps:
        return None
    hole = max(hole_comps, key=len)
    hole_cells = [(r0 + ar0 + rr, c0 + ac0 + cc) for rr, cc in hole]
    stamp_cells = [(r0 + rr, c0 + cc) for rr, cc in stamp]
    return hole_cells, stamp_cells


def _rot_to_match(stamp_mask, hole_mask):
    hole = _trim(hole_mask)
    for k in range(4):
        cand = _trim(np.rot90(stamp_mask, k))
        if cand.shape == hole.shape and np.array_equal(cand, hole):
            return cand
    return hole


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    rooms = _room_bounds(grid)
    side_colors = [_room_side_colors(grid, room) for room in rooms]
    counts = Counter(color for colors in side_colors for color in colors)

    for room, colors in zip(rooms, side_colors):
        found = _find_hole_and_stamp(grid, room)
        if found is None:
            continue
        fill_color = min(colors, key=lambda color: (counts[color], colors.index(color)))
        hole_cells, stamp_cells = found
        hole_mask = _mask_from_cells(hole_cells)
        stamp_mask = _mask_from_cells(stamp_cells)
        fitted = _rot_to_match(stamp_mask, hole_mask)
        hr0, hc0, _, _ = _bbox(hole_cells)
        for r, c in stamp_cells:
            out[r, c] = BLACK
        for rr in range(fitted.shape[0]):
            for cc in range(fitted.shape[1]):
                if fitted[rr, cc]:
                    out[hr0 + rr, hc0 + cc] = fill_color
    return out


def _draw_lines(grid, room_h, room_w, rows, cols, hcolors, vcolors):
    for i in range(rows + 1):
        r = i * (room_h + 1)
        grid[r, :] = hcolors[i]
    for j in range(cols + 1):
        c = j * (room_w + 1)
        grid[:, c] = vcolors[j]


def _place_room_content(grid, r0, c0):
    shape = random.choice(HOLE_SHAPES)
    hole = np.rot90(shape, random.randint(0, 3))
    tr, tc = r0 + 1, c0 + 1
    grid[tr : tr + 5, tc : tc + 5] = AZURE
    hole_origin = (tr + 1, tc + 1)
    for rr in range(hole.shape[0]):
        for cc in range(hole.shape[1]):
            if hole[rr, cc]:
                grid[hole_origin[0] + rr, hole_origin[1] + cc] = BLACK
    stamp = np.rot90(hole, random.randint(0, 3))
    sr, sc = r0 + 2, c0 + 6
    for rr in range(stamp.shape[0]):
        for cc in range(stamp.shape[1]):
            if stamp[rr, cc]:
                grid[sr + rr, sc + cc] = GREY


def generate_input() -> np.ndarray:
    rows = random.choice([2, 3])
    cols = 2 if rows == 3 else random.choice([2, 3])
    room_h = random.randint(8, 10)
    room_w = random.randint(10, 12)
    h = rows * room_h + rows + 1
    w = cols * room_w + cols + 1
    grid = np.zeros((h, w), dtype=int)
    hcolors = random.choices(LINE_COLORS, k=rows + 1)
    vcolors = random.choices(LINE_COLORS, k=cols + 1)
    _draw_lines(grid, room_h, room_w, rows, cols, hcolors, vcolors)
    for ri in range(rows):
        for ci in range(cols):
            r0 = ri * (room_h + 1) + 1
            c0 = ci * (room_w + 1) + 1
            _place_room_content(grid, r0, c0)
    return grid


if __name__ == "__main__":
    for _ in range(100):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
