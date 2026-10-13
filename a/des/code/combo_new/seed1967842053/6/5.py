import random
from collections import deque

import numpy as np


SWAP_COLORS = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    5: 5,
}

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _component_cells(grid, start, seen):
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    seen[start] = True
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return cells


def _marker_direction(cells, h, w):
    cells = set(cells)
    for r, c in cells:
        if r == 0 and (1, c) in cells:
            tips = [p for p in cells if p not in {(r, c), (1, c)}]
            if len(tips) == 1 and tips[0][0] == 1:
                return "left" if tips[0][1] < c else "right"
        if r == h - 1 and (h - 2, c) in cells:
            tips = [p for p in cells if p not in {(r, c), (h - 2, c)}]
            if len(tips) == 1 and tips[0][0] == h - 2:
                return "left" if tips[0][1] < c else "right"
        if c == 0 and (r, 1) in cells:
            tips = [p for p in cells if p not in {(r, c), (r, 1)}]
            if len(tips) == 1 and tips[0][1] == 1:
                return "up" if tips[0][0] < r else "down"
        if c == w - 1 and (r, w - 2) in cells:
            tips = [p for p in cells if p not in {(r, c), (r, w - 2)}]
            if len(tips) == 1 and tips[0][1] == w - 2:
                return "up" if tips[0][0] < r else "down"
    return None


def _find_markers(grid):
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    markers = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            cells = _component_cells(grid, (r, c), seen)
            if len(cells) == 3 and any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in cells):
                direction = _marker_direction(cells, h, w)
                if direction is not None:
                    markers[color] = (direction, set(cells))
    return markers


def _find_objects(grid, marker_cells):
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    for r, c in marker_cells:
        seen[r, c] = True
    objects = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != 0 and not seen[r, c]:
                objects.append((int(grid[r, c]), _component_cells(grid, (r, c), seen)))
    return objects


def _shift_to_edge(cells, direction, h, w):
    dr, dc = DIRECTIONS[direction]
    if dr == -1:
        step = min(r for r, _ in cells) - 1
    elif dr == 1:
        step = h - 2 - max(r for r, _ in cells)
    elif dc == -1:
        step = min(c for _, c in cells) - 1
    else:
        step = w - 2 - max(c for _, c in cells)
    return [(r + dr * step, c + dc * step) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    """Project each interior object toward its color-matched edge arrow and draw a swapped-color shadow.

    Input structure:
    The grid has a black background, several 3-pixel L-shaped arrow markers touching the outer border,
    and interior 4-connected monochrome objects. Each object color matches exactly one border marker color.

    Rule:
    Read the matched marker's L turn as a direction, translate the object in that direction until its front
    edge reaches the cell just inside the grid boundary, and draw an identical shadow there using the fixed
    visible color swaps: red/blue, green/yellow, magenta/orange, azure/maroon, with grey unchanged. Original
    object pixels remain in place and take priority where a shadow overlaps them.

    Key insight:
    The border marker color pairs it to an object, while the marker geometry gives the projection direction;
    the new terminal copy is not the same color but the object's complementary swapped color.
    """
    out = np.array(grid, copy=True)
    h, w = out.shape
    markers = _find_markers(out)
    marker_cells = set()
    for _, cells in markers.values():
        marker_cells.update(cells)
    original_object_cells = set()
    objects = _find_objects(out, marker_cells)
    for _, cells in objects:
        original_object_cells.update(cells)

    for color, cells in objects:
        if color not in markers:
            continue
        direction, _ = markers[color]
        shadow_color = SWAP_COLORS.get(color, color)
        for r, c in _shift_to_edge(cells, direction, h, w):
            if (r, c) not in original_object_cells and 0 <= r < h and 0 <= c < w:
                out[r, c] = shadow_color
    return out


def _marker_cells(edge, pos, direction, h, w):
    if edge == "top":
        base = (0, pos)
        mid = (1, pos)
        if direction == "left":
            tip = (1, pos - 1)
        else:
            tip = (1, pos + 1)
    elif edge == "bottom":
        base = (h - 1, pos)
        mid = (h - 2, pos)
        if direction == "left":
            tip = (h - 2, pos - 1)
        else:
            tip = (h - 2, pos + 1)
    elif edge == "left":
        base = (pos, 0)
        mid = (pos, 1)
        if direction == "up":
            tip = (pos - 1, 1)
        else:
            tip = (pos + 1, 1)
    else:
        base = (pos, w - 1)
        mid = (pos, w - 2)
        if direction == "up":
            tip = (pos - 1, w - 2)
        else:
            tip = (pos + 1, w - 2)
    return [base, mid, tip]


def _random_shape():
    patterns = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
    ]
    return random.choice(patterns)


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _rotated(cells, turns):
    cur = list(cells)
    for _ in range(turns):
        cur = [(c, -r) for r, c in cur]
        cur = _normalize(cur)
    return cur


def _can_place(occupied, cells, pad=1):
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def generate_input() -> np.ndarray:
    dirs = ["up", "down", "left", "right"]
    edge_for_dir = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["top", "bottom"],
        "right": ["top", "bottom"],
    }

    for _attempt in range(300):
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(3, 6))
        dir_cycle = dirs[:]
        random.shuffle(dir_cycle)
        occupied = set()
        specs = []
        ok = True

        for i, color in enumerate(colors):
            direction = dir_cycle[i % 4] if i < 4 else random.choice(dirs)
            edge = random.choice(edge_for_dir[direction])
            placed_marker = False
            for _ in range(160):
                if edge in ("top", "bottom"):
                    pos = random.randint(2, w - 3)
                else:
                    pos = random.randint(2, h - 3)
                marker = _marker_cells(edge, pos, direction, h, w)
                if all(0 <= r < h and 0 <= c < w for r, c in marker) and _can_place(occupied, marker, 1):
                    for r, c in marker:
                        grid[r, c] = color
                        occupied.add((r, c))
                    specs.append((color, direction))
                    placed_marker = True
                    break
            if not placed_marker:
                ok = False
                break
        if not ok:
            continue

        object_occupied = set(occupied)
        for color, direction in specs:
            shape = _rotated(_random_shape(), random.randint(0, 3))
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            placed = False
            for _ in range(240):
                r0 = random.randint(3, h - sh - 4)
                c0 = random.randint(3, w - sw - 4)
                cells = [(r0 + r, c0 + c) for r, c in shape]
                shadow = _shift_to_edge(cells, direction, h, w)
                if not _can_place(object_occupied, cells, 1):
                    continue
                if any((r, c) in occupied for r, c in shadow):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                    object_occupied.add((r, c))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid

    grid = np.zeros((16, 16), dtype=int)
    fixed = [
        (1, "up", "left", 5, [(9, 6), (10, 6), (10, 7)]),
        (2, "down", "right", 5, [(5, 9), (5, 10), (6, 10)]),
        (3, "left", "top", 6, [(8, 11), (9, 11)]),
        (4, "right", "bottom", 8, [(6, 4), (6, 5), (7, 4)]),
    ]
    for color, direction, edge, pos, cells in fixed:
        for r, c in _marker_cells(edge, pos, direction, 16, 16):
            grid[r, c] = color
        for r, c in cells:
            grid[r, c] = color
    return grid
