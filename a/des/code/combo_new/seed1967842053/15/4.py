from collections import deque
import random

import numpy as np


BLUE = 1
BACKGROUND = 0


def _most_common_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _blue_components(grid):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] != BLUE:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r1, c1, r2, c2):
    cells = []
    for c in range(c1, c2 + 1):
        cells.append((r1, c))
        cells.append((r2, c))
    for r in range(r1 + 1, r2):
        cells.append((r, c1))
        cells.append((r, c2))
    return cells


def _door_direction(door, r1, c1, r2, c2):
    r, c = door
    if r == r1 and c1 < c < c2:
        return -1, 0
    if r == r2 and c1 < c < c2:
        return 1, 0
    if c == c1 and r1 < r < r2:
        return 0, -1
    if c == c2 and r1 < r < r2:
        return 0, 1
    return 0, 0


def main(grid: np.ndarray) -> np.ndarray:
    """Push each room's inner object outward through its door by its own size.

    Input structure:
    A mostly black grid contains several disjoint one-pixel-thick blue rectangular
    room borders. Each room border has exactly one one-pixel gap on one side, and
    each room interior contains one small 4-connected object in a single non-blue,
    non-background color.

    Rule:
    For every blue room, find the missing border pixel to determine the door
    direction: up, down, left, or right. Count the non-background, non-blue pixels
    inside the room. Clear those original pixels, then translate that whole object
    by exactly that count of pixels in the door direction. The blue room border and
    its gap are otherwise left unchanged.

    Key insight:
    The door gives only the direction; the less obvious movement distance is the
    pixel count of the object being pushed.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    bg = _most_common_color(grid)

    for comp in _blue_components(grid):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r1, r2 = min(rs), max(rs)
        c1, c2 = min(cs), max(cs)
        if r2 - r1 < 2 or c2 - c1 < 2:
            continue

        missing = [(r, c) for r, c in _border_cells(r1, c1, r2, c2) if grid[r, c] != BLUE]
        if len(missing) != 1:
            continue

        dr, dc = _door_direction(missing[0], r1, c1, r2, c2)
        if dr == 0 and dc == 0:
            continue

        inner = grid[r1 + 1:r2, c1 + 1:c2]
        mask = (inner != bg) & (inner != BLUE)
        local = np.argwhere(mask)
        if len(local) == 0:
            continue

        coords = [(int(r + r1 + 1), int(c + c1 + 1)) for r, c in local]
        values = [grid[r, c] for r, c in coords]
        distance = len(coords)

        for r, c in coords:
            out[r, c] = bg
        for (r, c), value in zip(coords, values):
            out[r + dr * distance, c + dc * distance] = value

    return out


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_connected_shape(size, movement_axis):
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < size:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            cells.add((r + dr, c + dc))
        shape = _normalize(cells)
        for _ in range(random.randrange(4)):
            shape = _normalize([(c, -r) for r, c in shape])
        if random.choice((False, True)):
            shape = _normalize([(r, -c) for r, c in shape])
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if movement_axis == "vertical" and height < size:
            return shape
        if movement_axis == "horizontal" and width < size:
            return shape

    if movement_axis == "vertical":
        return [(0, c) for c in range(size)]
    return [(r, 0) for r in range(size)]


def _room_bounds_clear(occupied, r1, c1, r2, c2):
    rows, cols = occupied.shape
    rr1 = max(0, r1 - 1)
    cc1 = max(0, c1 - 1)
    rr2 = min(rows, r2 + 2)
    cc2 = min(cols, c2 + 2)
    return not occupied[rr1:rr2, cc1:cc2].any()


def _cells_clear(occupied, cells):
    rows, cols = occupied.shape
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        rr1 = max(0, r - 1)
        cc1 = max(0, c - 1)
        rr2 = min(rows, r + 2)
        cc2 = min(cols, c + 2)
        if occupied[rr1:rr2, cc1:cc2].any():
            return False
    return True


def _draw_room(grid, occupied, r1, c1, r2, c2, door, obj_cells, obj_color, final_cells):
    for r, c in _border_cells(r1, c1, r2, c2):
        if (r, c) != door:
            grid[r, c] = BLUE
            occupied[r, c] = True

    for r, c in obj_cells:
        grid[r, c] = obj_color
        occupied[r, c] = True

    for r, c in final_cells:
        occupied[r, c] = True


def _try_place_room(grid, occupied, direction):
    rows, cols = grid.shape
    axis = "vertical" if direction in ("up", "down") else "horizontal"
    colors = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(350):
        size = random.randint(2, 6)
        shape = _random_connected_shape(size, axis)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        if axis == "vertical" and shape_h >= size:
            continue
        if axis == "horizontal" and shape_w >= size:
            continue

        min_h = max(5, shape_h + 2)
        min_w = max(5, shape_w + 2)
        max_h = min(10, rows - 2)
        max_w = min(10, cols - 2)
        if min_h > max_h or min_w > max_w:
            continue

        room_h = random.randint(min_h, max_h)
        room_w = random.randint(min_w, max_w)

        r_low, r_high = 0, rows - room_h
        c_low, c_high = 0, cols - room_w
        if direction == "up":
            r_low = size - 1
        elif direction == "down":
            r_high = rows - room_h - size
        elif direction == "left":
            c_low = size - 1
        else:
            c_high = cols - room_w - size
        if r_low > r_high or c_low > c_high:
            continue

        r1 = random.randint(r_low, r_high)
        c1 = random.randint(c_low, c_high)
        r2 = r1 + room_h - 1
        c2 = c1 + room_w - 1
        if not _room_bounds_clear(occupied, r1, c1, r2, c2):
            continue

        if direction == "up":
            anchor_r = r1 + 1
            anchor_c = random.randint(c1 + 1, c2 - shape_w)
            front = [(r, c) for r, c in shape if r == 0]
            door = (r1, anchor_c + random.choice(front)[1])
            delta = (-size, 0)
        elif direction == "down":
            anchor_r = r2 - shape_h
            anchor_c = random.randint(c1 + 1, c2 - shape_w)
            front = [(r, c) for r, c in shape if r == shape_h - 1]
            door = (r2, anchor_c + random.choice(front)[1])
            delta = (size, 0)
        elif direction == "left":
            anchor_r = random.randint(r1 + 1, r2 - shape_h)
            anchor_c = c1 + 1
            front = [(r, c) for r, c in shape if c == 0]
            door = (anchor_r + random.choice(front)[0], c1)
            delta = (0, -size)
        else:
            anchor_r = random.randint(r1 + 1, r2 - shape_h)
            anchor_c = c2 - shape_w
            front = [(r, c) for r, c in shape if c == shape_w - 1]
            door = (anchor_r + random.choice(front)[0], c2)
            delta = (0, size)

        obj_cells = [(anchor_r + r, anchor_c + c) for r, c in shape]
        final_cells = [(r + delta[0], c + delta[1]) for r, c in obj_cells]
        if not _cells_clear(occupied, obj_cells):
            continue
        if not _cells_clear(occupied, final_cells):
            continue

        _draw_room(
            grid,
            occupied,
            r1,
            c1,
            r2,
            c2,
            door,
            obj_cells,
            random.choice(colors),
            final_cells,
        )
        return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(200):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        grid = np.full((rows, cols), BACKGROUND, dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        directions = ["up", "down", "left", "right"]
        random.shuffle(directions)
        room_count = random.randint(2, 4)

        placed = 0
        for direction in directions[:room_count]:
            if _try_place_room(grid, occupied, direction):
                placed += 1

        if placed >= 2:
            return grid

    grid = np.full((20, 20), BACKGROUND, dtype=int)
    occupied = np.zeros((20, 20), dtype=bool)
    _try_place_room(grid, occupied, "up")
    _try_place_room(grid, occupied, "right")
    return grid
