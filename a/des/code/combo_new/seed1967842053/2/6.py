import random
from collections import deque

import numpy as np


BG = 0
AZURE = 8


def _color_components(grid, predicate):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not predicate(int(grid[r, c])):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _perimeter_cells(r0, r1, c0, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_rooms(grid):
    rooms = []
    for _, cells in _color_components(grid, lambda v: v == AZURE):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perimeter = _perimeter_cells(r0, r1, c0, c1)
        missing = [p for p in perimeter if int(grid[p]) != AZURE]
        if len(missing) != 1 or len(cells) != len(perimeter) - 1:
            continue

        gap_r, gap_c = missing[0]
        if gap_r == r0:
            door = "top"
        elif gap_c == c1:
            door = "right"
        elif gap_r == r1:
            door = "bottom"
        elif gap_c == c0:
            door = "left"
        else:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_pos = np.argwhere((inner != BG) & (inner != AZURE))
        if len(marker_pos) != 1:
            continue
        mr, mc = marker_pos[0]
        marker = (r0 + 1 + int(mr), c0 + 1 + int(mc))
        rooms.append(
            {
                "bbox": (r0, r1, c0, c1),
                "inner": (r0 + 1, r1 - 1, c0 + 1, c1 - 1),
                "door": door,
                "marker": marker,
                "color": int(grid[marker]),
            }
        )
    return rooms


def _component_mask(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _rotate_for_door(mask, door):
    turns = {"top": 0, "right": 3, "bottom": 2, "left": 1}[door]
    return np.rot90(mask, turns)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each matching shelf shape into its walled room after door-directed rotation.

    Input structure:
    The grid has a black background, azure one-pixel rectangular room walls, one
    centered single-pixel colored marker inside each room, and one external
    4-connected shelf object with the same color as each marker. Each room wall
    has exactly one central one-pixel door gap on its top, right, bottom, or left
    side, and shelf objects are outside all rooms.

    Rule:
    Pair rooms and shelf objects by their visible color. Remove each marker and
    its original external shelf object. Rotate the shelf object's original
    bounding-box orientation so its natural top faces the room door: top keeps it
    unchanged, right rotates it clockwise 90 degrees, bottom rotates it 180
    degrees, and left rotates it counterclockwise 90 degrees. Place the rotated
    object centered in the room interior while leaving azure walls unchanged.

    Key insight:
    Color only identifies which object belongs to which room; the door gap, not
    the object's shelf position, determines the required rotation before centered
    placement.
    """
    out = grid.copy()
    rooms = _find_rooms(grid)
    marker_cells = {room["marker"] for room in rooms}

    shelves = {}
    for color, cells in _color_components(grid, lambda v: v != BG and v != AZURE):
        cell_set = set(cells)
        if len(cells) == 1 and next(iter(cell_set)) in marker_cells:
            continue
        shelves[color] = cells

    for room in rooms:
        out[room["marker"]] = BG
    for cells in shelves.values():
        for r, c in cells:
            out[r, c] = BG

    for room in rooms:
        color = room["color"]
        if color not in shelves:
            continue
        mask = _rotate_for_door(_component_mask(shelves[color]), room["door"])
        ir0, ir1, ic0, ic1 = room["inner"]
        inner_h = ir1 - ir0 + 1
        inner_w = ic1 - ic0 + 1
        top = ir0 + (inner_h - mask.shape[0]) // 2
        left = ic0 + (inner_w - mask.shape[1]) // 2
        for rr, cc in np.argwhere(mask):
            out[top + int(rr), left + int(cc)] = color
    return out


def _can_place_empty(grid, r0, c0, h, w, margin=1):
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    er0 = max(0, r0 - margin)
    ec0 = max(0, c0 - margin)
    er1 = min(gh, r0 + h + margin)
    ec1 = min(gw, c0 + w + margin)
    return not np.any(grid[er0:er1, ec0:ec1])


def _draw_room(grid, r0, c0, h, w, door, color):
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = AZURE
    grid[r1, c0 : c1 + 1] = AZURE
    grid[r0 : r1 + 1, c0] = AZURE
    grid[r0 : r1 + 1, c1] = AZURE
    if door == "top":
        grid[r0, (c0 + c1) // 2] = BG
    elif door == "right":
        grid[(r0 + r1) // 2, c1] = BG
    elif door == "bottom":
        grid[r1, (c0 + c1) // 2] = BG
    else:
        grid[(r0 + r1) // 2, c0] = BG
    grid[(r0 + r1) // 2, (c0 + c1) // 2] = color


def _random_shape(rng, limit):
    limit = max(4, min(5, limit))
    h = rng.randint(3, limit - 1)
    w = rng.randint(h + 1, limit)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    spine = rng.randint(1, w - 2)
    mask[:, spine] = True

    side = rng.choice([-1, 1])
    row = rng.randint(1, h - 1)
    mask[row, spine + side] = True
    if h > 3 and rng.random() < 0.65:
        row2 = rng.randint(1, h - 1)
        side2 = -side if rng.random() < 0.55 else side
        mask[row2, spine + side2] = True
    if w > 4 and rng.random() < 0.55:
        col = 0 if rng.random() < 0.5 else w - 1
        mask[1, col] = True
    return mask


def _place_shape_near_edge(grid, mask, color, rng):
    gh, gw = grid.shape
    mh, mw = mask.shape
    for _ in range(300):
        side = rng.choice(("top", "right", "bottom", "left"))
        if side == "top":
            r0 = rng.randint(1, min(4, gh - mh - 1))
            c0 = rng.randint(1, gw - mw - 1)
        elif side == "bottom":
            r0 = rng.randint(max(1, gh - mh - 5), gh - mh - 1)
            c0 = rng.randint(1, gw - mw - 1)
        elif side == "left":
            r0 = rng.randint(1, gh - mh - 1)
            c0 = rng.randint(1, min(4, gw - mw - 1))
        else:
            r0 = rng.randint(1, gh - mh - 1)
            c0 = rng.randint(max(1, gw - mw - 5), gw - mw - 1)
        if not _can_place_empty(grid, r0, c0, mh, mw, margin=1):
            continue
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
        return True
    return False


def _try_generate(rng):
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    max_rooms = 4 if h >= 28 and w >= 28 else 3
    n_rooms = rng.randint(2, max_rooms)
    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 9], n_rooms)
    doors = [rng.choice(("top", "right", "bottom", "left")) for _ in range(n_rooms)]
    rooms = []

    for i in range(n_rooms):
        placed = False
        for _ in range(200):
            rh = rng.choice((7, 9, 11 if n_rooms <= 2 else 9))
            rw = rng.choice((7, 9, 11 if n_rooms <= 2 else 9))
            if rh > h - 10 or rw > w - 10:
                continue
            r_low, r_high = 5, h - rh - 5
            c_low, c_high = 5, w - rw - 5
            if r_low > r_high or c_low > c_high:
                continue
            r0 = rng.randint(r_low, r_high)
            c0 = rng.randint(c_low, c_high)
            if not _can_place_empty(grid, r0, c0, rh, rw, margin=1):
                continue
            _draw_room(grid, r0, c0, rh, rw, doors[i], colors[i])
            rooms.append((rh - 2, rw - 2, colors[i]))
            placed = True
            break
        if not placed:
            return None

    shape_records = []
    for inner_h, inner_w, color in rooms:
        limit = min(inner_h, inner_w) - 1
        shape_records.append((_random_shape(rng, limit), color))

    rng.shuffle(shape_records)
    for mask, color in shape_records:
        if not _place_shape_near_edge(grid, mask, color, rng):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        rng = random.Random()
        grid = _try_generate(rng)
        if grid is not None:
            return grid

    grid = np.zeros((26, 26), dtype=int)
    _draw_room(grid, 6, 6, 7, 7, "top", 2)
    _draw_room(grid, 13, 14, 9, 9, "left", 3)
    for mask, color in ((_random_shape(random.Random(1), 4), 2), (_random_shape(random.Random(2), 4), 3)):
        _place_shape_near_edge(grid, mask, color, random.Random(color))
    return grid
