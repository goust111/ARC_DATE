import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

DIR_NAMES = ("up", "down", "left", "right")

ARROW_SHAPES = {
    "right": frozenset(((0, 0), (2, 0), (1, 1))),
    "left": frozenset(((0, 1), (1, 0), (2, 1))),
    "down": frozenset(((0, 0), (0, 2), (1, 1))),
    "up": frozenset(((0, 1), (1, 0), (1, 2))),
}


def _block_cells(r, c):
    return [(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)]


def _arrow_origin(r, c, direction):
    if direction == "right":
        return r, c + 3
    if direction == "left":
        return r, c - 2
    if direction == "down":
        return r + 3, c
    return r - 2, c


def _arrow_cells(r, c, direction):
    ar, ac = _arrow_origin(r, c, direction)
    return [(ar + dr, ac + dc) for dr, dc in ARROW_SHAPES[direction]]


def _object_cells(r, c, direction):
    return _block_cells(r, c) + _arrow_cells(r, c, direction)


def _decode_arrow(cells):
    if len(cells) != 3:
        return None
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = frozenset((r - min_r, c - min_c) for r, c in cells)
    for direction, shape in ARROW_SHAPES.items():
        if norm == shape:
            return direction
    return None


def _touches_block(cells, r, c):
    block = set(_block_cells(r, c))
    if block.intersection(cells):
        return False
    return any(
        abs(ar - br) + abs(ac - bc) == 1
        for ar, ac in cells
        for br, bc in block
    )


def _extract_objects(grid):
    h, w = grid.shape
    objects = []
    for color in [int(v) for v in np.unique(grid) if int(v) != 0]:
        coords = set(map(tuple, np.argwhere(grid == color)))
        found = None
        for r in range(h - 2):
            if found is not None:
                break
            for c in range(w - 2):
                block = set(_block_cells(r, c))
                if not block.issubset(coords):
                    continue
                arrow = coords - block
                direction = _decode_arrow(arrow)
                if direction is not None and _touches_block(arrow, r, c):
                    found = {
                        "r": r,
                        "c": c,
                        "color": color,
                        "direction": direction,
                    }
                    break
        if found is not None:
            objects.append(found)
    return objects


def _slide_distance(obj, h, w, fixed=None):
    dr, dc = DIRS[obj["direction"]]
    r, c = obj["r"], obj["c"]
    dist = 0
    while True:
        nr = r + dr * (dist + 1)
        nc = c + dc * (dist + 1)
        if nr < 0 or nc < 0 or nr + 3 > h or nc + 3 > w:
            return dist
        if fixed is not None and fixed[nr : nr + 3, nc : nc + 3].any():
            return dist
        dist += 1


def _draw_trail(out, fixed, obj, final_r, final_c, dist):
    if dist <= 0:
        return
    r, c = obj["r"], obj["c"]
    color = obj["color"]
    direction = obj["direction"]
    if direction == "right":
        rr = r + 1
        for cc in range(c, final_c):
            if not fixed[rr, cc]:
                out[rr, cc] = color
    elif direction == "left":
        rr = r + 1
        for cc in range(final_c + 3, c + 3):
            if not fixed[rr, cc]:
                out[rr, cc] = color
    elif direction == "down":
        cc = c + 1
        for rr in range(r, final_r):
            if not fixed[rr, cc]:
                out[rr, cc] = color
    else:
        cc = c + 1
        for rr in range(final_r + 3, r + 3):
            if not fixed[rr, cc]:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Move each 3x3 source block along its adjacent arrow and leave a one-pixel trail.

    Input structure:
    The grid has a black background, several differently colored solid 3x3
    blocks, and one same-colored three-pixel arrow marker touching each block
    from above, below, left, or right.  The marker is a tiny T-half arrowhead
    whose orientation gives the push direction.
    Rule:
    First erase all arrow markers.  Rank the blocks by their unobstructed slide
    distance toward the arrow direction, from longest to shortest.  In that
    order, slide each block until its next step would cross the grid boundary
    or overlap an already fixed final block.  Draw the final 3x3 block and draw
    a one-pixel-wide trail along the swept row or column, excluding the final
    block cells.
    Key insight:
    The three-pixel marker encodes direction, and the final arrangement depends
    on processing blocks by slide distance before treating earlier results as
    fixed obstacles.
    """
    arr = np.asarray(grid)
    h, w = arr.shape
    objects = _extract_objects(arr)
    for obj in objects:
        obj["rank_distance"] = _slide_distance(obj, h, w)
    objects.sort(key=lambda obj: (-obj["rank_distance"], obj["r"], obj["c"]))

    out = np.zeros_like(arr)
    fixed = np.zeros((h, w), dtype=bool)
    for obj in objects:
        dist = _slide_distance(obj, h, w, fixed)
        dr, dc = DIRS[obj["direction"]]
        final_r = obj["r"] + dr * dist
        final_c = obj["c"] + dc * dist
        _draw_trail(out, fixed, obj, final_r, final_c, dist)
        out[final_r : final_r + 3, final_c : final_c + 3] = obj["color"]
        fixed[final_r : final_r + 3, final_c : final_c + 3] = True
    return out


def _valid_range(h, w, direction):
    if direction == "right":
        return (0, h - 3), (0, w - 5)
    if direction == "left":
        return (0, h - 3), (2, w - 3)
    if direction == "down":
        return (0, h - 5), (0, w - 3)
    return (2, h - 3), (0, w - 3)


def _can_place(cells, h, w, occupied, pad=1):
    for r, c in cells:
        if r < 0 or c < 0 or r >= h or c >= w:
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paint_object(grid, occupied, r, c, direction, color):
    grid[r : r + 3, c : c + 3] = color
    for rr, cc in _arrow_cells(r, c, direction):
        grid[rr, cc] = color
    for rr, cc in _object_cells(r, c, direction):
        occupied[rr, cc] = True


def _random_place(grid, occupied, rng, direction, color):
    h, w = grid.shape
    rr, cr = _valid_range(h, w, direction)
    if rr[0] > rr[1] or cr[0] > cr[1]:
        return None
    for _ in range(100):
        r = int(rng.integers(rr[0], rr[1] + 1))
        c = int(rng.integers(cr[0], cr[1] + 1))
        cells = _object_cells(r, c, direction)
        if _can_place(cells, h, w, occupied):
            _paint_object(grid, occupied, r, c, direction, color)
            return {"r": r, "c": c, "color": int(color), "direction": direction}
    return None


def _place_chain(grid, occupied, rng, direction, colors):
    h, w = grid.shape
    color_a, color_b = int(colors[0]), int(colors[1])
    if direction == "right":
        if w < 14 or h < 3:
            return []
        r = int(rng.integers(0, h - 2))
        c1 = int(rng.integers(0, w - 10))
        c2 = int(rng.integers(c1 + 6, w - 4))
        pair = ((r, c1, color_a), (r, c2, color_b))
    elif direction == "left":
        if w < 14 or h < 3:
            return []
        r = int(rng.integers(0, h - 2))
        c1 = int(rng.integers(2, w - 8))
        c2 = int(rng.integers(c1 + 6, w - 2))
        pair = ((r, c1, color_a), (r, c2, color_b))
    elif direction == "down":
        if h < 14 or w < 3:
            return []
        c = int(rng.integers(0, w - 2))
        r1 = int(rng.integers(0, h - 10))
        r2 = int(rng.integers(r1 + 6, h - 4))
        pair = ((r1, c, color_a), (r2, c, color_b))
    else:
        if h < 14 or w < 3:
            return []
        c = int(rng.integers(0, w - 2))
        r1 = int(rng.integers(2, h - 8))
        r2 = int(rng.integers(r1 + 6, h - 2))
        pair = ((r1, c, color_a), (r2, c, color_b))

    temp = occupied.copy()
    for r, c, _ in pair:
        cells = _object_cells(r, c, direction)
        if not _can_place(cells, h, w, temp):
            return []
        for rr, cc in cells:
            temp[rr, cc] = True

    made = []
    for r, c, color in pair:
        _paint_object(grid, occupied, r, c, direction, color)
        made.append({"r": r, "c": c, "color": color, "direction": direction})
    return made


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        h = int(rng.integers(14, 31))
        w = int(rng.integers(14, 31))
        n = int(rng.integers(2, 7))
        colors = [int(v) for v in rng.choice(np.arange(1, 10), size=n, replace=False)]
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        objects = []

        start = 0
        if n >= 2 and rng.random() < 0.55:
            direction = str(rng.choice(DIR_NAMES))
            made = _place_chain(grid, occupied, rng, direction, colors[:2])
            if made:
                objects.extend(made)
                start = 2

        dirs = list(DIR_NAMES)
        rng.shuffle(dirs)
        dir_index = 0
        for color in colors[start:]:
            placed = None
            for _ in range(80):
                if rng.random() < 0.65:
                    direction = dirs[dir_index % len(dirs)]
                    dir_index += 1
                else:
                    direction = str(rng.choice(DIR_NAMES))
                placed = _random_place(grid, occupied, rng, direction, color)
                if placed is not None:
                    break
            if placed is None:
                break
            objects.append(placed)

        if len(objects) != n:
            continue
        distances = [_slide_distance(obj, h, w) for obj in objects]
        if len(set(distances)) != len(distances):
            continue
        return grid

    grid = np.zeros((18, 18), dtype=int)
    occupied = np.zeros((18, 18), dtype=bool)
    _paint_object(grid, occupied, 5, 1, "right", 1)
    _paint_object(grid, occupied, 5, 8, "right", 2)
    _paint_object(grid, occupied, 12, 10, "up", 3)
    return grid
