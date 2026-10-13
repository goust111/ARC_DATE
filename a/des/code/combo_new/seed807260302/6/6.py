from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _separator_lines(g: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    h, w = g.shape
    row_doors: dict[int, int] = {}
    col_doors: dict[int, int] = {}

    for r in range(h):
        row = g[r, :]
        if (
            int(np.count_nonzero(row == GREY)) == w - 1
            and int(np.count_nonzero(row == BG)) == 1
            and np.all((row == GREY) | (row == BG))
        ):
            row_doors[int(r)] = int(np.argwhere(row == BG)[0, 0])

    for c in range(w):
        col = g[:, c]
        if (
            int(np.count_nonzero(col == GREY)) == h - 1
            and int(np.count_nonzero(col == BG)) == 1
            and np.all((col == GREY) | (col == BG))
        ):
            col_doors[int(c)] = int(np.argwhere(col == BG)[0, 0])

    assert row_doors or col_doors, "expected at least one grey divider"
    assert all(0 < r < h - 1 for r in row_doors), "divider rows must be internal"
    assert all(0 < c < w - 1 for c in col_doors), "divider columns must be internal"
    return row_doors, col_doors


def _rooms(g: np.ndarray, row_doors: dict[int, int], col_doors: dict[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    row_cuts = [-1] + sorted(row_doors) + [h]
    col_cuts = [-1] + sorted(col_doors) + [w]

    rooms: list[tuple[int, int, int, int]] = []
    for i in range(len(row_cuts) - 1):
        r0, r1 = row_cuts[i] + 1, row_cuts[i + 1] - 1
        if r0 > r1:
            continue
        for j in range(len(col_cuts) - 1):
            c0, c1 = col_cuts[j] + 1, col_cuts[j + 1] - 1
            if c0 <= c1:
                rooms.append((int(r0), int(c0), int(r1), int(c1)))

    assert rooms, "expected rectangular rooms between dividers"
    return rooms


def _marker_direction(g: np.ndarray, room: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = room
    markers = np.argwhere(g[r0 : r1 + 1, c0 : c1 + 1] == MAROON)
    assert markers.shape[0] == 1, "each room must contain exactly one maroon marker"
    mr = int(markers[0, 0] + r0)
    mc = int(markers[0, 1] + c0)

    touched: list[str] = []
    if mr == r0:
        touched.append("up")
    if mr == r1:
        touched.append("down")
    if mc == c0:
        touched.append("left")
    if mc == c1:
        touched.append("right")
    assert len(touched) == 1, "marker must touch exactly one room wall"
    return touched[0], (mr, mc)


def _room_object(g: np.ndarray, room: tuple[int, int, int, int]) -> tuple[int, set[tuple[int, int]]]:
    r0, c0, r1, c1 = room
    cells: set[tuple[int, int]] = set()
    colors: set[int] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = int(g[r, c])
            if v not in (BG, GREY, MAROON):
                cells.add((r, c))
                colors.add(v)

    assert cells, "each room must contain one colored object"
    assert len(colors) == 1, "room object must be monochrome"
    assert _is_connected(cells), "room object must be 4-connected"
    return next(iter(colors)), cells


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_side_cell(cells: set[tuple[int, int]], side: str) -> tuple[int, int]:
    if side == "left":
        edge = min(c for _r, c in cells)
        pts = [(r, c) for r, c in cells if c == edge]
    elif side == "right":
        edge = max(c for _r, c in cells)
        pts = [(r, c) for r, c in cells if c == edge]
    elif side == "up":
        edge = min(r for r, _c in cells)
        pts = [(r, c) for r, c in cells if r == edge]
    else:
        edge = max(r for r, _c in cells)
        pts = [(r, c) for r, c in cells if r == edge]

    assert len(pts) == 1, "the doorway-facing side must have a unique contact pixel"
    return pts[0]


def _translation(
    g: np.ndarray,
    room: tuple[int, int, int, int],
    direction: str,
    cells: set[tuple[int, int]],
    row_doors: dict[int, int],
    col_doors: dict[int, int],
) -> tuple[int, int]:
    h, w = g.shape
    r0, c0, r1, c1 = room
    obj_r0, obj_c0, obj_r1, obj_c1 = _bbox(cells)

    if direction == "left":
        if c0 == 0:
            return 0, -obj_c0
        sep_c = c0 - 1
        assert sep_c in col_doors, "left wall must be a divider column"
        door_r = col_doors[sep_c]
        assert r0 <= door_r <= r1, "divider door must open into this room side"
        ar, ac = _unique_side_cell(cells, "right")
        return door_r - ar, (sep_c - 1) - ac

    if direction == "right":
        if c1 == w - 1:
            return 0, (w - 1) - obj_c1
        sep_c = c1 + 1
        assert sep_c in col_doors, "right wall must be a divider column"
        door_r = col_doors[sep_c]
        assert r0 <= door_r <= r1, "divider door must open into this room side"
        ar, ac = _unique_side_cell(cells, "left")
        return door_r - ar, (sep_c + 1) - ac

    if direction == "up":
        if r0 == 0:
            return -obj_r0, 0
        sep_r = r0 - 1
        assert sep_r in row_doors, "top wall must be a divider row"
        door_c = row_doors[sep_r]
        assert c0 <= door_c <= c1, "divider door must open into this room side"
        ar, ac = _unique_side_cell(cells, "down")
        return (sep_r - 1) - ar, door_c - ac

    assert direction == "down"
    if r1 == h - 1:
        return (h - 1) - obj_r1, 0
    sep_r = r1 + 1
    assert sep_r in row_doors, "bottom wall must be a divider row"
    door_c = row_doors[sep_r]
    assert c0 <= door_c <= c1, "divider door must open into this room side"
    ar, ac = _unique_side_cell(cells, "up")
    return (sep_r + 1) - ar, door_c - ac


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's colored object through the doorway indicated by its wall marker.

    Input structure:
    - Black is the background, and full grey divider rows or columns split the grid into rectangular rooms.
    - Every divider line is grey except for exactly one black one-cell doorway.
    - Each room contains one small monochrome colored object and one maroon pixel touching exactly one inside wall.
    - Object colors are not black, grey, or maroon; maroon pixels are direction markers, not part of the objects.

    Rule:
    - Read the wall touched by the maroon marker in each room: left, right, top, or bottom gives the movement direction.
    - Clear every original colored object and every maroon marker, while preserving all grey dividers and black doorways.
    - If the chosen side is a grey divider, move the object through that divider's doorway and place its doorway-facing edge just inside the adjacent room.
    - If the chosen side is the outer grid edge, slide the object in that direction until its leading edge is flush with the grid edge.
    - All rooms move simultaneously, so destinations are computed from the input before any moved object is painted.

    Key insight:
    - The maroon marker encodes direction only by which wall it touches, and divider crossings must use the single black doorway on that wall rather than the nearest grey segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    row_doors, col_doors = _separator_lines(g)
    rooms = _rooms(g, row_doors, col_doors)

    moves: list[tuple[int, set[tuple[int, int]], int, int]] = []
    markers: list[tuple[int, int]] = []
    sources: set[tuple[int, int]] = set()

    for room in rooms:
        direction, marker = _marker_direction(g, room)
        color, cells = _room_object(g, room)
        dr, dc = _translation(g, room, direction, cells, row_doors, col_doors)
        moves.append((color, cells, dr, dc))
        markers.append(marker)
        sources |= cells

    out = g.copy()
    for r, c in sources:
        out[r, c] = BG
    for r, c in markers:
        out[r, c] = BG

    for color, cells, dr, dc in moves:
        for r, c in cells:
            rr, cc = r + dr, c + dc
            assert 0 <= rr < h and 0 <= cc < w, "translated object must stay in bounds"
            assert int(out[rr, cc]) == BG, "translated objects must not collide with walls or each other"
            out[rr, cc] = color

    return out


# ---------------- generator ----------------

GEN_TOPOLOGIES = (
    (2, 1),
    (3, 1),
    (4, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (3, 2),
)
GEN_RETRY_LIMIT = 500


def _gen_band_sizes(count: int) -> list[int]:
    for _ in range(200):
        sizes = [random.randint(6, 8) for _ in range(count)]
        if sum(sizes) + count - 1 <= 30:
            return sizes
    raise RuntimeError("could not sample ARC-sized room bands")


def _gen_ranges(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    ranges: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        ranges.append((pos, pos + size - 1))
        pos += size
        if i != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return ranges, separators


def _gen_door_inside(lo: int, hi: int) -> int:
    if hi - lo >= 4:
        return random.randint(lo + 2, hi - 2)
    return random.randint(lo + 1, hi - 1)


def _gen_base_grid() -> tuple[np.ndarray, dict[int, int], dict[int, int], list[tuple[int, int, int, int]]]:
    row_count, col_count = random.choice(GEN_TOPOLOGIES)
    row_sizes = _gen_band_sizes(row_count)
    col_sizes = _gen_band_sizes(col_count)
    row_ranges, sep_rows = _gen_ranges(row_sizes)
    col_ranges, sep_cols = _gen_ranges(col_sizes)

    h = sum(row_sizes) + len(sep_rows)
    w = sum(col_sizes) + len(sep_cols)
    grid = np.full((h, w), BG, dtype=int)

    for r in sep_rows:
        grid[r, :] = GREY
    for c in sep_cols:
        grid[:, c] = GREY

    row_doors: dict[int, int] = {}
    col_doors: dict[int, int] = {}
    for r in sep_rows:
        c0, c1 = random.choice(col_ranges)
        door_c = _gen_door_inside(c0, c1)
        grid[r, door_c] = BG
        row_doors[int(r)] = int(door_c)

    for c in sep_cols:
        r0, r1 = random.choice(row_ranges)
        door_r = _gen_door_inside(r0, r1)
        grid[door_r, c] = BG
        col_doors[int(c)] = int(door_r)

    rooms = [(r0, c0, r1, c1) for r0, r1 in row_ranges for c0, c1 in col_ranges]
    return grid, row_doors, col_doors, rooms


def _gen_allowed_dirs(
    room: tuple[int, int, int, int],
    h: int,
    w: int,
    row_doors: dict[int, int],
    col_doors: dict[int, int],
) -> tuple[list[str], list[str]]:
    r0, c0, r1, c1 = room
    edge: list[str] = []
    divider: list[str] = []

    if c0 == 0:
        edge.append("left")
    elif c0 - 1 in col_doors and r0 <= col_doors[c0 - 1] <= r1:
        divider.append("left")

    if c1 == w - 1:
        edge.append("right")
    elif c1 + 1 in col_doors and r0 <= col_doors[c1 + 1] <= r1:
        divider.append("right")

    if r0 == 0:
        edge.append("up")
    elif r0 - 1 in row_doors and c0 <= row_doors[r0 - 1] <= c1:
        divider.append("up")

    if r1 == h - 1:
        edge.append("down")
    elif r1 + 1 in row_doors and c0 <= row_doors[r1 + 1] <= c1:
        divider.append("down")

    return edge, divider


def _gen_marker(room: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    if direction == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    if direction == "right":
        return random.randint(r0 + 1, r1 - 1), c1
    if direction == "up":
        return r0, random.randint(c0 + 1, c1 - 1)
    return r1, random.randint(c0 + 1, c1 - 1)


def _norm_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _shape_has_unique_side(cells: set[tuple[int, int]], side: str) -> bool:
    try:
        _unique_side_cell(cells, side)
    except AssertionError:
        return False
    return True


def _gen_shape(unique_side: str) -> set[tuple[int, int]]:
    for _ in range(300):
        target_area = random.randint(3, 5)
        cells = {(random.randrange(3), random.randrange(3))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        cells = _norm_shape(cells)
        if len(cells) == target_area and _shape_has_unique_side(cells, unique_side):
            return cells

    return {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def _shape_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _place_shape(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    marker: tuple[int, int],
    shape: set[tuple[int, int]],
    color: int,
) -> bool:
    r0, c0, r1, c1 = room
    sh, sw = _shape_size(shape)
    positions = [
        (r, c)
        for r in range(r0, r1 - sh + 2)
        for c in range(c0, c1 - sw + 2)
    ]
    random.shuffle(positions)

    for pr, pc in positions:
        cells = {(pr + dr, pc + dc) for dr, dc in shape}
        if marker in cells:
            continue
        if all(int(grid[r, c]) == BG for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def _background_is_clear_mode(grid: np.ndarray) -> bool:
    counts = {int(v): int(np.count_nonzero(grid == v)) for v in np.unique(grid)}
    return counts.get(BG, 0) > max(count for color, count in counts.items() if color != BG)


def _gen_attempt() -> np.ndarray | None:
    grid, row_doors, col_doors, rooms = _gen_base_grid()
    h, w = grid.shape
    colors = list(OBJECT_COLORS)
    random.shuffle(colors)

    plans: list[tuple[tuple[int, int, int, int], str, bool]] = []
    has_edge_move = False
    has_divider_move = False
    for room in rooms:
        edge_dirs, divider_dirs = _gen_allowed_dirs(room, h, w, row_doors, col_doors)
        if not edge_dirs and not divider_dirs:
            return None

        if edge_dirs and divider_dirs:
            use_divider = random.random() < 0.5
            direction = random.choice(divider_dirs if use_divider else edge_dirs)
        elif divider_dirs:
            use_divider = True
            direction = random.choice(divider_dirs)
        else:
            use_divider = False
            direction = random.choice(edge_dirs)

        has_edge_move |= not use_divider
        has_divider_move |= use_divider
        plans.append((room, direction, use_divider))

    if not (has_edge_move and has_divider_move):
        return None

    for idx, (room, direction, use_divider) in enumerate(plans):
        marker = _gen_marker(room, direction)
        if int(grid[marker]) != BG:
            return None
        grid[marker] = MAROON

        unique_side = OPPOSITE[direction] if use_divider else direction
        shape = _gen_shape(unique_side)
        color = colors[idx % len(colors)]
        if not _place_shape(grid, room, marker, shape, color):
            return None

    if not _background_is_clear_mode(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.any(out == MAROON):
        return None
    if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-escape input")
