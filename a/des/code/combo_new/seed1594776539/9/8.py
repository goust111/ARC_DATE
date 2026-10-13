import numpy as np


BLACK = 0
RED = 2
WALL = 9
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8)

_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400

_BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _normalize_shape(shape):
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_variants(shape):
    seen = set()
    cur = tuple(shape)
    for _ in range(4):
        cur = tuple((c, -r) for r, c in cur)
        for flip in (False, True):
            variant = tuple((r, -c) for r, c in cur) if flip else cur
            norm = _normalize_shape(variant)
            if norm not in seen:
                seen.add(norm)
                yield norm


SHAPES = tuple(
    variant
    for base in _BASE_SHAPES
    for variant in _shape_variants(base)
)


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_lines(counts):
    nz = counts[counts > 0]
    assert nz.size > 0, "expected wall pixels"
    low = int(nz.min())
    high = int(nz.max())
    assert high > low, "wall rows/cols must stand out from room-interior crossings"
    threshold = (low + high) // 2 + 1
    lines = np.flatnonzero(counts >= threshold)
    assert lines.size >= 3, "expected at least a 2x2 room grid"
    assert np.all(np.diff(lines) > 1), "wall lines must be one-pixel thick"
    return [int(x) for x in lines]


def _extract_object(g, ir0, ic0, ir1, ic1):
    patch = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    mask = patch != BLACK
    assert np.all(patch[mask] != WALL), "room interiors must not contain wall-colored pixels"
    coords = np.argwhere(mask)
    assert 2 <= coords.shape[0] <= 5, "each room must contain one small 2-5-cell object"
    colors = np.unique(patch[mask])
    assert colors.size == 1, "room object must be monochrome"
    comps = _components(mask)
    assert len(comps) == 1, "room object must be 4-connected"
    cells = [(int(ir0 + r), int(ic0 + c)) for r, c in coords]
    return int(colors[0]), cells


def _door_cells(g, r0, c0, r1, c1):
    doors = []
    for c in range(c0 + 1, c1):
        if int(g[r0, c]) == BLACK:
            doors.append((r0, c, -1, 0))
        if int(g[r1, c]) == BLACK:
            doors.append((r1, c, 1, 0))
    for r in range(r0 + 1, r1):
        if int(g[r, c0]) == BLACK:
            doors.append((r, c0, 0, -1))
        if int(g[r, c1]) == BLACK:
            doors.append((r, c1, 0, 1))
    return doors


def _analyze_rooms(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_counts = np.sum(g == WALL, axis=1)
    col_counts = np.sum(g == WALL, axis=0)
    wall_rows = _infer_lines(row_counts)
    wall_cols = _infer_lines(col_counts)

    rooms = []
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            r0, r1 = wall_rows[i], wall_rows[i + 1]
            c0, c1 = wall_cols[j], wall_cols[j + 1]
            assert r1 - r0 >= 3 and c1 - c0 >= 3, "each room needs interior space"
            color, cells = _extract_object(g, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
            doors = _door_cells(g, r0, c0, r1, c1)
            assert len(doors) <= 3, "room door count must be 0-3"
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                    "color": color,
                    "cells": cells,
                    "doors": doors,
                }
            )
    return g, rooms


def _reflect_cells(cells, interior):
    ir0, ic0, ir1, ic1 = interior
    return [(int(ir0 + ir1 - r), int(ic0 + ic1 - c)) for r, c in cells]


def _ray_path(g, door):
    h, w = g.shape
    r, c, dr, dc = door
    path = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and int(g[r, c]) != WALL:
        path.append((int(r), int(c)))
        r += dr
        c += dc
    return path


def _solve(g, rooms):
    out = g.copy()
    ray_specs = []
    for room in rooms:
        door_count = len(room["doors"])
        assert door_count in {0, 1, 2, 3}, "unsupported door count"
        cells = room["cells"]
        color = room["color"]
        if door_count == 0:
            for r, c in cells:
                out[r, c] = BLACK
            ray_color = None
        elif door_count == 1:
            ray_color = color
        elif door_count == 2:
            for r, c in _reflect_cells(cells, room["interior"]):
                out[r, c] = color
            ray_color = color
        else:
            for r, c in cells:
                out[r, c] = RED
            ray_color = RED

        if ray_color is not None:
            for door in room["doors"]:
                ray_specs.append((door, ray_color))

    for door, ray_color in ray_specs:
        for r, c in _ray_path(g, door):
            out[r, c] = ray_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each room object according to door count, then project door-aligned rays outward.

    Input structure:
    - The background is black.
    - Maroon one-pixel horizontal and vertical wall lines partition a larger rectangle into axis-aligned rectangular rooms.
    - Each room wall may contain 0-3 one-pixel black gaps that act as doors.
    - Every room interior contains exactly one small monochrome 4-connected object of size 2-5.

    Rule:
    - Count the doors on each room boundary.
    - Rooms with 0 doors lose their object; rooms with 1 door keep it unchanged.
    - Rooms with 2 doors gain a reflected copy of the object at the point-symmetric position across the room center.
    - Rooms with 3 doors recolor their object to red.
    - For every door, draw a one-pixel ray starting just outside the door and extending straight outward until the next maroon wall or the grid edge; the ray uses the room object's resulting color.

    Key insight:
    - Door count and door position are independent clues: count chooses the in-room object transformation, while the door locations choose where outward rays are cast.
    """
    g, rooms = _analyze_rooms(grid)
    return _solve(g, rooms)


def _sample_layout():
    for _ in range(100):
        room_rows, room_cols = ((3, 3), (2, 3), (3, 2))[int(_RNG.integers(0, 3))]
        heights = _RNG.integers(5, 8, size=room_rows)
        widths = _RNG.integers(5, 8, size=room_cols)
        top, bottom, left, right = _RNG.integers(1, 3, size=4)
        wall_rows = [int(top)]
        wall_cols = [int(left)]
        for h in heights:
            wall_rows.append(wall_rows[-1] + int(h) + 1)
        for w in widths:
            wall_cols.append(wall_cols[-1] + int(w) + 1)
        height = wall_rows[-1] + int(bottom) + 1
        width = wall_cols[-1] + int(right) + 1
        if height <= 30 and width <= 30:
            return room_rows, room_cols, heights, widths, wall_rows, wall_cols, height, width
    raise RuntimeError("failed to sample a valid room layout")


def _door_counts(open_h, open_v):
    room_rows = open_v.shape[0]
    room_cols = open_h.shape[1]
    counts = np.zeros((room_rows, room_cols), dtype=int)
    for i in range(room_rows):
        for j in range(room_cols):
            counts[i, j] = (
                int(open_h[i, j])
                + int(open_h[i + 1, j])
                + int(open_v[i, j])
                + int(open_v[i, j + 1])
            )
    return counts


def _sample_doors(room_rows, room_cols):
    for _ in range(500):
        p = float(_RNG.uniform(0.22, 0.55))
        open_h = _RNG.random((room_rows + 1, room_cols)) < p
        open_v = _RNG.random((room_rows, room_cols + 1)) < p
        counts = _door_counts(open_h, open_v)
        values = set(map(int, counts.ravel()))
        outer_open = bool(open_h[0].any() or open_h[-1].any() or open_v[:, 0].any() or open_v[:, -1].any())
        inner_open = bool(
            (room_rows > 1 and open_h[1:room_rows].any())
            or (room_cols > 1 and open_v[:, 1:room_cols].any())
        )
        if counts.max() <= 3 and {0, 1, 2, 3}.issubset(values) and outer_open and inner_open:
            return open_h, open_v, counts
    return None


def _paint_walls(grid, wall_rows, wall_cols):
    top, bottom = wall_rows[0], wall_rows[-1]
    left, right = wall_cols[0], wall_cols[-1]
    for r in wall_rows:
        grid[r, left : right + 1] = WALL
    for c in wall_cols:
        grid[top : bottom + 1, c] = WALL


def _door_positions(open_h, open_v, wall_rows, wall_cols):
    h_pos = {}
    v_pos = {}
    room_rows, room_cols = open_v.shape[0], open_h.shape[1]
    for i in range(room_rows + 1):
        for j in range(room_cols):
            if open_h[i, j]:
                h_pos[(i, j)] = int(_RNG.integers(wall_cols[j] + 1, wall_cols[j + 1]))
    for i in range(room_rows):
        for j in range(room_cols + 1):
            if open_v[i, j]:
                v_pos[(i, j)] = int(_RNG.integers(wall_rows[i] + 1, wall_rows[i + 1]))
    return h_pos, v_pos


def _carve_doors(grid, wall_rows, wall_cols, open_h, open_v, h_pos, v_pos):
    room_rows, room_cols = open_v.shape[0], open_h.shape[1]
    for i in range(room_rows + 1):
        for j in range(room_cols):
            if open_h[i, j]:
                grid[wall_rows[i], h_pos[(i, j)]] = BLACK
    for i in range(room_rows):
        for j in range(room_cols + 1):
            if open_v[i, j]:
                grid[v_pos[(i, j)], wall_cols[j]] = BLACK


def _room_forbidden_lines(i, j, open_h, open_v, h_pos, v_pos):
    forbidden_rows = set()
    forbidden_cols = set()
    if open_v[i, j]:
        forbidden_rows.add(v_pos[(i, j)])
    if open_v[i, j + 1]:
        forbidden_rows.add(v_pos[(i, j + 1)])
    if open_h[i, j]:
        forbidden_cols.add(h_pos[(i, j)])
    if open_h[i + 1, j]:
        forbidden_cols.add(h_pos[(i + 1, j)])
    return forbidden_rows, forbidden_cols


def _place_shape(ir0, ic0, ir1, ic1, forbidden_rows, forbidden_cols, door_count):
    shape_indices = _RNG.permutation(len(SHAPES))
    for idx in shape_indices:
        shape = SHAPES[int(idx)]
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        row_starts = list(range(ir0, ir1 - max_r + 1))
        col_starts = list(range(ic0, ic1 - max_c + 1))
        _RNG.shuffle(row_starts)
        _RNG.shuffle(col_starts)
        for base_r in row_starts:
            for base_c in col_starts:
                cells = [(base_r + r, base_c + c) for r, c in shape]
                if any(r in forbidden_rows or c in forbidden_cols for r, c in cells):
                    continue
                if door_count == 2:
                    mirrored = _reflect_cells(cells, (ir0, ic0, ir1, ic1))
                    if set(cells) & set(mirrored):
                        continue
                    if any(r in forbidden_rows or c in forbidden_cols for r, c in mirrored):
                        continue
                return cells
    return None


def _clear_output(rooms, g):
    occupied = {}
    for room in rooms:
        door_count = len(room["doors"])
        color = RED if door_count == 3 else room["color"]
        if door_count == 0:
            final_cells = []
        elif door_count == 2:
            final_cells = list(room["cells"]) + _reflect_cells(room["cells"], room["interior"])
        else:
            final_cells = list(room["cells"])
        if len(final_cells) != len(set(final_cells)):
            return False
        for cell in final_cells:
            if cell in occupied:
                return False
            occupied[cell] = color

    for room in rooms:
        door_count = len(room["doors"])
        if door_count == 0:
            continue
        for door in room["doors"]:
            for cell in _ray_path(g, door):
                if cell in occupied:
                    return False
                occupied[cell] = room["color"] if door_count != 3 else RED
    return True


def _gen_attempt():
    room_rows, room_cols, _heights, _widths, wall_rows, wall_cols, height, width = _sample_layout()
    sampled = _sample_doors(room_rows, room_cols)
    if sampled is None:
        return None
    open_h, open_v, counts = sampled

    grid = np.zeros((height, width), dtype=int)
    _paint_walls(grid, wall_rows, wall_cols)
    h_pos, v_pos = _door_positions(open_h, open_v, wall_rows, wall_cols)
    _carve_doors(grid, wall_rows, wall_cols, open_h, open_v, h_pos, v_pos)

    room_order = [(i, j) for i in range(room_rows) for j in range(room_cols)]
    room_order.sort(key=lambda ij: int(counts[ij]), reverse=True)

    for i, j in room_order:
        r0, r1 = wall_rows[i], wall_rows[i + 1]
        c0, c1 = wall_cols[j], wall_cols[j + 1]
        forbidden_rows, forbidden_cols = _room_forbidden_lines(i, j, open_h, open_v, h_pos, v_pos)
        cells = _place_shape(r0 + 1, c0 + 1, r1 - 1, c1 - 1, forbidden_rows, forbidden_cols, int(counts[i, j]))
        if cells is None:
            return None
        color = int(_RNG.choice(OBJECT_COLORS))
        for r, c in cells:
            grid[r, c] = color

    try:
        g, rooms = _analyze_rooms(grid)
    except AssertionError:
        return None

    door_values = [len(room["doors"]) for room in rooms]
    if set(door_values) != {0, 1, 2, 3}:
        return None
    if not _clear_output(rooms, g):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-door sample")
