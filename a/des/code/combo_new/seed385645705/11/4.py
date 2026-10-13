from __future__ import annotations

import random

import numpy as np


BLACK = 0
AZURE = 8
GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_single_component(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    pts = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)


def _rotate_shape(shape: np.ndarray) -> np.ndarray:
    rows = shape[:, 0]
    cols = shape[:, 1]
    max_r = int(rows.max())
    return np.array([(int(c), int(max_r - r)) for r, c in shape], dtype=int)


def _mirror_shape(shape: np.ndarray) -> np.ndarray:
    cols = shape[:, 1]
    max_c = int(cols.max())
    return np.array([(int(r), int(max_c - c)) for r, c in shape], dtype=int)


def _random_shape(area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand in cells:
                        continue
                    test = cells | {cand}
                    rows = [rr for rr, _cc in test]
                    cols = [cc for _rr, cc in test]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        frontier.add(cand)
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        shape = _normalize_shape(cells)
        for _ in range(random.randint(0, 3)):
            shape = _rotate_shape(shape)
            shape = _normalize_shape({tuple(map(int, rc)) for rc in shape})
        if random.random() < 0.5:
            shape = _mirror_shape(shape)
            shape = _normalize_shape({tuple(map(int, rc)) for rc in shape})

        height = int(shape[:, 0].max()) + 1
        width = int(shape[:, 1].max()) + 1
        if height <= max_h and width <= max_w:
            return shape
    raise RuntimeError("failed to sample a connected shape")


def _wall_rows(grid: np.ndarray) -> list[int]:
    if grid.shape[1] < 2:
        return []
    return [
        r
        for r in range(grid.shape[0])
        if bool(np.any((grid[r, :-1] == AZURE) & (grid[r, 1:] == AZURE)))
    ]


def _wall_cols(grid: np.ndarray) -> list[int]:
    if grid.shape[0] < 2:
        return []
    return [
        c
        for c in range(grid.shape[1])
        if bool(np.any((grid[:-1, c] == AZURE) & (grid[1:, c] == AZURE)))
    ]


def _extract_rooms(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    wall_rows = _wall_rows(g)
    wall_cols = _wall_cols(g)
    assert wall_rows and wall_cols, "expected horizontal and vertical azure walls"

    row_bounds = [-1] + wall_rows + [g.shape[0]]
    col_bounds = [-1] + wall_cols + [g.shape[1]]
    rooms: dict[tuple[int, int], dict] = {}

    for ri in range(len(row_bounds) - 1):
        r0 = row_bounds[ri] + 1
        r1 = row_bounds[ri + 1] - 1
        if r0 > r1:
            continue
        for ci in range(len(col_bounds) - 1):
            c0 = col_bounds[ci] + 1
            c1 = col_bounds[ci + 1] - 1
            if c0 > c1:
                continue

            room = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = (room != BLACK) & (room != AZURE)
            if not bool(mask.any()):
                continue

            rel = np.argwhere(mask)
            abs_coords = rel + np.array([r0, c0], dtype=int)
            colors = {int(g[r, c]) for r, c in abs_coords}
            assert len(colors) == 1, "each room must contain exactly one monochrome pattern"
            assert _is_single_component(abs_coords), "room pattern must be 4-connected"

            if r0 > 0:
                assert not bool(mask[0].any()), "pattern must not touch the top wall"
            if r1 + 1 < g.shape[0]:
                assert not bool(mask[-1].any()), "pattern must not touch the bottom wall"
            if c0 > 0:
                assert not bool(mask[:, 0].any()), "pattern must not touch the left wall"
            if c1 + 1 < g.shape[1]:
                assert not bool(mask[:, -1].any()), "pattern must not touch the right wall"

            rooms[(ri, ci)] = {
                "bbox": (r0, c0, r1, c1),
                "coords": abs_coords.astype(int),
                "color": next(iter(colors)),
                "area": int(abs_coords.shape[0]),
            }

    assert len(rooms) >= 2, "expected at least two patterned rooms"
    return rooms, row_bounds, col_bounds


def _vertical_door(grid: np.ndarray, wall_c: int, r0: int, r1: int) -> tuple[int, int]:
    segment = grid[r0 : r1 + 1, wall_c]
    assert set(map(int, np.unique(segment))) <= {BLACK, AZURE}
    doors = np.flatnonzero(segment == BLACK)
    assert doors.size == 1, "each shared vertical wall must have exactly one door"
    return r0 + int(doors[0]), wall_c


def _horizontal_door(grid: np.ndarray, wall_r: int, c0: int, c1: int) -> tuple[int, int]:
    segment = grid[wall_r, c0 : c1 + 1]
    assert set(map(int, np.unique(segment))) <= {BLACK, AZURE}
    doors = np.flatnonzero(segment == BLACK)
    assert doors.size == 1, "each shared horizontal wall must have exactly one door"
    return wall_r, c0 + int(doors[0])


def _translate_copy(
    coords: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    direction: str,
    door: tuple[int, int],
) -> list[tuple[int, int]]:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    door_r, door_c = map(int, door)
    tr0, tc0, tr1, tc1 = map(int, target_bbox)

    if direction == "right":
        edge_c = int(cols.min())
        edge_rows = sorted(int(r) for r, c in coords if int(c) == edge_c)
        anchor = min(edge_rows, key=lambda r: (abs(r - door_r), r))
        dr = door_r - anchor
        dc = tc0 - edge_c
    elif direction == "left":
        edge_c = int(cols.max())
        edge_rows = sorted(int(r) for r, c in coords if int(c) == edge_c)
        anchor = min(edge_rows, key=lambda r: (abs(r - door_r), r))
        dr = door_r - anchor
        dc = tc1 - edge_c
    elif direction == "down":
        edge_r = int(rows.min())
        edge_cols = sorted(int(c) for r, c in coords if int(r) == edge_r)
        anchor = min(edge_cols, key=lambda c: (abs(c - door_c), c))
        dr = tr0 - edge_r
        dc = door_c - anchor
    else:
        assert direction == "up"
        edge_r = int(rows.max())
        edge_cols = sorted(int(c) for r, c in coords if int(r) == edge_r)
        anchor = min(edge_cols, key=lambda c: (abs(c - door_c), c))
        dr = tr1 - edge_r
        dc = door_c - anchor

    moved = [(int(r + dr), int(c + dc)) for r, c in coords]
    for r, c in moved:
        assert tr0 <= r <= tr1 and tc0 <= c <= tc1, "copied pattern must fit inside the target room"
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each smaller room pattern through its door into the adjacent larger room, aligned just inside the doorway.

    Input structure:
    - Background is black.
    - Azure one-cell-thick horizontal and vertical walls partition a rectangular enclosure into rectangular rooms.
    - Every shared wall segment between two neighboring rooms contains exactly one black one-pixel door.
    - Each room contains one 4-connected monochrome pattern whose color is neither black nor azure, and the pattern does not touch the room walls.

    Rule:
    - Recover the room lattice from the azure wall rows and columns, then find every adjacent room pair separated by a wall segment with one door.
    - Compare the two room-pattern areas; the smaller pattern is the source and the larger room is the target, while equal areas cause no change.
    - Copy the source pattern without rotation or reflection, translate it into the target room so the placed copy sits flush against the target side of the shared wall, and align the copy along that wall using the door coordinate.
    - Keep all original patterns, keep azure walls unchanged, and keep every door pixel black.

    Key insight:
    - The puzzle requires three linked detections: infer rooms from wall lines, interpret the black wall gaps as doors connecting room pairs, then use doorway-aligned translation of the smaller neighboring pattern into the larger room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms, row_bounds, col_bounds = _extract_rooms(g)
    out = g.copy()

    for (ri, ci), left_room in rooms.items():
        right_key = (ri, ci + 1)
        if right_key in rooms:
            wall_c = col_bounds[ci + 1]
            if 0 <= wall_c < g.shape[1]:
                r0, _c0, r1, _c1 = map(int, left_room["bbox"])
                door = _vertical_door(g, wall_c, r0, r1)
                right_room = rooms[right_key]
                left_area = int(left_room["area"])
                right_area = int(right_room["area"])
                if left_area < right_area:
                    placements = _translate_copy(left_room["coords"], right_room["bbox"], "right", door)
                    color = int(left_room["color"])
                elif right_area < left_area:
                    placements = _translate_copy(right_room["coords"], left_room["bbox"], "left", door)
                    color = int(right_room["color"])
                else:
                    placements = []
                    color = BLACK

                for r, c in placements:
                    if out[r, c] == BLACK:
                        out[r, c] = color

        down_key = (ri + 1, ci)
        if down_key in rooms:
            wall_r = row_bounds[ri + 1]
            if 0 <= wall_r < g.shape[0]:
                r0, c0, _r1, c1 = map(int, left_room["bbox"])
                door = _horizontal_door(g, wall_r, c0, c1)
                down_room = rooms[down_key]
                top_area = int(left_room["area"])
                down_area = int(down_room["area"])
                if top_area < down_area:
                    placements = _translate_copy(left_room["coords"], down_room["bbox"], "down", door)
                    color = int(left_room["color"])
                elif down_area < top_area:
                    placements = _translate_copy(down_room["coords"], left_room["bbox"], "up", door)
                    color = int(down_room["color"])
                else:
                    placements = []
                    color = BLACK

                for r, c in placements:
                    if out[r, c] == BLACK:
                        out[r, c] = color

    return out


def _area_template(room_count: int) -> list[int]:
    if room_count == 2:
        if random.random() < 0.5:
            template = random.choice([(1, 2), (2, 1)])
        else:
            template = (1, 1)
    else:
        strict = [(1, 2, 3), (3, 2, 1), (3, 1, 3)]
        equal = [(1, 1, 3), (3, 1, 1), (1, 3, 3), (3, 3, 1)]
        template = random.choice(strict if random.random() < 0.5 else equal)

    levels = sorted(set(template))
    values = sorted(random.sample([2, 3, 4, 5], len(levels)))
    mapping = {level: values[i] for i, level in enumerate(levels)}
    return [mapping[level] for level in template]


def _sample_structure() -> tuple[str, list[tuple[int, int, int, int]], np.ndarray]:
    orientation = random.choice(["horizontal", "vertical"])
    room_count = random.choice([2, 3])

    if orientation == "horizontal":
        room_h = random.randint(7, 10)
        width_hi = 10 if room_count == 2 else 8
        widths = [random.randint(7, width_hi) for _ in range(room_count)]
        struct_h = room_h + 2
        struct_w = sum(widths) + room_count + 1
        rooms: list[tuple[int, int, int, int]] = []
        c = 1
        for width in widths:
            rooms.append((1, c, room_h, c + width - 1))
            c += width + 1
    else:
        room_w = random.randint(7, 10)
        height_hi = 10 if room_count == 2 else 8
        heights = [random.randint(7, height_hi) for _ in range(room_count)]
        struct_h = sum(heights) + room_count + 1
        struct_w = room_w + 2
        rooms = []
        r = 1
        for height in heights:
            rooms.append((r, 1, r + height - 1, room_w))
            r += height + 1

    grid = np.full((struct_h, struct_w), BLACK, dtype=int)
    grid[0, :] = AZURE
    grid[-1, :] = AZURE
    grid[:, 0] = AZURE
    grid[:, -1] = AZURE

    if orientation == "horizontal":
        cursor = rooms[0][3] + 1
        for _ in range(room_count - 1):
            grid[:, cursor] = AZURE
            cursor += rooms[_ + 1][3] - rooms[_ + 1][1] + 2
    else:
        cursor = rooms[0][2] + 1
        for _ in range(room_count - 1):
            grid[cursor, :] = AZURE
            cursor += rooms[_ + 1][2] - rooms[_ + 1][0] + 2

    return orientation, rooms, grid


def _place_shape_in_room(
    room_bbox: tuple[int, int, int, int],
    shape: np.ndarray,
    orientation: str,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = map(int, room_bbox)
    sh = int(shape[:, 0].max()) + 1
    sw = int(shape[:, 1].max()) + 1

    if orientation == "horizontal":
        row_lo = r0 + 1
        row_hi = r1 - sh
        col_lo = c0 + 2
        col_hi = c1 - sw - 1
    else:
        row_lo = r0 + 2
        row_hi = r1 - sh - 1
        col_lo = c0 + 1
        col_hi = c1 - sw

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _shape_cells(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in shape]


def _copy_fits(
    source_cells: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    direction: str,
    door: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    try:
        placed = _translate_copy(source_cells, target_bbox, direction, door)
    except AssertionError:
        return False
    return not any(cell in occupied for cell in placed)


def _embed_with_padding(structure: np.ndarray, rooms: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    max_extra_h = 30 - structure.shape[0]
    max_extra_w = 30 - structure.shape[1]
    assert max_extra_h >= 0 and max_extra_w >= 0

    pad_top = random.randint(0, min(2, max_extra_h))
    pad_bottom = random.randint(0, min(2, max_extra_h - pad_top))
    pad_left = random.randint(0, min(2, max_extra_w))
    pad_right = random.randint(0, min(2, max_extra_w - pad_left))

    grid = np.full(
        (structure.shape[0] + pad_top + pad_bottom, structure.shape[1] + pad_left + pad_right),
        BLACK,
        dtype=int,
    )
    grid[pad_top : pad_top + structure.shape[0], pad_left : pad_left + structure.shape[1]] = structure
    shifted = [
        (r0 + pad_top, c0 + pad_left, r1 + pad_top, c1 + pad_left)
        for r0, c0, r1, c1 in rooms
    ]
    return grid, shifted


def _generate_attempt() -> np.ndarray | None:
    orientation, struct_rooms, structure = _sample_structure()
    grid, rooms = _embed_with_padding(structure, struct_rooms)
    room_count = len(rooms)
    areas = _area_template(room_count)
    colors = random.sample(GEN_COLORS, room_count)

    shapes: list[np.ndarray] = []
    placed_cells: list[np.ndarray] = []
    occupied_by_room: list[set[tuple[int, int]]] = []

    if orientation == "horizontal":
        max_h, max_w = 3, 2
    else:
        max_h, max_w = 2, 3

    for room_bbox, area, color in zip(rooms, areas, colors):
        shape = _random_shape(area, max_h=max_h, max_w=max_w)
        top_left = _place_shape_in_room(room_bbox, shape, orientation)
        if top_left is None:
            return None
        top, left = top_left
        cells = np.array(_shape_cells(shape, top, left), dtype=int)
        for r, c in cells:
            grid[r, c] = int(color)
        shapes.append(shape)
        placed_cells.append(cells)
        occupied_by_room.append({(int(r), int(c)) for r, c in cells})

    if orientation == "horizontal":
        for i in range(room_count - 1):
            wall_c = rooms[i][3] + 1
            rows = range(rooms[i][0], rooms[i][2] + 1)
            left_area = areas[i]
            right_area = areas[i + 1]
            if left_area == right_area:
                door_r = random.choice(tuple(rows))
            else:
                if left_area < right_area:
                    source_cells = placed_cells[i]
                    target_bbox = rooms[i + 1]
                    direction = "right"
                    occupied = occupied_by_room[i + 1]
                else:
                    source_cells = placed_cells[i + 1]
                    target_bbox = rooms[i]
                    direction = "left"
                    occupied = occupied_by_room[i]

                candidates = [
                    r
                    for r in rows
                    if _copy_fits(source_cells, target_bbox, direction, (r, wall_c), occupied)
                ]
                if not candidates:
                    return None
                door_r = random.choice(candidates)
            grid[door_r, wall_c] = BLACK
    else:
        for i in range(room_count - 1):
            wall_r = rooms[i][2] + 1
            cols = range(rooms[i][1], rooms[i][3] + 1)
            top_area = areas[i]
            bottom_area = areas[i + 1]
            if top_area == bottom_area:
                door_c = random.choice(tuple(cols))
            else:
                if top_area < bottom_area:
                    source_cells = placed_cells[i]
                    target_bbox = rooms[i + 1]
                    direction = "down"
                    occupied = occupied_by_room[i + 1]
                else:
                    source_cells = placed_cells[i + 1]
                    target_bbox = rooms[i]
                    direction = "up"
                    occupied = occupied_by_room[i]

                candidates = [
                    c
                    for c in cols
                    if _copy_fits(source_cells, target_bbox, direction, (wall_r, c), occupied)
                ]
                if not candidates:
                    return None
                door_c = random.choice(candidates)
            grid[wall_r, door_c] = BLACK

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid doorway-room sample after {GEN_RETRIES} attempts")
