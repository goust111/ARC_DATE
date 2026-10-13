from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ROOM_GRID_CHOICES = ((3, 3), (3, 4), (4, 3), (4, 4))
ROOM_SIZE_RANGE = (2, 5)
MARGIN_RANGE = (0, 2)
GEN_ATTEMPTS = 300
SYMMETRIES = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti")


def _wall_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        line = grid[r] == WALL
        if np.any(line[:-1] & line[1:]):
            rows.append(int(r))
    return rows


def _wall_cols(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        line = grid[:, c] == WALL
        if np.any(line[:-1] & line[1:]):
            cols.append(int(c))
    return cols


def _valid_boundary(line: np.ndarray) -> bool:
    vals = set(map(int, np.unique(line)))
    return vals <= {BG, WALL} and WALL in vals


def _extract_rooms(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    wall_rows = _wall_rows(grid)
    wall_cols = _wall_cols(grid)
    candidates: list[dict] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()

    for top_idx, top in enumerate(wall_rows):
        for bottom in wall_rows[top_idx + 1 :]:
            if bottom - top < 2:
                continue
            for left_idx, left in enumerate(wall_cols):
                for right in wall_cols[left_idx + 1 :]:
                    if right - left < 2:
                        continue
                    bbox = (int(top), int(left), int(bottom), int(right))
                    if bbox in seen_bboxes:
                        continue
                    if (
                        int(grid[top, left]) != WALL
                        or int(grid[top, right]) != WALL
                        or int(grid[bottom, left]) != WALL
                        or int(grid[bottom, right]) != WALL
                    ):
                        continue

                    interior = grid[top + 1 : bottom, left + 1 : right]
                    if interior.size == 0 or np.any(interior == WALL):
                        continue

                    if not _valid_boundary(grid[top, left : right + 1]):
                        continue
                    if not _valid_boundary(grid[bottom, left : right + 1]):
                        continue
                    if not _valid_boundary(grid[top : bottom + 1, left]):
                        continue
                    if not _valid_boundary(grid[top : bottom + 1, right]):
                        continue

                    candidates.append(
                        {
                            "wall_bbox": bbox,
                            "interior_bbox": (top + 1, left + 1, bottom - 1, right - 1),
                            "area": int(interior.shape[0] * interior.shape[1]),
                        }
                    )
                    seen_bboxes.add(bbox)

    candidates.sort(key=lambda room: (room["area"], room["interior_bbox"]))
    rooms: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    cell_to_room: dict[tuple[int, int], int] = {}

    for room in candidates:
        r0, c0, r1, c1 = room["interior_bbox"]
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if any(cell in occupied for cell in cells):
            continue
        idx = len(rooms)
        room["cells"] = cells
        rooms.append(room)
        for cell in cells:
            occupied.add(cell)
            cell_to_room[cell] = idx

    rooms.sort(key=lambda room: room["interior_bbox"])
    remapped: dict[tuple[int, int], int] = {}
    for idx, room in enumerate(rooms):
        for cell in room["cells"]:
            remapped[cell] = idx
    return rooms, remapped


def _classify_room(grid: np.ndarray, room: dict) -> tuple[str, int | None]:
    r0, c0, r1, c1 = room["interior_bbox"]
    interior = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals = set(map(int, np.unique(interior)))
    if vals == {BG}:
        return "empty", None
    if len(vals) == 1 and BG not in vals:
        return "source", next(iter(vals))
    return "other", None


def _collect_doors(grid: np.ndarray, room: dict, cell_to_room: dict[tuple[int, int], int]) -> list[dict]:
    top, left, bottom, right = room["wall_bbox"]
    doors: list[dict] = []

    for c in range(left + 1, right):
        if int(grid[top, c]) == BG:
            doors.append({"side": "top", "pos": (top, c), "neighbor": cell_to_room.get((top - 1, c))})
    for c in range(left + 1, right):
        if int(grid[bottom, c]) == BG:
            doors.append({"side": "bottom", "pos": (bottom, c), "neighbor": cell_to_room.get((bottom + 1, c))})
    for r in range(top + 1, bottom):
        if int(grid[r, left]) == BG:
            doors.append({"side": "left", "pos": (r, left), "neighbor": cell_to_room.get((r, left - 1))})
    for r in range(top + 1, bottom):
        if int(grid[r, right]) == BG:
            doors.append({"side": "right", "pos": (r, right), "neighbor": cell_to_room.get((r, right + 1))})
    return doors


def _paint_pattern(out: np.ndarray, room: dict, door_count: int, color: int) -> None:
    r0, c0, r1, c1 = room["interior_bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for dr in range(height):
        for dc in range(width):
            fill = False
            if door_count == 1:
                fill = True
            elif door_count == 2:
                fill = dr % 2 == 1
            elif door_count == 3:
                fill = (dr + dc) % 2 == 0
            else:
                fill = dr in (0, height - 1) or dc in (0, width - 1)
            if fill:
                out[r0 + dr, c0 + dc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular rooms bounded by grey walls. For each empty room, use the
    number of wall gaps as the pattern selector and use adjacent source rooms seen
    through those gaps as the color vote.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms, cell_to_room = _extract_rooms(g)
    kinds: list[tuple[str, int | None]] = [_classify_room(g, room) for room in rooms]
    door_lists = [_collect_doors(g, room, cell_to_room) for room in rooms]

    for idx, room in enumerate(rooms):
        kind, _ = kinds[idx]
        if kind != "empty":
            continue

        doors = door_lists[idx]
        if not doors:
            continue

        votes: dict[int, int] = {}
        first_seen: dict[int, int] = {}
        for order, door in enumerate(doors):
            neighbor = door["neighbor"]
            if neighbor is None:
                continue
            neighbor_kind, neighbor_color = kinds[neighbor]
            if neighbor_kind != "source" or neighbor_color is None:
                continue
            color = int(neighbor_color)
            votes[color] = votes.get(color, 0) + 1
            first_seen.setdefault(color, order)

        if not votes:
            continue

        fill_color = max(votes, key=lambda color: (votes[color], -first_seen[color]))
        _paint_pattern(out, room, len(doors), fill_color)

    return out


def _symmetry_pos(pos: tuple[int, int], symmetry: str) -> tuple[int, int]:
    r, c = pos
    if symmetry == "id":
        return r, c
    if symmetry == "rot90":
        return c, 2 - r
    if symmetry == "rot180":
        return 2 - r, 2 - c
    if symmetry == "rot270":
        return 2 - c, r
    if symmetry == "flip_h":
        return r, 2 - c
    if symmetry == "flip_v":
        return 2 - r, c
    if symmetry == "diag":
        return c, r
    if symmetry == "anti":
        return 2 - c, 2 - r
    raise ValueError(symmetry)


def _build_geometry(
    room_rows: int,
    room_cols: int,
    heights: list[int],
    widths: list[int],
    margin_top: int,
    margin_left: int,
    margin_bottom: int,
    margin_right: int,
) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    h_lines = [int(margin_top)]
    for height in heights:
        h_lines.append(h_lines[-1] + int(height) + 1)
    v_lines = [int(margin_left)]
    for width in widths:
        v_lines.append(v_lines[-1] + int(width) + 1)

    grid_h = h_lines[-1] + int(margin_bottom) + 1
    grid_w = v_lines[-1] + int(margin_right) + 1
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for row in h_lines:
        grid[row, v_lines[0] : v_lines[-1] + 1] = WALL
    for col in v_lines:
        grid[h_lines[0] : h_lines[-1] + 1, col] = WALL

    room_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr in range(room_rows):
        for cc in range(room_cols):
            room_boxes[(rr, cc)] = (
                h_lines[rr] + 1,
                v_lines[cc] + 1,
                h_lines[rr + 1] - 1,
                v_lines[cc + 1] - 1,
            )
    return grid, h_lines, v_lines, room_boxes


def _fill_room(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _open_door(
    grid: np.ndarray,
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    h_lines: list[int],
    v_lines: list[int],
) -> None:
    ar, ac = room_a
    br, bc = room_b
    if ar == br and abs(ac - bc) == 1:
        row = random.randint(h_lines[ar] + 1, h_lines[ar + 1] - 1)
        col = v_lines[max(ac, bc)]
        grid[row, col] = BG
        return
    if ac == bc and abs(ar - br) == 1:
        row = h_lines[max(ar, br)]
        col = random.randint(v_lines[ac] + 1, v_lines[ac + 1] - 1)
        grid[row, col] = BG
        return
    raise ValueError((room_a, room_b))


def _normalize_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _role_template(
    offset_r: int,
    offset_c: int,
    symmetry: str,
) -> dict[str, tuple[int, int]]:
    base = {
        "src_a1": (0, 0),
        "d2": (0, 1),
        "src_b": (0, 2),
        "d3": (1, 0),
        "d4": (1, 1),
        "src_c1": (1, 2),
        "src_a2": (2, 0),
        "src_c2": (2, 1),
        "d1": (2, 2),
    }
    out: dict[str, tuple[int, int]] = {}
    for role, pos in base.items():
        rr, cc = _symmetry_pos(pos, symmetry)
        out[role] = (offset_r + rr, offset_c + cc)
    return out


def _expected_output(
    inp: np.ndarray,
    room_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    roles: dict[str, tuple[int, int]],
    d2_color: int,
    d1_color: int,
    color_a: int,
    color_c: int,
) -> np.ndarray:
    out = inp.copy()
    _paint_pattern(out, {"interior_bbox": room_boxes[roles["d1"]]}, 1, d1_color)
    _paint_pattern(out, {"interior_bbox": room_boxes[roles["d2"]]}, 2, d2_color)
    _paint_pattern(out, {"interior_bbox": room_boxes[roles["d3"]]}, 3, color_a)
    _paint_pattern(out, {"interior_bbox": room_boxes[roles["d4"]]}, 4, color_c)
    return out


def _adjacent_pairs(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if rr + 1 < room_rows:
                edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < room_cols:
                edges.append(((rr, cc), (rr, cc + 1)))
    return edges


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_GRID_CHOICES)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    margins = [random.randint(*MARGIN_RANGE) for _ in range(4)]
    grid, h_lines, v_lines, room_boxes = _build_geometry(
        room_rows,
        room_cols,
        heights,
        widths,
        margins[0],
        margins[1],
        margins[2],
        margins[3],
    )

    offset_r = random.randint(0, room_rows - 3)
    offset_c = random.randint(0, room_cols - 3)
    symmetry = random.choice(SYMMETRIES)
    roles = _role_template(offset_r, offset_c, symmetry)
    empty_rooms = {roles["d1"], roles["d2"], roles["d3"], roles["d4"]}

    color_a, color_b, color_c = random.sample(PALETTE, 3)
    room_colors: dict[tuple[int, int], int] = {}
    for rr in range(room_rows):
        for cc in range(room_cols):
            pos = (rr, cc)
            if pos in empty_rooms:
                continue
            room_colors[pos] = random.choice(PALETTE)

    room_colors[roles["src_a1"]] = color_a
    room_colors[roles["src_a2"]] = color_a
    room_colors[roles["src_b"]] = color_b
    room_colors[roles["src_c1"]] = color_c
    room_colors[roles["src_c2"]] = color_c

    for pos, color in room_colors.items():
        _fill_room(grid, room_boxes[pos], color)

    base_edges = [
        ("d4", "d2"),
        ("d4", "d3"),
        ("d4", "src_c1"),
        ("d4", "src_c2"),
        ("d3", "src_a1"),
        ("d3", "src_a2"),
    ]
    d2_side = random.choice(("src_a1", "src_b"))
    d1_side = random.choice(("src_c1", "src_c2"))

    open_edges = {
        _normalize_edge(roles[a], roles[b]) for a, b in base_edges
    }
    open_edges.add(_normalize_edge(roles["d2"], roles[d2_side]))
    open_edges.add(_normalize_edge(roles["d1"], roles[d1_side]))

    source_edges = []
    for a, b in _adjacent_pairs(room_rows, room_cols):
        if a in empty_rooms or b in empty_rooms:
            continue
        edge = _normalize_edge(a, b)
        if edge in open_edges:
            continue
        source_edges.append(edge)
    random.shuffle(source_edges)
    for edge in source_edges[: random.randint(0, min(4, len(source_edges)))]:
        open_edges.add(edge)

    for room_a, room_b in open_edges:
        _open_door(grid, room_a, room_b, h_lines, v_lines)

    expected = _expected_output(
        grid,
        room_boxes,
        roles,
        color_a if d2_side == "src_a1" else color_b,
        color_c,
        color_a,
        color_c,
    )

    rooms, _ = _extract_rooms(grid)
    if len(rooms) != room_rows * room_cols:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid room-door ARC input")
