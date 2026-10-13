from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TRANSFORMS = ("id", "flip_lr", "flip_ud", "rot180")
GEN_MAX_ATTEMPTS = 300
ROW_HEIGHT_RANGE = (3, 6)
COL_WIDTH_RANGE = (3, 6)
START_EXTRA_WIDTH_RANGE = (0, 3)


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "expected a unique mode"
    return int(ranked[0][0])


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = map(int, mask.shape)
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue

            room_id = len(components)
            stack = [(r, c)]
            labels[r, c] = room_id
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = room_id
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return labels, components


def _find_doors(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], str]]:
    h, w = map(int, grid.shape)
    doors: list[tuple[tuple[int, int], str]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(bg):
                continue

            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])

            if up == BLACK and down == BLACK and left != BLACK and right != BLACK:
                doors.append(((r, c), "vertical"))
            elif left == BLACK and right == BLACK and up != BLACK and down != BLACK:
                doors.append(((r, c), "horizontal"))

    return doors


def _extract_layout(grid: np.ndarray) -> tuple[dict[object, dict], object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _unique_mode(g)
    doors = _find_doors(g, bg)
    sealed = g.copy()
    for (r, c), _orientation in doors:
        sealed[r, c] = BLACK

    labels, components = _components(sealed != BLACK)
    rooms: dict[object, dict] = {}
    start_candidates: list[object] = []

    for room_id, coords in enumerate(components):
        vals = g[coords[:, 0], coords[:, 1]]
        marker_mask = ~np.isin(vals, [bg, BLACK, GREY])
        marker_coords = coords[marker_mask]
        assert marker_coords.shape[0] == 1, "each room must contain exactly one colored marker"
        marker_color = int(g[marker_coords[0, 0], marker_coords[0, 1]])

        grey_coords = [tuple(map(int, pt)) for pt in coords[vals == GREY]]
        if grey_coords:
            start_candidates.append(room_id)
            grey_coords = sorted(grey_coords, key=lambda p: (p[0], p[1]))
            rows = {r for r, _c in grey_coords}
            cols = [c for _r, c in grey_coords]
            assert len(rows) == 1, "start-room blanks must form one horizontal row"
            assert cols == list(range(cols[0], cols[0] + len(cols))), "start-room blanks must be contiguous"

        rooms[room_id] = {
            "marker_color": marker_color,
            "grey_coords": grey_coords,
            "doors": [],
        }

    for door_id, ((r, c), orientation) in enumerate(doors):
        if orientation == "vertical":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "each door must separate two rooms"

        pos = (int(r), int(c))
        rooms[a]["doors"].append({"door_id": door_id, "other": b, "pos": pos})
        rooms[b]["doors"].append({"door_id": door_id, "other": a, "pos": pos})

    assert len(start_candidates) == 1, "expected exactly one start room"
    start_id = start_candidates[0]
    assert len(rooms[start_id]["doors"]) == 1, "start room must have exactly one door"
    assert len(rooms[start_id]["grey_coords"]) == len(rooms) - 1, "blank count must match other-room count"

    return rooms, start_id


def _simulate_walk(
    room_doors: dict[object, list[dict]],
    marker_by_room: dict[object, int],
    start_id: object,
) -> tuple[list[int], set[object], str, bool]:
    current = start_id
    incoming = None
    visited = {start_id}
    sequence: list[int] = []
    had_choice = False

    while True:
        eligible = [door for door in room_doors[current] if door["door_id"] != incoming]
        if not eligible:
            return sequence, visited, "dead_end", had_choice

        unvisited = [door for door in eligible if door["other"] not in visited]
        if not unvisited:
            return sequence, visited, "visited_exhausted", had_choice

        if len(unvisited) >= 2:
            had_choice = True

        chosen = min(unvisited, key=lambda door: (door["pos"][0], door["pos"][1]))
        current = chosen["other"]
        incoming = chosen["door_id"]
        visited.add(current)
        sequence.append(int(marker_by_room[current]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse rooms through wall doors, collect visited marker colors, and write the sequence into the start room's grey blanks.

    Input structure
    - The dominant color is the background.
    - Black one-pixel walls partition the grid into rectangular rooms.
    - Doorways are one-pixel gaps in shared walls.
    - Each room contains one unique colored marker pixel.
    - Exactly one start room also contains a horizontal run of grey blanks.

    Rule
    - Seal the doors conceptually to recover the individual rooms.
    - Start from the room with grey blanks and leave through its only door.
    - In each visited room, among exits other than the incoming door, choose the topmost door,
      breaking ties by the leftmost door, while avoiding revisiting rooms.
    - Write the encountered marker colors into the start-room grey blanks from left to right.
    """
    g = np.asarray(grid, dtype=int)
    rooms, start_id = _extract_layout(g)
    marker_by_room = {room_id: int(info["marker_color"]) for room_id, info in rooms.items()}
    room_doors = {room_id: list(info["doors"]) for room_id, info in rooms.items()}
    sequence, _visited, _reason, _had_choice = _simulate_walk(room_doors, marker_by_room, start_id)

    out = g.copy()
    grey_coords = rooms[start_id]["grey_coords"]
    assert len(sequence) <= len(grey_coords), "collected sequence must fit the blank row"
    for (r, c), color in zip(grey_coords, sequence):
        out[r, c] = int(color)
    return out


def _band_walls(sizes: list[int]) -> list[int]:
    walls = [0]
    for size in sizes:
        walls.append(int(walls[-1] + size + 1))
    return walls


def _draw_lattice(row_heights: list[int], col_widths: list[int], bg: int) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int]]:
    wall_rows = _band_walls(row_heights)
    wall_cols = _band_walls(col_widths)
    h = int(wall_rows[-1] + 1)
    w = int(wall_cols[-1] + 1)

    grid = np.full((h, w), int(bg), dtype=int)
    grid[wall_rows, :] = BLACK
    grid[:, wall_cols] = BLACK

    room_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr in range(len(row_heights)):
        for cc in range(len(col_widths)):
            room_boxes[(rr, cc)] = (
                int(wall_rows[rr] + 1),
                int(wall_cols[cc] + 1),
                int(wall_rows[rr + 1] - 1),
                int(wall_cols[cc + 1] - 1),
            )

    return grid, room_boxes, wall_rows, wall_cols


def _tree_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    edges = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    random.shuffle(edges)

    parent = {node: node for node in nodes}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for edge in edges:
        if union(edge[0], edge[1]):
            tree.append(edge)
        if len(tree) == len(nodes) - 1:
            break

    return tree


def _door_candidates(
    room_boxes: dict[object, tuple[int, int, int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    a: object,
    b: object,
) -> list[tuple[int, int]]:
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        raise AssertionError("lattice candidate lookup expects tuple room ids")

    ar, ac = a
    br, bc = b
    if ar == br and abs(ac - bc) == 1:
        wall_c = int(wall_cols[max(ac, bc)])
        box = room_boxes[a]
        return [(r, wall_c) for r in range(int(box[0]), int(box[2]) + 1)]
    if ac == bc and abs(ar - br) == 1:
        wall_r = int(wall_rows[max(ar, br)])
        box = room_boxes[a]
        return [(wall_r, c) for c in range(int(box[1]), int(box[3]) + 1)]
    raise AssertionError("rooms are not orthogonally adjacent")


def _transform_grid(grid: np.ndarray, code: str) -> np.ndarray:
    if code == "id":
        return grid.copy()
    if code == "flip_lr":
        return np.fliplr(grid).copy()
    if code == "flip_ud":
        return np.flipud(grid).copy()
    if code == "rot180":
        return np.flipud(np.fliplr(grid)).copy()
    raise AssertionError("unknown transform")


def _transform_point(point: tuple[int, int], shape: tuple[int, int], code: str) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, point)
    if code == "id":
        return r, c
    if code == "flip_lr":
        return r, w - 1 - c
    if code == "flip_ud":
        return h - 1 - r, c
    if code == "rot180":
        return h - 1 - r, w - 1 - c
    raise AssertionError("unknown transform")


def _random_box_position(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return random.randint(r0, r1), random.randint(c0, c1)


def _paint_room_contents(
    grid: np.ndarray,
    room_boxes: dict[object, tuple[int, int, int, int]],
    start_id: object,
    marker_by_room: dict[object, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = grid.copy()
    grey_len = len(room_boxes) - 1
    start_box = room_boxes[start_id]
    sr0, sc0, sr1, sc1 = map(int, start_box)
    width = sc1 - sc0 + 1
    assert width >= grey_len, "start room must fit the whole grey sequence"

    grey_row = random.randint(sr0, sr1)
    grey_col0 = random.randint(sc0, sc1 - grey_len + 1)
    grey_coords = [(grey_row, grey_col0 + i) for i in range(grey_len)]
    for r, c in grey_coords:
        out[r, c] = GREY

    for room_id, box in room_boxes.items():
        color = int(marker_by_room[room_id])
        if room_id == start_id:
            marker_rows = [r for r in range(sr0, sr1 + 1) if r != grey_row]
            assert marker_rows, "start room needs a separate marker row"
            marker_r = random.choice(marker_rows)
            marker_c = random.randint(sc0, sc1)
        else:
            marker_r, marker_c = _random_box_position(box)
        out[marker_r, marker_c] = color

    return out, grey_coords


def _build_room_doors(
    room_ids: list[object],
    edge_positions: list[tuple[int, object, object, tuple[int, int]]],
) -> dict[object, list[dict]]:
    room_doors: dict[object, list[dict]] = {room_id: [] for room_id in room_ids}
    for door_id, a, b, pos in edge_positions:
        room_doors[a].append({"door_id": door_id, "other": b, "pos": pos})
        room_doors[b].append({"door_id": door_id, "other": a, "pos": pos})
    return room_doors


def _make_expected(inp: np.ndarray, grey_coords: list[tuple[int, int]], sequence: list[int]) -> np.ndarray:
    out = inp.copy()
    ordered = sorted(grey_coords, key=lambda p: (p[0], p[1]))
    for (r, c), color in zip(ordered, sequence):
        out[r, c] = int(color)
    return out


def _gen_dead_end_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = random.choice(((2, 3), (3, 2)))
    room_count = rows * cols

    tree = _tree_edges(rows, cols)
    degrees = Counter()
    for a, b in tree:
        degrees[a] += 1
        degrees[b] += 1
    if max(degrees.values()) > 3:
        return None

    leaves = [node for node in degrees if int(degrees[node]) == 1]
    if not leaves:
        return None
    start_id = random.choice(leaves)

    bg = random.choice([c for c in range(10) if c not in (BLACK, GREY)])
    colors = random.sample([c for c in range(10) if c not in (BLACK, GREY, bg)], room_count)
    marker_by_room = {(r, c): color for (r, c), color in zip([(r, c) for r in range(rows) for c in range(cols)], colors)}

    row_heights = [random.randint(*ROW_HEIGHT_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*COL_WIDTH_RANGE) for _ in range(cols)]
    start_col = int(start_id[1])
    col_widths[start_col] = max(
        col_widths[start_col],
        room_count - 1 + random.randint(*START_EXTRA_WIDTH_RANGE),
    )

    grid, room_boxes, wall_rows, wall_cols = _draw_lattice(row_heights, col_widths, bg)
    edge_positions: list[tuple[int, object, object, tuple[int, int]]] = []
    for door_id, (a, b) in enumerate(tree):
        pos = random.choice(_door_candidates(room_boxes, wall_rows, wall_cols, a, b))
        grid[pos[0], pos[1]] = bg
        edge_positions.append((door_id, a, b, pos))

    inp, grey_coords = _paint_room_contents(grid, room_boxes, start_id, marker_by_room)
    transform = random.choice(TRANSFORMS)
    inp = _transform_grid(inp, transform)
    transformed_doors = [
        (door_id, a, b, _transform_point(pos, grid.shape, transform))
        for door_id, a, b, pos in edge_positions
    ]
    transformed_grey = [_transform_point(pos, grid.shape, transform) for pos in grey_coords]

    room_doors = _build_room_doors(list(room_boxes.keys()), transformed_doors)
    sequence, _visited, reason, had_choice = _simulate_walk(room_doors, marker_by_room, start_id)
    if reason != "dead_end" or not had_choice:
        return None

    expected = _make_expected(inp, transformed_grey, sequence)
    if _unique_mode(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def _gen_cycle_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice([c for c in range(10) if c not in (BLACK, GREY)])
    room_ids = ["S", "A", "B", "C", "D"]
    marker_colors = random.sample([c for c in range(10) if c not in (BLACK, GREY, bg)], len(room_ids))
    marker_by_room = dict(zip(room_ids, marker_colors))

    row_heights = [random.randint(*ROW_HEIGHT_RANGE) for _ in range(2)]
    col_widths = [random.randint(4, 7), random.randint(*COL_WIDTH_RANGE), random.randint(*COL_WIDTH_RANGE)]
    col_widths[0] = max(col_widths[0], len(room_ids) - 1 + random.randint(*START_EXTRA_WIDTH_RANGE))

    grid, cell_boxes, wall_rows, wall_cols = _draw_lattice(row_heights, col_widths, bg)
    split_row = int(wall_rows[1])
    start_col0 = int(wall_cols[0] + 1)
    start_col1 = int(wall_cols[1] - 1)
    grid[split_row, start_col0 : start_col1 + 1] = bg

    room_boxes: dict[object, tuple[int, int, int, int]] = {
        "S": (1, start_col0, int(grid.shape[0] - 2), start_col1),
        "A": cell_boxes[(0, 1)],
        "B": cell_boxes[(0, 2)],
        "C": cell_boxes[(1, 1)],
        "D": cell_boxes[(1, 2)],
    }

    def vertical_segment(left_room: object, right_room: object) -> list[tuple[int, int]]:
        left_box = room_boxes[left_room]
        right_box = room_boxes[right_room]
        wall_c = int(left_box[3] + 1)
        r0 = max(int(left_box[0]), int(right_box[0]))
        r1 = min(int(left_box[2]), int(right_box[2]))
        return [(r, wall_c) for r in range(r0, r1 + 1)]

    def horizontal_segment(top_room: object, bottom_room: object) -> list[tuple[int, int]]:
        top_box = room_boxes[top_room]
        bottom_box = room_boxes[bottom_room]
        wall_r = int(top_box[2] + 1)
        c0 = max(int(top_box[1]), int(bottom_box[1]))
        c1 = min(int(top_box[3]), int(bottom_box[3]))
        return [(wall_r, c) for c in range(c0, c1 + 1)]

    attach_room = random.choice(["A", "C"])
    edge_defs = [
        ("cycle_0", "A", "B", vertical_segment("A", "B")),
        ("cycle_1", "A", "C", horizontal_segment("A", "C")),
        ("cycle_2", "B", "D", horizontal_segment("B", "D")),
        ("cycle_3", "C", "D", vertical_segment("C", "D")),
        ("start", "S", attach_room, vertical_segment("S", attach_room)),
    ]

    edge_positions: list[tuple[int, object, object, tuple[int, int]]] = []
    for door_id, (_name, a, b, candidates) in enumerate(edge_defs):
        pos = random.choice(candidates)
        grid[pos[0], pos[1]] = bg
        edge_positions.append((door_id, a, b, pos))

    inp, grey_coords = _paint_room_contents(grid, room_boxes, "S", marker_by_room)
    transform = random.choice(TRANSFORMS)
    inp = _transform_grid(inp, transform)
    transformed_doors = [
        (door_id, a, b, _transform_point(pos, grid.shape, transform))
        for door_id, a, b, pos in edge_positions
    ]
    transformed_grey = [_transform_point(pos, grid.shape, transform) for pos in grey_coords]

    room_doors = _build_room_doors(room_ids, transformed_doors)
    sequence, _visited, reason, had_choice = _simulate_walk(room_doors, marker_by_room, "S")
    if reason != "visited_exhausted" or not had_choice:
        return None

    expected = _make_expected(inp, transformed_grey, sequence)
    if _unique_mode(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _gen_dead_end_case() if random.random() < 0.5 else _gen_cycle_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
