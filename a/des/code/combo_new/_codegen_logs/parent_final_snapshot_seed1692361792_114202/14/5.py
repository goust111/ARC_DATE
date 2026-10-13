import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]

GEN_MIN_BANDS = 2
GEN_MAX_BANDS = 4
GEN_ROOM_SIZE_RANGE = (4, 6)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == len(cells)


def _wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] != BLACK)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != BLACK)]
    assert len(rows) >= 2 and len(cols) >= 2
    assert rows[0] == 0 and rows[-1] == grid.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    return rows, cols


def _room_ids(wall_rows: list[int], wall_cols: list[int]):
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            yield (i, j)


def _room_bounds(wall_rows: list[int], wall_cols: list[int], room_id: tuple[int, int]) -> tuple[int, int, int, int]:
    i, j = room_id
    return wall_rows[i], wall_rows[i + 1], wall_cols[j], wall_cols[j + 1]


def _extract_room(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int], room_id: tuple[int, int]) -> dict:
    top, bottom, left, right = _room_bounds(wall_rows, wall_cols, room_id)
    interior = grid[top + 1 : bottom, left + 1 : right]
    cells = np.argwhere(interior != BLACK)
    assert len(cells) > 0
    colors = np.unique(interior[interior != BLACK])
    assert len(colors) == 1
    color = int(colors[0])
    assert color not in (BLACK, YELLOW)
    mask = interior == color
    assert _is_connected(mask)
    coords = cells + np.array([top + 1, left + 1])
    return {"bounds": (top, bottom, left, right), "color": color, "coords": coords}


def _segment_window(segment: np.ndarray) -> int | None:
    window_cells = np.argwhere(segment != YELLOW)
    assert len(window_cells) <= 1
    if len(window_cells) == 0:
        return None
    color = int(segment[tuple(window_cells[0])])
    assert color not in (BLACK, YELLOW)
    return color


def _parse_scene(grid: np.ndarray):
    wall_rows, wall_cols = _wall_lines(grid)
    rooms = {
        room_id: _extract_room(grid, wall_rows, wall_cols, room_id)
        for room_id in _room_ids(wall_rows, wall_cols)
    }
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {
        room_id: [] for room_id in rooms
    }
    incident_windows: dict[tuple[int, int], list[int]] = {room_id: [] for room_id in rooms}

    for i in range(len(wall_rows) - 1):
        top = wall_rows[i]
        bottom = wall_rows[i + 1]
        for j in range(len(wall_cols) - 2):
            left_room = (i, j)
            right_room = (i, j + 1)
            wall_c = wall_cols[j + 1]
            segment = grid[top + 1 : bottom, wall_c]
            color = _segment_window(segment)
            if color is None:
                continue
            adjacency[left_room].append((right_room, color))
            adjacency[right_room].append((left_room, color))
            incident_windows[left_room].append(color)
            incident_windows[right_room].append(color)

    for i in range(len(wall_rows) - 2):
        wall_r = wall_rows[i + 1]
        for j in range(len(wall_cols) - 1):
            top_room = (i, j)
            bottom_room = (i + 1, j)
            left = wall_cols[j]
            right = wall_cols[j + 1]
            segment = grid[wall_r, left + 1 : right]
            color = _segment_window(segment)
            if color is None:
                continue
            adjacency[top_room].append((bottom_room, color))
            adjacency[bottom_room].append((top_room, color))
            incident_windows[top_room].append(color)
            incident_windows[bottom_room].append(color)

    return wall_rows, wall_cols, rooms, adjacency, incident_windows


def _choose_source(rooms: dict, incident_windows: dict) -> tuple[int, int]:
    matching_candidates = [
        room_id
        for room_id, room in rooms.items()
        if incident_windows[room_id]
        and all(color == room["color"] for color in incident_windows[room_id])
    ]
    if len(matching_candidates) == 1:
        return matching_candidates[0]

    zero_window_candidates = [
        room_id for room_id, colors in incident_windows.items() if not colors
    ]
    assert len(zero_window_candidates) == 1
    return zero_window_candidates[0]


def _propagated_colors(rooms: dict, adjacency: dict, source: tuple[int, int]) -> dict[tuple[int, int], int]:
    colors = {source: int(rooms[source]["color"])}
    queue = deque([source])
    while queue:
        room_id = queue.popleft()
        current = colors[room_id]
        for neighbor, window_color in adjacency[room_id]:
            if neighbor in colors:
                continue
            colors[neighbor] = current if window_color == current else int(window_color)
            queue.append(neighbor)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each room's inner motif by propagating colors through windowed shared walls.

    Input structure:
    - The background is black, and full yellow rows and columns partition the grid into rectangular rooms.
    - Each room contains one connected monochrome motif separated from the yellow walls by black space.
    - Adjacent rooms may share a single non-yellow window pixel on their common yellow wall segment.
    - Exactly one room acts as the seed: either it has no windows, or every window on its boundary matches its motif color.

    Rule:
    - Keep the seed room's motif color unchanged.
    - Traverse the room graph through windowed shared walls starting from that seed.
    - When a room of current color c reaches an unseen neighbor through a window of color w, repaint the neighbor motif to c if w == c, otherwise repaint it to w.
    - Rooms unreachable from the seed keep their original motif colors, and all yellow walls and window pixels stay unchanged.

    Key insight:
    - The task requires separating wall geometry from room interiors, then following color propagation on the room adjacency graph rather than treating windows as ordinary pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _wall_rows, _wall_cols, rooms, adjacency, incident_windows = _parse_scene(g)
    source = _choose_source(rooms, incident_windows)
    new_colors = _propagated_colors(rooms, adjacency, source)

    out = g.copy()
    for room_id, color in new_colors.items():
        coords = rooms[room_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _sample_band_sizes(count: int) -> list[int]:
    return [random.randint(*GEN_ROOM_SIZE_RANGE) for _ in range(count)]


def _layout_from_sizes(heights: list[int], widths: list[int]) -> tuple[list[int], list[int]]:
    wall_rows = [0]
    for size in heights:
        wall_rows.append(wall_rows[-1] + size + 1)
    wall_cols = [0]
    for size in widths:
        wall_cols.append(wall_cols[-1] + size + 1)
    return wall_rows, wall_cols


def _neighbor_map(n_rows: int, n_cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            room_id = (i, j)
            neighbors = []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni = i + di
                nj = j + dj
                if 0 <= ni < n_rows and 0 <= nj < n_cols:
                    neighbors.append((ni, nj))
            out[room_id] = neighbors
    return out


def _grow_tree(
    root: tuple[int, int],
    target_size: int,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    allowed: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]] | None:
    if allowed is None:
        allowed = set(neighbors)
    if root not in allowed or target_size < 1 or target_size > len(allowed):
        return None

    chosen = {root}
    order = [root]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while len(chosen) < target_size:
        frontier = [
            (room_id, nbr)
            for room_id in chosen
            for nbr in neighbors[room_id]
            if nbr in allowed and nbr not in chosen
        ]
        if not frontier:
            return None
        room_id, nbr = random.choice(frontier)
        chosen.add(nbr)
        order.append(nbr)
        parent[nbr] = room_id
    return order, parent


def _line_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if height == 1 or (width > 1 and random.random() < 0.5):
        r = random.randrange(height)
        c0 = random.randrange(width)
        c1 = random.randrange(c0, width)
        mask[r, c0 : c1 + 1] = True
    else:
        c = random.randrange(width)
        r0 = random.randrange(height)
        r1 = random.randrange(r0, height)
        mask[r0 : r1 + 1, c] = True
    return mask


def _l_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    corner = random.choice(((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    r0, c0 = corner
    if r0 == 0:
        mask[0 : random.randint(1, height), c0] = True
    else:
        mask[random.randint(0, height - 1) : height, c0] = True
    if c0 == 0:
        mask[r0, 0 : random.randint(1, width)] = True
    else:
        mask[r0, random.randint(0, width - 1) : width] = True
    return mask


def _blob_mask(height: int, width: int) -> np.ndarray:
    target = random.randint(1, min(height * width, 6))
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        boundary = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, height, width):
                if (nr, nc) not in cells:
                    boundary.append((nr, nc))
        if not boundary:
            break
        cells.add(random.choice(boundary))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_mask(height: int, width: int) -> np.ndarray:
    kinds = ["blob", "rect"]
    if height > 1 or width > 1:
        kinds.append("line")
    if height > 1 and width > 1:
        kinds.append("l")
    kind = random.choice(kinds)
    if kind == "rect":
        return np.ones((height, width), dtype=bool)
    if kind == "line":
        return _line_mask(height, width)
    if kind == "l":
        return _l_mask(height, width)
    return _blob_mask(height, width)


def _place_pattern(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: int,
) -> np.ndarray:
    top, bottom, left, right = bounds
    room_h = bottom - top - 1
    room_w = right - left - 1
    safe_h = room_h - 2
    safe_w = room_w - 2
    assert safe_h >= 1 and safe_w >= 1

    for _ in range(60):
        box_h = random.randint(1, safe_h)
        box_w = random.randint(1, safe_w)
        mask = _shape_mask(box_h, box_w)
        if not _is_connected(mask):
            continue
        r0 = top + 2 + random.randint(0, safe_h - box_h)
        c0 = left + 2 + random.randint(0, safe_w - box_w)
        coords = np.argwhere(mask) + np.array([r0, c0])
        grid[coords[:, 0], coords[:, 1]] = color
        return coords
    raise RuntimeError("failed to place room motif")


def _set_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    grid[wall_rows, :] = YELLOW
    grid[:, wall_cols] = YELLOW


def _paint_window(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    a: tuple[int, int],
    b: tuple[int, int],
    color: int,
) -> None:
    ai, aj = a
    bi, bj = b
    if ai == bi:
        i = ai
        wall_c = wall_cols[max(aj, bj)]
        top = wall_rows[i]
        bottom = wall_rows[i + 1]
        row = random.randint(top + 1, bottom - 1)
        grid[row, wall_c] = color
    else:
        j = aj
        wall_r = wall_rows[max(ai, bi)]
        left = wall_cols[j]
        right = wall_cols[j + 1]
        col = random.randint(left + 1, right - 1)
        grid[wall_r, col] = color


def _build_expected(
    inp: np.ndarray,
    room_coords: dict[tuple[int, int], np.ndarray],
    output_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = inp.copy()
    for room_id, color in output_colors.items():
        coords = room_coords[room_id]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _planned_windows(
    root: tuple[int, int],
    root_order: list[tuple[int, int]],
    root_parent: dict[tuple[int, int], tuple[int, int]],
) -> tuple[dict[frozenset, int], dict[tuple[int, int], int]]:
    root_color = random.choice(PALETTE)
    window_colors: dict[frozenset, int] = {}
    output_colors: dict[tuple[int, int], int] = {root: root_color}
    deep_children = [room_id for room_id in root_order[1:] if root_parent[room_id] != root]
    forced_mismatch = random.choice(deep_children) if deep_children else None

    for room_id in root_order[1:]:
        parent = root_parent[room_id]
        parent_color = output_colors[parent]
        if parent == root:
            window_color = parent_color
        else:
            if room_id == forced_mismatch:
                window_color = random.choice([c for c in PALETTE if c != parent_color])
            elif random.random() < 0.5:
                window_color = parent_color
            else:
                window_color = random.choice([c for c in PALETTE if c != parent_color])
        window_colors[frozenset((parent, room_id))] = window_color
        output_colors[room_id] = parent_color if window_color == parent_color else window_color

    return window_colors, output_colors


def _incident_windows(
    rooms: list[tuple[int, int]],
    edge_windows: dict[frozenset, int],
) -> dict[tuple[int, int], list[int]]:
    incident = {room_id: [] for room_id in rooms}
    for edge, color in edge_windows.items():
        a, b = tuple(edge)
        incident[a].append(int(color))
        incident[b].append(int(color))
    return incident


def _choose_input_color(
    room_id: tuple[int, int],
    incident: dict[tuple[int, int], list[int]],
    root: tuple[int, int],
    root_color: int,
    desired: int | None,
) -> int:
    if room_id == root:
        return root_color

    options = PALETTE[:]
    windows = incident[room_id]
    if windows and len(set(windows)) == 1:
        options = [color for color in options if color != windows[0]]
    if desired in options and random.random() < 0.5:
        return int(desired)
    return int(random.choice(options))


def _valid_layout(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> bool:
    try:
        found_rows, found_cols = _wall_lines(grid)
    except AssertionError:
        return False
    return found_rows == wall_rows and found_cols == wall_cols


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(GEN_MIN_BANDS, GEN_MAX_BANDS)
    n_cols = random.randint(GEN_MIN_BANDS, GEN_MAX_BANDS)
    heights = _sample_band_sizes(n_rows)
    widths = _sample_band_sizes(n_cols)
    wall_rows, wall_cols = _layout_from_sizes(heights, widths)
    h = wall_rows[-1] + 1
    w = wall_cols[-1] + 1
    if h > 30 or w > 30:
        return None

    rooms = list(_room_ids(wall_rows, wall_cols))
    neighbors = _neighbor_map(n_rows, n_cols)
    root = random.choice(rooms)
    room_count = len(rooms)

    infect_all = room_count <= 3 or random.random() < 0.5
    if infect_all:
        root_size = room_count
    else:
        root_size = random.randint(2 if room_count == 2 else 3, room_count - 1)
    root_tree = _grow_tree(root, root_size, neighbors)
    if root_tree is None:
        return None
    root_order, root_parent = root_tree
    edge_windows, output_colors = _planned_windows(root, root_order, root_parent)
    root_color = output_colors[root]

    remaining = [room_id for room_id in rooms if room_id not in set(root_order)]
    if len(remaining) >= 2 and random.random() < 0.7:
        extra_root = random.choice(remaining)
        extra_size = random.randint(2, len(remaining))
        extra_tree = _grow_tree(extra_root, extra_size, neighbors, allowed=set(remaining))
        if extra_tree is None:
            return None
        extra_order, extra_parent = extra_tree
        for room_id in extra_order[1:]:
            parent = extra_parent[room_id]
            color = random.choice(PALETTE)
            edge_windows[frozenset((parent, room_id))] = color

    incident = _incident_windows(rooms, edge_windows)
    grid = np.full((h, w), BLACK, dtype=int)
    _set_walls(grid, wall_rows, wall_cols)

    room_coords: dict[tuple[int, int], np.ndarray] = {}
    for room_id in rooms:
        planned = output_colors.get(room_id)
        input_color = _choose_input_color(room_id, incident, root, root_color, planned)
        bounds = _room_bounds(wall_rows, wall_cols, room_id)
        room_coords[room_id] = _place_pattern(grid, bounds, input_color)

    for edge, color in edge_windows.items():
        a, b = tuple(edge)
        _paint_window(grid, wall_rows, wall_cols, a, b, int(color))

    if not _valid_layout(grid, wall_rows, wall_cols):
        return None

    expected = _build_expected(grid, room_coords, output_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
