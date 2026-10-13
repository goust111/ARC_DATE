from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 1
MARKER = 6
DOOR_COLORS = [2, 3, 4, 5, 7, 8, 9]
MIN_ROOM_SIDE = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    candidates = [color for color in counts if color not in (WALL, MARKER)]
    assert candidates, "background cannot be inferred"
    best = max(counts[color] for color in candidates)
    bg_candidates = [color for color in candidates if counts[color] == best]
    assert len(bg_candidates) == 1, "background color must be uniquely dominant among non-wall colors"
    return int(bg_candidates[0])


def _structural_wall_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    structural = np.zeros((h, w), dtype=bool)
    for comp in _components(grid == WALL):
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 1] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == w - 1)
        )
        if touches_border:
            structural[comp[:, 0], comp[:, 1]] = True
    return structural


def _label_rooms(passable: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []
    for sr, sc in np.argwhere(passable):
        sr = int(sr)
        sc = int(sc)
        if labels[sr, sc] != -1:
            continue
        label = len(rooms)
        stack = [(sr, sc)]
        labels[sr, sc] = label
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds((h, w), nr, nc) and passable[nr, nc] and labels[nr, nc] == -1:
                    labels[nr, nc] = label
                    stack.append((nr, nc))
        rooms.append(np.array(coords, dtype=int))
    return labels, rooms


def _infer_layout(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    structural = _structural_wall_mask(g)

    door_mask = np.zeros((h, w), dtype=bool)
    door_cells: list[tuple[int, int, int]] = []
    for color in set(int(v) for v in g.ravel()) - {bg, WALL, MARKER}:
        coords = np.argwhere(g == int(color))
        assert coords.shape[0] == 1, "each door color must appear as one visible pixel"
        r, c = map(int, coords[0])
        door_mask[r, c] = True
        door_cells.append((r, c, int(color)))

    passable = ~structural & ~door_mask
    labels, rooms = _label_rooms(passable)
    assert rooms, "at least one room is required"

    edges: list[list[tuple[int, int, tuple[int, int]]]] = [[] for _ in rooms]
    for r, c, color in door_cells:
        adjacent_rooms = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(r, c)
            if _in_bounds((h, w), nr, nc) and labels[nr, nc] >= 0
        }
        assert len(adjacent_rooms) == 2, "each colored door must touch exactly two rooms"
        a, b = tuple(adjacent_rooms)
        edges[a].append((b, int(color), (r, c)))
        edges[b].append((a, int(color), (r, c)))

    return bg, structural, door_mask, door_cells, labels, rooms, edges


def _entry_colors(edges, start_label: int, bg: int) -> list[int]:
    edge_count = sum(len(e) for e in edges) // 2
    assert edge_count == len(edges) - 1, "room connections must form a tree"

    colors: list[int | None] = [None] * len(edges)
    colors[start_label] = int(bg)
    q = deque([start_label])
    while q:
        room = q.popleft()
        for nxt, door_color, _pos in edges[room]:
            if colors[nxt] is None:
                colors[nxt] = int(door_color)
                q.append(nxt)

    assert all(color is not None for color in colors), "all rooms must be reachable from the marker room"
    return [int(color) for color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each reachable room by the colored door used to enter it from the marker room.

    Input structure:
    - A dominant background is enclosed and divided by blue structural walls into rooms whose door graph is a chain or tree.
    - Every connection between adjacent rooms is one unique non-blue, non-background, non-magenta door pixel embedded in a wall.
    - One magenta marker pixel lies in a corner of the starting room.
    - Extra blue fragments may appear inside rooms, separated from the structural wall network.

    Rule:
    - Treat only border-connected blue components as structural walls; isolated blue room fragments are passable room contents.
    - Build the room graph through the colored door pixels and traverse it from the room containing the magenta marker.
    - The starting room is restored to background; every other room is filled with the color of the door crossed to enter it.
    - Structural walls and door pixels stay unchanged, while the magenta marker and interior blue fragments are overwritten by their room color.

    Key insight:
    - In rooms touching multiple doors, the fill color is the entry door on the unique path from the marker, not an arbitrary adjacent door.
    """
    g = np.asarray(grid, dtype=int)
    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape[0] == 1, "expected exactly one magenta marker"
    mr, mc = map(int, marker_coords[0])

    bg, structural, _door_mask, door_cells, labels, rooms, edges = _infer_layout(g)
    start_label = int(labels[mr, mc])
    assert start_label >= 0, "marker must be inside a room"
    fill_colors = _entry_colors(edges, start_label, bg)

    out = g.copy()
    for room_label, coords in enumerate(rooms):
        out[coords[:, 0], coords[:, 1]] = fill_colors[room_label]

    out[structural] = WALL
    for r, c, color in door_cells:
        out[r, c] = int(color)
    return out


def _split_options(grid: np.ndarray, room: tuple[int, int, int, int]):
    r0, c0, r1, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options = []

    if width >= 2 * MIN_ROOM_SIDE + 1 and height >= MIN_ROOM_SIDE:
        for c in range(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE + 1):
            if grid[r0 - 1, c] == WALL and grid[r1 + 1, c] == WALL:
                door_rows = [r for r in range(r0 + 1, r1) if grid[r, c] == BG]
                if door_rows:
                    options.append(("vertical", c, door_rows))

    if height >= 2 * MIN_ROOM_SIDE + 1 and width >= MIN_ROOM_SIDE:
        for r in range(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE + 1):
            if grid[r, c0 - 1] == WALL and grid[r, c1 + 1] == WALL:
                door_cols = [c for c in range(c0 + 1, c1) if grid[r, c] == BG]
                if door_cols:
                    options.append(("horizontal", r, door_cols))

    return options


def _apply_split(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    option,
    door_color: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = room
    kind, line, door_positions = option
    if kind == "vertical":
        c = int(line)
        door_r = int(random.choice(door_positions))
        grid[r0 - 1 : r1 + 2, c] = WALL
        grid[door_r, c] = int(door_color)
        return (r0, c0, r1, c - 1), (r0, c + 1, r1, c1)

    r = int(line)
    door_c = int(random.choice(door_positions))
    grid[r, c0 - 1 : c1 + 2] = WALL
    grid[r, door_c] = int(door_color)
    return (r0, c0, r - 1, c1), (r + 1, c0, r1, c1)


def _room_corner(labels: np.ndarray, room_label: int) -> tuple[int, int]:
    coords = np.argwhere(labels == int(room_label))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    corners = [(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))]
    random.shuffle(corners)
    for r, c in corners:
        if labels[r, c] == room_label:
            return r, c
    r, c = coords[0]
    return int(r), int(c)


def _random_fragment_shape() -> list[tuple[int, int]]:
    target = random.choice([1, 1, 2, 2, 3])
    cells = {(0, 0)}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        nr, nc = random.choice([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        cells.add((nr, nc))
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm = [(r - min_r, c - min_c) for r, c in cells]
    for _ in range(random.randrange(4)):
        max_r = max(r for r, _c in norm)
        norm = [(c, max_r - r) for r, c in norm]
    if random.random() < 0.5:
        max_c = max(c for _r, c in norm)
        norm = [(r, max_c - c) for r, c in norm]
    min_r = min(r for r, _c in norm)
    min_c = min(c for _r, c in norm)
    return [(int(r - min_r), int(c - min_c)) for r, c in norm]


def _fragment_is_clear(
    grid: np.ndarray,
    labels: np.ndarray,
    structural: np.ndarray,
    door_mask: np.ndarray,
    blocked: set[tuple[int, int]],
    room_label: int,
    cells: list[tuple[int, int]],
) -> bool:
    shape = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if labels[r, c] != room_label or grid[r, c] != BG or (r, c) in blocked:
            return False
        for nr, nc in list(_neighbors4(r, c)) + list(_neighbors8(r, c)):
            if not _in_bounds(shape, nr, nc):
                continue
            if (nr, nc) in cell_set:
                continue
            if structural[nr, nc] or door_mask[nr, nc] or (nr, nc) in blocked:
                return False
    return True


def _place_fragment(
    grid: np.ndarray,
    labels: np.ndarray,
    structural: np.ndarray,
    door_mask: np.ndarray,
    blocked: set[tuple[int, int]],
    room_label: int,
) -> bool:
    anchors = [(int(r), int(c)) for r, c in np.argwhere(labels == int(room_label))]
    random.shuffle(anchors)
    for ar, ac in anchors:
        for _ in range(8):
            shape = _random_fragment_shape()
            cells = [(ar + dr, ac + dc) for dr, dc in shape]
            if _fragment_is_clear(grid, labels, structural, door_mask, blocked, int(room_label), cells):
                for r, c in cells:
                    grid[r, c] = WALL
                    blocked.add((r, c))
                    for nr, nc in _neighbors8(r, c):
                        if _in_bounds(grid.shape, nr, nc):
                            blocked.add((nr, nc))
                return True
    return False


def _connected_from(edges, start: int) -> set[int]:
    seen = {int(start)}
    q = deque([int(start)])
    while q:
        room = q.popleft()
        for nxt, _color, _pos in edges[room]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        marker_coords = np.argwhere(g == MARKER)
        if marker_coords.shape[0] != 1:
            return False
        mr, mc = map(int, marker_coords[0])
        bg, structural, door_mask, door_cells, labels, rooms, edges = _infer_layout(g)
        start_label = int(labels[mr, mc])
        if start_label < 0 or len(rooms) < 4:
            return False
        if len(door_cells) != len(rooms) - 1:
            return False
        if len(_connected_from(edges, start_label)) != len(rooms):
            return False

        degrees = [len(edge_list) for edge_list in edges]
        if not any(label != start_label and degree >= 2 for label, degree in enumerate(degrees)):
            return False

        decorative = (g == WALL) & ~structural
        if int(decorative.sum()) < 2:
            return False
        decorative_rooms = {int(labels[r, c]) for r, c in np.argwhere(decorative)}
        if start_label in decorative_rooms and len(decorative_rooms) == 1:
            return False
        if not any(room != start_label for room in decorative_rooms):
            return False

        expected_colors = _entry_colors(edges, start_label, bg)
        out = main(g)
        if out.shape != g.shape or np.array_equal(out, g):
            return False
        if not np.all(out[structural] == WALL):
            return False
        for r, c, color in door_cells:
            if int(out[r, c]) != int(color):
                return False
        for label, coords in enumerate(rooms):
            if not np.all(out[coords[:, 0], coords[:, 1]] == expected_colors[label]):
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    target_rooms = random.randint(5, 8)

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rooms: list[tuple[int, int, int, int]] = [(1, 1, h - 2, w - 2)]
    door_colors = random.sample(DOOR_COLORS, target_rooms - 1)

    for door_color in door_colors:
        candidates = []
        for index, room in enumerate(rooms):
            for option in _split_options(grid, room):
                candidates.append((index, option))
        if not candidates:
            return None

        index, option = random.choice(candidates)
        room = rooms.pop(index)
        a, b = _apply_split(grid, room, option, door_color)
        rooms.extend([a, b])

    try:
        bg, structural, door_mask, _door_cells, labels, parsed_rooms, edges = _infer_layout(grid)
    except AssertionError:
        return None
    if bg != BG or len(parsed_rooms) != target_rooms or len(_connected_from(edges, 0)) != target_rooms:
        return None

    degrees = [len(edge_list) for edge_list in edges]
    leaves = [label for label, degree in enumerate(degrees) if degree == 1]
    start_label = random.choice(leaves if leaves and random.random() < 0.85 else list(range(target_rooms)))
    if not any(label != start_label and degree >= 2 for label, degree in enumerate(degrees)):
        return None

    mr, mc = _room_corner(labels, start_label)
    grid[mr, mc] = MARKER

    blocked = {(mr, mc)}
    for nr, nc in _neighbors8(mr, mc):
        if _in_bounds(grid.shape, nr, nc):
            blocked.add((nr, nc))

    room_order = [label for label in range(target_rooms) if label != start_label]
    random.shuffle(room_order)
    room_order.insert(random.randrange(len(room_order) + 1), start_label)
    desired_fragments = random.randint(2, min(7, target_rooms + 2))
    placed = 0
    rounds = 0
    while placed < desired_fragments and rounds < target_rooms * 3:
        room_label = room_order[rounds % len(room_order)]
        if _place_fragment(grid, labels, structural, door_mask, blocked, room_label):
            placed += 1
        rounds += 1

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for door-color chained flood fill")
