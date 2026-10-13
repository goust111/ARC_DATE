from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(10))
BG_COLOR = 0
GEN_ATTEMPTS = 400
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (4, 7)


@dataclass(frozen=True)
class Room:
    idx: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass(frozen=True)
class Door:
    idx: int
    room_a: int
    room_b: int
    pos: tuple[int, int]
    color: int
    orientation: str
    anchor_a: tuple[int, int]
    anchor_b: tuple[int, int]

    def other(self, room_idx: int) -> int:
        if room_idx == self.room_a:
            return self.room_b
        if room_idx == self.room_b:
            return self.room_a
        raise ValueError("room not attached to door")

    def target_anchor(self, target_room: int) -> tuple[int, int]:
        if target_room == self.room_a:
            return self.anchor_a
        if target_room == self.room_b:
            return self.anchor_b
        raise ValueError("room not attached to door")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    boundary = np.concatenate(
        [grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]]
    )
    boundary = boundary[boundary != bg]
    if boundary.size:
        values, counts = np.unique(boundary, return_counts=True)
        return int(values[int(np.argmax(counts))])

    non_bg = grid[grid != bg]
    if non_bg.size == 0:
        raise AssertionError("grid must contain walls")
    values, counts = np.unique(non_bg, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_doors(
    grid: np.ndarray, bg: int, wall: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    door_mask = np.zeros((h, w), dtype=bool)
    door_specs: list[dict[str, object]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color == bg or color == wall:
                continue

            vertical = (
                int(grid[r - 1, c]) == wall
                and int(grid[r + 1, c]) == wall
                and int(grid[r, c - 1]) != wall
                and int(grid[r, c + 1]) != wall
            )
            horizontal = (
                int(grid[r, c - 1]) == wall
                and int(grid[r, c + 1]) == wall
                and int(grid[r - 1, c]) != wall
                and int(grid[r + 1, c]) != wall
            )
            if vertical == horizontal:
                continue

            door_mask[r, c] = True
            door_specs.append(
                {
                    "pos": (r, c),
                    "color": color,
                    "orientation": "V" if vertical else "H",
                }
            )

    return door_mask, door_specs


def _extract_rooms(
    grid: np.ndarray, barrier: np.ndarray
) -> tuple[list[Room], np.ndarray]:
    h, w = grid.shape
    room_of = np.full((h, w), -1, dtype=int)
    rooms: list[Room] = []

    for sr in range(h):
        for sc in range(w):
            if barrier[sr, sc] or room_of[sr, sc] != -1:
                continue

            q = deque([(sr, sc)])
            room_of[sr, sc] = len(rooms)
            coords: list[tuple[int, int]] = []
            r_min = r_max = sr
            c_min = c_max = sc

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not barrier[nr, nc] and room_of[nr, nc] == -1:
                        room_of[nr, nc] = len(rooms)
                        q.append((nr, nc))

            room = Room(len(rooms), r_min, c_min, r_max, c_max)
            if len(coords) != room.height * room.width:
                raise AssertionError("rooms must be solid rectangles")
            rooms.append(room)

    return rooms, room_of


def _build_door_graph(
    room_of: np.ndarray, door_specs: list[dict[str, object]]
) -> tuple[list[Door], dict[int, list[int]]]:
    doors: list[Door] = []
    room_doors: dict[int, list[int]] = defaultdict(list)

    for spec in door_specs:
        r, c = spec["pos"]  # type: ignore[index]
        orientation = spec["orientation"]  # type: ignore[index]
        color = int(spec["color"])  # type: ignore[index]
        if orientation == "V":
            room_a = int(room_of[r, c - 1])
            room_b = int(room_of[r, c + 1])
            anchor_a = (r, c - 1)
            anchor_b = (r, c + 1)
        else:
            room_a = int(room_of[r - 1, c])
            room_b = int(room_of[r + 1, c])
            anchor_a = (r - 1, c)
            anchor_b = (r + 1, c)

        if room_a < 0 or room_b < 0 or room_a == room_b:
            raise AssertionError("door must separate two rooms")

        door = Door(
            idx=len(doors),
            room_a=room_a,
            room_b=room_b,
            pos=(int(r), int(c)),
            color=color,
            orientation=str(orientation),
            anchor_a=anchor_a,
            anchor_b=anchor_b,
        )
        doors.append(door)
        room_doors[room_a].append(door.idx)
        room_doors[room_b].append(door.idx)

    return doors, room_doors


def _room_pattern_cells(
    grid: np.ndarray, room: Room, bg: int, door_mask: np.ndarray
) -> np.ndarray:
    sub = grid[room.r0 : room.r1 + 1, room.c0 : room.c1 + 1]
    sub_doors = door_mask[room.r0 : room.r1 + 1, room.c0 : room.c1 + 1]
    coords = np.argwhere((sub != bg) & ~sub_doors)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int)
    coords[:, 0] += room.r0
    coords[:, 1] += room.c0
    return coords.astype(int, copy=False)


def _translate_pattern(
    coords: np.ndarray,
    door_pos: tuple[int, int],
    target_anchor: tuple[int, int],
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, bool, bool]:
    if coords.size == 0:
        return np.empty((0, 2), dtype=int), False, False

    dr = np.abs(coords[:, 0] - door_pos[0])
    dc = np.abs(coords[:, 1] - door_pos[1])
    distances = dr + dc
    min_dist = int(distances.min())
    nearest = coords[distances == min_dist]
    order = np.lexsort((nearest[:, 1], nearest[:, 0]))
    source_anchor = nearest[order[0]]
    shift_r = int(target_anchor[0] - source_anchor[0])
    shift_c = int(target_anchor[1] - source_anchor[1])
    moved = coords + np.array([shift_r, shift_c], dtype=int)

    in_bounds = (
        (moved[:, 0] >= 0)
        & (moved[:, 0] < keep_mask.shape[0])
        & (moved[:, 1] >= 0)
        & (moved[:, 1] < keep_mask.shape[1])
    )
    moved = moved[in_bounds]
    if moved.size == 0:
        return np.empty((0, 2), dtype=int), len(nearest) == 1, True

    kept = keep_mask[moved[:, 0], moved[:, 1]]
    moved = moved[kept]
    clipped = len(moved) != len(coords)
    return moved.astype(int, copy=False), len(nearest) == 1, clipped


def _solve_from_layout(
    grid: np.ndarray,
    rooms: list[Room],
    doors: list[Door],
    initial_patterns: dict[int, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, dict[str, int | bool]]:
    out = grid.copy()
    room_masks = []
    for room in rooms:
        mask = np.zeros(grid.shape, dtype=bool)
        mask[room.r0 : room.r1 + 1, room.c0 : room.c1 + 1] = True
        room_masks.append(mask)

    room_doors: dict[int, list[int]] = defaultdict(list)
    for door in doors:
        room_doors[door.room_a].append(door.idx)
        room_doors[door.room_b].append(door.idx)

    room_patterns = {
        room_idx: (coords.copy(), int(color))
        for room_idx, (coords, color) in initial_patterns.items()
    }
    queue = deque(sorted(room_patterns))
    seen_clip = False
    unique_nearest = True
    filled_rooms = 0

    while queue:
        source_room = queue.popleft()
        coords, _source_color = room_patterns[source_room]
        for door_idx in sorted(room_doors[source_room]):
            door = doors[door_idx]
            target_room = door.other(source_room)
            if target_room in room_patterns:
                continue

            target_anchor = door.target_anchor(target_room)
            moved, unique, clipped = _translate_pattern(
                coords,
                door.pos,
                target_anchor,
                room_masks[target_room],
            )
            unique_nearest = unique_nearest and unique
            if moved.size == 0:
                continue

            out[moved[:, 0], moved[:, 1]] = door.color
            room_patterns[target_room] = (moved, door.color)
            queue.append(target_room)
            filled_rooms += 1
            seen_clip = seen_clip or clipped

    return out, {
        "seen_clip": seen_clip,
        "unique_nearest": unique_nearest,
        "filled_rooms": filled_rooms,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each room's interior pattern through colored doors into adjacent empty rooms.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = _most_common_color(g)
    wall = _detect_wall_color(g, bg)
    door_mask, door_specs = _detect_doors(g, bg, wall)
    barrier = (g == wall) | door_mask
    rooms, room_of = _extract_rooms(g, barrier)
    doors, room_doors = _build_door_graph(room_of, door_specs)

    out = g.copy()
    room_patterns: dict[int, np.ndarray] = {}
    queue = deque()

    for room in rooms:
        coords = _room_pattern_cells(g, room, bg, door_mask)
        if coords.size == 0:
            continue
        room_patterns[room.idx] = coords
        queue.append(room.idx)

    while queue:
        source_room = queue.popleft()
        coords = room_patterns[source_room]
        for door_idx in sorted(room_doors.get(source_room, [])):
            door = doors[door_idx]
            target_room = door.other(source_room)
            if target_room in room_patterns:
                continue

            target_anchor = door.target_anchor(target_room)
            keep_mask = room_of == target_room
            moved, _unique, _clipped = _translate_pattern(
                coords,
                door.pos,
                target_anchor,
                keep_mask,
            )
            if moved.size == 0:
                continue

            out[moved[:, 0], moved[:, 1]] = door.color
            room_patterns[target_room] = moved
            queue.append(target_room)

    return out


def _room_id(r: int, c: int, cols: int) -> int:
    return r * cols + c


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    if area < 1 or area > height * width:
        return None

    for _ in range(80):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = [start]
        while len(cells) < area and frontier:
            base = random.choice(frontier)
            neighbors = []
            for dr, dc in DIR4:
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.remove(base)
                continue
            nxt = random.choice(neighbors)
            cells.add(nxt)
            frontier.append(nxt)

        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask

    return None


def _grow_tree(
    rows: int,
    cols: int,
    size: int,
    used: set[int],
) -> tuple[list[int], list[tuple[int, int]]] | None:
    available = [idx for idx in range(rows * cols) if idx not in used]
    if len(available) < size:
        return None

    start = random.choice(available)
    nodes = [start]
    node_set = {start}
    edges: list[tuple[int, int]] = []

    while len(nodes) < size:
        frontier: list[tuple[int, int]] = []
        for idx in nodes:
            rr, cc = divmod(idx, cols)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nbr = _room_id(nr, nc, cols)
                    if nbr in node_set or nbr in used:
                        continue
                    frontier.append((idx, nbr))
        if not frontier:
            return None
        parent, child = random.choice(frontier)
        nodes.append(child)
        node_set.add(child)
        edges.append((parent, child))

    return nodes, edges


def _pick_sources_for_components(
    components: list[tuple[list[int], list[tuple[int, int]]]]
) -> list[int]:
    sources: list[int] = []
    for nodes, edges in components:
        deg = {node: 0 for node in nodes}
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        leaves = [node for node, d in deg.items() if d <= 1]
        sources.append(random.choice(leaves if leaves else nodes))
    return sources


def _layout_rooms(
    row_heights: list[int],
    col_widths: list[int],
) -> list[Room]:
    row_starts = []
    pos = 1
    for height in row_heights:
        row_starts.append(pos)
        pos += height + 1

    col_starts = []
    pos = 1
    for width in col_widths:
        col_starts.append(pos)
        pos += width + 1

    rooms: list[Room] = []
    idx = 0
    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            rooms.append(
                Room(
                    idx=idx,
                    r0=r0,
                    c0=c0,
                    r1=r0 + row_heights[rr] - 1,
                    c1=c0 + col_widths[cc] - 1,
                )
            )
            idx += 1
    return rooms


def _door_for_edge(
    edge: tuple[int, int],
    rooms: list[Room],
    cols: int,
    color: int,
    idx: int,
) -> Door | None:
    a, b = edge
    room_a = rooms[a]
    room_b = rooms[b]
    ar, ac = divmod(a, cols)
    br, bc = divmod(b, cols)

    if ar == br and abs(ac - bc) == 1:
        left = room_a if room_a.c0 < room_b.c0 else room_b
        right = room_b if left.idx == room_a.idx else room_a
        rows = list(range(left.r0 + 1, left.r1))
        if not rows:
            return None
        if random.random() < 0.6:
            door_r = random.choice([rows[0], rows[-1]])
        else:
            door_r = random.choice(rows)
        door_c = left.c1 + 1
        return Door(
            idx=idx,
            room_a=left.idx,
            room_b=right.idx,
            pos=(door_r, door_c),
            color=color,
            orientation="V",
            anchor_a=(door_r, door_c - 1),
            anchor_b=(door_r, door_c + 1),
        )

    if ac == bc and abs(ar - br) == 1:
        top = room_a if room_a.r0 < room_b.r0 else room_b
        bottom = room_b if top.idx == room_a.idx else room_a
        cols_range = list(range(top.c0 + 1, top.c1))
        if not cols_range:
            return None
        if random.random() < 0.6:
            door_c = random.choice([cols_range[0], cols_range[-1]])
        else:
            door_c = random.choice(cols_range)
        door_r = top.r1 + 1
        return Door(
            idx=idx,
            room_a=top.idx,
            room_b=bottom.idx,
            pos=(door_r, door_c),
            color=color,
            orientation="H",
            anchor_a=(door_r - 1, door_c),
            anchor_b=(door_r + 1, door_c),
        )

    return None


def _make_source_pattern(room: Room) -> np.ndarray | None:
    max_h = room.height - 2
    max_w = room.width - 2
    if max_h < 1 or max_w < 1:
        return None

    patch_h = random.randint(1, min(4, max_h))
    patch_w = random.randint(1, min(4, max_w))
    area_hi = min(patch_h * patch_w, 6)
    area = random.randint(1, area_hi)
    mask = _random_connected_mask(patch_h, patch_w, area)
    if mask is None:
        return None

    top = random.randint(room.r0 + 1, room.r1 - patch_h)
    left = random.randint(room.c0 + 1, room.c1 - patch_w)
    coords = np.argwhere(mask)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords.astype(int, copy=False)


def _render_input(
    grid_shape: tuple[int, int],
    rooms: list[Room],
    wall_color: int,
    doors: list[Door],
    initial_patterns: dict[int, tuple[np.ndarray, int]],
) -> np.ndarray:
    grid = np.full(grid_shape, wall_color, dtype=int)
    for room in rooms:
        grid[room.r0 : room.r1 + 1, room.c0 : room.c1 + 1] = BG_COLOR
    for door in doors:
        r, c = door.pos
        grid[r, c] = door.color
    for coords, color in initial_patterns.values():
        grid[coords[:, 0], coords[:, 1]] = color
    return grid


def _edge_orientations(edges: list[tuple[int, int]], cols: int) -> set[str]:
    result = set()
    for a, b in edges:
        ar, ac = divmod(a, cols)
        br, bc = divmod(b, cols)
        result.add("H" if ar != br else "V")
    return result


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROOM_ROWS_RANGE)
    cols = random.randint(*ROOM_COLS_RANGE)
    total_rooms = rows * cols

    n_sources = 1 if random.random() < 0.5 else 2
    min_needed = 3 if n_sources == 1 else 5
    if total_rooms < min_needed:
        return None

    max_comp1 = min(5, total_rooms - (2 if n_sources == 2 else 0))
    comp1_size = random.randint(3, max_comp1)
    comp2_size = 0
    if n_sources == 2:
        max_comp2 = min(3, total_rooms - comp1_size)
        if max_comp2 < 2:
            return None
        comp2_size = random.randint(2, max_comp2)

    row_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(cols)]
    grid_h = sum(row_heights) + rows + 1
    grid_w = sum(col_widths) + cols + 1
    rooms = _layout_rooms(row_heights, col_widths)

    used: set[int] = set()
    comp1 = _grow_tree(rows, cols, comp1_size, used)
    if comp1 is None:
        return None
    used.update(comp1[0])
    components = [comp1]
    if n_sources == 2:
        comp2 = _grow_tree(rows, cols, comp2_size, used)
        if comp2 is None:
            return None
        used.update(comp2[0])
        components.append(comp2)

    all_edges = [edge for _nodes, edges in components for edge in edges]
    if len(all_edges) < 2:
        return None
    if rows > 1 and cols > 1 and _edge_orientations(all_edges, cols) != {"H", "V"}:
        return None

    sources = _pick_sources_for_components(components)
    colors = ALL_COLORS.copy()
    colors.remove(BG_COLOR)
    random.shuffle(colors)
    if len(colors) < 1 + n_sources + len(all_edges):
        return None

    wall_color = colors.pop()
    source_colors = [colors.pop() for _ in range(n_sources)]
    door_colors = [colors.pop() for _ in range(len(all_edges))]

    doors: list[Door] = []
    for idx, (edge, color) in enumerate(zip(all_edges, door_colors)):
        door = _door_for_edge(edge, rooms, cols, color, idx)
        if door is None:
            return None
        doors.append(door)

    initial_patterns: dict[int, tuple[np.ndarray, int]] = {}
    for source_room, color in zip(sources, source_colors):
        coords = _make_source_pattern(rooms[source_room])
        if coords is None:
            return None
        initial_patterns[source_room] = (coords, color)

    inp = _render_input((grid_h, grid_w), rooms, wall_color, doors, initial_patterns)
    expected, stats = _solve_from_layout(inp, rooms, doors, initial_patterns)
    if not bool(stats["unique_nearest"]):
        return None
    if int(stats["filled_rooms"]) < len(all_edges):
        return None

    require_clip = random.random() < 0.5
    if require_clip and not bool(stats["seen_clip"]):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_ATTEMPTS} attempts")
