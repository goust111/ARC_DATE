from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 30
OUTER_ROOM_SIZE_MIN = 11
OUTER_ROOM_SIZE_MAX = 24
MIN_ROOM_SIDE = 3
ROOM_COUNT_MIN = 3
ROOM_COUNT_MAX = 7
GENERATION_ATTEMPTS = 300


@dataclass(frozen=True)
class Room:
    top: int
    bottom: int
    left: int
    right: int
    color: int
    doors: tuple[tuple[int, int], ...] = ()

    @property
    def area(self) -> int:
        return (self.bottom - self.top + 1) * (self.right - self.left + 1)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.top, self.bottom, self.left, self.right)

    @property
    def center2(self) -> tuple[int, int]:
        return (self.top + self.bottom, self.left + self.right)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each connected group of rectangular rooms with the largest room's border color.

    Input structure:
    - Background is black.
    - Several axis-aligned rectangular rooms have black floors and 1-pixel-thick colored borders.
    - Adjacent rooms share wall segments; some shared walls contain a single black door-gap.

    Rule:
    - Detect every room as a black rectangular floor enclosed by colored walls, allowing isolated single-pixel wall gaps.
    - Use the shared wall gaps as doors: rooms whose borders contain the same gap belong to the same connected region, transitively.
    - For each connected region, find the room with the largest black floor area and copy that room's border color into every room floor of the region.
    - Keep every wall pixel unchanged, including the black door gaps.

    Key insight:
    The black rectangles are individual room floors, while single black pixels inside shared walls are doors that define a graph over rooms; the fill color is inherited from the largest room in each graph component.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)
    if not rooms:
        return g.copy()

    edges = _room_edges(rooms)
    components = _components_from_edges(len(rooms), edges)

    out = g.copy()
    for component in components:
        winner = max(component, key=lambda idx: rooms[idx].area)
        fill_color = rooms[winner].color
        for idx in component:
            room = rooms[idx]
            out[room.top : room.bottom + 1, room.left : room.right + 1] = fill_color
    return out


def _find_rooms(grid: np.ndarray) -> list[Room]:
    h, w = grid.shape
    enclosed = _enclosed_black_mask(grid)
    prefix = _prefix_sum(enclosed)
    rooms: list[Room] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not enclosed[r, c]:
                continue
            if grid[r - 1, c] == BG or grid[r, c - 1] == BG:
                continue

            width = 0
            while c + width < w - 1 and enclosed[r, c + width]:
                width += 1
            height = 0
            while r + height < h - 1 and enclosed[r + height, c]:
                height += 1

            if width <= 0 or height <= 0:
                continue

            bottom = r + height - 1
            right = c + width - 1
            rect = (r, bottom, c, right)
            if rect in seen:
                continue
            if _rect_sum(prefix, r, c, bottom, right) != width * height:
                continue

            doors = _room_doors(grid, enclosed, r, bottom, c, right)
            if doors is None:
                continue

            color = _room_color(grid, r, bottom, c, right)
            if color == BG:
                continue

            rooms.append(Room(r, bottom, c, right, color, doors))
            seen.add(rect)

    return rooms


def _room_doors(
    grid: np.ndarray,
    enclosed: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> tuple[tuple[int, int], ...] | None:
    h, w = grid.shape
    if top <= 0 or left <= 0 or bottom >= h - 1 or right >= w - 1:
        return None

    corners = (
        (top - 1, left - 1),
        (top - 1, right + 1),
        (bottom + 1, left - 1),
        (bottom + 1, right + 1),
    )
    if any(grid[r, c] == BG for r, c in corners):
        return None

    doors: list[tuple[int, int]] = []

    for c in range(left, right + 1):
        if grid[top - 1, c] != BG:
            continue
        if top - 2 < 0 or not enclosed[top - 2, c]:
            return None
        if grid[top - 1, c - 1] == BG or grid[top - 1, c + 1] == BG:
            return None
        doors.append((top - 1, c))

    for c in range(left, right + 1):
        if grid[bottom + 1, c] != BG:
            continue
        if bottom + 2 >= h or not enclosed[bottom + 2, c]:
            return None
        if grid[bottom + 1, c - 1] == BG or grid[bottom + 1, c + 1] == BG:
            return None
        doors.append((bottom + 1, c))

    for r in range(top, bottom + 1):
        if grid[r, left - 1] != BG:
            continue
        if left - 2 < 0 or not enclosed[r, left - 2]:
            return None
        if grid[r - 1, left - 1] == BG or grid[r + 1, left - 1] == BG:
            return None
        doors.append((r, left - 1))

    for r in range(top, bottom + 1):
        if grid[r, right + 1] != BG:
            continue
        if right + 2 >= w or not enclosed[r, right + 2]:
            return None
        if grid[r - 1, right + 1] == BG or grid[r + 1, right + 1] == BG:
            return None
        doors.append((r, right + 1))

    return tuple(sorted(set(doors)))


def _room_color(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    colors: list[int] = []
    for c in range(left - 1, right + 2):
        if grid[top - 1, c] != BG:
            colors.append(int(grid[top - 1, c]))
        if grid[bottom + 1, c] != BG:
            colors.append(int(grid[bottom + 1, c]))
    for r in range(top, bottom + 1):
        if grid[r, left - 1] != BG:
            colors.append(int(grid[r, left - 1]))
        if grid[r, right + 1] != BG:
            colors.append(int(grid[r, right + 1]))

    if not colors:
        return BG

    first_seen: dict[int, int] = {}
    for idx, color in enumerate(colors):
        first_seen.setdefault(color, idx)

    counts = Counter(colors)
    return max(counts, key=lambda color: (counts[color], -first_seen[color]))


def _room_edges(rooms: list[Room]) -> list[tuple[int, int]]:
    door_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        for door in room.doors:
            door_map[door].append(idx)

    edges = {
        tuple(sorted(indices))
        for indices in door_map.values()
        if len(indices) == 2
    }
    return sorted(edges)


def _components_from_edges(n_rooms: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    graph = [[] for _ in range(n_rooms)]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    seen = [False] * n_rooms
    components: list[list[int]] = []
    for start in range(n_rooms):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in graph[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(component)
    return components


def _enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    black = grid == BG
    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return black & ~exterior


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    return int(
        prefix[bottom + 1, right + 1]
        - prefix[top, right + 1]
        - prefix[bottom + 1, left]
        + prefix[top, left]
    )


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rooms = rng.randint(ROOM_COUNT_MIN, ROOM_COUNT_MAX)

    grid_h = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid_w = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    outer_h_max = min(OUTER_ROOM_SIZE_MAX, grid_h - 2)
    outer_w_max = min(OUTER_ROOM_SIZE_MAX, grid_w - 2)
    if outer_h_max < OUTER_ROOM_SIZE_MIN or outer_w_max < OUTER_ROOM_SIZE_MIN:
        return None

    outer_h = rng.randint(OUTER_ROOM_SIZE_MIN, outer_h_max)
    outer_w = rng.randint(OUTER_ROOM_SIZE_MIN, outer_w_max)
    top0 = rng.randint(1, grid_h - outer_h - 1)
    left0 = rng.randint(1, grid_w - outer_w - 1)

    floor_top = top0 + 1
    floor_left = left0 + 1
    floor_bottom = top0 + outer_h - 2
    floor_right = left0 + outer_w - 2

    rooms0 = _partition_floor(
        rng,
        Room(floor_top, floor_bottom, floor_left, floor_right, 0),
        n_rooms,
    )
    if rooms0 is None:
        return None

    if len({room.area for room in rooms0}) != len(rooms0):
        return None

    colors = list(rng.sample(COLOR_POOL, len(rooms0)))
    rooms_colored = [replace(room, color=colors[idx]) for idx, room in enumerate(rooms0)]

    adjacencies = _adjacency_segments(rooms_colored)
    if not adjacencies:
        return None

    if not _graph_connected(len(rooms_colored), adjacencies):
        return None

    target_components = _target_component_count(rng, len(rooms_colored))
    open_edges = _choose_open_edges(rng, len(rooms_colored), adjacencies, target_components)
    if open_edges is None:
        return None

    doors = _assign_doors(rng, adjacencies, open_edges)
    if doors is None:
        return None

    room_doors: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(rooms_colored))}
    for edge_key, door in doors.items():
        a, b = edge_key
        room_doors[a].append(door)
        room_doors[b].append(door)
    rooms = [replace(room, doors=tuple(sorted(room_doors[idx]))) for idx, room in enumerate(rooms_colored)]

    grid = _render_layout((grid_h, grid_w), rooms, doors)
    expected = _expected_output(grid, rooms, open_edges)
    if expected is None:
        return None

    detected = _find_rooms(grid)
    if len(detected) != len(rooms):
        return None

    detected_map = {room.rect: room.color for room in detected}
    target_map = {room.rect: room.color for room in rooms}
    if detected_map != target_map:
        return None

    rect_to_id = {room.rect: idx for idx, room in enumerate(rooms)}
    detected_edges = {
        tuple(sorted((rect_to_id[detected[a].rect], rect_to_id[detected[b].rect])))
        for a, b in _room_edges(detected)
    }
    if detected_edges != open_edges:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def _partition_floor(rng: random.Random, root: Room, target_rooms: int) -> list[Room] | None:
    rooms = [root]
    while len(rooms) < target_rooms:
        candidates: list[tuple[int, str]] = []
        for idx, room in enumerate(rooms):
            height = room.bottom - room.top + 1
            width = room.right - room.left + 1
            if height >= 2 * MIN_ROOM_SIDE + 1:
                candidates.append((idx, "h"))
            if width >= 2 * MIN_ROOM_SIDE + 1:
                candidates.append((idx, "v"))
        if not candidates:
            return None

        idx, orientation = rng.choice(candidates)
        room = rooms.pop(idx)
        if orientation == "h":
            split = rng.randint(room.top + MIN_ROOM_SIDE, room.bottom - MIN_ROOM_SIDE)
            a = Room(room.top, split - 1, room.left, room.right, room.color)
            b = Room(split + 1, room.bottom, room.left, room.right, room.color)
        else:
            split = rng.randint(room.left + MIN_ROOM_SIDE, room.right - MIN_ROOM_SIDE)
            a = Room(room.top, room.bottom, room.left, split - 1, room.color)
            b = Room(room.top, room.bottom, split + 1, room.right, room.color)
        rooms.extend([a, b])

    return rooms


def _adjacency_segments(rooms: list[Room]) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            if a.right + 2 == b.left or b.right + 2 == a.left:
                left_idx, right_idx = (i, j) if a.right < b.left else (j, i)
                left_room = rooms[left_idx]
                right_room = rooms[right_idx]
                lo = max(left_room.top, right_room.top)
                hi = min(left_room.bottom, right_room.bottom)
                if hi - lo + 1 >= 3:
                    segments.append(
                        {
                            "a": left_idx,
                            "b": right_idx,
                            "axis": "v",
                            "wall": left_room.right + 1,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
            if a.bottom + 2 == b.top or b.bottom + 2 == a.top:
                top_idx, bottom_idx = (i, j) if a.bottom < b.top else (j, i)
                top_room = rooms[top_idx]
                bottom_room = rooms[bottom_idx]
                lo = max(top_room.left, bottom_room.left)
                hi = min(top_room.right, bottom_room.right)
                if hi - lo + 1 >= 3:
                    segments.append(
                        {
                            "a": top_idx,
                            "b": bottom_idx,
                            "axis": "h",
                            "wall": top_room.bottom + 1,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    return segments


def _graph_connected(n_rooms: int, adjacencies: list[dict[str, int | str]]) -> bool:
    graph = [[] for _ in range(n_rooms)]
    for segment in adjacencies:
        a = int(segment["a"])
        b = int(segment["b"])
        graph[a].append(b)
        graph[b].append(a)

    seen = [False] * n_rooms
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)
    return all(seen)


def _target_component_count(rng: random.Random, n_rooms: int) -> int:
    if n_rooms <= 3:
        return 1 if rng.random() < 0.5 else 2
    if rng.random() < 0.5:
        return 1
    return rng.randint(2, min(3, n_rooms - 1))


def _choose_open_edges(
    rng: random.Random,
    n_rooms: int,
    adjacencies: list[dict[str, int | str]],
    target_components: int,
) -> set[tuple[int, int]] | None:
    edges = [tuple(sorted((int(seg["a"]), int(seg["b"])))) for seg in adjacencies]
    edges = sorted(set(edges))
    order = edges[:]
    rng.shuffle(order)

    parent = list(range(n_rooms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    tree: list[tuple[int, int]] = []
    for edge in order:
        if union(*edge):
            tree.append(edge)
    if len(tree) != n_rooms - 1:
        return None

    keep = tree[:]
    if target_components > 1:
        to_remove = rng.sample(range(len(tree)), target_components - 1)
        keep = [edge for idx, edge in enumerate(tree) if idx not in set(to_remove)]

    parent2 = list(range(n_rooms))

    def find2(x: int) -> int:
        while parent2[x] != x:
            parent2[x] = parent2[parent2[x]]
            x = parent2[x]
        return x

    def union2(a: int, b: int) -> None:
        ra = find2(a)
        rb = find2(b)
        if ra != rb:
            parent2[rb] = ra

    open_edges = set(keep)
    for edge in keep:
        union2(*edge)

    extras = [edge for edge in edges if edge not in open_edges]
    rng.shuffle(extras)
    for edge in extras:
        if find2(edge[0]) == find2(edge[1]) and rng.random() < 0.35:
            open_edges.add(edge)

    component_sizes = Counter(find2(idx) for idx in range(n_rooms))
    if max(component_sizes.values()) <= 1:
        return None
    return open_edges


def _assign_doors(
    rng: random.Random,
    adjacencies: list[dict[str, int | str]],
    open_edges: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int]] | None:
    doors: dict[tuple[int, int], tuple[int, int]] = {}
    for segment in adjacencies:
        edge = tuple(sorted((int(segment["a"]), int(segment["b"]))))
        if edge not in open_edges:
            continue
        lo = int(segment["lo"])
        hi = int(segment["hi"])
        if hi - lo + 1 < 3:
            return None
        pos = rng.randint(lo + 1, hi - 1)
        if segment["axis"] == "v":
            door = (pos, int(segment["wall"]))
        else:
            door = (int(segment["wall"]), pos)
        doors[edge] = door
    return doors


def _render_layout(
    shape: tuple[int, int],
    rooms: list[Room],
    doors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    door_cells = set(doors.values())
    votes: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, room in enumerate(rooms):
        for c in range(room.left - 1, room.right + 2):
            votes[(room.top - 1, c)].append(idx)
            votes[(room.bottom + 1, c)].append(idx)
        for r in range(room.top, room.bottom + 1):
            votes[(r, room.left - 1)].append(idx)
            votes[(r, room.right + 1)].append(idx)

    for cell, voters in votes.items():
        if cell in door_cells:
            continue
        color = _resolve_wall_color(cell[0], cell[1], voters, rooms)
        grid[cell] = color

    return grid


def _resolve_wall_color(r: int, c: int, voters: list[int], rooms: list[Room]) -> int:
    if len(voters) == 1:
        return rooms[voters[0]].color

    left = [idx for idx in voters if rooms[idx].right < c]
    right = [idx for idx in voters if rooms[idx].left > c]
    above = [idx for idx in voters if rooms[idx].bottom < r]
    below = [idx for idx in voters if rooms[idx].top > r]

    if left and right:
        a = rooms[left[0]]
        b = rooms[right[0]]
        lo = max(a.top, b.top)
        hi = min(a.bottom, b.bottom)
        mid = (lo + hi) // 2
        return a.color if r <= mid else b.color

    if above and below:
        a = rooms[above[0]]
        b = rooms[below[0]]
        lo = max(a.left, b.left)
        hi = min(a.right, b.right)
        mid = (lo + hi) // 2
        return a.color if c <= mid else b.color

    best_idx = min(
        voters,
        key=lambda idx: (
            abs(rooms[idx].center2[0] - 2 * r) + abs(rooms[idx].center2[1] - 2 * c),
            idx,
        ),
    )
    return rooms[best_idx].color


def _expected_output(
    grid: np.ndarray,
    rooms: list[Room],
    open_edges: set[tuple[int, int]],
) -> np.ndarray | None:
    components = _components_from_edges(len(rooms), sorted(open_edges))
    out = grid.copy()
    for component in components:
        areas = [rooms[idx].area for idx in component]
        if len(areas) != len(set(areas)):
            return None
        winner = max(component, key=lambda idx: rooms[idx].area)
        fill = rooms[winner].color
        for idx in component:
            room = rooms[idx]
            out[room.top : room.bottom + 1, room.left : room.right + 1] = fill
    return out
