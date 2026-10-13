from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, components_mask


MAX_GRID_SIZE = 30
GEN_MAX_ATTEMPTS = 400
MIN_ROOM_DIM = 3
SPLITTABLE_LEN = 2 * MIN_ROOM_DIM + 1
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _separator_row(grid: np.ndarray) -> int:
    for r in range(1, grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]):
            return int(r)
    raise ValueError("expected a horizontal separator wall row")


def _palette_from_row(row: np.ndarray) -> list[int]:
    palette: list[int] = []
    for value in map(int, row):
        if not palette or palette[-1] != value:
            palette.append(value)
    if not palette:
        raise ValueError("expected a non-empty palette legend")
    return palette


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    height, width = map(int, mask.shape)
    dist = np.full((height, width), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        is_boundary = False
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not mask[nr, nc]:
                is_boundary = True
                break
        if is_boundary:
            dist[rr, cc] = 0
            queue.append((rr, cc))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _room_ids(body: np.ndarray, wall: int) -> tuple[list[np.ndarray], np.ndarray]:
    rooms = components_mask(body != wall, connectivity=4)
    room_map = np.full(body.shape, -1, dtype=int)
    for idx, coords in enumerate(rooms):
        for r, c in coords:
            room_map[int(r), int(c)] = int(idx)
    return rooms, room_map


def _find_source_room(body: np.ndarray, rooms: list[np.ndarray]) -> int:
    source_room = -1
    for idx, coords in enumerate(rooms):
        values = np.array([int(body[int(r), int(c)]) for r, c in coords], dtype=int)
        counts = Counter(map(int, values))
        if len(counts) <= 1:
            continue
        minority_area = sum(count for count in counts.values() if count != max(counts.values()))
        if minority_area == 1:
            if source_room != -1:
                raise ValueError("expected exactly one seeded room")
            source_room = int(idx)
    if source_room == -1:
        raise ValueError("missing seed room")
    return source_room


def _build_room_graph(body: np.ndarray, wall: int, room_map: np.ndarray) -> list[set[int]]:
    graph = [set() for _ in range(int(room_map.max()) + 1)]
    height, width = map(int, body.shape)
    for r in range(height):
        for c in range(width):
            if int(body[r, c]) != int(wall):
                continue

            if 0 < c < width - 1:
                left = int(room_map[r, c - 1])
                right = int(room_map[r, c + 1])
                if left != -1 and right != -1 and left != right:
                    graph[left].add(right)
                    graph[right].add(left)

            if 0 < r < height - 1:
                up = int(room_map[r - 1, c])
                down = int(room_map[r + 1, c])
                if up != -1 and down != -1 and up != down:
                    graph[up].add(down)
                    graph[down].add(up)
    return graph


def _room_distances(graph: list[set[int]], source_room: int) -> list[int]:
    distances = [-1] * len(graph)
    queue: deque[int] = deque([int(source_room)])
    distances[int(source_room)] = 0

    while queue:
        room = queue.popleft()
        for nxt in graph[room]:
            if distances[nxt] != -1:
                continue
            distances[nxt] = distances[room] + 1
            queue.append(nxt)

    if any(distance < 0 for distance in distances):
        raise ValueError("room graph is disconnected")
    return distances


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-separated room with inward Chebyshev bands.
    The palette phase for a room is offset by its graph distance from the seeded room.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    separator = _separator_row(grid)
    wall = int(grid[separator, 0])
    palette = _palette_from_row(grid[0])
    body = grid[separator + 1 :]

    rooms, room_map = _room_ids(body, wall)
    if not rooms:
        raise ValueError("expected at least one room")

    source_room = _find_source_room(body, rooms)
    graph = _build_room_graph(body, wall, room_map)
    room_distance = _room_distances(graph, source_room)

    out = grid.copy()
    for idx, coords in enumerate(rooms):
        mask = np.zeros(body.shape, dtype=bool)
        for r, c in coords:
            mask[int(r), int(c)] = True
        layer = _boundary_distance(mask)
        phase = int(room_distance[idx])
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            out[separator + 1 + rr, cc] = int(palette[(phase + int(layer[rr, cc])) % len(palette)])
    return out


def _split_lengths(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    lengths: list[int] = []
    for cut in cuts + [total]:
        lengths.append(cut - prev)
        prev = cut
    return lengths


def _make_palette_row(width: int, palette: list[int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    start = 0
    for color, run in zip(palette, _split_lengths(width, len(palette))):
        row[start : start + run] = int(color)
        start += run
    return row


def _build_layout(body_h: int, body_w: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    wall_mask = np.zeros((body_h, body_w), dtype=bool)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True

    rects = [(1, 1, body_h - 2, body_w - 2)]
    split_count = random.randint(2, 5)

    for _ in range(split_count):
        candidates: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            if r1 - r0 + 1 >= SPLITTABLE_LEN:
                candidates.append((idx, "h"))
            if c1 - c0 + 1 >= SPLITTABLE_LEN:
                candidates.append((idx, "v"))
        if not candidates:
            break

        rect_idx, axis = random.choice(candidates)
        r0, c0, r1, c1 = rects.pop(rect_idx)

        if axis == "h":
            split_r = random.randint(r0 + MIN_ROOM_DIM, r1 - MIN_ROOM_DIM)
            wall_mask[split_r, c0 : c1 + 1] = True
            rects.append((r0, c0, split_r - 1, c1))
            rects.append((split_r + 1, c0, r1, c1))
        else:
            split_c = random.randint(c0 + MIN_ROOM_DIM, c1 - MIN_ROOM_DIM)
            wall_mask[r0 : r1 + 1, split_c] = True
            rects.append((r0, c0, r1, split_c - 1))
            rects.append((r0, split_c + 1, r1, c1))

    return wall_mask, rects


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        body_h = random.randint(8, 22)
        body_w = random.randint(10, 26)
        if body_h + 2 > MAX_GRID_SIZE or body_w > MAX_GRID_SIZE:
            continue

        palette_len = random.randint(3, 6)
        total_colors = palette_len + 3
        if total_colors > len(Color.ALL_COLORS):
            continue

        wall_mask, rooms = _build_layout(body_h, body_w)
        if not (3 <= len(rooms) <= 6):
            continue

        color_pool = random.sample(Color.ALL_COLORS, total_colors)
        wall = int(color_pool[0])
        base = int(color_pool[1])
        marker = int(color_pool[2])
        palette = [int(value) for value in color_pool[3:]]

        grid = np.full((body_h + 2, body_w), base, dtype=int)
        grid[0] = _make_palette_row(body_w, palette)
        grid[1, :] = wall
        grid[2:, :] = base
        grid[2:, :][wall_mask] = wall

        room = random.choice(rooms)
        r0, c0, r1, c1 = room
        marker_r = random.randint(r0, r1)
        marker_c = random.randint(c0, c1)
        grid[2 + marker_r, marker_c] = marker
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
