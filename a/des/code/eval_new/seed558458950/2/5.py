"""Fill the rooms of a tree maze from the start room to the dead-end room by the
sequence of door colors encountered along that path.

Input structure:
- Grid background is black, walls are one-cell grey lines, and rooms are
  axis-aligned black rectangles.
- Rooms are connected by door gaps in the wall network; the maze topology is a tree.
- Exactly one room has a maroon start marker.
- Some doors expose a same-colored pair of marker pixels on opposite sides of the
  gap (red/blue/green/yellow/orange-like palette in this implementation).

Rule:
- Build room components from black/maroon cells and build doorway edges from marker
  pairs separated by one black gap.
- Find the room graph, locate the start room, and walk along the unique path to a
  leaf room.
- Record the doorway colors in that room-traversal order, then recolor rooms on the
  path cyclically with these colors.
- Only black cells inside visited rooms are replaced; maroon start marker and all
  doorway pixels remain unchanged; grey walls are preserved.

Key insight:
- Doorway pairs determine traversal order, and that ordered palette is the only
  signal needed to assign colors back to rooms.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np

BG = 0
WALL = 5
MAROON = 9
DOOR_PALETTE = (1, 2, 3, 4, 6)

_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _connected_components(mask: np.ndarray) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    comp_id = -np.ones((h, w), dtype=np.int32)
    comps: List[List[Tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or comp_id[r, c] != -1:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells: List[Tuple[int, int]] = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if not mask[nr, nc] or comp_id[nr, nc] != -1:
                        continue
                    comp_id[nr, nc] = cid
                    q.append((nr, nc))
                    cells.append((nr, nc))

            comps.append(cells)

    return comps, comp_id


def _safe_unique_color(grid: np.ndarray) -> int | None:
    vals, counts = np.unique(grid, return_counts=True)
    if len(vals) == 0:
        return None

    bg = int(vals[np.argmax(counts)])
    for color, count in zip(vals, counts):
        color = int(color)
        if color == bg:
            continue
        if color == WALL:
            continue
        if count == 1:
            return color
    return None


def _pair_door_markers(
    grid: np.ndarray,
    bg: int,
    wall: int,
    start_color: int | None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
    h, w = grid.shape
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
    used = set()

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or color == wall or (start_color is not None and color == start_color):
                continue
            if (r, c) in used:
                continue

            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + 2 * dr, c + 2 * dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if (rr, cc) in used:
                    continue
                if int(grid[rr, cc]) != color:
                    continue
                mr, mc = r + dr, c + dc
                if grid[mr, mc] != bg:
                    continue

                a = (r, c)
                b = (rr, cc)
                if b < a:
                    a, b = b, a
                pairs.append((a, b, color))
                used.add(a)
                used.add(b)
                break

    return pairs


def _room_components(grid: np.ndarray, bg: int, start_color: int | None) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
    mask = grid == bg
    if start_color is not None:
        mask = np.logical_or(mask, grid == start_color)

    comps, comp_id = _connected_components(mask)

    room_id_map = -np.ones(comp_id.shape, dtype=np.int32)
    rooms: List[List[Tuple[int, int]]] = []
    for cid, cells in enumerate(comps):
        if len(cells) < 4:
            continue
        rid = len(rooms)
        for rr, cc in cells:
            room_id_map[rr, cc] = rid
        rooms.append(cells)

    return rooms, room_id_map


def _build_room_graph(
    grid: np.ndarray,
    rooms: List[List[Tuple[int, int]]],
    room_id_map: np.ndarray,
    bg: int,
    wall: int,
    start_color: int | None,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], List[int]]]:
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(rooms))}
    edge_colors: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    pairs = _pair_door_markers(grid, bg, wall, start_color)
    h, w = grid.shape

    def marker_room(r: int, c: int) -> int | None:
        candidates = set()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            rid = int(room_id_map[nr, nc])
            if rid >= 0:
                candidates.add(rid)
        for rid in candidates:
            if len(rooms[rid]) >= 4:
                return rid
        return None

    for (r1, c1), (r2, c2), color in pairs:
        ra = marker_room(r1, c1)
        rb = marker_room(r2, c2)
        if ra is None or rb is None or ra == rb:
            continue

        a, b = sorted((ra, rb))
        adj[a].add(b)
        adj[b].add(a)
        edge_colors[(a, b)].append(color)

    return adj, edge_colors


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    wall_candidates = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg]
    if wall_candidates:
        wall = max(wall_candidates, key=lambda x: x[1])[0]
    else:
        wall = int(bg)

    start_color = _safe_unique_color(grid)
    if start_color is None:
        return grid

    rooms, room_id_map = _room_components(grid, bg, start_color)
    if len(rooms) <= 1:
        return grid

    start_room = -1
    for rid, cells in enumerate(rooms):
        if any(grid[r, c] == start_color for r, c in cells):
            start_room = rid
            break
    if start_room < 0:
        return grid

    adj, edge_colors = _build_room_graph(grid, rooms, room_id_map, bg, wall, start_color)
    if start_room not in adj:
        return grid

    leaves = [rid for rid, neigh in adj.items() if len(neigh) <= 1]
    if leaves:
        targets = [rid for rid in leaves if rid != start_room]
        target = targets[0] if targets else start_room
    else:
        target = None

    path = [start_room]
    prev = -1
    cur = start_room
    visited = {start_room}
    while target is not None and cur != target:
        nxts = [n for n in sorted(adj[cur]) if n != prev]
        nxt = None
        for v in nxts:
            if v not in visited:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    if len(path) <= 1:
        return grid

    color_seq: List[int] = []
    for a, b in zip(path, path[1:]):
        key = (a, b) if a < b else (b, a)
        cands = edge_colors.get(key)
        if cands:
            color_seq.append(int(cands[0]))

    if not color_seq:
        return grid

    out = np.array(grid, copy=True)
    for i, rid in enumerate(path):
        color = color_seq[i % len(color_seq)]
        for r, c in rooms[rid]:
            cell = int(out[r, c])
            if cell == bg or cell == start_color:
                out[r, c] = color

    return out


def _random_path(rng: np.random.Generator, rows: int, cols: int, target_len: int) -> List[Tuple[int, int]]:
    if rows <= 0 or cols <= 0:
        return []

    start_r = int(rng.integers(rows))
    start_c = int(rng.integers(cols))
    path = [(start_r, start_c)]
    used = {path[0]}

    while len(path) < target_len:
        r, c = path[-1]
        candidates = []
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in used:
                continue
            candidates.append((nr, nc))
        if not candidates:
            break
        nxt = candidates[int(rng.integers(len(candidates)))]
        used.add(nxt)
        path.append(nxt)

    return path


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(240):
        slot_rows = int(rng.integers(2, 5))
        slot_cols = int(rng.integers(2, 5))
        target_rooms = int(rng.integers(3, 9))

        path = _random_path(rng, slot_rows, slot_cols, target_rooms)
        if len(path) < 3:
            continue

        room_h = int(rng.integers(3, 5))
        room_w = int(rng.integers(3, 5))

        min_r = min(r for r, _ in path)
        min_c = min(c for _, c in path)
        max_r = max(r for r, _ in path)
        max_c = max(c for _, c in path)

        row_slots = max_r - min_r + 1
        col_slots = max_c - min_c + 1

        h = row_slots * (room_h + 1) + 1
        w = col_slots * (room_w + 1) + 1
        if h > 30 or w > 30:
            continue

        grid = np.full((h, w), BG, dtype=int)

        # outer boundary walls
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # paint each room interior and its surrounding walls
        slot_to_grid: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for rs, cs in path:
            top = 1 + (rs - min_r) * (room_h + 1)
            left = 1 + (cs - min_c) * (room_w + 1)
            bottom = top + room_h - 1
            right = left + room_w - 1
            slot_to_grid[(rs, cs)] = (top, left)

            grid[top - 1, left - 1 : right + 2] = WALL
            grid[bottom + 1, left - 1 : right + 2] = WALL
            grid[top - 1 : bottom + 2, left - 1] = WALL
            grid[top - 1 : bottom + 2, right + 1] = WALL

        # start marker
        sr, sc = slot_to_grid[path[0]]
        grid[sr + room_h // 2, sc + room_w // 2] = MAROON

        # open doorways for each adjacency in the path
        for i in range(len(path) - 1):
            (r0, c0) = path[i]
            (r1, c1) = path[i + 1]
            color = int(rng.choice(DOOR_PALETTE))

            if r0 == r1:
                row_idx = int(rng.integers(room_h))
                gap_row = 1 + (r0 - min_r) * (room_h + 1) + row_idx
                gap_col = 1 + (min(c0, c1) - min_c) * (room_w + 1) + room_w
            else:
                col_idx = int(rng.integers(room_w))
                gap_col = 1 + (c0 - min_c) * (room_w + 1) + col_idx
                gap_row = 1 + (min(r0, r1) - min_r) * (room_h + 1) + room_h

            if r0 == r1:
                grid[gap_row, gap_col - 1] = color
                grid[gap_row, gap_col] = BG
                grid[gap_row, gap_col + 1] = color
            else:
                grid[gap_row - 1, gap_col] = color
                grid[gap_row, gap_col] = BG
                grid[gap_row + 1, gap_col] = color

        return grid

    # fallback deterministic sample
    return np.array(
        [
            [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
            [WALL, BG, BG, BG, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, BG, MAROON, BG, BG, WALL, BG, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, 1, 1, 2, BG, BG, WALL],
            [WALL, BG, WALL, WALL, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, WALL, WALL, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, BG, BG, BG, BG, BG, BG, BG, BG, BG, WALL],
            [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
