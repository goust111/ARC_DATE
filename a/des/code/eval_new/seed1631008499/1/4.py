from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_GEN_ATTEMPTS = 400
_GEN_MODES = ("closed", "mixed", "contest")
_GEN_MODE_OFFSET = random.randrange(len(_GEN_MODES))
_GEN_COUNTER = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _lock_orientation(grid: np.ndarray, r: int, c: int) -> str | None:
    if int(grid[r, c]) in {BLACK, GREY}:
        return None

    up = int(grid[r - 1, c]) == GREY
    down = int(grid[r + 1, c]) == GREY
    left = int(grid[r, c - 1]) == GREY
    right = int(grid[r, c + 1]) == GREY

    vertical = up and down and (not left) and (not right)
    horizontal = left and right and (not up) and (not down)
    if vertical == horizontal:
        return None
    return "v" if vertical else "h"


def _classify_locks(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    h, w = grid.shape
    lock_mask = np.zeros((h, w), dtype=bool)
    orientations: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            orient = _lock_orientation(grid, r, c)
            if orient is None:
                continue
            lock_mask[r, c] = True
            orientations[(r, c)] = orient
    return lock_mask, orientations


def _label_rooms(
    grid: np.ndarray, lock_mask: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    free = (grid != GREY) & (~lock_mask)
    room_id = np.full((h, w), -1, dtype=int)
    rooms: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not free[r, c] or room_id[r, c] >= 0:
                continue

            rid = len(rooms)
            queue = deque([(r, c)])
            room_id[r, c] = rid
            cells: list[tuple[int, int]] = []
            key_cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if int(grid[rr, cc]) != BLACK:
                    key_cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and free[nr, nc] and room_id[nr, nc] < 0:
                        room_id[nr, nc] = rid
                        queue.append((nr, nc))

            assert len(key_cells) == 1, "each room must contain exactly one key"
            kr, kc = key_cells[0]
            rooms.append(
                {
                    "cells": cells,
                    "key_pos": (kr, kc),
                    "key_color": int(grid[kr, kc]),
                }
            )

    assert rooms, "expected at least one room"
    return room_id, rooms


def _open_mask(
    grid: np.ndarray,
    room_id: np.ndarray,
    rooms: list[dict[str, object]],
    orientations: dict[tuple[int, int], str],
) -> np.ndarray:
    open_mask = np.zeros_like(grid, dtype=bool)
    for (r, c), orient in orientations.items():
        if orient == "v":
            a = int(room_id[r, c - 1])
            b = int(room_id[r, c + 1])
        else:
            a = int(room_id[r - 1, c])
            b = int(room_id[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "each lock must separate two rooms"
        lock_color = int(grid[r, c])
        if lock_color in {int(rooms[a]["key_color"]), int(rooms[b]["key_color"])}:
            open_mask[r, c] = True
    return open_mask


def _multi_source_fill(
    grid: np.ndarray, rooms: list[dict[str, object]], open_mask: np.ndarray
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[open_mask] = BLACK

    passable = (grid == BLACK) | open_mask
    key_sources = sorted(
        [
            (int(room["key_pos"][0]), int(room["key_pos"][1]), int(room["key_color"]))
            for room in rooms
        ]
    )
    for r, c, _ in key_sources:
        passable[r, c] = True

    owner = np.full(grid.shape, -1, dtype=int)
    dist = np.full(grid.shape, -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for sid, (r, c, _color) in enumerate(key_sources):
        owner[r, c] = sid
        dist[r, c] = 0
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc) or not passable[nr, nc]:
                continue
            if dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            owner[nr, nc] = owner[r, c]
            queue.append((nr, nc))

    black_cells = np.argwhere(grid == BLACK)
    for r, c in black_cells:
        sid = int(owner[r, c])
        assert sid >= 0, "every floor cell must be reachable from some key"
        out[r, c] = int(key_sources[sid][2])

    return out


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    lock_mask, orientations = _classify_locks(g)
    room_id, rooms = _label_rooms(g, lock_mask)
    open_mask = _open_mask(g, room_id, rooms, orientations)
    out = _multi_source_fill(g, rooms, open_mask)
    return out, {
        "lock_mask": lock_mask,
        "orientations": orientations,
        "room_id": room_id,
        "rooms": rooms,
        "open_mask": open_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-matching locks between rectangular rooms, then spread each key color through the reachable floors.

    Input structure:
    - Black is empty floor and grey is a one-pixel wall network that partitions the grid into closed rectangular rooms.
    - Each room contains exactly one non-black interior key pixel plus otherwise black floor.
    - Some wall pixels are replaced by non-grey colored lock pixels, each sitting on a wall segment between two adjacent rooms.

    Rule:
    - First identify the rooms and the lock pixels on the walls.
    - A lock opens if either adjacent room contains a key of the same color; opened locks become black door cells, while unopened locks keep their original color.
    - After all doors open simultaneously, start a 4-connected multi-source flood fill from every key, moving through black floor and through opened door cells but not through unopened locks or grey walls.
    - Recolor every original black floor cell with the color of its nearest reachable key, while keeping the key pixels unchanged and keeping opened door pixels black.

    Key insight:
    - The task is a two-stage graph problem: colored wall pixels must first be interpreted as doors whose openness depends on the neighboring rooms' keys, and only then can the floor be colored by nearest-key distances in the resulting connectivity graph.
    """
    return _solve_details(grid)[0]


def _room_layout() -> dict[str, object] | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    heights = [random.randint(3, 6) for _ in range(room_rows)]
    widths = [random.randint(3, 6) for _ in range(room_cols)]
    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    if h > 30 or w > 30:
        return None

    row_walls = [0]
    for height in heights:
        row_walls.append(row_walls[-1] + height + 1)
    col_walls = [0]
    for width in widths:
        col_walls.append(col_walls[-1] + width + 1)

    grid = np.full((h, w), BLACK, dtype=int)
    for r in row_walls:
        grid[r, :] = GREY
    for c in col_walls:
        grid[:, c] = GREY

    rooms: list[dict[str, object]] = []
    room_index = np.full((h, w), -1, dtype=int)
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = row_walls[rr] + 1
            r1 = row_walls[rr + 1] - 1
            c0 = col_walls[cc] + 1
            c1 = col_walls[cc + 1] - 1
            rid = len(rooms)
            room_index[r0 : r1 + 1, c0 : c1 + 1] = rid
            rooms.append({"bbox": (r0, r1, c0, c1), "grid_pos": (rr, cc)})

    edges: list[dict[str, object]] = []
    edge_lookup: dict[frozenset[int], int] = {}
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            left = rr * room_cols + cc
            right = left + 1
            r0, r1, _, _ = rooms[left]["bbox"]
            wall_c = col_walls[cc + 1]
            positions = [(r, wall_c) for r in range(r0, r1 + 1)]
            idx = len(edges)
            edges.append({"rooms": (left, right), "orientation": "v", "positions": positions})
            edge_lookup[frozenset((left, right))] = idx
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            top = rr * room_cols + cc
            bottom = top + room_cols
            _, _, c0, c1 = rooms[top]["bbox"]
            wall_r = row_walls[rr + 1]
            positions = [(wall_r, c) for c in range(c0, c1 + 1)]
            idx = len(edges)
            edges.append({"rooms": (top, bottom), "orientation": "h", "positions": positions})
            edge_lookup[frozenset((top, bottom))] = idx

    return {
        "grid": grid,
        "rooms": rooms,
        "room_index": room_index,
        "edges": edges,
        "edge_lookup": edge_lookup,
        "room_rows": room_rows,
        "room_cols": room_cols,
    }


def _assign_keys(layout: dict[str, object], mode: str) -> list[dict[str, object]]:
    rooms = layout["rooms"]
    assert isinstance(rooms, list)

    palette_size = random.randint(3 if mode == "contest" else 2, min(6, len(PALETTE)))
    active = random.sample(PALETTE, palette_size)
    key_colors = [random.choice(active) for _ in rooms]
    if len(set(key_colors)) == 1:
        flip = random.randrange(len(key_colors))
        alternatives = [color for color in active if color != key_colors[flip]]
        key_colors[flip] = random.choice(alternatives)

    keyed_rooms: list[dict[str, object]] = []
    for rid, room in enumerate(rooms):
        r0, r1, c0, c1 = room["bbox"]
        kr = random.randint(r0, r1)
        kc = random.randint(c0, c1)
        keyed_rooms.append(
            {
                "bbox": room["bbox"],
                "grid_pos": room["grid_pos"],
                "key_pos": (kr, kc),
                "key_color": int(key_colors[rid]),
            }
        )
    return keyed_rooms


def _mismatching_color(exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    assert choices, "a closed lock needs some color different from both room keys"
    return random.choice(choices)


def _build_contest_tree(layout: dict[str, object]) -> tuple[set[int], list[int]] | None:
    edges = layout["edges"]
    rooms = layout["rooms"]
    edge_lookup = layout["edge_lookup"]
    assert isinstance(edges, list) and isinstance(rooms, list) and isinstance(edge_lookup, dict)

    target_size = random.randint(2, min(4, len(rooms)))
    chosen_rooms = {random.randrange(len(rooms))}
    chosen_edges: list[int] = []

    while len(chosen_rooms) < target_size:
        frontier: list[tuple[int, int]] = []
        for room in chosen_rooms:
            rr, cc = rooms[room]["grid_pos"]
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < layout["room_rows"] and 0 <= nc < layout["room_cols"]):
                    continue
                other = nr * layout["room_cols"] + nc
                if other in chosen_rooms:
                    continue
                edge_idx = edge_lookup[frozenset((room, other))]
                frontier.append((other, edge_idx))
        if not frontier:
            return None
        other, edge_idx = random.choice(frontier)
        chosen_rooms.add(other)
        chosen_edges.append(edge_idx)

    return chosen_rooms, chosen_edges


def _choose_locks(layout: dict[str, object], rooms: list[dict[str, object]], mode: str) -> tuple[dict[int, int], set[int]]:
    edges = layout["edges"]
    edge_indices = list(range(len(edges)))
    key_colors = [int(room["key_color"]) for room in rooms]
    locks: dict[int, int] = {}
    open_edges: set[int] = set()

    if mode == "closed":
        n_closed = random.randint(1, min(6, len(edges)))
        for edge_idx in random.sample(edge_indices, n_closed):
            a, b = edges[edge_idx]["rooms"]
            locks[edge_idx] = _mismatching_color({key_colors[a], key_colors[b]})
        return locks, open_edges

    if mode == "mixed":
        n_total = random.randint(2, min(8, len(edges)))
        selected = random.sample(edge_indices, n_total)
        n_open = random.randint(1, n_total - 1)
        open_sample = set(random.sample(selected, n_open))
        for edge_idx in selected:
            a, b = edges[edge_idx]["rooms"]
            if edge_idx in open_sample:
                locks[edge_idx] = random.choice((key_colors[a], key_colors[b]))
                open_edges.add(edge_idx)
            else:
                locks[edge_idx] = _mismatching_color({key_colors[a], key_colors[b]})
        return locks, open_edges

    contest = _build_contest_tree(layout)
    if contest is None:
        return {}, set()
    contest_rooms, contest_edges = contest
    contest_colors = {key_colors[rid] for rid in contest_rooms}
    if len(contest_colors) < 2:
        pivot = random.choice(sorted(contest_rooms))
        alternatives = [color for color in PALETTE if color != key_colors[pivot]]
        key_colors[pivot] = random.choice(alternatives)
        rooms[pivot]["key_color"] = key_colors[pivot]

    for edge_idx in contest_edges:
        a, b = edges[edge_idx]["rooms"]
        locks[edge_idx] = random.choice((key_colors[a], key_colors[b]))
        open_edges.add(edge_idx)

    remaining = [idx for idx in edge_indices if idx not in open_edges]
    if remaining and random.random() < 0.75:
        n_closed = random.randint(1, min(4, len(remaining)))
        for edge_idx in random.sample(remaining, n_closed):
            a, b = edges[edge_idx]["rooms"]
            locks[edge_idx] = _mismatching_color({key_colors[a], key_colors[b]})
    return locks, open_edges


def _render_scene(
    layout: dict[str, object], rooms: list[dict[str, object]], locks: dict[int, int]
) -> np.ndarray:
    grid = np.array(layout["grid"], copy=True)
    for room in rooms:
        kr, kc = room["key_pos"]
        grid[kr, kc] = int(room["key_color"])
    for edge_idx, color in locks.items():
        pos = random.choice(layout["edges"][edge_idx]["positions"])
        grid[pos] = int(color)
    return grid


def _distance_map(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    queue = deque([start])
    dist[start] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(passable.shape, nr, nc) or not passable[nr, nc] or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


def _has_multicolor_tie(grid: np.ndarray, open_mask: np.ndarray, rooms: list[dict[str, object]]) -> bool:
    passable = (grid == BLACK) | open_mask
    sources = [(int(room["key_pos"][0]), int(room["key_pos"][1]), int(room["key_color"])) for room in rooms]
    for r, c, _ in sources:
        passable[r, c] = True
    dist_maps = [_distance_map(passable, (r, c)) for r, c, _ in sources]

    for r, c in np.argwhere(grid == BLACK):
        best = None
        colors: set[int] = set()
        for dist, (_, _, color) in zip(dist_maps, sources, strict=False):
            d = int(dist[r, c])
            if d < 0:
                continue
            if best is None or d < best:
                best = d
                colors = {int(color)}
            elif d == best:
                colors.add(int(color))
        if best is None or len(colors) > 1:
            return True
    return False


def _cross_room_spread(out: np.ndarray, room_index: np.ndarray, rooms: list[dict[str, object]], grid: np.ndarray) -> bool:
    for r, c in np.argwhere(grid == BLACK):
        rid = int(room_index[r, c])
        assert rid >= 0
        if int(out[r, c]) != int(rooms[rid]["key_color"]):
            return True
    return False


def _gen_attempt(mode: str) -> np.ndarray | None:
    layout = _room_layout()
    if layout is None:
        return None

    rooms = _assign_keys(layout, mode)
    locks, expected_open_edges = _choose_locks(layout, rooms, mode)
    if not locks:
        return None

    grid = _render_scene(layout, rooms, locks)
    out, details = _solve_details(grid)

    parsed_lock_positions = {tuple(map(int, rc)) for rc in np.argwhere(details["lock_mask"])}
    parsed_open_positions = {tuple(map(int, rc)) for rc in np.argwhere(details["open_mask"])}
    expected_lock_positions = set()
    expected_open_positions = set()
    for edge_idx, color in locks.items():
        del color
        cell = next(
            (pos for pos in layout["edges"][edge_idx]["positions"] if int(grid[pos]) != GREY),
            None,
        )
        assert cell is not None
        expected_lock_positions.add(cell)
        if edge_idx in expected_open_edges:
            expected_open_positions.add(cell)

    if parsed_lock_positions != expected_lock_positions:
        return None
    if parsed_open_positions != expected_open_positions:
        return None
    if _has_multicolor_tie(grid, details["open_mask"], rooms):
        return None

    if mode == "closed" and parsed_open_positions:
        return None
    if mode == "mixed" and (not parsed_open_positions or len(parsed_open_positions) == len(parsed_lock_positions)):
        return None
    if mode == "contest":
        component_colors = {int(rooms[rid]["key_color"]) for rid in range(len(rooms))}
        if len(component_colors) < 2:
            return None
        if not parsed_open_positions:
            return None
        if not _cross_room_spread(out, layout["room_index"], rooms, grid):
            return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(MAX_GEN_ATTEMPTS):
        mode = _GEN_MODES[(_GEN_MODE_OFFSET + _GEN_COUNTER) % len(_GEN_MODES)]
        _GEN_COUNTER += 1
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-lock scene")
