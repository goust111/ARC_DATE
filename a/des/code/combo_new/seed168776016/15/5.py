from __future__ import annotations

import random

import numpy as np


BG = 0
DOOR = 5
WALL = 9
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _door_adjacent(g: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == DOOR:
            return True
    return False


def _extract_rooms(g: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    walkable = (g != WALL) & (g != DOOR)
    room_id = np.full(g.shape, -1, dtype=int)
    rooms: list[dict[str, object]] = []

    for coords in _components(walkable):
        if _touches_border(coords, g.shape):
            continue

        rid = len(rooms)
        for r, c in coords:
            room_id[r, c] = rid

        key_cells: list[tuple[int, int, int]] = []
        marker_cells: list[tuple[int, int, int]] = []
        bg_area = 0
        for r, c in coords:
            value = int(g[r, c])
            if value == BG:
                bg_area += 1
                continue
            if _door_adjacent(g, r, c):
                marker_cells.append((int(r), int(c), value))
            else:
                key_cells.append((int(r), int(c), value))

        assert len(key_cells) == 1, "each room must have exactly one non-door-adjacent key"
        kr, kc, key_color = key_cells[0]
        rooms.append(
            {
                "cells": coords,
                "key_pos": (int(kr), int(kc)),
                "key_color": int(key_color),
                "bg_area": int(bg_area),
                "markers": marker_cells,
            }
        )

    key_colors = [int(room["key_color"]) for room in rooms]
    assert len(key_colors) == len(set(key_colors)), "room key colors must be globally unique"
    return rooms, room_id


def _door_sides(room_id: np.ndarray, r: int, c: int) -> list[tuple[tuple[int, int], int]]:
    up = int(room_id[r - 1, c]) if r > 0 else -1
    down = int(room_id[r + 1, c]) if r + 1 < room_id.shape[0] else -1
    left = int(room_id[r, c - 1]) if c > 0 else -1
    right = int(room_id[r, c + 1]) if c + 1 < room_id.shape[1] else -1

    lr = []
    ud = []
    if left >= 0:
        lr.append(((int(r), int(c - 1)), left))
    if right >= 0:
        lr.append(((int(r), int(c + 1)), right))
    if up >= 0:
        ud.append(((int(r - 1), int(c)), up))
    if down >= 0:
        ud.append(((int(r + 1), int(c)), down))

    if len(lr) == 2 and not ud:
        return lr
    if len(ud) == 2 and not lr:
        return ud
    if len(lr) == 2:
        return lr
    if len(ud) == 2:
        return ud
    raise AssertionError("each door must separate exactly two opposite rooms")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-matched doors between walled rooms, merge the connected rooms, and repaint the surviving walls by the largest room's key color.

    Input structure:
    - Black background surrounds one or more maroon wall compounds made of rectangular rooms.
    - Inside each room there is exactly one colored key pixel that is not adjacent to any grey door.
    - Some wall pixels are grey doors, and each grey door has exactly one colored marker pixel immediately inside one of its two adjacent rooms.

    Rule:
    - For each grey door, compare the marker color with the key color of the room on the opposite side of that door.
    - If the colors match, open the door by turning the grey pixel to background, erase that used marker, and merge the two rooms through the opening.
    - After all doors are resolved, erase every key pixel; each merged room group repaints its remaining maroon walls with the key color of the original room in that group that has the most background interior cells; unused grey doors stay grey.

    Key insight:
    - The colored door-adjacent pixels are not extra keys: each one is a directional requirement attached to one side of a door, so solving the task means pairing that requirement with the opposite room, then using the resulting room connectivity to choose one dominant key color for the whole merged wall network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms, room_id = _extract_rooms(g)
    assert rooms, "expected at least one enclosed room"

    dsu = _DSU(len(rooms))
    opened_doors: set[tuple[int, int]] = set()
    erased_markers: set[tuple[int, int]] = set()

    for r, c in np.argwhere(g == DOOR):
        rr = int(r)
        cc = int(c)
        sides = _door_sides(room_id, rr, cc)
        colored = [i for i, (pos, _rid) in enumerate(sides) if int(g[pos]) != BG]
        assert len(colored) == 1, "each door must have exactly one colored adjacent marker"

        marker_index = colored[0]
        marker_pos, marker_room = sides[marker_index]
        other_pos, other_room = sides[1 - marker_index]
        _ = other_pos
        marker_color = int(g[marker_pos])
        target_key = int(rooms[other_room]["key_color"])

        if marker_color == target_key:
            opened_doors.add((rr, cc))
            erased_markers.add(marker_pos)
            dsu.union(marker_room, other_room)

    comp_members: dict[int, list[int]] = {}
    for rid in range(len(rooms)):
        root = dsu.find(rid)
        comp_members.setdefault(root, []).append(rid)

    comp_color: dict[int, int] = {}
    comp_strength: dict[int, int] = {}
    for root, members in comp_members.items():
        best_area = max(int(rooms[rid]["bg_area"]) for rid in members)
        best_rooms = [rid for rid in members if int(rooms[rid]["bg_area"]) == best_area]
        assert len(best_rooms) == 1, "largest room in each merged group must be unique"
        best_room = best_rooms[0]
        comp_color[root] = int(rooms[best_room]["key_color"])
        comp_strength[root] = best_area

    out = g.copy()
    for room in rooms:
        kr, kc = room["key_pos"]
        out[int(kr), int(kc)] = BG

    for r, c in erased_markers:
        out[int(r), int(c)] = BG
    for r, c in opened_doors:
        out[int(r), int(c)] = BG

    for r, c in np.argwhere(g == WALL):
        rr = int(r)
        cc = int(c)
        roots = {
            dsu.find(int(room_id[nr, nc]))
            for dr, dc in DIR4
            if _in_bounds(g.shape, rr + dr, cc + dc)
            for nr, nc in [(rr + dr, cc + dc)]
            if int(room_id[nr, nc]) >= 0
        }
        if not roots:
            continue
        winner = max(sorted(roots), key=lambda root: comp_strength[root])
        out[rr, cc] = comp_color[winner]

    return out


# --------------- generator (compound_rooms) ---------------

ONE_COMPOUND_CHOICES = [(2, 2), (2, 2), (2, 3), (3, 2), (1, 3), (3, 1), (1, 2), (2, 1)]
TWO_COMPOUND_MAIN_CHOICES = [(2, 2), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1)]
TWO_COMPOUND_SIDE_CHOICES = [(1, 1), (1, 1), (1, 2), (2, 1)]
ROOM_SPAN_RANGE = (4, 6)
OUTER_MARGIN_RANGE = (1, 3)
COMPOUND_GAP_RANGE = (2, 4)
JITTER_RANGE = (0, 2)
GEN_ATTEMPTS = 400


def _weighted_shape_pair() -> list[tuple[int, int]]:
    if random.random() < 0.45:
        return [random.choice(ONE_COMPOUND_CHOICES)]
    first = random.choice(TWO_COMPOUND_MAIN_CHOICES)
    second = random.choice(TWO_COMPOUND_SIDE_CHOICES)
    return [first, second]


def _room_starts(spans: list[int]) -> list[int]:
    starts = []
    cursor = 1
    for span in spans:
        starts.append(int(cursor))
        cursor += int(span) + 1
    return starts


def _pick_open_edges(room_count: int, edges: list[dict[str, object]]) -> set[int]:
    if room_count <= 1:
        return set()

    by_room: dict[int, list[tuple[int, int]]] = {rid: [] for rid in range(room_count)}
    for idx, edge in enumerate(edges):
        a, b = edge["rooms"]
        by_room[int(a)].append((idx, int(b)))
        by_room[int(b)].append((idx, int(a)))

    start = random.randrange(room_count)
    visited = {int(start)}
    open_edges: set[int] = set()
    while len(visited) < room_count:
        frontier = [
            (idx, neighbor)
            for room in visited
            for idx, neighbor in by_room[room]
            if int(neighbor) not in visited
        ]
        assert frontier, "room adjacency must be connected"
        edge_idx, neighbor = random.choice(frontier)
        open_edges.add(int(edge_idx))
        visited.add(int(neighbor))

    extras = [idx for idx in range(len(edges)) if idx not in open_edges]
    forced_closed = random.choice(extras) if extras and random.random() < 0.85 else None
    for idx in extras:
        if idx == forced_closed:
            continue
        if random.random() < 0.30:
            open_edges.add(int(idx))
    return open_edges


def _build_compound(
    shape: tuple[int, int],
    local_key_colors: list[int],
    all_key_colors: list[int],
) -> tuple[np.ndarray, list[int]] | None:
    rows, cols = shape
    room_count = int(rows * cols)
    heights = [random.randint(*ROOM_SPAN_RANGE) for _ in range(rows)]
    widths = [random.randint(*ROOM_SPAN_RANGE) for _ in range(cols)]

    h = sum(heights) + rows + 1
    w = sum(widths) + cols + 1
    patch = np.full((h, w), WALL, dtype=int)

    row_starts = _room_starts(heights)
    col_starts = _room_starts(widths)
    room_boxes: list[dict[str, int]] = []
    for i in range(rows):
        for j in range(cols):
            r0 = row_starts[i]
            c0 = col_starts[j]
            rh = heights[i]
            cw = widths[j]
            patch[r0 : r0 + rh, c0 : c0 + cw] = BG
            room_boxes.append({"r0": r0, "c0": c0, "h": rh, "w": cw})

    def rid(i: int, j: int) -> int:
        return int(i * cols + j)

    edges: list[dict[str, object]] = []
    for i in range(rows):
        for j in range(cols):
            room = room_boxes[rid(i, j)]
            if j + 1 < cols:
                door_r = random.randint(room["r0"] + 1, room["r0"] + room["h"] - 2)
                door_c = room["c0"] + room["w"]
                edges.append(
                    {
                        "rooms": (rid(i, j), rid(i, j + 1)),
                        "door": (int(door_r), int(door_c)),
                        "markers": {
                            rid(i, j): (int(door_r), int(door_c - 1)),
                            rid(i, j + 1): (int(door_r), int(door_c + 1)),
                        },
                    }
                )
            if i + 1 < rows:
                door_r = room["r0"] + room["h"]
                door_c = random.randint(room["c0"] + 1, room["c0"] + room["w"] - 2)
                edges.append(
                    {
                        "rooms": (rid(i, j), rid(i + 1, j)),
                        "door": (int(door_r), int(door_c)),
                        "markers": {
                            rid(i, j): (int(door_r - 1), int(door_c)),
                            rid(i + 1, j): (int(door_r + 1), int(door_c)),
                        },
                    }
                )

    open_edges = _pick_open_edges(room_count, edges)
    marker_positions_by_room: dict[int, set[tuple[int, int]]] = {room: set() for room in range(room_count)}

    for idx, edge in enumerate(edges):
        a, b = edge["rooms"]
        door_r, door_c = edge["door"]
        patch[int(door_r), int(door_c)] = DOOR

        marker_side = random.choice([int(a), int(b)])
        other_side = int(b) if marker_side == int(a) else int(a)
        marker_pos = edge["markers"][marker_side]

        if idx in open_edges:
            marker_color = int(local_key_colors[other_side])
        else:
            choices = [color for color in all_key_colors if color != int(local_key_colors[other_side])]
            marker_color = int(random.choice(choices))
        patch[marker_pos] = marker_color
        marker_positions_by_room[marker_side].add(marker_pos)

    for room_idx, room in enumerate(room_boxes):
        candidates = []
        for r in range(room["r0"], room["r0"] + room["h"]):
            for c in range(room["c0"], room["c0"] + room["w"]):
                if (r, c) in marker_positions_by_room[room_idx]:
                    continue
                if any(
                    _in_bounds(patch.shape, r + dr, c + dc) and int(patch[r + dr, c + dc]) == DOOR
                    for dr, dc in DIR4
                ):
                    continue
                candidates.append((int(r), int(c)))
        if not candidates:
            return None
        key_r, key_c = random.choice(candidates)
        patch[key_r, key_c] = int(local_key_colors[room_idx])

    bg_counts = []
    for room in room_boxes:
        sub = patch[room["r0"] : room["r0"] + room["h"], room["c0"] : room["c0"] + room["w"]]
        bg_counts.append(int(np.sum(sub == BG)))
    if len(set(bg_counts)) != len(bg_counts):
        return None

    return patch, bg_counts


def _place_compounds(compounds: list[np.ndarray]) -> np.ndarray | None:
    top = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)

    if len(compounds) == 1:
        h, w = compounds[0].shape
        canvas = np.full((top + h + bottom, left + w + right), BG, dtype=int)
        canvas[top : top + h, left : left + w] = compounds[0]
        return canvas

    first, second = compounds
    h1, w1 = first.shape
    h2, w2 = second.shape
    gap = random.randint(*COMPOUND_GAP_RANGE)
    jitter = random.randint(*JITTER_RANGE)
    orientation = random.choice(["horizontal", "vertical", "diagonal"])

    if orientation == "horizontal":
        r1 = top
        c1 = left
        r2 = top + jitter
        c2 = c1 + w1 + gap
    elif orientation == "vertical":
        r1 = top
        c1 = left
        r2 = r1 + h1 + gap
        c2 = left + jitter
    else:
        r1 = top
        c1 = left
        r2 = r1 + h1 + gap
        c2 = c1 + w1 + gap

    final_h = max(r1 + h1, r2 + h2) + bottom
    final_w = max(c1 + w1, c2 + w2) + right
    if final_h > 30 or final_w > 30:
        return None

    canvas = np.full((final_h, final_w), BG, dtype=int)
    canvas[r1 : r1 + h1, c1 : c1 + w1] = first
    canvas[r2 : r2 + h2, c2 : c2 + w2] = second
    return canvas


def _gen_attempt() -> np.ndarray | None:
    shapes = _weighted_shape_pair()
    total_rooms = sum(rows * cols for rows, cols in shapes)
    if total_rooms > len(KEY_COLORS):
        return None

    sampled_colors = random.sample(KEY_COLORS, total_rooms)
    compounds: list[np.ndarray] = []
    offset = 0
    for shape in shapes:
        room_count = int(shape[0] * shape[1])
        local_colors = sampled_colors[offset : offset + room_count]
        offset += room_count
        built = _build_compound(shape, local_colors, sampled_colors)
        if built is None:
            return None
        patch, _bg_counts = built
        compounds.append(patch)

    canvas = _place_compounds(compounds)
    if canvas is None or max(canvas.shape) > 30:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != canvas.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-key-door input")
