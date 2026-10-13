import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8]
COLOR_PRIORITY = {color: idx for idx, color in enumerate(KEY_COLORS)}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _parse_rooms(grid: np.ndarray):
    room_mask = (grid != MAROON) & (grid != GREY)
    rooms = []
    coord_to_room: dict[tuple[int, int], int] = {}
    for coords in _components(room_mask):
        key_cells = [
            (r, c)
            for r, c in coords
            if int(grid[r, c]) not in (BLACK, GREY, MAROON)
        ]
        assert len(key_cells) == 1, "each room must contain exactly one key pixel"
        key_r, key_c = key_cells[0]
        anchor = min(coords)
        black_area = sum(int(grid[r, c] == BLACK) for r, c in coords)
        rid = len(rooms)
        for rc in coords:
            coord_to_room[rc] = rid
        rooms.append(
            {
                "cells": coords,
                "anchor": anchor,
                "key_color": int(grid[key_r, key_c]),
                "area": int(black_area),
            }
        )
    assert rooms, "expected at least one room"
    return rooms, coord_to_room


def _door_rooms(
    coord_to_room: dict[tuple[int, int], int], r: int, c: int
) -> tuple[int, int]:
    left = coord_to_room.get((r, c - 1))
    right = coord_to_room.get((r, c + 1))
    up = coord_to_room.get((r - 1, c))
    down = coord_to_room.get((r + 1, c))

    horizontal_split = left is not None and right is not None and left != right
    vertical_split = up is not None and down is not None and up != down
    assert horizontal_split ^ vertical_split, "door must separate exactly two rooms"

    if horizontal_split:
        return int(left), int(right)
    return int(up), int(down)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open same-key doors, close different-key doors, merge connected rooms, and
    fill each merged region using the largest room's key color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms, coord_to_room = _parse_rooms(g)
    parent = list(range(len(rooms)))
    out = g.copy()
    open_doors: list[tuple[int, int, int, int]] = []

    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        a, b = _door_rooms(coord_to_room, rr, cc)
        if rooms[a]["key_color"] == rooms[b]["key_color"]:
            out[rr, cc] = BLACK
            open_doors.append((rr, cc, a, b))
            _union(parent, a, b)
        else:
            out[rr, cc] = MAROON

    groups: dict[int, list[int]] = defaultdict(list)
    for rid in range(len(rooms)):
        groups[_find(parent, rid)].append(rid)

    def room_rank(rid: int):
        info = rooms[rid]
        return (
            -int(info["area"]),
            COLOR_PRIORITY.get(int(info["key_color"]), 99),
            int(info["anchor"][0]),
            int(info["anchor"][1]),
        )

    for root, members in groups.items():
        if len(members) == 1:
            continue
        chosen = min(members, key=room_rank)
        fill_color = int(rooms[chosen]["key_color"])

        for rid in members:
            for r, c in rooms[rid]["cells"]:
                if out[r, c] == BLACK:
                    out[r, c] = fill_color

        for r, c, a, b in open_doors:
            if _find(parent, a) == root and _find(parent, b) == root and out[r, c] == BLACK:
                out[r, c] = fill_color

    return out


def _room_id(ri: int, ci: int, n_cols: int) -> int:
    return ri * n_cols + ci


def _row_col_edges(sizes: list[int]) -> list[int]:
    edges = [0]
    cur = 0
    for size in sizes:
        cur += size + 1
        edges.append(cur)
    return edges


def _room_cells(
    row_edges: list[int], col_edges: list[int], ri: int, ci: int
) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(row_edges[ri] + 1, row_edges[ri + 1])
        for c in range(col_edges[ci] + 1, col_edges[ci + 1])
    ]


def _adjacency_specs(heights: list[int], widths: list[int]):
    n_rows = len(heights)
    n_cols = len(widths)
    row_edges = _row_col_edges(heights)
    col_edges = _row_col_edges(widths)
    specs = []

    for ri in range(n_rows):
        for ci in range(n_cols - 1):
            wall_c = col_edges[ci + 1]
            span = [(r, wall_c) for r in range(row_edges[ri] + 1, row_edges[ri + 1])]
            specs.append(
                {
                    "a": _room_id(ri, ci, n_cols),
                    "b": _room_id(ri, ci + 1, n_cols),
                    "orientation": "vertical",
                    "span": span,
                }
            )

    for ri in range(n_rows - 1):
        for ci in range(n_cols):
            wall_r = row_edges[ri + 1]
            span = [(wall_r, c) for c in range(col_edges[ci] + 1, col_edges[ci + 1])]
            specs.append(
                {
                    "a": _room_id(ri, ci, n_cols),
                    "b": _room_id(ri + 1, ci, n_cols),
                    "orientation": "horizontal",
                    "span": span,
                }
            )

    return specs, row_edges, col_edges


def _pick_chain(adjacencies: list[dict], room_count: int):
    by_room: dict[int, list[int]] = defaultdict(list)
    for idx, adj in enumerate(adjacencies):
        by_room[int(adj["a"])].append(idx)
        by_room[int(adj["b"])].append(idx)

    candidates = []
    for mid in range(room_count):
        incident = by_room[mid]
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                e1 = adjacencies[incident[i]]
                e2 = adjacencies[incident[j]]
                left = int(e1["b"] if e1["a"] == mid else e1["a"])
                right = int(e2["b"] if e2["a"] == mid else e2["a"])
                if left == right:
                    continue
                candidates.append((incident[i], incident[j], {left, mid, right}))
    if not candidates:
        return None
    return random.choice(candidates)


def _base_colors(n_rows: int, n_cols: int, palette: list[int]) -> dict[int, int]:
    mode = random.choice(["random", "rows", "cols", "patch"])
    colors: dict[int, int] = {}

    if mode == "rows":
        row_colors = [random.choice(palette) for _ in range(n_rows)]
        for ri in range(n_rows):
            for ci in range(n_cols):
                colors[_room_id(ri, ci, n_cols)] = row_colors[ri]
    elif mode == "cols":
        col_colors = [random.choice(palette) for _ in range(n_cols)]
        for ri in range(n_rows):
            for ci in range(n_cols):
                colors[_room_id(ri, ci, n_cols)] = col_colors[ci]
    elif mode == "patch":
        for ri in range(n_rows):
            for ci in range(n_cols):
                rid = _room_id(ri, ci, n_cols)
                choices = [random.choice(palette)]
                if ri > 0:
                    choices.extend([colors[_room_id(ri - 1, ci, n_cols)]] * 2)
                if ci > 0:
                    choices.extend([colors[_room_id(ri, ci - 1, n_cols)]] * 2)
                colors[rid] = random.choice(choices)
    else:
        for ri in range(n_rows):
            for ci in range(n_cols):
                colors[_room_id(ri, ci, n_cols)] = random.choice(palette)

    return colors


def _door_region_stats(
    room_areas: dict[int, int],
    room_colors: dict[int, int],
    door_edges: set[int],
    adjacencies: list[dict],
):
    room_count = len(room_areas)
    parent = list(range(room_count))
    open_edges = []
    closed_edges = []

    for idx in door_edges:
        adj = adjacencies[idx]
        a = int(adj["a"])
        b = int(adj["b"])
        if room_colors[a] == room_colors[b]:
            open_edges.append(idx)
            _union(parent, a, b)
        else:
            closed_edges.append(idx)

    groups: dict[int, list[int]] = defaultdict(list)
    for rid in range(room_count):
        groups[_find(parent, rid)].append(rid)

    return {
        "groups": groups,
        "open_edges": open_edges,
        "closed_edges": closed_edges,
    }


def _largest_unique(groups: dict[int, list[int]], room_areas: dict[int, int]) -> bool:
    for members in groups.values():
        if len(members) == 1:
            continue
        areas = [room_areas[rid] for rid in members]
        max_area = max(areas)
        if sum(area == max_area for area in areas) != 1:
            return False
    return True


def _place_keys(
    grid: np.ndarray,
    row_edges: list[int],
    col_edges: list[int],
    n_rows: int,
    n_cols: int,
    room_colors: dict[int, int],
) -> None:
    for ri in range(n_rows):
        for ci in range(n_cols):
            cells = _room_cells(row_edges, col_edges, ri, ci)
            kr, kc = random.choice(cells)
            grid[kr, kc] = room_colors[_room_id(ri, ci, n_cols)]


def _build_grid(
    heights: list[int],
    widths: list[int],
    room_colors: dict[int, int],
    door_edges: set[int],
    adjacencies: list[dict],
    row_edges: list[int],
    col_edges: list[int],
) -> np.ndarray:
    h = row_edges[-1] + 1
    w = col_edges[-1] + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[row_edges, :] = MAROON
    grid[:, col_edges] = MAROON

    for idx in door_edges:
        span = adjacencies[idx]["span"]
        door_count = 2 if len(span) >= 4 and random.random() < 0.35 else 1
        for r, c in random.sample(span, door_count):
            grid[r, c] = GREY

    _place_keys(grid, row_edges, col_edges, len(heights), len(widths), room_colors)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    heights = [random.randint(2, 5) for _ in range(n_rows)]
    widths = [random.randint(2, 5) for _ in range(n_cols)]
    adjacencies, row_edges, col_edges = _adjacency_specs(heights, widths)
    room_count = n_rows * n_cols
    if len(adjacencies) < 3:
        return None

    palette_size = random.randint(3, min(5, len(KEY_COLORS)))
    palette = random.sample(KEY_COLORS, palette_size)
    room_colors = _base_colors(n_rows, n_cols, palette)
    room_areas = {
        _room_id(ri, ci, n_cols): heights[ri] * widths[ci] - 1
        for ri in range(n_rows)
        for ci in range(n_cols)
    }

    need_chain = random.random() < 0.5
    chain = _pick_chain(adjacencies, room_count) if need_chain else None
    chain_edges: set[int] = set()
    chain_rooms: set[int] = set()
    if chain is not None:
        e1, e2, rooms = chain
        chain_edges = {int(e1), int(e2)}
        chain_rooms = {int(r) for r in rooms}
        chain_color = random.choice(palette)
        for rid in chain_rooms:
            room_colors[rid] = chain_color

    closed_candidates = [
        idx
        for idx, adj in enumerate(adjacencies)
        if idx not in chain_edges
        and not (int(adj["a"]) in chain_rooms and int(adj["b"]) in chain_rooms)
    ]
    if not closed_candidates:
        closed_candidates = [idx for idx in range(len(adjacencies)) if idx not in chain_edges]
    if not closed_candidates:
        return None

    forced_closed = random.choice(closed_candidates)
    a = int(adjacencies[forced_closed]["a"])
    b = int(adjacencies[forced_closed]["b"])
    if room_colors[a] == room_colors[b]:
        choices = [color for color in palette if color != room_colors[a]]
        if not choices:
            return None
        if b not in chain_rooms:
            room_colors[b] = random.choice(choices)
        elif a not in chain_rooms:
            room_colors[a] = random.choice(choices)
        else:
            return None

    door_edges = set(chain_edges)
    door_edges.add(int(forced_closed))
    for idx in range(len(adjacencies)):
        if idx in door_edges:
            continue
        if random.random() < 0.55:
            door_edges.add(idx)

    if len(door_edges) < 3:
        extras = [idx for idx in range(len(adjacencies)) if idx not in door_edges]
        random.shuffle(extras)
        for idx in extras[: 3 - len(door_edges)]:
            door_edges.add(idx)

    stats = _door_region_stats(room_areas, room_colors, door_edges, adjacencies)
    groups = stats["groups"]
    open_edges = stats["open_edges"]
    closed_edges = stats["closed_edges"]

    if not open_edges or not closed_edges:
        return None
    if need_chain and max(len(members) for members in groups.values()) < 3:
        return None
    if not any(len(members) > 1 for members in groups.values()):
        return None
    if not any(len(members) == 1 for members in groups.values()):
        return None
    if not _largest_unique(groups, room_areas):
        return None

    grid = _build_grid(
        heights, widths, room_colors, door_edges, adjacencies, row_edges, col_edges
    )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(grid == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
