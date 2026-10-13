from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 1
MARKER = 2
STRIP_PALETTE = [3, 4, 5, 6, 7, 8, 9]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            label = len(components)
            labels[r, c] = label
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(cells)
    return labels, components


def _find_strip(grid: np.ndarray, bg: int, wall_top: int) -> tuple[int, int, list[int]]:
    candidates: list[tuple[int, int, int, list[int]]] = []
    for r in range(wall_top):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) < 2:
            continue
        if cols[-1] - cols[0] + 1 != len(cols):
            continue
        colors = [int(v) for v in grid[r, cols]]
        if WALL in colors or MARKER in colors:
            continue
        if len(set(colors)) != len(colors):
            continue
        candidates.append((len(cols), r, int(cols[0]), colors))
    assert candidates, "could not find the color strip"
    length, row, left, colors = max(candidates, key=lambda item: (item[0], -item[1]))
    assert length == len(colors)
    return row, left, colors


def _gap_orientation(grid: np.ndarray, r: int, c: int) -> str | None:
    if grid[r, c] != BG:
        return None
    vertical_wall_gap = (
        grid[r - 1, c] == WALL
        and grid[r + 1, c] == WALL
        and grid[r, c - 1] != WALL
        and grid[r, c + 1] != WALL
    )
    horizontal_wall_gap = (
        grid[r, c - 1] == WALL
        and grid[r, c + 1] == WALL
        and grid[r - 1, c] != WALL
        and grid[r + 1, c] != WALL
    )
    if vertical_wall_gap:
        return "vertical"
    if horizontal_wall_gap:
        return "horizontal"
    return None


def _parse_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    wall_coords = np.argwhere(g == WALL)
    assert wall_coords.size > 0, "missing walls"
    wall_bbox = bbox_coords(wall_coords)
    r0, c0, r1, c1 = wall_bbox

    marker_cells = np.argwhere(g == MARKER)
    assert marker_cells.shape == (1, 2), "expected one source marker"
    marker = tuple(int(v) for v in marker_cells[0])
    assert r0 < marker[0] < r1 and c0 < marker[1] < c1, "marker must be inside the chamber"

    strip_row, strip_left, strip_colors = _find_strip(g, bg, r0)

    gap_cells: list[tuple[int, int, str]] = []
    gap_mask = np.zeros(g.shape, dtype=bool)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            orient = _gap_orientation(g, r, c)
            if orient is None:
                continue
            gap_cells.append((r, c, orient))
            gap_mask[r, c] = True
    assert gap_cells, "expected wall gaps"

    domain = np.zeros(g.shape, dtype=bool)
    domain[r0 : r1 + 1, c0 : c1 + 1] = True
    sealed_passable = domain & (g != WALL) & (~gap_mask)
    labels, components = _label_components(sealed_passable)
    source_label = int(labels[marker])
    assert source_label >= 0

    graph = {idx: set() for idx in range(len(components))}
    for r, c, orient in gap_cells:
        if orient == "vertical":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        assert a >= 0 and b >= 0 and a != b
        graph[a].add(b)
        graph[b].add(a)

    distances = {source_label: 0}
    queue = deque([source_label])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in distances:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    assert len(distances) == len(components), "all rooms must be reachable"
    assert max(distances.values()) < len(strip_colors), "strip too short for room distances"

    return {
        "bg": bg,
        "wall_bbox": wall_bbox,
        "strip": (strip_row, strip_left, strip_colors),
        "marker": marker,
        "gap_cells": gap_cells,
        "labels": labels,
        "components": components,
        "distances": distances,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Color each rectangular room by the strip color indexed by how many wall gaps separate it from the source room.

    Input structure:
    - One color forms a one-cell-thick wall network that encloses a rectangular chamber and subdivides it into rectangular rooms.
    - Each internal wall segment contains a single one-cell background gap, and one room contains a unique source marker pixel.
    - Above the chamber sits a horizontal strip of distinct colors separated from the walls by background.

    Rule:
    - Treat every gap as if the missing wall cell were restored, and use that sealed wall network to split the chamber into rooms.
    - Then treat each gap as a doorway between the two rooms on its opposite sides, run breadth-first search from the source room, and assign distance d rooms the (d+1)-th strip color.
    - Recolor only background cells inside each room; keep gaps as background and leave the walls, marker, and top strip unchanged.

    Key insight:
    - A gap has two roles at once: it must be closed to identify the rooms, but reopened as a graph edge to measure doorway distance and pick the correct strip color.
    """
    parsed = _parse_case(np.asarray(input_grid, dtype=int))
    g = np.asarray(input_grid, dtype=int)
    out = g.copy()

    for label, cells in enumerate(parsed["components"]):
        color = parsed["strip"][2][parsed["distances"][label]]
        for r, c in cells:
            if out[r, c] == parsed["bg"]:
                out[r, c] = color
    return out


def _room_spans(lengths: list[int], offset: int) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    walls = [offset]
    cursor = offset + 1
    for length in lengths:
        spans.append((cursor, cursor + length - 1))
        cursor += length
        walls.append(cursor)
        cursor += 1
    return spans, walls


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for room, bounds in meta["rooms"].items():
        color = meta["strip_colors"][meta["distances"][room]]
        r0, r1, c0, c1 = bounds
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    mr, mc = meta["marker"]
    out[mr, mc] = MARKER
    for r, c, _orient in meta["gap_cells"]:
        out[r, c] = BG
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, dict]:
    for _ in range(500):
        room_rows = rng.randint(2, 4)
        room_cols = rng.randint(2, 4)
        row_heights = [rng.randint(2, 4) for _ in range(room_rows)]
        col_widths = [rng.randint(2, 4) for _ in range(room_cols)]

        chamber_h = sum(row_heights) + room_rows + 1
        chamber_w = sum(col_widths) + room_cols + 1

        chamber_top = rng.randint(2, 4)
        chamber_left = rng.randint(1, 4)
        bottom_margin = rng.randint(1, 4)
        right_margin = rng.randint(1, 4)

        source_room = (rng.randrange(room_rows), rng.randrange(room_cols))
        max_distance = max(
            abs(r - source_room[0]) + abs(c - source_room[1])
            for r in range(room_rows)
            for c in range(room_cols)
        )
        if max_distance + 1 > len(STRIP_PALETTE):
            continue
        if max_distance < 2:
            continue

        strip_colors = rng.sample(STRIP_PALETTE, max_distance + 1)

        h = chamber_top + chamber_h + bottom_margin
        w = chamber_left + chamber_w + right_margin
        w = max(w, len(strip_colors) + 2)
        if h > 30 or w > 30:
            continue

        grid = np.full((h, w), BG, dtype=int)
        row_spans, horizontal_walls = _room_spans(row_heights, chamber_top)
        col_spans, vertical_walls = _room_spans(col_widths, chamber_left)

        top_wall = horizontal_walls[0]
        bottom_wall = horizontal_walls[-1]
        left_wall = vertical_walls[0]
        right_wall = vertical_walls[-1]

        grid[top_wall : bottom_wall + 1, left_wall] = WALL
        grid[top_wall : bottom_wall + 1, right_wall] = WALL
        grid[top_wall, left_wall : right_wall + 1] = WALL
        grid[bottom_wall, left_wall : right_wall + 1] = WALL
        for wall_r in horizontal_walls[1:-1]:
            grid[wall_r, left_wall : right_wall + 1] = WALL
        for wall_c in vertical_walls[1:-1]:
            grid[top_wall : bottom_wall + 1, wall_c] = WALL

        gap_cells: list[tuple[int, int, str]] = []
        for wall_c in vertical_walls[1:-1]:
            for rr, (room_r0, room_r1) in enumerate(row_spans):
                gap_r = rng.randint(room_r0, room_r1)
                grid[gap_r, wall_c] = BG
                gap_cells.append((gap_r, wall_c, "vertical"))
        for wall_r in horizontal_walls[1:-1]:
            for cc, (room_c0, room_c1) in enumerate(col_spans):
                gap_c = rng.randint(room_c0, room_c1)
                grid[wall_r, gap_c] = BG
                gap_cells.append((wall_r, gap_c, "horizontal"))

        rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for rr, (room_r0, room_r1) in enumerate(row_spans):
            for cc, (room_c0, room_c1) in enumerate(col_spans):
                rooms[(rr, cc)] = (room_r0, room_r1, room_c0, room_c1)

        sr0, sr1, sc0, sc1 = rooms[source_room]
        marker = (rng.randint(sr0, sr1), rng.randint(sc0, sc1))
        grid[marker] = MARKER

        strip_left = rng.randint(0, w - len(strip_colors))
        grid[0, strip_left : strip_left + len(strip_colors)] = np.array(strip_colors, dtype=int)

        distances = {
            room: abs(room[0] - source_room[0]) + abs(room[1] - source_room[1])
            for room in rooms
        }
        meta = {
            "room_rows": room_rows,
            "room_cols": room_cols,
            "rooms": rooms,
            "source_room": source_room,
            "distances": distances,
            "strip_colors": strip_colors,
            "marker": marker,
            "gap_cells": gap_cells,
            "wall_bbox": (top_wall, left_wall, bottom_wall, right_wall),
        }
        expected = _expected_from_meta(grid, meta)
        meta["expected"] = expected
        return grid, meta
    raise RuntimeError("failed to build a valid case")


def _validate_generated_case(grid: np.ndarray, meta: dict) -> bool:
    grid = np.asarray(grid, dtype=int)
    expected = np.asarray(meta["expected"], dtype=int)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return False

    distinct_distances = set(meta["distances"].values())
    if len(distinct_distances) < 3:
        return False

    top_wall, left_wall, bottom_wall, right_wall = meta["wall_bbox"]
    domain = np.zeros(grid.shape, dtype=bool)
    domain[top_wall : bottom_wall + 1, left_wall : right_wall + 1] = True
    sealed_gap_mask = np.zeros(grid.shape, dtype=bool)
    for r, c, _orient in meta["gap_cells"]:
        sealed_gap_mask[r, c] = True

    sealed_passable = domain & (grid != WALL) & (~sealed_gap_mask)
    open_passable = domain & (grid != WALL)
    _sealed_labels, sealed_components = _label_components(sealed_passable)
    _open_labels, open_components = _label_components(open_passable)
    if len(sealed_components) != meta["room_rows"] * meta["room_cols"]:
        return False
    if len(open_components) != 1:
        return False

    strip_row, strip_left, strip_colors = _find_strip(grid, BG, top_wall)
    if strip_row != 0 or strip_left < 0 or strip_colors != meta["strip_colors"]:
        return False

    if actual[meta["marker"]] != MARKER:
        return False
    for r, c, _orient in meta["gap_cells"]:
        if actual[r, c] != BG:
            return False
    if not np.array_equal(actual[0, strip_left : strip_left + len(strip_colors)], np.array(strip_colors)):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid, meta = _generate_case(rng)
        if _validate_generated_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
