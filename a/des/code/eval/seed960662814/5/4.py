from __future__ import annotations

import random
from collections import Counter

import numpy as np


GEN_RETRIES = 400
ASSIGNMENT_RETRIES = 200
PATTERN_RETRIES = 80
ROOM_COUNT_RANGE = (2, 4)
ROOM_HEIGHT_RANGE = (5, 8)
ROOM_WIDTH_RANGE = (5, 8)
MAX_SOURCE_ROOMS = 6


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        mask = grid == color
        row_max = int(mask.sum(axis=1).max())
        col_max = int(mask.sum(axis=0).max())
        total = int(mask.sum())
        score = (min(row_max, col_max), max(row_max, col_max), total)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None
    return best_color


def _boundary_lines(counts: np.ndarray, major_len: int) -> list[int]:
    lines = [int(i) for i, value in enumerate(counts) if int(value) * 2 > int(major_len)]
    assert len(lines) >= 2
    assert lines[0] == 0 and lines[-1] == len(counts) - 1
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:]))
    return lines


def _door_in_vertical_wall(
    grid: np.ndarray,
    bg: int,
    wall_col: int,
    row_start: int,
    row_end: int,
) -> tuple[int, int]:
    rows = [int(r) for r in range(row_start, row_end + 1) if int(grid[int(r), int(wall_col)]) == int(bg)]
    assert len(rows) == 1
    return rows[0], int(wall_col)


def _door_in_horizontal_wall(
    grid: np.ndarray,
    bg: int,
    wall_row: int,
    col_start: int,
    col_end: int,
) -> tuple[int, int]:
    cols = [int(c) for c in range(col_start, col_end + 1) if int(grid[int(wall_row), int(c)]) == int(bg)]
    assert len(cols) == 1
    return int(wall_row), cols[0]


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = map(int, g.shape)

    bg = _most_common_color(g)
    wall = _wall_color(g, bg)
    wall_mask = g == wall

    row_lines = _boundary_lines(wall_mask.sum(axis=1), w)
    col_lines = _boundary_lines(wall_mask.sum(axis=0), h)
    room_rows = len(row_lines) - 1
    room_cols = len(col_lines) - 1

    rooms: dict[tuple[int, int], dict] = {}
    for i in range(room_rows):
        for j in range(room_cols):
            r0 = int(row_lines[i] + 1)
            r1 = int(row_lines[i + 1] - 1)
            c0 = int(col_lines[j] + 1)
            c1 = int(col_lines[j + 1] - 1)
            assert r0 <= r1 and c0 <= c1
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == wall)
            cells = [
                (int(r0 + rr), int(c0 + cc), int(patch[rr, cc]))
                for rr, cc in np.argwhere((patch != bg) & (patch != wall))
            ]
            rooms[(i, j)] = {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "cells": cells,
            }

    vertical_doors: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(room_rows):
        row_start = int(row_lines[i] + 1)
        row_end = int(row_lines[i + 1] - 1)
        for j in range(room_cols - 1):
            wall_col = int(col_lines[j + 1])
            vertical_doors[(i, j)] = _door_in_vertical_wall(g, bg, wall_col, row_start, row_end)

    horizontal_doors: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(room_rows - 1):
        wall_row = int(row_lines[i + 1])
        for j in range(room_cols):
            col_start = int(col_lines[j] + 1)
            col_end = int(col_lines[j + 1] - 1)
            horizontal_doors[(i, j)] = _door_in_horizontal_wall(g, bg, wall_row, col_start, col_end)

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "room_rows": room_rows,
        "room_cols": room_cols,
        "rooms": rooms,
        "vertical_doors": vertical_doors,
        "horizontal_doors": horizontal_doors,
    }


def _candidate_sources(structure: dict, target_pos: tuple[int, int]) -> list[dict]:
    rooms = structure["rooms"]
    vertical_doors = structure["vertical_doors"]
    horizontal_doors = structure["horizontal_doors"]
    room_rows = int(structure["room_rows"])
    room_cols = int(structure["room_cols"])
    i, j = target_pos
    target = rooms[target_pos]
    center_r, center_c = target["center"]
    candidates: list[dict] = []

    if j > 0 and rooms[(i, j - 1)]["cells"]:
        door_r, door_c = vertical_doors[(i, j - 1)]
        candidates.append(
            {
                "source": (i, j - 1),
                "flip": "horizontal",
                "wall_col": int(door_c),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if j + 1 < room_cols and rooms[(i, j + 1)]["cells"]:
        door_r, door_c = vertical_doors[(i, j)]
        candidates.append(
            {
                "source": (i, j + 1),
                "flip": "horizontal",
                "wall_col": int(door_c),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if i > 0 and rooms[(i - 1, j)]["cells"]:
        door_r, door_c = horizontal_doors[(i - 1, j)]
        candidates.append(
            {
                "source": (i - 1, j),
                "flip": "vertical",
                "wall_row": int(door_r),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if i + 1 < room_rows and rooms[(i + 1, j)]["cells"]:
        door_r, door_c = horizontal_doors[(i, j)]
        candidates.append(
            {
                "source": (i + 1, j),
                "flip": "vertical",
                "wall_row": int(door_r),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )

    return candidates


def _project_room(
    out: np.ndarray,
    bg: int,
    source_cells: list[tuple[int, int, int]],
    target_room: dict,
    candidate: dict,
) -> None:
    if candidate["flip"] == "horizontal":
        wall_col = int(candidate["wall_col"])
        for r, c, color in source_cells:
            rr = int(r)
            cc = int(2 * wall_col - c)
            assert target_room["r0"] <= rr <= target_room["r1"]
            assert target_room["c0"] <= cc <= target_room["c1"]
            assert int(out[rr, cc]) == int(bg)
            out[rr, cc] = int(color)
    else:
        wall_row = int(candidate["wall_row"])
        for r, c, color in source_cells:
            rr = int(2 * wall_row - r)
            cc = int(c)
            assert target_room["r0"] <= rr <= target_room["r1"]
            assert target_room["c0"] <= cc <= target_room["c1"]
            assert int(out[rr, cc]) == int(bg)
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each room pattern through one door into adjacent empty rooms.

    The source room is chosen per empty room by the closest door-to-center
    distance, and the copied pattern is mirrored across the shared wall.
    """
    structure = _parse_structure(np.asarray(grid, dtype=int))
    rooms = structure["rooms"]
    bg = int(structure["bg"])
    out = structure["grid"].copy()

    for target_pos, room in rooms.items():
        if room["cells"]:
            continue
        candidates = _candidate_sources(structure, target_pos)
        if not candidates:
            continue
        best_dist = min(float(item["dist2"]) for item in candidates)
        best = [item for item in candidates if float(item["dist2"]) == best_dist]
        assert len(best) == 1
        chosen = best[0]
        source_cells = rooms[chosen["source"]]["cells"]
        _project_room(out, bg, source_cells, room, chosen)

    return out


def _abstract_boxes(room_rows: int, room_cols: int, room_h: int, room_w: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes = {}
    for i in range(room_rows):
        for j in range(room_cols):
            r0 = 1 + i * (room_h + 1)
            c0 = 1 + j * (room_w + 1)
            boxes[(i, j)] = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
    return boxes


def _abstract_doors(
    room_rows: int,
    room_cols: int,
    room_h: int,
    room_w: int,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    vertical_doors = {}
    for i in range(room_rows):
        for j in range(room_cols - 1):
            r0, _c0, _r1, c1 = boxes[(i, j)]
            door_r = int(r0 + random.randrange(room_h))
            wall_col = int(c1 + 1)
            vertical_doors[(i, j)] = (door_r, wall_col)

    horizontal_doors = {}
    for i in range(room_rows - 1):
        for j in range(room_cols):
            _r0, c0, r1, _c1 = boxes[(i, j)]
            wall_row = int(r1 + 1)
            door_c = int(c0 + random.randrange(room_w))
            horizontal_doors[(i, j)] = (wall_row, door_c)

    return vertical_doors, horizontal_doors


def _abstract_candidates(
    pos: tuple[int, int],
    source_rooms: set[tuple[int, int]],
    room_rows: int,
    room_cols: int,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    vertical_doors: dict[tuple[int, int], tuple[int, int]],
    horizontal_doors: dict[tuple[int, int], tuple[int, int]],
) -> list[dict]:
    i, j = pos
    r0, c0, r1, c1 = boxes[pos]
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    candidates: list[dict] = []

    if j > 0 and (i, j - 1) in source_rooms:
        door_r, door_c = vertical_doors[(i, j - 1)]
        candidates.append(
            {
                "source": (i, j - 1),
                "flip": "horizontal",
                "wall_col": int(door_c),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if j + 1 < room_cols and (i, j + 1) in source_rooms:
        door_r, door_c = vertical_doors[(i, j)]
        candidates.append(
            {
                "source": (i, j + 1),
                "flip": "horizontal",
                "wall_col": int(door_c),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if i > 0 and (i - 1, j) in source_rooms:
        door_r, door_c = horizontal_doors[(i - 1, j)]
        candidates.append(
            {
                "source": (i - 1, j),
                "flip": "vertical",
                "wall_row": int(door_r),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )
    if i + 1 < room_rows and (i + 1, j) in source_rooms:
        door_r, door_c = horizontal_doors[(i, j)]
        candidates.append(
            {
                "source": (i + 1, j),
                "flip": "vertical",
                "wall_row": int(door_r),
                "dist2": (door_r - center_r) ** 2 + (door_c - center_c) ** 2,
            }
        )

    return candidates


def _analyze_assignment(
    source_rooms: set[tuple[int, int]],
    room_rows: int,
    room_cols: int,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    vertical_doors: dict[tuple[int, int], tuple[int, int]],
    horizontal_doors: dict[tuple[int, int], tuple[int, int]],
) -> list[dict] | None:
    fills: list[dict] = []
    for i in range(room_rows):
        for j in range(room_cols):
            pos = (i, j)
            if pos in source_rooms:
                continue
            candidates = _abstract_candidates(
                pos,
                source_rooms,
                room_rows,
                room_cols,
                boxes,
                vertical_doors,
                horizontal_doors,
            )
            if not candidates:
                continue
            best_dist = min(float(item["dist2"]) for item in candidates)
            best = [item for item in candidates if float(item["dist2"]) == best_dist]
            if len(best) != 1:
                return None
            chosen = dict(best[0])
            chosen["target"] = pos
            chosen["candidate_count"] = len(candidates)
            fills.append(chosen)
    return fills


def _all_rooms(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(room_rows) for j in range(room_cols)]


def _choose_sources(
    mode: str,
    room_rows: int,
    room_cols: int,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    vertical_doors: dict[tuple[int, int], tuple[int, int]],
    horizontal_doors: dict[tuple[int, int], tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[dict]] | None:
    rooms = _all_rooms(room_rows, room_cols)
    total_rooms = len(rooms)
    min_sources = 2
    max_sources = min(MAX_SOURCE_ROOMS, total_rooms - 1)
    for _ in range(ASSIGNMENT_RETRIES):
        source_count = random.randint(min_sources, max_sources)
        source_rooms = set(random.sample(rooms, source_count))
        fills = _analyze_assignment(source_rooms, room_rows, room_cols, boxes, vertical_doors, horizontal_doors)
        if fills is None:
            continue
        if len(fills) < 2:
            continue
        horizontal_count = sum(1 for item in fills if item["flip"] == "horizontal")
        vertical_count = sum(1 for item in fills if item["flip"] == "vertical")
        multi_count = sum(1 for item in fills if int(item["candidate_count"]) >= 2)
        if horizontal_count == 0 or vertical_count == 0:
            continue
        if mode == "multi":
            if multi_count == 0 or len(fills) < 3:
                continue
        return source_rooms, fills
    return None


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_pattern_mask(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(PATTERN_RETRIES):
        ph = random.randint(1, min(3, max_h))
        pw = random.randint(1, min(3, max_w))
        area = ph * pw
        if area < 3:
            continue
        min_cells = 2 if area > 1 else 1
        max_cells = area if area == 1 else area - 1
        count = random.randint(min_cells, max_cells)
        flat = np.zeros(area, dtype=bool)
        flat[random.sample(range(area), count)] = True
        mask = flat.reshape(ph, pw)
        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if area > 1 and not _connected(mask):
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric pattern mask")


def _paint_lattice(
    grid: np.ndarray,
    wall: int,
    room_rows: int,
    room_cols: int,
    room_h: int,
    room_w: int,
) -> None:
    height, width = grid.shape
    for i in range(room_rows + 1):
        row = int(i * (room_h + 1))
        grid[row, :] = wall
    for j in range(room_cols + 1):
        col = int(j * (room_w + 1))
        grid[:, col] = wall
    assert int(height - 1) == int(room_rows * (room_h + 1))
    assert int(width - 1) == int(room_cols * (room_w + 1))


def _carve_doors(
    grid: np.ndarray,
    bg: int,
    vertical_doors: dict[tuple[int, int], tuple[int, int]],
    horizontal_doors: dict[tuple[int, int], tuple[int, int]],
) -> None:
    for door_r, wall_col in vertical_doors.values():
        grid[int(door_r), int(wall_col)] = int(bg)
    for wall_row, door_c in horizontal_doors.values():
        grid[int(wall_row), int(door_c)] = int(bg)


def _place_patterns(
    grid: np.ndarray,
    source_rooms: set[tuple[int, int]],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    colors: list[int],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    placements: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for room, color in zip(sorted(source_rooms), colors):
        r0, c0, r1, c1 = boxes[room]
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        mask = _random_pattern_mask(room_h, room_w)
        ph, pw = mask.shape
        top = random.randint(0, room_h - ph)
        left = random.randint(0, room_w - pw)
        cells: list[tuple[int, int, int]] = []
        for rr, cc in np.argwhere(mask):
            r = int(r0 + top + rr)
            c = int(c0 + left + cc)
            grid[r, c] = int(color)
            cells.append((r, c, int(color)))
        placements[room] = cells
    return placements


def _expected_output(
    inp: np.ndarray,
    bg: int,
    fills: list[dict],
    placements: dict[tuple[int, int], list[tuple[int, int, int]]],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for fill in fills:
        target = fill["target"]
        source_cells = placements[fill["source"]]
        r0, c0, r1, c1 = boxes[target]
        target_room = {"r0": r0, "c0": c0, "r1": r1, "c1": c1}
        _project_room(out, bg, source_cells, target_room, fill)
    return out


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["basic", "multi"])
    room_rows = random.randint(max(3, ROOM_COUNT_RANGE[0]), ROOM_COUNT_RANGE[1]) if mode == "multi" else random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(max(3, ROOM_COUNT_RANGE[0]), ROOM_COUNT_RANGE[1]) if mode == "multi" else random.randint(*ROOM_COUNT_RANGE)
    room_h = random.randint(*ROOM_HEIGHT_RANGE)
    room_w = random.randint(*ROOM_WIDTH_RANGE)

    height = room_rows * room_h + room_rows + 1
    width = room_cols * room_w + room_cols + 1
    bg, wall = random.sample(range(10), 2)
    grid = np.full((height, width), int(bg), dtype=int)

    boxes = _abstract_boxes(room_rows, room_cols, room_h, room_w)
    vertical_doors, horizontal_doors = _abstract_doors(room_rows, room_cols, room_h, room_w, boxes)
    selection = _choose_sources(mode, room_rows, room_cols, boxes, vertical_doors, horizontal_doors)
    if selection is None:
        return None
    source_rooms, fills = selection
    if len(source_rooms) < 2:
        return None

    palette = [color for color in range(10) if color not in {bg, wall}]
    if len(source_rooms) > len(palette):
        return None
    random.shuffle(palette)
    source_colors = palette[: len(source_rooms)]

    _paint_lattice(grid, wall, room_rows, room_cols, room_h, room_w)
    _carve_doors(grid, bg, vertical_doors, horizontal_doors)
    placements = _place_patterns(grid, source_rooms, boxes, source_colors)
    expected = _expected_output(grid, bg, fills, placements, boxes)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    before = int(np.sum(grid != solved))
    if before == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-projection puzzle")
