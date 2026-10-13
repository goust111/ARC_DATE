import random

import numpy as np


BG = 0
WALL = 9
FLOOR_COLORS = (1, 2, 3, 4, 5)
WINDOW_COLORS = (6, 7, 8)

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_HEIGHT_RANGE = (2, 5)
ROOM_WIDTH_RANGE = (2, 5)
MARGIN_RANGE = (1, 3)
MAX_SIDE = 30
GEN_RETRIES = 300


def _bbox_nonblack(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid != BG)
    if pts.size == 0:
        raise ValueError("grid must contain a non-black structure")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1])


def _validate_wall_positions(positions: list[int], low: int, high: int) -> None:
    if len(positions) < 2 or positions[0] != low or positions[-1] != high:
        raise ValueError("wall frame must bound the full non-black structure")
    for prev, cur in zip(positions, positions[1:]):
        if cur - prev < 3:
            raise ValueError("rooms must be at least two cells wide/tall")


def _dominant_color(block: np.ndarray) -> int:
    colors, counts = np.unique(block, return_counts=True)
    return int(colors[np.argmax(counts)])


def _room_bounds(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rooms = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            rooms[(rr, cc)] = (
                wall_rows[rr] + 1,
                wall_rows[rr + 1],
                wall_cols[cc] + 1,
                wall_cols[cc + 1],
            )
    return rooms


def _first_window_in_segment(segment: np.ndarray) -> tuple[int, int] | None:
    idxs = np.flatnonzero(segment != WALL)
    if idxs.size == 0:
        return None
    offset = int(idxs[0])
    color = int(_dominant_color(segment[idxs]))
    return offset, color


def _analyze_layout(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    r0, r1, c0, c1 = _bbox_nonblack(g)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    wall_rows = [r for r in range(r0, r1 + 1) if np.count_nonzero(g[r, c0 : c1 + 1] == WALL) >= width - 2]
    wall_cols = [c for c in range(c0, c1 + 1) if np.count_nonzero(g[r0 : r1 + 1, c] == WALL) >= height - 2]
    _validate_wall_positions(wall_rows, r0, r1)
    _validate_wall_positions(wall_cols, c0, c1)
    rooms = _room_bounds(wall_rows, wall_cols)

    initial_colors = {}
    candidates = {room: [] for room in rooms}
    for room, (top, bottom, left, right) in rooms.items():
        initial_colors[room] = _dominant_color(g[top:bottom, left:right])

    for rr in range(1, len(wall_rows) - 1):
        wall_r = wall_rows[rr]
        for cc in range(len(wall_cols) - 1):
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1]
            info = _first_window_in_segment(g[wall_r, left:right])
            if info is None:
                continue
            offset, color = info
            pos = (wall_r, left + offset)
            candidates[(rr - 1, cc)].append((pos, color))
            candidates[(rr, cc)].append((pos, color))

    for cc in range(1, len(wall_cols) - 1):
        wall_c = wall_cols[cc]
        for rr in range(len(wall_rows) - 1):
            top = wall_rows[rr] + 1
            bottom = wall_rows[rr + 1]
            info = _first_window_in_segment(g[top:bottom, wall_c])
            if info is None:
                continue
            offset, color = info
            pos = (top + offset, wall_c)
            candidates[(rr, cc - 1)].append((pos, color))
            candidates[(rr, cc)].append((pos, color))

    return {
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "rooms": rooms,
        "initial_colors": initial_colors,
        "candidates": candidates,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor room floors from colored windows on shared maroon walls.

    The non-black structure is a one-cell-thick rectangular wall grid. Each wall
    line is almost entirely maroon, except for at most a tiny 1-2 pixel colored
    window on some internal shared walls. Each window recolors the two touching
    rooms; when a room sees multiple window colors, the earliest boundary
    position in row-major order wins.
    """
    g = np.asarray(grid, dtype=int)
    layout = _analyze_layout(g)
    out = g.copy()

    final_colors = dict(layout["initial_colors"])
    for room, hits in layout["candidates"].items():
        if hits:
            final_colors[room] = min(hits, key=lambda item: item[0])[1]

    for room, (top, bottom, left, right) in layout["rooms"].items():
        out[top:bottom, left:right] = final_colors[room]
    return out


def _wall_positions(start: int, sizes: list[int]) -> list[int]:
    positions = [start]
    cur = start
    for size in sizes:
        cur += size + 1
        positions.append(cur)
    return positions


def _segment_catalog(wall_rows: list[int], wall_cols: list[int]) -> list[dict[str, object]]:
    segments = []
    seg_id = 0

    for rr in range(len(wall_rows) - 2):
        wall_r = wall_rows[rr + 1]
        for cc in range(len(wall_cols) - 1):
            segments.append(
                {
                    "id": seg_id,
                    "orientation": "h",
                    "wall": wall_r,
                    "start": wall_cols[cc] + 1,
                    "end": wall_cols[cc + 1] - 1,
                    "rooms": ((rr, cc), (rr + 1, cc)),
                }
            )
            seg_id += 1

    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 2):
            wall_c = wall_cols[cc + 1]
            segments.append(
                {
                    "id": seg_id,
                    "orientation": "v",
                    "wall": wall_c,
                    "start": wall_rows[rr] + 1,
                    "end": wall_rows[rr + 1] - 1,
                    "rooms": ((rr, cc), (rr, cc + 1)),
                }
            )
            seg_id += 1

    return segments


def _realize_window(segment: dict[str, object], color: int, rng: random.Random) -> dict[str, object]:
    span = int(segment["end"]) - int(segment["start"]) + 1
    length = 2 if span >= 2 and rng.random() < 0.5 else 1
    begin = rng.randint(int(segment["start"]), int(segment["end"]) - length + 1)

    if segment["orientation"] == "h":
        cells = [(int(segment["wall"]), c) for c in range(begin, begin + length)]
    else:
        cells = [(r, int(segment["wall"])) for r in range(begin, begin + length)]

    return {"segment": segment, "color": color, "cells": cells}


def _select_windows(
    segments: list[dict[str, object]],
    room_ids: list[tuple[int, int]],
    force_conflict: bool,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    incident = {room: [] for room in room_ids}
    for seg in segments:
        a, b = seg["rooms"]
        incident[a].append(seg)
        incident[b].append(seg)

    chosen = []
    chosen_ids = set()
    used_h_walls = set()
    used_v_walls = set()

    def can_take(segment: dict[str, object], blocked_rooms: set[tuple[int, int]]) -> bool:
        a, b = segment["rooms"]
        if a in blocked_rooms or b in blocked_rooms:
            return False
        if segment["orientation"] == "h":
            return int(segment["wall"]) not in used_h_walls
        return int(segment["wall"]) not in used_v_walls

    def register(segment: dict[str, object]) -> None:
        if segment["orientation"] == "h":
            used_h_walls.add(int(segment["wall"]))
        else:
            used_v_walls.add(int(segment["wall"]))

    if force_conflict:
        candidates = [room for room, segs in incident.items() if len(segs) >= 2]
        if not candidates:
            return None
        focus = rng.choice(candidates)
        pool = incident[focus][:]
        rng.shuffle(pool)
        pair = None
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                first = pool[i]
                second = pool[j]
                if first["orientation"] == second["orientation"] and int(first["wall"]) == int(second["wall"]):
                    continue
                pair = [first, second]
                break
            if pair is not None:
                break
        if pair is None:
            return None
        conflict_colors = rng.sample(WINDOW_COLORS, 2)
        for seg, color in zip(pair, conflict_colors):
            chosen.append(_realize_window(seg, color, rng))
            chosen_ids.add(int(seg["id"]))
            register(seg)

        blocked = {focus}
        for seg in pair:
            blocked.update(seg["rooms"])

        extra_goal = rng.randint(0, max(0, len(room_ids) // 3))
        shuffled = segments[:]
        rng.shuffle(shuffled)
        for seg in shuffled:
            if int(seg["id"]) in chosen_ids:
                continue
            if not can_take(seg, blocked):
                continue
            chosen.append(_realize_window(seg, rng.choice(WINDOW_COLORS), rng))
            chosen_ids.add(int(seg["id"]))
            register(seg)
            a, b = seg["rooms"]
            blocked.add(a)
            blocked.add(b)
            if len(chosen) >= 2 + extra_goal:
                break
        return chosen

    max_count = max(1, len(room_ids) // 2)
    goal = rng.randint(1, max_count)
    blocked = set()
    shuffled = segments[:]
    rng.shuffle(shuffled)
    for seg in shuffled:
        if not can_take(seg, blocked):
            continue
        chosen.append(_realize_window(seg, rng.choice(WINDOW_COLORS), rng))
        register(seg)
        a, b = seg["rooms"]
        blocked.add(a)
        blocked.add(b)
        if len(chosen) >= goal:
            break
    return chosen or None


def _expected_output(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    windows: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    room_hits = {room: [] for room in rooms}

    for spec in windows:
        pos = min(spec["cells"])
        color = int(spec["color"])
        for room in spec["segment"]["rooms"]:
            room_hits[room].append((pos, color))

    for room, (top, bottom, left, right) in rooms.items():
        color = _dominant_color(grid[top:bottom, left:right])
        if room_hits[room]:
            color = min(room_hits[room], key=lambda item: item[0])[1]
        out[top:bottom, left:right] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(*ROOM_ROWS_RANGE)
    room_cols = rng.randint(*ROOM_COLS_RANGE)
    heights = [rng.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    widths = [rng.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    inner_h = sum(heights) + room_rows + 1
    inner_w = sum(widths) + room_cols + 1
    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)
    total_h = inner_h + top_margin + bottom_margin
    total_w = inner_w + left_margin + right_margin
    if total_h > MAX_SIDE or total_w > MAX_SIDE:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    wall_rows = _wall_positions(top_margin, heights)
    wall_cols = _wall_positions(left_margin, widths)

    r0, r1 = wall_rows[0], wall_rows[-1]
    c0, c1 = wall_cols[0], wall_cols[-1]
    for wall_r in wall_rows:
        grid[wall_r, c0 : c1 + 1] = WALL
    for wall_c in wall_cols:
        grid[r0 : r1 + 1, wall_c] = WALL

    rooms = _room_bounds(wall_rows, wall_cols)
    for room, (top, bottom, left, right) in rooms.items():
        grid[top:bottom, left:right] = rng.choice(FLOOR_COLORS)

    segments = _segment_catalog(wall_rows, wall_cols)
    force_conflict = rng.random() < 0.5
    windows = _select_windows(segments, list(rooms.keys()), force_conflict, rng)
    if not windows:
        return None

    for spec in windows:
        for r, c in spec["cells"]:
            grid[r, c] = int(spec["color"])

    expected = _expected_output(grid, rooms, windows)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-window puzzle")
