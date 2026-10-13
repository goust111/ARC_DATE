from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
ARC_COLORS = tuple(range(10))
OBJECT_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, GREEN))
MAX_GRID_SIZE = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _separator_rows(g: np.ndarray) -> list[int]:
    h, w = g.shape
    threshold = max(3, w // 2 + 1)
    rows: list[int] = []
    for r in range(h):
        row = g[r, :]
        if int(np.count_nonzero(row == GREEN)) >= threshold and np.all((row == GREEN) | (row == BLACK)):
            rows.append(int(r))
    return rows


def _separator_cols(g: np.ndarray) -> list[int]:
    h, w = g.shape
    threshold = max(3, h // 2 + 1)
    cols: list[int] = []
    for c in range(w):
        col = g[:, c]
        if int(np.count_nonzero(col == GREEN)) >= threshold and np.all((col == GREEN) | (col == BLACK)):
            cols.append(int(c))
    return cols


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    edges = [-1] + [int(s) for s in separators] + [int(length)]
    return [
        (edges[i] + 1, edges[i + 1] - 1)
        for i in range(len(edges) - 1)
        if edges[i] + 1 <= edges[i + 1] - 1
    ]


def _door_count(g: np.ndarray, r0: int, r1: int, c0: int, c1: int, sep_rows: set[int], sep_cols: set[int]) -> int:
    count = 0
    top = r0 - 1
    bottom = r1 + 1
    left = c0 - 1
    right = c1 + 1

    if top in sep_rows:
        count += int(np.count_nonzero(g[top, c0 : c1 + 1] == BLACK))
    if bottom in sep_rows:
        count += int(np.count_nonzero(g[bottom, c0 : c1 + 1] == BLACK))
    if left in sep_cols:
        count += int(np.count_nonzero(g[r0 : r1 + 1, left] == BLACK))
    if right in sep_cols:
        count += int(np.count_nonzero(g[r0 : r1 + 1, right] == BLACK))
    return int(count)


def _room_object(sub: np.ndarray) -> tuple[int, np.ndarray] | None:
    fg = (sub != BLACK) & (sub != GREEN)
    if not bool(fg.any()):
        return None

    comps = _components(fg)
    assert len(comps) == 1, "each room must contain exactly one foreground object"
    pts = comps[0]
    colors = {int(sub[r, c]) for r, c in pts}
    assert len(colors) == 1, "room object must be monochrome"
    color = next(iter(colors))

    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return int(color), mask


def _axis_positions(start: int, end: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [(int(start) + int(end)) // 2]
    return [
        int(round(int(start) + i * (int(end) - int(start)) / (count - 1)))
        for i in range(count)
    ]


def _paint_mask(out: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        out[int(top) + int(rr), int(left) + int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each room's corner object according to the number of doors in its green walls.

    Input structure:
    - The background is black.
    - One-pixel green full-row and full-column dividers split the grid into rectangular rooms.
    - Some divider cells are black one-pixel gaps; these gaps are doors in the adjacent room walls.
    - Each room contains exactly one small monochrome non-black, non-green foreground object.

    Rule:
    - Infer the rectangular rooms from the green divider rows and columns.
    - For each room, count every black gap on its four green divider walls as a door d.
    - Clear the original object, then redraw its shape d times along the room's longest axis; width ties use the horizontal axis.
    - The first and last copies sit at the two ends of that axis, intermediate copies are evenly spaced, and a single copy is centered.
    - Green dividers and black door gaps are preserved unchanged.

    Key insight: The black cells interrupting the green dividers are doors, and the number of doors around a room is the hidden repeat count for that room's object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= MAX_GRID_SIZE and w <= MAX_GRID_SIZE, "ARC grids must be at most 30x30"

    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    assert sep_rows or sep_cols, "expected at least one green separator"

    row_rooms = _intervals(h, sep_rows)
    col_rooms = _intervals(w, sep_cols)
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    out = g.copy()
    for r0, r1 in row_rooms:
        for c0, c1 in col_rooms:
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            parsed = _room_object(sub)
            if parsed is None:
                continue
            color, mask = parsed
            obj_h, obj_w = map(int, mask.shape)
            room_h = int(r1 - r0 + 1)
            room_w = int(c1 - c0 + 1)
            assert obj_h <= room_h and obj_w <= room_w, "object must fit inside its room"

            d = _door_count(g, r0, r1, c0, c1, sep_row_set, sep_col_set)
            out[r0 : r1 + 1, c0 : c1 + 1][sub != BLACK] = BLACK

            if room_w >= room_h:
                top = int(r0 + (room_h - obj_h) // 2)
                start = int(c0)
                end = int(c1 - obj_w + 1)
                for left in _axis_positions(start, end, d):
                    _paint_mask(out, top, left, mask, color)
            else:
                left = int(c0 + (room_w - obj_w) // 2)
                start = int(r0)
                end = int(r1 - obj_h + 1)
                for top in _axis_positions(start, end, d):
                    _paint_mask(out, top, left, mask, color)

    return out


def _compose_sizes(rng: random.Random, count: int, low: int = 5, high: int = 10) -> list[int]:
    return [rng.randint(low, high) for _ in range(count)]


def _line_positions(sizes: list[int]) -> list[int]:
    pos: list[int] = []
    cur = int(sizes[0])
    for size in sizes[1:]:
        pos.append(cur)
        cur += 1 + int(size)
    return pos


def _room_intervals_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cur = 0
    for size in sizes:
        intervals.append((cur, cur + int(size) - 1))
        cur += int(size) + 1
    return intervals


def _wall_segments(
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> list[dict]:
    segments: list[dict] = []

    for ri, sep_r in enumerate(sep_rows):
        for cj, (c0, c1) in enumerate(col_ranges):
            segments.append(
                {
                    "rooms": ((ri, cj), (ri + 1, cj)),
                    "cells": [(int(sep_r), c) for c in range(c0, c1 + 1)],
                    "orient": "h",
                }
            )

    for cj, sep_c in enumerate(sep_cols):
        for ri, (r0, r1) in enumerate(row_ranges):
            segments.append(
                {
                    "rooms": ((ri, cj), (ri, cj + 1)),
                    "cells": [(r, int(sep_c)) for r in range(r0, r1 + 1)],
                    "orient": "v",
                }
            )

    return segments


def _door_candidates(segment: dict, doors: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    orient = segment["orient"]
    for r, c in segment["cells"]:
        if (r, c) in doors:
            continue
        if orient == "h" and any(rr == r and abs(cc - c) <= 1 for rr, cc in doors):
            continue
        if orient == "v" and any(cc == c and abs(rr - r) <= 1 for rr, cc in doors):
            continue
        out.append((int(r), int(c)))
    return out


def _add_door(
    rng: random.Random,
    segment: dict,
    doors: set[tuple[int, int]],
    counts: dict[tuple[int, int], int],
    max_doors: int,
) -> bool:
    if any(counts[room] >= max_doors for room in segment["rooms"]):
        return False
    choices = _door_candidates(segment, doors)
    if not choices:
        return False
    cell = rng.choice(choices)
    doors.add(cell)
    for room in segment["rooms"]:
        counts[room] += 1
    return True


def _make_doors(
    rng: random.Random,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]] | None:
    room_count = len(row_ranges) * len(col_ranges)
    counts = {(ri, cj): 0 for ri in range(len(row_ranges)) for cj in range(len(col_ranges))}
    doors: set[tuple[int, int]] = set()
    segments = _wall_segments(row_ranges, col_ranges, sep_rows, sep_cols)
    max_doors = 4

    for _ in range(200):
        zero_rooms = {room for room, count in counts.items() if count == 0}
        if not zero_rooms:
            break
        candidates = [
            seg
            for seg in segments
            if any(room in zero_rooms for room in seg["rooms"])
            and all(counts[room] < max_doors for room in seg["rooms"])
            and _door_candidates(seg, doors)
        ]
        if not candidates:
            return None
        _add_door(rng, rng.choice(candidates), doors, counts, max_doors)

    if any(count == 0 for count in counts.values()):
        return None

    for _ in range(rng.randint(1, room_count + 2)):
        candidates = [
            seg
            for seg in segments
            if all(counts[room] < max_doors for room in seg["rooms"])
            and _door_candidates(seg, doors)
        ]
        if not candidates:
            break
        if rng.random() < 0.65:
            _add_door(rng, rng.choice(candidates), doors, counts, max_doors)

    values = list(counts.values())
    if not (1 in values and any(v > 1 for v in values)):
        return None
    return doors, counts


def _random_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    area_max = min(4, int(height) * int(width))
    area_min = 1 if area_max == 1 else 2
    target = rng.randint(area_min, area_max)
    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < target:
        frontier = [
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
        ]
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _place_room_objects(
    rng: random.Random,
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
) -> bool:
    for ri, (r0, r1) in enumerate(row_ranges):
        for cj, (c0, c1) in enumerate(col_ranges):
            room_h = r1 - r0 + 1
            room_w = c1 - c0 + 1
            d = counts[(ri, cj)]

            if room_w >= room_h:
                max_w = max(1, min(3, room_w // max(1, d)))
                max_h = max(1, min(3, room_h // 2))
            else:
                max_h = max(1, min(3, room_h // max(1, d)))
                max_w = max(1, min(3, room_w // 2))

            if max_h < 1 or max_w < 1:
                return False
            obj_h = rng.randint(1, max_h)
            obj_w = rng.randint(1, max_w)
            mask = _random_mask(rng, obj_h, obj_w)
            color = rng.choice(OBJECT_COLORS)

            vertical_side = rng.choice(["top", "bottom"])
            horizontal_side = rng.choice(["left", "right"])
            top = r0 if vertical_side == "top" else r1 - obj_h + 1
            left = c0 if horizontal_side == "left" else c1 - obj_w + 1
            for rr, cc in np.argwhere(mask):
                grid[top + int(rr), left + int(cc)] = int(color)

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    room_rows = rng.choice([2, 3])
    room_cols = rng.choice([2, 3])

    row_sizes = _compose_sizes(rng, room_rows)
    col_sizes = _compose_sizes(rng, room_cols)
    h = sum(row_sizes) + room_rows - 1
    w = sum(col_sizes) + room_cols - 1
    if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
        return None

    row_ranges = _room_intervals_from_sizes(row_sizes)
    col_ranges = _room_intervals_from_sizes(col_sizes)
    axes = [
        "h" if (c1 - c0 + 1) >= (r1 - r0 + 1) else "v"
        for r0, r1 in row_ranges
        for c0, c1 in col_ranges
    ]
    if not ("h" in axes and "v" in axes):
        return None

    sep_rows = _line_positions(row_sizes)
    sep_cols = _line_positions(col_sizes)
    door_info = _make_doors(rng, row_ranges, col_ranges, sep_rows, sep_cols)
    if door_info is None:
        return None
    doors, counts = door_info

    grid = np.full((h, w), BLACK, dtype=int)
    for r in sep_rows:
        grid[int(r), :] = GREEN
    for c in sep_cols:
        grid[:, int(c)] = GREEN
    for r, c in doors:
        grid[int(r), int(c)] = BLACK

    if not _place_room_objects(rng, grid, row_ranges, col_ranges, counts):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > MAX_GRID_SIZE:
        return None
    if not np.all(out[np.array(grid == GREEN)] == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door repetition input")
