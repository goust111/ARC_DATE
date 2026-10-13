from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (4, 6)
MAX_GEN_ATTEMPTS = 500

SEED_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (
        ((0, 0),),
    ),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ),
}


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbor(room_idx: tuple[int, int], side: str, n_room_rows: int, n_room_cols: int) -> tuple[int, int] | None:
    ri, ci = room_idx
    if side == "top":
        return (ri - 1, ci) if ri > 0 else None
    if side == "bottom":
        return (ri + 1, ci) if ri + 1 < n_room_rows else None
    if side == "left":
        return (ri, ci - 1) if ci > 0 else None
    return (ri, ci + 1) if ci + 1 < n_room_cols else None


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _candidate_placements(room: dict[str, int], seed_count: int) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]]:
    placements: list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]] = []
    room_h = room["r1"] - room["r0"] + 1
    room_w = room["c1"] - room["c0"] + 1
    for shape in SEED_SHAPES[seed_count]:
        shape_h, shape_w = _shape_size(shape)
        if shape_h > room_h or shape_w > room_w:
            continue
        for top in range(room["r0"], room["r1"] - shape_h + 2):
            for left in range(room["c0"], room["c1"] - shape_w + 2):
                cells = [(top + dr, left + dc) for dr, dc in shape]
                placements.append((cells, _bbox(cells)))
    return placements


def _build_room_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    h = sum(row_sizes) + len(row_sizes) + 1
    w = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((h, w), BG, dtype=int)

    wall_rows = [0]
    pos = 0
    for size in row_sizes:
        pos += size + 1
        wall_rows.append(pos)

    wall_cols = [0]
    pos = 0
    for size in col_sizes:
        pos += size + 1
        wall_cols.append(pos)

    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL
    return grid, wall_rows, wall_cols


def _rooms_from_lines(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], dict[str, int]]:
    rooms: dict[tuple[int, int], dict[str, int]] = {}
    for ri in range(len(wall_rows) - 1):
        for ci in range(len(wall_cols) - 1):
            rooms[(ri, ci)] = {
                "r0": wall_rows[ri] + 1,
                "r1": wall_rows[ri + 1] - 1,
                "c0": wall_cols[ci] + 1,
                "c1": wall_cols[ci + 1] - 1,
                "top_wall": wall_rows[ri],
                "bottom_wall": wall_rows[ri + 1],
                "left_wall": wall_cols[ci],
                "right_wall": wall_cols[ci + 1],
            }
    return rooms


def _wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_counts = np.count_nonzero(grid == WALL, axis=1)
    col_counts = np.count_nonzero(grid == WALL, axis=0)

    row_threshold = max(3, w - 5)
    col_threshold = max(3, h - 5)
    wall_rows = [int(r) for r, count in enumerate(row_counts) if int(count) >= row_threshold]
    wall_cols = [int(c) for c, count in enumerate(col_counts) if int(count) >= col_threshold]

    assert len(wall_rows) >= 2 and wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert len(wall_cols) >= 2 and wall_cols[0] == 0 and wall_cols[-1] == w - 1
    return wall_rows, wall_cols


def _extract_gaps(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], dict[str, int]],
    n_room_rows: int,
    n_room_cols: int,
) -> dict[tuple[tuple[int, int], str], list[tuple[int, int, tuple[int, int]]]]:
    gaps: dict[tuple[tuple[int, int], str], list[tuple[int, int, tuple[int, int]]]] = {}
    for room_idx, room in rooms.items():
        ri, ci = room_idx

        key = (room_idx, "top")
        gaps[key] = []
        if ri > 0:
            target = (ri - 1, ci)
            for c in range(room["c0"], room["c1"] + 1):
                if int(grid[room["top_wall"], c]) != WALL:
                    gaps[key].append((room["top_wall"], c, target))

        key = (room_idx, "bottom")
        gaps[key] = []
        if ri + 1 < n_room_rows:
            target = (ri + 1, ci)
            for c in range(room["c0"], room["c1"] + 1):
                if int(grid[room["bottom_wall"], c]) != WALL:
                    gaps[key].append((room["bottom_wall"], c, target))

        key = (room_idx, "left")
        gaps[key] = []
        if ci > 0:
            target = (ri, ci - 1)
            for r in range(room["r0"], room["r1"] + 1):
                if int(grid[r, room["left_wall"]]) != WALL:
                    gaps[key].append((r, room["left_wall"], target))

        key = (room_idx, "right")
        gaps[key] = []
        if ci + 1 < n_room_cols:
            target = (ri, ci + 1)
            for r in range(room["r0"], room["r1"] + 1):
                if int(grid[r, room["right_wall"]]) != WALL:
                    gaps[key].append((r, room["right_wall"], target))

    return gaps


def _room_seed_info(grid: np.ndarray, room: dict[str, int]) -> tuple[int, list[tuple[int, int]], tuple[int, int, int, int]] | None:
    coords: list[tuple[int, int]] = []
    colors: set[int] = set()
    for r in range(room["r0"], room["r1"] + 1):
        for c in range(room["c0"], room["c1"] + 1):
            value = int(grid[r, c])
            if value in (BG, WALL):
                continue
            coords.append((r, c))
            colors.add(value)
    if not coords:
        return None
    assert len(colors) == 1, "all seeds in one room must share a color"
    coords.sort()
    return colors.pop(), coords, _bbox(coords)


def _distance_to_side(room: dict[str, int], bbox: tuple[int, int, int, int], side: str) -> int:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - room["top_wall"]
    if side == "bottom":
        return room["bottom_wall"] - r1
    if side == "left":
        return c0 - room["left_wall"]
    return room["right_wall"] - c1


def _gap_choices_for_bbox(
    room: dict[str, int],
    side: str,
    bbox: tuple[int, int, int, int],
    distance: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    choices: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        lo = max(room["c0"], c0 - distance)
        hi = min(room["c1"], c1 + distance)
        wall_r = room["top_wall"] if side == "top" else room["bottom_wall"]
        for c in range(lo, hi + 1):
            choices.append((wall_r, c))
        return choices

    lo = max(room["r0"], r0 - distance)
    hi = min(room["r1"], r1 + distance)
    wall_c = room["left_wall"] if side == "left" else room["right_wall"]
    for r in range(lo, hi + 1):
        choices.append((r, wall_c))
    return choices


def _priority_is_better(
    depth: int,
    total_layers: int,
    source_order: int,
    best_depth: np.ndarray,
    best_total: np.ndarray,
    best_order: np.ndarray,
    r: int,
    c: int,
) -> bool:
    current = (int(best_depth[r, c]), int(best_total[r, c]), int(best_order[r, c]))
    return (depth, total_layers, source_order) > current


def _solve(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    wall_rows, wall_cols = _wall_lines(g)
    rooms = _rooms_from_lines(wall_rows, wall_cols)
    n_room_rows = len(wall_rows) - 1
    n_room_cols = len(wall_cols) - 1
    gaps = _extract_gaps(g, rooms, n_room_rows, n_room_cols)

    out = g.copy()
    best_depth = np.full((h, w), -1, dtype=int)
    best_total = np.full((h, w), -1, dtype=int)
    best_order = np.full((h, w), -1, dtype=int)
    source_transitions: dict[int, int] = {}
    transition_events: list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
    seen_calls: set[tuple[tuple[int, int], tuple[int, int, int, int], int, int, int]] = set()

    def paint(r: int, c: int, color: int, depth: int, total_layers: int, source_order: int) -> None:
        if not (0 <= r < h and 0 <= c < w):
            return
        if int(g[r, c]) == WALL:
            return
        if _priority_is_better(depth, total_layers, source_order, best_depth, best_total, best_order, r, c):
            best_depth[r, c] = depth
            best_total[r, c] = total_layers
            best_order[r, c] = source_order
            out[r, c] = color

    def paint_room_ring(
        room: dict[str, int],
        bbox: tuple[int, int, int, int],
        local_layer: int,
        color: int,
        depth: int,
        total_layers: int,
        source_order: int,
    ) -> None:
        r0, c0, r1, c1 = bbox
        top = r0 - local_layer
        bottom = r1 + local_layer
        left = c0 - local_layer
        right = c1 + local_layer

        for c in range(left, right + 1):
            if room["r0"] <= top <= room["r1"] and room["c0"] <= c <= room["c1"]:
                paint(top, c, color, depth, total_layers, source_order)
            if bottom != top and room["r0"] <= bottom <= room["r1"] and room["c0"] <= c <= room["c1"]:
                paint(bottom, c, color, depth, total_layers, source_order)

        for r in range(top + 1, bottom):
            if room["c0"] <= left <= room["c1"] and room["r0"] <= r <= room["r1"]:
                paint(r, left, color, depth, total_layers, source_order)
            if right != left and room["c0"] <= right <= room["c1"] and room["r0"] <= r <= room["r1"]:
                paint(r, right, color, depth, total_layers, source_order)

    def expand(
        room_idx: tuple[int, int],
        bbox: tuple[int, int, int, int],
        layers: int,
        depth_offset: int,
        color: int,
        total_layers: int,
        source_order: int,
    ) -> None:
        if layers <= 0:
            return
        key = (room_idx, tuple(map(int, bbox)), int(layers), int(depth_offset), int(source_order))
        if key in seen_calls:
            return
        seen_calls.add(key)

        room = rooms[room_idx]
        for local_layer in range(1, layers + 1):
            depth = depth_offset + local_layer
            paint_room_ring(room, bbox, local_layer, color, depth, total_layers, source_order)

            top = bbox[0] - local_layer
            bottom = bbox[2] + local_layer
            left = bbox[1] - local_layer
            right = bbox[3] + local_layer

            for side, side_value in (
                ("top", top),
                ("bottom", bottom),
                ("left", left),
                ("right", right),
            ):
                room_wall = room["top_wall"] if side == "top" else room["bottom_wall"] if side == "bottom" else room["left_wall"] if side == "left" else room["right_wall"]
                if side_value != room_wall:
                    continue
                for gap_r, gap_c, target_room in gaps[(room_idx, side)]:
                    if side in ("top", "bottom"):
                        if not (left <= gap_c <= right):
                            continue
                    else:
                        if not (top <= gap_r <= bottom):
                            continue
                    paint(gap_r, gap_c, color, depth, total_layers, source_order)
                    remaining = layers - local_layer
                    if remaining <= 0:
                        continue
                    source_transitions[source_order] = source_transitions.get(source_order, 0) + 1
                    transition_events.append((source_order, room_idx, target_room, (gap_r, gap_c), depth))
                    expand(target_room, (gap_r, gap_c, gap_r, gap_c), remaining, depth, color, total_layers, source_order)

    source_order = 0
    for room_idx in sorted(rooms):
        info = _room_seed_info(g, rooms[room_idx])
        if info is None:
            continue
        color, coords, bbox = info
        total_layers = len(coords)
        source_transitions[source_order] = 0
        for r, c in coords:
            paint(r, c, color, 0, total_layers, source_order)
        expand(room_idx, bbox, total_layers, 0, color, total_layers, source_order)
        source_order += 1

    if not with_trace:
        return out
    trace = {
        "rooms": len(rooms),
        "sources": source_order,
        "changed": int(np.count_nonzero(out != g)),
        "transition_events": transition_events,
        "transitions_by_source": source_transitions,
        "max_penetrations": max(source_transitions.values(), default=0),
    }
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow concentric room-bounded rings from per-room seed sets, with optional
    continuation through single-cell wall gaps into adjacent rooms.
    """
    return _solve(grid, with_trace=False)


def _plan_none(rooms: dict[tuple[int, int], dict[str, int]]) -> dict[str, object] | None:
    room_indices = list(rooms)
    random.shuffle(room_indices)
    for room_idx in room_indices:
        seed_count = random.randint(1, 4)
        placements = _candidate_placements(rooms[room_idx], seed_count)
        if not placements:
            continue
        cells, bbox = random.choice(placements)
        return {
            "source": room_idx,
            "cells": cells,
            "bbox": bbox,
            "seed_count": seed_count,
            "path": [],
        }
    return None


def _plan_single(
    rooms: dict[tuple[int, int], dict[str, int]],
    n_room_rows: int,
    n_room_cols: int,
) -> dict[str, object] | None:
    room_indices = list(rooms)
    random.shuffle(room_indices)
    for room_idx in room_indices:
        side_options = [side for side in SIDES if _neighbor(room_idx, side, n_room_rows, n_room_cols) is not None]
        random.shuffle(side_options)
        for side in side_options:
            for seed_count in random.sample([2, 3, 4], 3):
                placements = _candidate_placements(rooms[room_idx], seed_count)
                random.shuffle(placements)
                for cells, bbox in placements:
                    distance = _distance_to_side(rooms[room_idx], bbox, side)
                    if not (1 <= distance < seed_count):
                        continue
                    gap_choices = _gap_choices_for_bbox(rooms[room_idx], side, bbox, distance)
                    if not gap_choices:
                        continue
                    gap_r, gap_c = random.choice(gap_choices)
                    return {
                        "source": room_idx,
                        "cells": cells,
                        "bbox": bbox,
                        "seed_count": seed_count,
                        "path": [
                            {
                                "side": side,
                                "gap": (gap_r, gap_c),
                                "target": _neighbor(room_idx, side, n_room_rows, n_room_cols),
                            }
                        ],
                    }
    return None


def _plan_chain(
    rooms: dict[tuple[int, int], dict[str, int]],
    n_room_rows: int,
    n_room_cols: int,
) -> dict[str, object] | None:
    room_indices = list(rooms)
    random.shuffle(room_indices)
    for room_idx in room_indices:
        first_sides = [side for side in SIDES if _neighbor(room_idx, side, n_room_rows, n_room_cols) is not None]
        random.shuffle(first_sides)
        for side1 in first_sides:
            room1_idx = _neighbor(room_idx, side1, n_room_rows, n_room_cols)
            assert room1_idx is not None
            for seed_count in random.sample([3, 4], 2):
                placements = _candidate_placements(rooms[room_idx], seed_count)
                random.shuffle(placements)
                for cells, bbox in placements:
                    distance1 = _distance_to_side(rooms[room_idx], bbox, side1)
                    if not (1 <= distance1 <= seed_count - 2):
                        continue
                    gap1_choices = _gap_choices_for_bbox(rooms[room_idx], side1, bbox, distance1)
                    random.shuffle(gap1_choices)
                    for gap1 in gap1_choices:
                        remaining = seed_count - distance1
                        room1 = rooms[room1_idx]
                        second_sides = []
                        for side2 in SIDES:
                            target2 = _neighbor(room1_idx, side2, n_room_rows, n_room_cols)
                            if target2 is None or target2 == room_idx:
                                continue
                            second_sides.append(side2)
                        random.shuffle(second_sides)
                        for side2 in second_sides:
                            point_bbox = (gap1[0], gap1[1], gap1[0], gap1[1])
                            distance2 = _distance_to_side(room1, point_bbox, side2)
                            if not (1 <= distance2 < remaining):
                                continue
                            gap2_choices = _gap_choices_for_bbox(room1, side2, point_bbox, distance2)
                            random.shuffle(gap2_choices)
                            for gap2 in gap2_choices:
                                target2 = _neighbor(room1_idx, side2, n_room_rows, n_room_cols)
                                if target2 is None:
                                    continue
                                return {
                                    "source": room_idx,
                                    "cells": cells,
                                    "bbox": bbox,
                                    "seed_count": seed_count,
                                    "path": [
                                        {
                                            "side": side1,
                                            "gap": gap1,
                                            "target": room1_idx,
                                        },
                                        {
                                            "side": side2,
                                            "gap": gap2,
                                            "target": target2,
                                        },
                                    ],
                                }
    return None


def _random_extra_seed(
    room: dict[str, int],
    seed_count: int,
) -> list[tuple[int, int]] | None:
    placements = _candidate_placements(room, seed_count)
    if not placements:
        return None
    cells, _bbox_value = random.choice(placements)
    return cells


def _gen_attempt() -> np.ndarray | None:
    n_room_rows = random.randint(*ROOM_COUNT_RANGE)
    n_room_cols = random.randint(*ROOM_COUNT_RANGE)
    row_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_rows)]
    col_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_cols)]

    grid, wall_rows, wall_cols = _build_room_grid(row_sizes, col_sizes)
    rooms = _rooms_from_lines(wall_rows, wall_cols)

    branch_mode = random.choice(("none", "single", "chain"))
    if branch_mode == "none":
        feature = _plan_none(rooms)
    elif branch_mode == "single":
        feature = _plan_single(rooms, n_room_rows, n_room_cols)
    else:
        feature = _plan_chain(rooms, n_room_rows, n_room_cols)
    if feature is None:
        return None

    for step in feature["path"]:
        gap_r, gap_c = step["gap"]
        grid[gap_r, gap_c] = BG

    colors = list(PALETTE)
    random.shuffle(colors)

    seeded_rooms: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}
    seeded_rooms[feature["source"]] = (colors.pop(), list(feature["cells"]))

    blocked_rooms = {feature["source"]}
    for step in feature["path"]:
        blocked_rooms.add(step["target"])

    available_rooms = [room_idx for room_idx in rooms if room_idx not in blocked_rooms]
    random.shuffle(available_rooms)

    extra_cap = 3 if branch_mode == "none" else 2
    extra_count = random.randint(0, min(extra_cap, len(available_rooms)))
    for room_idx in available_rooms:
        if len(seeded_rooms) >= 1 + extra_count or not colors:
            break
        seed_count = random.randint(1, 4)
        cells = _random_extra_seed(rooms[room_idx], seed_count)
        if cells is None:
            continue
        seeded_rooms[room_idx] = (colors.pop(), cells)

    for color, cells in seeded_rooms.values():
        for r, c in cells:
            grid[r, c] = color

    out, trace = _solve(grid, with_trace=True)
    if np.array_equal(grid, out):
        return None
    if trace["changed"] < 8:
        return None
    if branch_mode == "none" and trace["max_penetrations"] != 0:
        return None
    if branch_mode == "single" and trace["max_penetrations"] < 1:
        return None
    if branch_mode == "chain" and trace["max_penetrations"] < 2:
        return None
    if trace["sources"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth ARC input")
