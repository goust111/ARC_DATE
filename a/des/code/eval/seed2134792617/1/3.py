from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 8
SOURCE_PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (5, 6)
GEN_RETRIES = 500
MAX_SOURCES = 5
EXTRA_DOOR_PROB = 0.35


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    for color, _count in Counter(map(int, grid.ravel())).most_common():
        if int(color) != int(bg):
            return int(color)
    raise AssertionError("expected a non-background wall color")


def _split_high_indices(counts: np.ndarray) -> list[int]:
    values = sorted({int(v) for v in counts.tolist()})
    assert len(values) >= 2, "expected separate low/high count bands"
    best_gap = -1
    threshold = None
    for lo, hi in zip(values, values[1:]):
        gap = int(hi - lo)
        if gap > best_gap:
            best_gap = gap
            threshold = (lo + hi) / 2.0
    assert threshold is not None and best_gap > 0
    return [idx for idx, value in enumerate(counts.tolist()) if float(value) > float(threshold)]


def _wall_rows_cols(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    wall_mask = grid == int(wall)
    rows = _split_high_indices(wall_mask.sum(axis=1))
    cols = _split_high_indices(wall_mask.sum(axis=0))
    assert rows and cols
    assert rows[0] == 0 and rows[-1] == grid.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "room rows must have positive height"
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:])), "room cols must have positive width"
    return rows, cols


def _room_bounds(
    wall_rows: list[int], wall_cols: list[int]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            bounds[(rr, cc)] = (
                int(wall_rows[rr] + 1),
                int(wall_cols[cc] + 1),
                int(wall_rows[rr + 1] - 1),
                int(wall_cols[cc + 1] - 1),
            )
    return bounds


def _read_rooms(
    grid: np.ndarray,
    bg: int,
    wall: int,
    wall_rows: list[int],
    wall_cols: list[int],
) -> dict[tuple[int, int], dict[str, object]]:
    rooms: dict[tuple[int, int], dict[str, object]] = {}
    for room, bounds in _room_bounds(wall_rows, wall_cols).items():
        top, left, bottom, right = bounds
        window = grid[top : bottom + 1, left : right + 1]
        signal = window[(window != int(bg)) & (window != int(wall))]
        colors = {int(v) for v in signal.tolist()}
        assert len(colors) <= 1, "each room may contain at most one source color"
        count = int(signal.size)
        assert 0 <= count <= 3
        rooms[room] = {
            "bounds": bounds,
            "count": count,
            "color": None if count == 0 else int(next(iter(colors))),
        }
    return rooms


def _collect_events_from_rooms(
    grid: np.ndarray,
    bg: int,
    rooms: dict[tuple[int, int], dict[str, object]],
    room_rows: int,
    room_cols: int,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            room = rooms[(rr, cc)]
            count = int(room["count"])
            if count == 0:
                continue
            color = int(room["color"])
            top, left, bottom, right = map(int, room["bounds"])
            top_wall = top - 1
            bottom_wall = bottom + 1
            left_wall = left - 1
            right_wall = right + 1

            if rr > 0:
                target = rooms[(rr - 1, cc)]
                if int(target["count"]) == 0:
                    for col in range(left, right + 1):
                        if int(grid[top_wall, col]) == int(bg):
                            events.append(
                                {
                                    "door": (int(top_wall), int(col)),
                                    "color": color,
                                    "length": count,
                                    "start": (int(top_wall - 1), int(col)),
                                    "step": (-1, 0),
                                    "target_bounds": target["bounds"],
                                }
                            )
            if rr + 1 < room_rows:
                target = rooms[(rr + 1, cc)]
                if int(target["count"]) == 0:
                    for col in range(left, right + 1):
                        if int(grid[bottom_wall, col]) == int(bg):
                            events.append(
                                {
                                    "door": (int(bottom_wall), int(col)),
                                    "color": color,
                                    "length": count,
                                    "start": (int(bottom_wall + 1), int(col)),
                                    "step": (1, 0),
                                    "target_bounds": target["bounds"],
                                }
                            )
            if cc > 0:
                target = rooms[(rr, cc - 1)]
                if int(target["count"]) == 0:
                    for row in range(top, bottom + 1):
                        if int(grid[row, left_wall]) == int(bg):
                            events.append(
                                {
                                    "door": (int(row), int(left_wall)),
                                    "color": color,
                                    "length": count,
                                    "start": (int(row), int(left_wall - 1)),
                                    "step": (0, -1),
                                    "target_bounds": target["bounds"],
                                }
                            )
            if cc + 1 < room_cols:
                target = rooms[(rr, cc + 1)]
                if int(target["count"]) == 0:
                    for row in range(top, bottom + 1):
                        if int(grid[row, right_wall]) == int(bg):
                            events.append(
                                {
                                    "door": (int(row), int(right_wall)),
                                    "color": color,
                                    "length": count,
                                    "start": (int(row), int(right_wall + 1)),
                                    "step": (0, 1),
                                    "target_bounds": target["bounds"],
                                }
                            )
    events.sort(key=lambda item: tuple(item["door"]))
    return events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays through wall openings from source rooms into adjacent empty rooms.

    The dominant background color fills room interiors. A second color forms a
    1-pixel-thick rectangular wall lattice with 1-pixel openings. Some rooms
    contain 1-3 same-colored source pixels. Each opening on a source room wall
    launches a perpendicular ray of matching color into the neighboring room if,
    and only if, that neighboring room has no source pixels. Ray length equals
    the number of source pixels in the emitting room. Later doors in top-to-bottom,
    left-to-right order overwrite earlier ray pixels on overlap.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(grid)
    wall = _most_common_non_bg(grid, bg)
    wall_rows, wall_cols = _wall_rows_cols(grid, wall)
    rooms = _read_rooms(grid, bg, wall, wall_rows, wall_cols)

    out = grid.copy()
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    for event in _collect_events_from_rooms(grid, bg, rooms, room_rows, room_cols):
        top, left, bottom, right = map(int, event["target_bounds"])
        row, col = map(int, event["start"])
        dr, dc = map(int, event["step"])
        for _ in range(int(event["length"])):
            if not (top <= row <= bottom and left <= col <= right):
                break
            out[row, col] = int(event["color"])
            row += dr
            col += dc
    return out


def _make_wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _adjacent_pairs(room_rows: int, room_cols: int) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if rr + 1 < room_rows:
                pairs.append(("h", (rr, cc), (rr + 1, cc)))
            if cc + 1 < room_cols:
                pairs.append(("v", (rr, cc), (rr, cc + 1)))
    return pairs


def _neighbors(room: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    rr, cc = room
    out = []
    if rr > 0:
        out.append((rr - 1, cc))
    if rr + 1 < room_rows:
        out.append((rr + 1, cc))
    if cc > 0:
        out.append((rr, cc - 1))
    if cc + 1 < room_cols:
        out.append((rr, cc + 1))
    return out


def _border_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, int, int]:
    if a[0] == b[0]:
        rr = int(a[0])
        cc = min(int(a[1]), int(b[1]))
        return ("v", rr, cc)
    rr = min(int(a[0]), int(b[0]))
    cc = int(a[1])
    return ("h", rr, cc)


def _border_candidates(
    key: tuple[str, int, int], wall_rows: list[int], wall_cols: list[int]
) -> list[tuple[int, int]]:
    kind, rr, cc = key
    if kind == "h":
        wall_row = int(wall_rows[rr + 1])
        return [(wall_row, col) for col in range(int(wall_cols[cc] + 1), int(wall_cols[cc + 1]))]
    wall_col = int(wall_cols[cc + 1])
    return [(row, wall_col) for row in range(int(wall_rows[rr] + 1), int(wall_rows[rr + 1]))]


def _room_bounds_from_layout(
    room_rows: int, room_cols: int, wall_rows: list[int], wall_cols: list[int]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return _room_bounds(wall_rows, wall_cols)


def _choose_aligned_opposite_target(
    room_rows: int, room_cols: int
) -> tuple[tuple[int, int], list[tuple[int, int]], str] | None:
    choices = []
    for rr in range(1, room_rows - 1):
        for cc in range(room_cols):
            choices.append(((rr, cc), [(rr - 1, cc), (rr + 1, cc)], "vertical"))
    for rr in range(room_rows):
        for cc in range(1, room_cols - 1):
            choices.append(((rr, cc), [(rr, cc - 1), (rr, cc + 1)], "horizontal"))
    return random.choice(choices) if choices else None


def _sample_positions(
    candidates: list[tuple[int, int]], desired: int, fixed: tuple[int, int] | None = None
) -> list[tuple[int, int]]:
    if fixed is not None:
        assert fixed in candidates
        return [fixed]
    return random.sample(candidates, k=min(len(candidates), desired))


def _add_doors(
    door_map: dict[tuple[str, int, int], set[tuple[int, int]]],
    key: tuple[str, int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    *,
    count: int = 1,
    fixed: tuple[int, int] | None = None,
) -> None:
    candidates = _border_candidates(key, wall_rows, wall_cols)
    assert candidates
    door_map.setdefault(key, set()).update(_sample_positions(candidates, count, fixed=fixed))


def _build_sources(
    room_rows: int,
    room_cols: int,
    mode: str,
    wall_rows: list[int],
    wall_cols: list[int],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[str, int, int], set[tuple[int, int]]],
    dict[str, object],
]:
    rooms = [(rr, cc) for rr in range(room_rows) for cc in range(room_cols)]
    pairs = _adjacent_pairs(room_rows, room_cols)
    source_counts: dict[tuple[int, int], int] = {}
    forced_empty: set[tuple[int, int]] = set()
    door_map: dict[tuple[str, int, int], set[tuple[int, int]]] = {}
    meta: dict[str, object] = {"mode": mode}

    if mode == "multi":
        chosen = _choose_aligned_opposite_target(room_rows, room_cols)
        if chosen is None:
            mode = "basic"
            meta["mode"] = mode
        else:
            target, neighbors, axis = chosen
            forced_empty.add(target)
            meta["multi_target"] = target
            for room in neighbors:
                source_counts[room] = 3
            if axis == "vertical":
                candidates = _border_candidates(_border_key(neighbors[0], target), wall_rows, wall_cols)
                door = random.choice(candidates)
                _add_doors(
                    door_map,
                    _border_key(neighbors[0], target),
                    wall_rows,
                    wall_cols,
                    fixed=door,
                )
                _add_doors(
                    door_map,
                    _border_key(target, neighbors[1]),
                    wall_rows,
                    wall_cols,
                    fixed=(int(_border_candidates(_border_key(target, neighbors[1]), wall_rows, wall_cols)[0][0]), int(door[1])),
                )
            else:
                candidates = _border_candidates(_border_key(neighbors[0], target), wall_rows, wall_cols)
                door = random.choice(candidates)
                _add_doors(
                    door_map,
                    _border_key(neighbors[0], target),
                    wall_rows,
                    wall_cols,
                    fixed=door,
                )
                _add_doors(
                    door_map,
                    _border_key(target, neighbors[1]),
                    wall_rows,
                    wall_cols,
                    fixed=(int(door[0]), int(_border_candidates(_border_key(target, neighbors[1]), wall_rows, wall_cols)[0][1])),
                )

    if mode == "blocked":
        blocked_kind, blocked_a, blocked_b = random.choice(pairs)
        source_counts.setdefault(blocked_a, random.randint(1, 3))
        source_counts.setdefault(blocked_b, random.randint(1, 3))
        _add_doors(
            door_map,
            (blocked_kind, min(blocked_a[0], blocked_b[0]), min(blocked_a[1], blocked_b[1])),
            wall_rows,
            wall_cols,
            count=random.randint(1, 2),
        )
        meta["blocked_pair"] = (blocked_a, blocked_b)

    if mode in {"basic", "blocked"}:
        emit_candidates = []
        for kind, a, b in pairs:
            if a in source_counts and b not in source_counts and b not in forced_empty:
                emit_candidates.append((kind, a, b))
            if b in source_counts and a not in source_counts and a not in forced_empty:
                emit_candidates.append((kind, b, a))
        if not emit_candidates:
            kind, a, b = random.choice(pairs)
            source_room, empty_room = (a, b) if random.random() < 0.5 else (b, a)
            source_counts.setdefault(source_room, random.randint(1, 3))
            forced_empty.add(empty_room)
            emit_candidates.append((kind, source_room, empty_room))
        emit_kind, source_room, empty_room = random.choice(emit_candidates)
        forced_empty.add(empty_room)
        source_counts.setdefault(source_room, random.randint(1, 3))
        emit_key = (
            emit_kind,
            min(source_room[0], empty_room[0]),
            min(source_room[1], empty_room[1]),
        )
        _add_doors(
            door_map,
            emit_key,
            wall_rows,
            wall_cols,
            count=random.randint(1, 2),
        )
        meta["emit_pair"] = (source_room, empty_room)

    free_rooms = [
        room for room in rooms if room not in source_counts and room not in forced_empty
    ]
    random.shuffle(free_rooms)
    extra_limit = min(MAX_SOURCES - len(source_counts), len(free_rooms), 2 + random.randint(0, 1))
    for room in free_rooms[:extra_limit]:
        if random.random() < 0.5:
            source_counts[room] = random.randint(1, 3)

    return source_counts, door_map, meta


def _place_extra_doors(
    door_map: dict[tuple[str, int, int], set[tuple[int, int]]],
    room_rows: int,
    room_cols: int,
    wall_rows: list[int],
    wall_cols: list[int],
) -> None:
    for kind, a, b in _adjacent_pairs(room_rows, room_cols):
        key = (kind, min(a[0], b[0]), min(a[1], b[1]))
        existing = len(door_map.get(key, set()))
        max_extra = min(2, len(_border_candidates(key, wall_rows, wall_cols)))
        if existing >= max_extra:
            continue
        if random.random() >= EXTRA_DOOR_PROB:
            continue
        add_count = random.randint(1, max_extra - existing)
        _add_doors(door_map, key, wall_rows, wall_cols, count=add_count)


def _sample_source_cells(
    bounds: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]]:
    top, left, bottom, right = bounds
    cells = [(row, col) for row in range(top, bottom + 1) for col in range(left, right + 1)]
    return random.sample(cells, k=count)


def _flatten_doors(
    door_map: dict[tuple[str, int, int], set[tuple[int, int]]]
) -> list[dict[str, object]]:
    doors: list[dict[str, object]] = []
    for key, coords in door_map.items():
        kind, rr, cc = key
        a = (rr, cc)
        b = (rr + 1, cc) if kind == "h" else (rr, cc + 1)
        for coord in sorted(coords):
            doors.append({"kind": kind, "a": a, "b": b, "coord": coord})
    return doors


def _render_input(
    room_rows: int,
    room_cols: int,
    wall_rows: list[int],
    wall_cols: list[int],
    sources: dict[tuple[int, int], dict[str, object]],
    doors: list[dict[str, object]],
) -> np.ndarray:
    height = int(wall_rows[-1] + 1)
    width = int(wall_cols[-1] + 1)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for row in wall_rows:
        grid[int(row), :] = WALL_COLOR
    for col in wall_cols:
        grid[:, int(col)] = WALL_COLOR
    for door in doors:
        row, col = map(int, door["coord"])
        grid[row, col] = BG_COLOR
    for data in sources.values():
        color = int(data["color"])
        for row, col in data["cells"]:
            grid[int(row), int(col)] = color
    return grid


def _emit_events_from_case(
    sources: dict[tuple[int, int], dict[str, object]],
    bounds: dict[tuple[int, int], tuple[int, int, int, int]],
    doors: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int, dict[tuple[int, int], int]]:
    events: list[dict[str, object]] = []
    blocked = 0
    incoming: dict[tuple[int, int], int] = defaultdict(int)
    for door in doors:
        a = tuple(door["a"])
        b = tuple(door["b"])
        coord = tuple(door["coord"])
        a_src = a in sources
        b_src = b in sources
        if a_src and b_src:
            blocked += 1
            continue
        if not (a_src ^ b_src):
            continue
        source_room = a if a_src else b
        target_room = b if a_src else a
        source = sources[source_room]
        incoming[target_room] += 1
        if door["kind"] == "h":
            if source_room == a:
                start = (int(coord[0] + 1), int(coord[1]))
                step = (1, 0)
            else:
                start = (int(coord[0] - 1), int(coord[1]))
                step = (-1, 0)
        else:
            if source_room == a:
                start = (int(coord[0]), int(coord[1] + 1))
                step = (0, 1)
            else:
                start = (int(coord[0]), int(coord[1] - 1))
                step = (0, -1)
        events.append(
            {
                "door": coord,
                "color": int(source["color"]),
                "length": int(source["count"]),
                "start": start,
                "step": step,
                "target_bounds": bounds[target_room],
            }
        )
    events.sort(key=lambda item: tuple(item["door"]))
    return events, blocked, incoming


def _render_output(
    grid: np.ndarray,
    events: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for event in events:
        top, left, bottom, right = map(int, event["target_bounds"])
        row, col = map(int, event["start"])
        dr, dc = map(int, event["step"])
        for _ in range(int(event["length"])):
            if not (top <= row <= bottom and left <= col <= right):
                break
            out[row, col] = int(event["color"])
            row += dr
            col += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("basic", "blocked", "multi"))
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)

    wall_rows = _make_wall_positions([random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)])
    wall_cols = _make_wall_positions([random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)])
    bounds = _room_bounds_from_layout(room_rows, room_cols, wall_rows, wall_cols)

    try:
        source_counts, door_map, meta = _build_sources(room_rows, room_cols, mode, wall_rows, wall_cols)
    except AssertionError:
        return None

    _place_extra_doors(door_map, room_rows, room_cols, wall_rows, wall_cols)

    source_rooms = sorted(source_counts)
    if not source_rooms or len(source_rooms) > len(SOURCE_PALETTE):
        return None

    colors = random.sample(list(SOURCE_PALETTE), k=len(source_rooms))
    sources: dict[tuple[int, int], dict[str, object]] = {}
    for room, color in zip(source_rooms, colors):
        count = int(source_counts[room])
        sources[room] = {
            "count": count,
            "color": int(color),
            "cells": _sample_source_cells(bounds[room], count),
        }

    doors = _flatten_doors(door_map)
    if not doors:
        return None

    grid = _render_input(room_rows, room_cols, wall_rows, wall_cols, sources, doors)
    expected_events, blocked, incoming = _emit_events_from_case(sources, bounds, doors)
    if not expected_events:
        return None

    if meta["mode"] == "blocked" and blocked == 0:
        return None
    if meta["mode"] == "multi":
        target = meta.get("multi_target")
        if target is None or int(incoming.get(tuple(target), 0)) < 2:
            return None

    counts = Counter(map(int, grid.ravel()))
    if int(max(counts, key=counts.get)) != BG_COLOR:
        return None
    non_bg = [item for item in counts.items() if int(item[0]) != BG_COLOR]
    if not non_bg or int(max(non_bg, key=lambda item: item[1])[0]) != WALL_COLOR:
        return None

    expected = _render_output(grid, expected_events)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
