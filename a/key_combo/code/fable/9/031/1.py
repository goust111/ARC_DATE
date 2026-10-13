from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 6
COLOR_POOL = (1, 2, 3, 4, 5, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_color_components(grid: np.ndarray, colors: Iterable[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    allowed = set(int(c) for c in colors)
    seen = np.zeros(g.shape, dtype=bool)
    comps = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color not in allowed or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _has_enclosed_hole(cells: set[tuple[int, int]]) -> bool:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    q = deque()
    for rr in range(h):
        for cc in range(w):
            if (rr in (0, h - 1) or cc in (0, w - 1)) and not mask[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~mask) & (~exterior)))


def _initial_dir(marker: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = marker
    h, w = shape
    sides = []
    if r == 0:
        sides.append(2)
    if r == h - 1:
        sides.append(0)
    if c == 0:
        sides.append(1)
    if c == w - 1:
        sides.append(3)
    assert len(sides) == 1, "marker must lie on exactly one border side"
    return sides[0]


def _turn(dir_idx: int, closed: bool) -> int:
    return (dir_idx - 1) % 4 if closed else (dir_idx + 1) % 4


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int], list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert len(marker_coords) == 1, "expected exactly one magenta border marker"
    marker = marker_coords[0]
    assert marker[0] in (0, g.shape[0] - 1) or marker[1] in (0, g.shape[1] - 1), "marker must be on border"
    assert not (
        marker[0] in (0, g.shape[0] - 1) and marker[1] in (0, g.shape[1] - 1)
    ), "marker must not be on a corner"

    fg_colors = sorted(int(v) for v in np.unique(g) if v not in (BG, MARKER))
    objects = _find_color_components(g, fg_colors)
    assert objects, "expected at least one foreground object"

    coord_to_obj: dict[tuple[int, int], int] = {}
    for obj_id, obj in enumerate(objects):
        closed = _has_enclosed_hole(obj["cells"])
        obj["closed"] = closed
        for cell in obj["cells"]:
            assert cell not in coord_to_obj, "objects must not overlap"
            coord_to_obj[cell] = obj_id

    all_object_cells = set(coord_to_obj)
    for cell in all_object_cells:
        for nbr in _neighbors8(*cell):
            if nbr in all_object_cells and nbr not in _find_owner_neighbors(cell, objects, coord_to_obj):
                raise AssertionError("objects must not touch each other")
    return marker, objects, coord_to_obj


def _find_owner_neighbors(
    cell: tuple[int, int], objects: list[dict], coord_to_obj: dict[tuple[int, int], int]
) -> set[tuple[int, int]]:
    obj_id = coord_to_obj[cell]
    return {nbr for nbr in _neighbors8(*cell) if nbr in objects[obj_id]["cells"]}


def _simulate(
    grid: np.ndarray, override_closed: dict[int, bool] | None = None
) -> tuple[np.ndarray, list[int], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    marker, objects, coord_to_obj = _parse_scene(g)
    out = g.copy()
    pos = marker
    dir_idx = _initial_dir(marker, g.shape)
    color = MARKER
    hit_ids: list[int] = []
    segments: list[list[tuple[int, int]]] = []
    current_segment: list[tuple[int, int]] = []
    seen_states = {(pos, dir_idx)}

    while True:
        dr, dc = DIRS[dir_idx]
        nr = pos[0] + dr
        nc = pos[1] + dc
        if not _in_bounds(g.shape, nr, nc):
            segments.append(current_segment)
            break

        next_cell = (nr, nc)
        obj_id = coord_to_obj.get(next_cell)
        if obj_id is not None:
            segments.append(current_segment)
            current_segment = []
            hit_ids.append(obj_id)
            closed = objects[obj_id]["closed"]
            if override_closed and obj_id in override_closed:
                closed = override_closed[obj_id]
            dir_idx = _turn(dir_idx, closed)
            color = int(objects[obj_id]["color"])
            state = (pos, dir_idx)
            assert state not in seen_states, "ray routing looped"
            seen_states.add(state)
            continue

        pos = next_cell
        if int(g[pos]) == BG:
            out[pos] = color
            current_segment.append(pos)

    return out, hit_ids, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch a ray from the border marker and let each hit object reroute and recolor the next segment.

    Input structure:
    - The background is black, and exactly one magenta single-pixel marker lies on the border but not on a corner.
    - The interior contains several separated monochrome objects.
    - Some objects are closed hollow frames with an enclosed hole; the rest are open line-like objects with no hole.

    Rule:
    - Start a ray at the marker and move inward perpendicular to the touched border, painting one background cell per step.
    - Whenever the next cell in the current direction is an object pixel, stop before it and turn: left by 90 degrees for a closed object, right by 90 degrees for an open object.
    - After each collision, continue from the same turning cell and recolor all later painted cells with the color of the most recently hit object.
    - The ray terminates as soon as it leaves the grid; the original marker and all object pixels stay unchanged.

    Key insight:
    - The objects are routers rather than decorations: hole-vs-no-hole encodes left-vs-right, and the polyline color records the latest router that redirected the ray.
    """
    out, _hits, _segments = _simulate(np.asarray(grid, dtype=int))
    return out


def _span_range(center: int, span: int, bias: int) -> tuple[int, int]:
    if bias == 0:
        half = span // 2
        return center - half, center + half
    if bias < 0:
        return center - span + 1, center
    return center, center + span - 1


def _ring_cells(hit: tuple[int, int], dir_idx: int, depth: int, span: int, bias: int = 0) -> set[tuple[int, int]]:
    assert depth >= 3 and span >= 3 and span % 2 == 1
    hr, hc = hit
    cells = set()
    if dir_idx == 1:
        r0, r1 = _span_range(hr, span, bias)
        c0, c1 = hc, hc + depth - 1
    elif dir_idx == 3:
        r0, r1 = _span_range(hr, span, bias)
        c0, c1 = hc - depth + 1, hc
    elif dir_idx == 2:
        r0, r1 = hr, hr + depth - 1
        c0, c1 = _span_range(hc, span, bias)
    else:
        r0, r1 = hr - depth + 1, hr
        c0, c1 = _span_range(hc, span, bias)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.add((r, c))
    return cells


def _bar_cells(hit: tuple[int, int], dir_idx: int, span: int, bias: int = 0) -> set[tuple[int, int]]:
    assert span >= 3 and span % 2 == 1
    hr, hc = hit
    if dir_idx in (1, 3):
        r0, r1 = _span_range(hr, span, bias)
        return {(r, hc) for r in range(r0, r1 + 1)}
    c0, c1 = _span_range(hc, span, bias)
    return {(hr, c) for c in range(c0, c1 + 1)}


def _hit_object_options(hit: tuple[int, int], dir_idx: int, closed: bool) -> list[set[tuple[int, int]]]:
    options = []
    out_dir = _turn(dir_idx, closed)
    if dir_idx in (1, 3):
        bias = 1 if out_dir == 0 else -1
    else:
        bias = 1 if out_dir == 3 else -1
    if closed:
        for depth, span in ((3, 3), (4, 3), (5, 3), (3, 5), (5, 5)):
            options.append(_ring_cells(hit, dir_idx, depth, span, bias))
    else:
        for span in (3, 5):
            options.append(_bar_cells(hit, dir_idx, span, bias))
    random.shuffle(options)
    return options


def _segment_clear(
    cells: list[tuple[int, int]],
    start: tuple[int, int],
    path_cells: set[tuple[int, int]],
    marker: tuple[int, int],
) -> bool:
    prev = start
    new_cells: set[tuple[int, int]] = set()
    for cell in cells:
        if cell == marker or cell in path_cells or cell in new_cells:
            return False
        for nbr in _neighbors4(*cell):
            if nbr in path_cells and nbr != prev:
                return False
            if nbr in new_cells and nbr != prev:
                return False
        new_cells.add(cell)
        prev = cell
    return True


def _object_clear(
    obj_cells: set[tuple[int, int]],
    hit: tuple[int, int],
    end_pos: tuple[int, int],
    path_cells: set[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    marker: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    if marker in obj_cells:
        return False
    for cell in obj_cells:
        if not _in_bounds(shape, *cell) or cell in path_cells or cell in object_cells:
            return False
        for nbr in _neighbors8(*cell):
            if nbr in object_cells:
                return False
        for nbr in _neighbors4(*cell):
            if nbr in path_cells and not (cell == hit and nbr == end_pos):
                return False
    return True


def _path_step_buffer_clear(
    cell: tuple[int, int], path_cells: set[tuple[int, int]], object_cells: set[tuple[int, int]], marker: tuple[int, int]
) -> bool:
    if cell == marker or cell in path_cells or cell in object_cells:
        return False
    return not any(nbr in object_cells for nbr in _neighbors4(*cell))


def _steps_to_exit(pos: tuple[int, int], dir_idx: int, shape: tuple[int, int]) -> int:
    dr, dc = DIRS[dir_idx]
    steps = 0
    r, c = pos
    while _in_bounds(shape, r + dr, c + dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _plan_route(
    shape: tuple[int, int], marker: tuple[int, int], turn_types: list[bool]
) -> tuple[list[list[tuple[int, int]]], list[dict]] | None:
    path_cells: set[tuple[int, int]] = set()
    segments: list[list[tuple[int, int]]] = []
    hits: list[dict] = []
    start_dir = _initial_dir(marker, shape)

    def local_object_ok(hit: tuple[int, int], end_pos: tuple[int, int], dir_idx: int, closed: bool) -> bool:
        turned = _turn(dir_idx, closed)
        next_step = (end_pos[0] + DIRS[turned][0], end_pos[1] + DIRS[turned][1])
        if not _in_bounds(shape, *next_step):
            return False
        for obj in _hit_object_options(hit, dir_idx, closed):
            if all(_in_bounds(shape, *cell) for cell in obj) and next_step not in obj:
                return True
        return False

    def dfs(pos: tuple[int, int], dir_idx: int, hit_idx: int) -> bool:
        if hit_idx == len(turn_types):
            final_len = _steps_to_exit(pos, dir_idx, shape)
            if final_len < 2:
                return False
            dr, dc = DIRS[dir_idx]
            segment = [(pos[0] + dr * step, pos[1] + dc * step) for step in range(1, final_len + 1)]
            if not _segment_clear(segment, pos, path_cells, marker):
                return False
            segments.append(segment)
            path_cells.update(segment)
            return True

        max_steps = _steps_to_exit(pos, dir_idx, shape)
        if max_steps < 3:
            return False
        lengths = list(range(2, max_steps))
        random.shuffle(lengths)

        for length in lengths:
            dr, dc = DIRS[dir_idx]
            segment = [(pos[0] + dr * step, pos[1] + dc * step) for step in range(1, length + 1)]
            hit = (pos[0] + dr * (length + 1), pos[1] + dc * (length + 1))
            end_pos = segment[-1]
            if not _segment_clear(segment, pos, path_cells, marker):
                continue

            closed = bool(turn_types[hit_idx])
            if not local_object_ok(hit, end_pos, dir_idx, closed):
                continue
            new_dir = _turn(dir_idx, closed)
            next_step = (end_pos[0] + DIRS[new_dir][0], end_pos[1] + DIRS[new_dir][1])
            if not _in_bounds(shape, *next_step) or next_step in path_cells or next_step in segment:
                continue
            if any(nbr in path_cells for nbr in _neighbors4(*next_step) if nbr != end_pos):
                continue

            segments.append(segment)
            path_cells.update(segment)
            hits.append({"closed": closed, "hit": hit, "dir_idx": dir_idx, "end_pos": end_pos})
            if dfs(end_pos, new_dir, hit_idx + 1):
                return True
            hits.pop()
            for cell in segment:
                path_cells.remove(cell)
            segments.pop()
        return False

    if dfs(marker, start_dir, 0):
        return segments, hits
    return None


def _place_hit_objects(
    shape: tuple[int, int], marker: tuple[int, int], segments: list[list[tuple[int, int]]], hits: list[dict]
) -> list[dict] | None:
    path_cells = set().union(*(set(seg) for seg in segments)) if segments else set()
    object_cells: set[tuple[int, int]] = set()
    objects: list[dict] = []
    for hit_info in hits:
        hit = hit_info["hit"]
        end_pos = hit_info["end_pos"]
        closed = bool(hit_info["closed"])
        dir_idx = int(hit_info["dir_idx"])
        placed = False
        for obj in _hit_object_options(hit, dir_idx, closed):
            if _object_clear(obj, hit, end_pos, path_cells, object_cells, marker, shape):
                object_cells.update(obj)
                objects.append({"cells": obj, "closed": closed, "hit": hit})
                placed = True
                break
        if not placed:
            return None
    return objects


def _random_marker(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(2, w - 3))
    if side == "bottom":
        return (h - 1, random.randint(2, w - 3))
    if side == "left":
        return (random.randint(2, h - 3), 0)
    return (random.randint(2, h - 3), w - 1)


def _paint_grid(
    shape: tuple[int, int], marker: tuple[int, int], objects: list[dict]
) -> tuple[np.ndarray, list[int]]:
    grid = np.full(shape, BG, dtype=int)
    grid[marker] = MARKER
    colors = random.sample(COLOR_POOL, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)
        for r, c in obj["cells"]:
            grid[r, c] = color
    return grid, colors


def _extra_candidates(
    shape: tuple[int, int], path_cells: set[tuple[int, int]], objects: list[dict], marker: tuple[int, int]
) -> list[dict]:
    h, w = shape
    object_cells = set().union(*(obj["cells"] for obj in objects)) if objects else set()
    candidates = []

    def far_from_path(cells: set[tuple[int, int]]) -> bool:
        for cell in cells:
            if cell == marker:
                return False
            for nbr in [cell, *_neighbors8(*cell)]:
                if nbr in path_cells or nbr == marker:
                    return False
            for nbr in _neighbors8(*cell):
                if nbr in object_cells:
                    return False
        return True

    for _ in range(80):
        closed = bool(random.getrandbits(1))
        if closed:
            horizontal = bool(random.getrandbits(1))
            if horizontal:
                span = random.choice((3, 5))
                depth = random.choice((3, 4, 5))
                r = random.randint(1 + span // 2, h - 2 - span // 2)
                c = random.randint(1, w - depth - 1)
                cells = _ring_cells((r, c), 1, depth, span)
            else:
                span = random.choice((3, 5))
                depth = random.choice((3, 4, 5))
                r = random.randint(1, h - depth - 1)
                c = random.randint(1 + span // 2, w - 2 - span // 2)
                cells = _ring_cells((r, c), 2, depth, span)
        else:
            if random.getrandbits(1):
                span = random.choice((3, 5))
                r = random.randint(1 + span // 2, h - 2 - span // 2)
                c = random.randint(1, w - 2)
                cells = _bar_cells((r, c), 1, span)
            else:
                span = random.choice((3, 5))
                r = random.randint(1, h - 2)
                c = random.randint(1 + span // 2, w - 2 - span // 2)
                cells = _bar_cells((r, c), 2, span)
        if all(_in_bounds(shape, *cell) for cell in cells) and far_from_path(cells):
            candidates.append({"cells": cells, "closed": closed, "hit": None})
    random.shuffle(candidates)
    return candidates


def _render_expected(grid: np.ndarray, segments: list[list[tuple[int, int]]], hit_order: list[int], objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    current = MARKER
    for seg_idx, segment in enumerate(segments):
        if seg_idx > 0:
            current = int(objects[hit_order[seg_idx - 1]]["color"])
        for r, c in segment:
            out[r, c] = current
    return out


def _validate_attempt(
    grid: np.ndarray, expected: np.ndarray, planned_segments: list[list[tuple[int, int]]], planned_hits: int
) -> bool:
    try:
        actual, hit_order, detected_segments = _simulate(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if len(hit_order) != planned_hits or len(detected_segments) != len(planned_segments):
        return False
    if [len(seg) for seg in detected_segments] != [len(seg) for seg in planned_segments]:
        return False

    marker, objects, _coord_to_obj = _parse_scene(grid)
    hit_types = [bool(objects[obj_id]["closed"]) for obj_id in hit_order]
    if len(hit_types) < 2 or len(set(hit_types)) < 2:
        return False

    if np.array_equal(grid, actual):
        return False
    if len({tuple(seg[0]) for seg in detected_segments if seg}) < 2:
        return False

    for obj_id in hit_order:
        try:
            flipped, _flipped_hits, _flipped_segments = _simulate(
                grid, override_closed={obj_id: not objects[obj_id]["closed"]}
            )
        except AssertionError:
            continue
        if np.array_equal(flipped, actual):
            return False

    path_colors = {int(actual[r, c]) for segment in detected_segments for r, c in segment}
    if MARKER not in path_colors or len(path_colors) < 2:
        return False
    if marker in set().union(*[set(seg) for seg in detected_segments if seg] or [set()]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    shape = (h, w)
    marker = _random_marker(shape)
    hit_count = random.choices([2, 3, 4], weights=[3, 4, 2])[0]
    turn_types = [True, False]
    while len(turn_types) < hit_count:
        turn_types.append(bool(random.getrandbits(1)))
    random.shuffle(turn_types)

    planned = _plan_route(shape, marker, turn_types)
    if planned is None:
        return None
    segments, hits = planned
    routed_objects = _place_hit_objects(shape, marker, segments, hits)
    if routed_objects is None:
        return None
    hit_object_count = len(routed_objects)

    path_cells = set().union(*(set(seg) for seg in segments)) if segments else set()
    target_extra_count = random.choice((0, 1, 2))
    for candidate in _extra_candidates(shape, path_cells, routed_objects, marker):
        if len(routed_objects) - hit_object_count >= target_extra_count:
            break
        routed_objects.append(candidate)

    if len(routed_objects) > len(COLOR_POOL):
        return None

    grid, _colors = _paint_grid(shape, marker, routed_objects)
    expected = _render_expected(grid, segments, list(range(hit_object_count)), routed_objects)
    if not _validate_attempt(grid, expected, segments, hit_object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routing-ray input")
