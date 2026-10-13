from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

UR_PATTERN = [(-1, 0), (-1, 0), (0, 1), (0, 1)]
LD_PATTERN = [(1, 0), (1, 0), (0, -1), (0, -1)]
PATTERNS = {"ur": UR_PATTERN, "ld": LD_PATTERN}

GRID_HEIGHT_RANGE = (14, 19)
GRID_WIDTH_RANGE = (14, 19)
MAX_GENERATION_ATTEMPTS = 500
MIN_COMPLETE_STEPS = 2
MIN_STAIR_PIXELS = MIN_COMPLETE_STEPS * 4
OPTIONAL_BRANCH_PROBABILITY = 0.45

BASE_ANCHOR = (7, 5)
BASE_UR_STOP = [(r, 10) for r in range(7)]
BASE_LD_STOP = [(13, c) for c in range(8)]
THREE_WALL_EXTRAS = [
    [(0, 6), (0, 7), (0, 8)],
    [(0, 5), (0, 6), (0, 7), (0, 8)],
    [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8)],
    [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)],
    [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)],
    [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)],
    [(8, 0), (9, 0), (10, 0)],
    [(7, 0), (8, 0), (9, 0), (10, 0)],
    [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
    [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
    [(4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
    [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
]
FOUR_WALL_FIRST_EXTRAS = [
    [(8, 0), (9, 0), (10, 0)],
    [(7, 0), (8, 0), (9, 0), (10, 0)],
    [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
    [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
]
FOUR_WALL_SECOND_EXTRA = [(10, c) for c in range(7, 12)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    label_id = 0
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label_id += 1
            labels[r, c] = label_id
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label_id
                        queue.append((nr, nc))
    return labels, label_id


def _trace_direction(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    wall_mask: np.ndarray,
    direction: str,
) -> dict[str, object]:
    pattern = PATTERNS[direction]
    r, c = anchor
    path: list[tuple[int, int]] = []
    move_index = 0
    while True:
        dr, dc = pattern[move_index % len(pattern)]
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return {
                "path": path,
                "stop_reason": "boundary",
                "stop_next": (nr, nc),
                "full_steps": len(path) // 4,
            }
        if wall_mask[nr, nc]:
            return {
                "path": path,
                "stop_reason": "wall",
                "stop_next": (nr, nc),
                "full_steps": len(path) // 4,
            }
        path.append((nr, nc))
        r, c = nr, nc
        move_index += 1


def _region_graph(barrier_mask: np.ndarray) -> tuple[np.ndarray, int, dict[int, set[int]]]:
    labels, region_count = _label_mask(~barrier_mask)
    graph = {region_id: set() for region_id in range(1, region_count + 1)}
    height, width = barrier_mask.shape

    def _single_region(cells: list[tuple[int, int]]) -> int | None:
        region_ids = {
            int(labels[r, c])
            for r, c in cells
            if _in_bounds(barrier_mask.shape, r, c) and int(labels[r, c]) > 0
        }
        if len(region_ids) == 1:
            return next(iter(region_ids))
        return None

    for r in range(height):
        for c in range(width):
            if barrier_mask[r, c] and c + 1 < width and barrier_mask[r, c + 1]:
                a = _single_region([(r - 1, c), (r - 1, c + 1)])
                b = _single_region([(r + 1, c), (r + 1, c + 1)])
                if a is not None and b is not None and a != b:
                    graph[a].add(b)
                    graph[b].add(a)
            if barrier_mask[r, c] and r + 1 < height and barrier_mask[r + 1, c]:
                a = _single_region([(r, c - 1), (r + 1, c - 1)])
                b = _single_region([(r, c + 1), (r + 1, c + 1)])
                if a is not None and b is not None and a != b:
                    graph[a].add(b)
                    graph[b].add(a)
    return labels, region_count, graph


def _bipartite_coloring(graph: dict[int, set[int]], start: int) -> dict[int, int]:
    parity = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            expected = 1 - parity[cur]
            if nxt in parity:
                assert parity[nxt] == expected, "region graph must be bipartite"
            else:
                parity[nxt] = expected
                queue.append(nxt)
    return parity


def _transform_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height <= 30 and width <= 30, "grid must satisfy ARC size bounds"

    colors = set(map(int, g.flat))
    assert colors.issubset({BLACK, RED, TEAL}), "input should only contain black, red, and teal"

    anchors = np.argwhere(g == TEAL)
    assert len(anchors) == 1, "expected exactly one teal anchor"
    anchor = tuple(map(int, anchors[0]))
    wall_mask = g == RED
    for nr, nc in _neighbors4(*anchor):
        if _in_bounds(g.shape, nr, nc):
            assert not wall_mask[nr, nc], "anchor cannot be wall-adjacent"

    traces = {
        direction: _trace_direction(g.shape, anchor, wall_mask, direction)
        for direction in ("ur", "ld")
    }

    out = g.copy()
    stair_cells: list[tuple[int, int]] = []
    for trace in traces.values():
        for r, c in trace["path"]:
            out[r, c] = GREY
            stair_cells.append((r, c))
    out[anchor] = TEAL

    barrier_mask = out != BLACK
    labels, region_count, graph = _region_graph(barrier_mask)
    start_region = int(labels[height - 1, 0])
    assert start_region > 0, "the lower-left corner must stay in a background region"

    if region_count == 1:
        parity = {start_region: 0}
    else:
        parity = _bipartite_coloring(graph, start_region)
        assert len(parity) == region_count, "all background regions must be connected through barrier adjacencies"

    colored = out.copy()
    for region_id, region_parity in parity.items():
        colored[labels == region_id] = GREEN if region_parity == 0 else BLUE

    info = {
        "anchor": anchor,
        "traces": traces,
        "stair_cells": tuple(stair_cells),
        "barrier_mask": barrier_mask,
        "labels": labels,
        "region_count": region_count,
        "graph": graph,
        "parity": parity,
    }
    return colored, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw wall-stopped mirrored staircases from the teal anchor, then alternately color the wall-and-stair-separated background regions.

    Input structure:
    - The grid is black except for one teal anchor pixel and several red one-cell-thick wall segments.
    - Every red wall is a straight horizontal or vertical segment, and the anchor is not wall-adjacent.
    - The teal anchor is placed so the upper-right and lower-left staircase arms can each advance for at least two full step-length-two turns before stopping.

    Rule:
    - From the teal anchor, draw a grey staircase toward the upper-right by repeating: move up two cells, then right two cells; also draw the point-symmetric lower-left staircase by repeating: down two, then left two.
    - Before placing each staircase pixel, test the next target cell: if it is outside the grid or already a red wall, stop that whole arm immediately without covering the wall or leaving the grid.
    - Keep the teal anchor teal and keep all red walls unchanged.
    - Treat the red walls, the grey staircase pixels, and the teal anchor together as one barrier structure; split the remaining black cells into 4-connected regions and color the region containing the lower-left corner green, then alternate blue and green across every barrier-separated region adjacency.

    Key insight:
    - The red walls do double duty: they decide where each staircase arm terminates and, together with those truncated stairs, create the final barrier graph that determines the alternating flood-fill colors.
    """
    out, _info = _transform_details(grid)
    return out


def _horizontal_segment(row: int, col0: int, col1: int) -> list[tuple[int, int]]:
    return [(row, c) for c in range(col0, col1 + 1)]


def _vertical_segment(col: int, row0: int, row1: int) -> list[tuple[int, int]]:
    return [(r, col) for r in range(row0, row1 + 1)]


def _boundary_segments_through(shape: tuple[int, int], cell: tuple[int, int], side: str) -> list[list[tuple[int, int]]]:
    height, width = shape
    r, c = cell
    segments: list[list[tuple[int, int]]] = []
    if side == "top":
        for length in range(max(3, r + 1), min(8, height) + 1):
            segments.append(_vertical_segment(c, 0, length - 1))
    elif side == "bottom":
        dist = height - 1 - r
        for length in range(max(3, dist + 1), min(8, height) + 1):
            segments.append(_vertical_segment(c, height - length, height - 1))
    elif side == "left":
        for length in range(max(3, c + 1), min(8, width) + 1):
            segments.append(_horizontal_segment(r, 0, length - 1))
    elif side == "right":
        dist = width - 1 - c
        for length in range(max(3, dist + 1), min(8, width) + 1):
            segments.append(_horizontal_segment(r, width - length, width - 1))
    else:
        raise ValueError(side)
    return segments


def _wall_cells(segments: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for segment in segments:
        cells.update(segment)
    return cells


def _segment_ok_for_input(
    segment: list[tuple[int, int]],
    existing_walls: set[tuple[int, int]],
    anchor: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    for r, c in segment:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) == anchor:
            return False
        if (r, c) in existing_walls:
            return False
        if abs(r - anchor[0]) + abs(c - anchor[1]) == 1:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in existing_walls:
                    return False
    return True


def _segment_contact_count(segment: list[tuple[int, int]], contact_cells: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in segment:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in contact_cells:
                count += 1
    return count


def _empty_trace(shape: tuple[int, int], anchor: tuple[int, int], direction: str) -> dict[str, object]:
    wall_mask = np.zeros(shape, dtype=bool)
    return _trace_direction(shape, anchor, wall_mask, direction)


def _stop_candidates(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    direction: str,
    existing_walls: set[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], set[tuple[int, int]]]]:
    trace = _empty_trace(shape, anchor, direction)
    path = list(trace["path"])
    if len(path) <= MIN_STAIR_PIXELS:
        return []
    sides = {"ur": ("top",), "ld": ("left",)}[direction]
    candidates: list[tuple[list[tuple[int, int]], set[tuple[int, int]]]] = []
    for visited in range(MIN_STAIR_PIXELS, len(path)):
        last_r, last_c = path[visited - 1]
        dr, dc = PATTERNS[direction][visited % 4]
        next_cell = (last_r + dr, last_c + dc)
        for side in sides:
            for segment in _boundary_segments_through(shape, next_cell, side):
                if _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    candidates.append((segment, set(segment)))
    random.shuffle(candidates)
    return candidates


def _loop_candidates(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    existing_walls: set[tuple[int, int]],
    stair_cells: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    stair_set = set(stair_cells)
    candidates: list[list[tuple[int, int]]] = []
    height, width = shape

    def contact_positions(segment: list[tuple[int, int]]) -> list[int]:
        positions = []
        for idx, (r, c) in enumerate(segment):
            hits = 0
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in stair_set:
                    hits += 1
            if hits:
                if hits != 1:
                    return []
                positions.append(idx)
        return positions

    for row in range(height):
        for col0 in range(width):
            for length in range(3, min(8, width - col0) + 1):
                segment = _horizontal_segment(row, col0, col0 + length - 1)
                if not _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    continue
                positions = contact_positions(segment)
                if len(positions) == 2 and positions[0] < positions[1]:
                    candidates.append(segment)

    for col in range(width):
        for row0 in range(height):
            for length in range(3, min(8, height - row0) + 1):
                segment = _vertical_segment(col, row0, row0 + length - 1)
                if not _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    continue
                positions = contact_positions(segment)
                if len(positions) == 2 and positions[0] < positions[1]:
                    candidates.append(segment)

    random.shuffle(candidates)
    return candidates


def _branch_candidates(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    existing_walls: set[tuple[int, int]],
    stair_cells: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    stair_set = set(stair_cells)
    candidates: list[list[tuple[int, int]]] = []
    side_specs = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    for r, c in stair_cells:
        for side, (dr, dc) in side_specs.items():
            adj = (r + dr, c + dc)
            if not _in_bounds(shape, *adj):
                continue
            if adj == anchor or adj in stair_set or adj in existing_walls:
                continue
            for segment in _boundary_segments_through(shape, adj, side):
                if not _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    continue
                if _segment_contact_count(segment, stair_set) != 1:
                    continue
                candidates.append(segment)
    random.shuffle(candidates)
    return candidates


def _free_segment_candidates(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    existing_walls: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    height, width = shape
    candidates: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col0 in range(width):
            for length in range(3, min(8, width - col0) + 1):
                segment = _horizontal_segment(row, col0, col0 + length - 1)
                if _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    candidates.append(segment)
    for col in range(width):
        for row0 in range(height):
            for length in range(3, min(8, height - row0) + 1):
                segment = _vertical_segment(col, row0, row0 + length - 1)
                if _segment_ok_for_input(segment, existing_walls, anchor, shape):
                    candidates.append(segment)
    random.shuffle(candidates)
    return candidates


def _build_input(shape: tuple[int, int], anchor: tuple[int, int], segments: list[list[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in _wall_cells(segments):
        grid[r, c] = RED
    grid[anchor] = TEAL
    return grid


def _wall_component_lengths(grid: np.ndarray) -> list[int]:
    labels, count = _label_mask(grid == RED)
    lengths = []
    for label_id in range(1, count + 1):
        coords = np.argwhere(labels == label_id)
        rows = {int(r) for r, _c in coords}
        cols = {int(c) for _r, c in coords}
        if not (len(rows) == 1 or len(cols) == 1):
            return []
        lengths.append(len(coords))
    return lengths


def _region_count_from_barrier(barrier_mask: np.ndarray) -> int:
    _labels, count = _label_mask(~barrier_mask)
    return count


def _stairs_only_region_count(shape: tuple[int, int], anchor: tuple[int, int]) -> int:
    wall_mask = np.zeros(shape, dtype=bool)
    barrier = np.zeros(shape, dtype=bool)
    barrier[anchor] = True
    for direction in ("ur", "ld"):
        trace = _trace_direction(shape, anchor, wall_mask, direction)
        for cell in trace["path"]:
            barrier[cell] = True
    return _region_count_from_barrier(barrier)


def _validate_generated(
    grid: np.ndarray,
    mode: str,
    stop_targets: dict[str, set[tuple[int, int]]],
) -> bool:
    try:
        out, info = _transform_details(grid)
    except AssertionError:
        return False

    wall_lengths = _wall_component_lengths(grid)
    if not wall_lengths or any(length < 3 or length > 8 for length in wall_lengths):
        return False

    wall_only_regions = _region_count_from_barrier(grid == RED)
    if wall_only_regions != 1:
        return False

    stair_only_regions = _stairs_only_region_count(grid.shape, info["anchor"])
    if stair_only_regions >= int(info["region_count"]):
        return False

    for direction in ("ur", "ld"):
        trace = info["traces"][direction]
        if len(trace["path"]) < MIN_STAIR_PIXELS:
            return False
        if mode == "ur_boundary" and direction == "ur":
            if trace["stop_reason"] != "boundary":
                return False
        elif mode == "ld_boundary" and direction == "ld":
            if trace["stop_reason"] != "boundary":
                return False
        else:
            if trace["stop_reason"] != "wall":
                return False
            if tuple(trace["stop_next"]) not in stop_targets[direction]:
                return False

    if int(info["region_count"]) < 3:
        return False

    if not np.any(out == BLUE):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(17, 19)
    width = random.randint(14, 18)
    shape = (height, width)
    anchor = BASE_ANCHOR
    segments = [BASE_UR_STOP, BASE_LD_STOP]

    if random.random() < OPTIONAL_BRANCH_PROBABILITY:
        segments.append(random.choice(FOUR_WALL_FIRST_EXTRAS))
        segments.append(FOUR_WALL_SECOND_EXTRA)
    else:
        segments.append(random.choice(THREE_WALL_EXTRAS))

    grid = _build_input(shape, anchor, segments)
    stop_targets = {"ur": set(BASE_UR_STOP), "ld": set(BASE_LD_STOP)}
    if _validate_generated(grid, "wall_wall", stop_targets):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staircase-wall separator puzzle")
