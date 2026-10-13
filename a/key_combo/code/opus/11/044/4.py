from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8_including_self(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            yield r + dr, c + dc


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _bbox_from_cells(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return _tight_mask(np.rot90(mask, k=-(turns % 4)))


def _is_connected(mask: np.ndarray) -> bool:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not cells:
        return False
    seen = set()
    q = deque([next(iter(cells))])
    seen.add(q[0])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == cells


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
    ]
    return all(mask.shape != other.shape or not np.array_equal(mask, other) for other in variants)


def _build_shape_library() -> list[np.ndarray]:
    base_shapes = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ]

    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    shapes: list[np.ndarray] = []
    for base in base_shapes:
        for turns in range(4):
            mask = _rotate_mask_cw(base, turns)
            key = (tuple(mask.shape), tuple(int(v) for v in mask.ravel()))
            if key in seen:
                continue
            seen.add(key)
            if 3 <= int(mask.sum()) <= 6 and max(mask.shape) <= 4 and _is_connected(mask) and _is_asymmetric(mask):
                shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _build_shape_library()
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_RANGE = (12, 15)
MAX_ATTEMPTS = 400


def _object_record(color: int, cells: set[Coord]) -> dict:
    return {
        "color": int(color),
        "cells": cells,
        "bbox": _bbox_from_cells(cells),
    }


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            objects.append(_object_record(color, cells))
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _next_outline_layer(current: set[Coord], shape: tuple[int, int]) -> set[Coord] | None:
    layer: set[Coord] = set()
    for r, c in current:
        for nr, nc in _neighbors8_including_self(r, c):
            if (nr, nc) in current:
                continue
            if not _in_bounds(shape, nr, nc):
                return None
            layer.add((nr, nc))
    return layer


def _touches_other_foreground(layer: set[Coord], other_foreground: set[Coord]) -> bool:
    for r, c in layer:
        for nr, nc in _neighbors8_including_self(r, c):
            if (nr, nc) in other_foreground:
                return True
    return False


def _conflicts_reserved(layer: set[Coord], reserved: set[Coord]) -> bool:
    if not reserved:
        return False
    for r, c in layer:
        for nr, nc in _neighbors8_including_self(r, c):
            if (nr, nc) in reserved:
                return True
    return False


def _depth_for_object(grid: np.ndarray, obj: dict, reserved: set[Coord]) -> tuple[int, set[Coord]]:
    foreground = {(int(r), int(c)) for r, c in np.argwhere(grid != 0)}
    other_foreground = foreground - obj["cells"]
    current = set(obj["cells"])
    claimed: set[Coord] = set()
    depth = 0
    while True:
        layer = _next_outline_layer(current, grid.shape)
        if not layer:
            break
        if any(int(grid[r, c]) != 0 for r, c in layer):
            break
        if _touches_other_foreground(layer, other_foreground):
            break
        if _conflicts_reserved(layer, reserved):
            break
        claimed |= layer
        current |= layer
        depth += 1
    return depth, claimed


def _sequential_depths(grid: np.ndarray, objects: list[dict]) -> list[int]:
    reserved: set[Coord] = set()
    depths: list[int] = []
    for obj in objects:
        depth, claimed = _depth_for_object(grid, obj, reserved)
        depths.append(depth)
        reserved |= claimed
    return depths


def _free_depths(grid: np.ndarray, objects: list[dict]) -> list[int]:
    return [_depth_for_object(grid, obj, set())[0] for obj in objects]


def _rotate_cell_cw(r: int, c: int, pivot: Coord, turns: int) -> Coord:
    pr, pc = pivot
    dr = r - pr
    dc = c - pc
    turns %= 4
    for _ in range(turns):
        dr, dc = dc, -dr
    return pr + dr, pc + dc


def _render_rotated(grid: np.ndarray, objects: list[dict], depths: list[int]) -> np.ndarray:
    out = np.zeros_like(grid)
    for obj, depth in zip(objects, depths):
        r0, c0, r1, c1 = obj["bbox"]
        pivot = ((r0 + r1) // 2, (c0 + c1) // 2)
        turns = depth % 4
        for r, c in obj["cells"]:
            nr, nc = _rotate_cell_cw(r, c, pivot, turns)
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if int(out[nr, nc]) != 0:
                raise AssertionError("rotated objects must not overlap")
            out[nr, nc] = obj["color"]
    return out


def _analyze(grid: np.ndarray) -> dict:
    objects = _extract_objects(grid)
    sequential = _sequential_depths(grid, objects)
    free = _free_depths(grid, objects)
    out_seq = _render_rotated(grid, objects, sequential)
    out_free = _render_rotated(grid, objects, free)
    return {
        "objects": objects,
        "sequential_depths": sequential,
        "free_depths": free,
        "output": out_seq,
        "free_output": out_free,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure each object's ordered outline depth and rotate it clockwise by that many quarter-turns.

    Input structure:
    - A black background contains 3 or 4 disconnected single-color foreground objects.
    - Each object is a small 4-connected asymmetric shape, and objects are processed in bbox-top-left order.
    - Empty black space between objects and borders is the only space outlines may claim.

    Rule:
    - For each object, iteratively grow one 8-neighbor outline layer at a time around its current occupied region.
    - A layer is valid only if every required cell stays in bounds, lands on black background, does not 8-touch another foreground object, and does not conflict with outline space already claimed by earlier objects.
    - Let depth be the number of valid layers. Rotate the original object clockwise by `(depth mod 4) * 90` around the floored center of its tight bbox, and output only the rotated object pixels.

    Key insight:
    - The outlines are not drawn for decoration: their ordered growth depth acts as a hidden space meter, and that measured depth is the rotation instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    info = _analyze(g)
    assert 3 <= len(info["objects"]) <= 4, "expected 3 or 4 objects"
    return info["output"]


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[Coord]:
    return {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}


def _paint_objects(shape: tuple[int, int], placements: list[tuple[np.ndarray, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for mask, top, left, color in placements:
        for r, c in _mask_cells(mask, top, left):
            if int(grid[r, c]) != 0:
                raise AssertionError("input objects must not overlap")
            grid[r, c] = color
    return grid


def _candidate_positions(mask: np.ndarray, grid_shape: tuple[int, int]) -> list[Coord]:
    h, w = grid_shape
    mh, mw = mask.shape
    return [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]


def _position_later_than(pos: Coord, other: Coord) -> bool:
    return pos[0] > other[0] or (pos[0] == other[0] and pos[1] > other[1])


def _same_area_after_rotation(info: dict) -> bool:
    out = info["output"]
    for obj in info["objects"]:
        if int(np.count_nonzero(out == obj["color"])) != len(obj["cells"]):
            return False
    return True


def _output_component_count(out: np.ndarray) -> int:
    return len(_extract_objects(out))


def _instance_is_valid(grid: np.ndarray, expected_count: int) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    objects = info["objects"]
    seq = info["sequential_depths"]
    free = info["free_depths"]
    out = info["output"]

    if len(objects) != expected_count:
        return False
    if any(len(obj["cells"]) < 3 or len(obj["cells"]) > 6 for obj in objects):
        return False
    if any(max(obj["bbox"][2] - obj["bbox"][0] + 1, obj["bbox"][3] - obj["bbox"][1] + 1) > 4 for obj in objects):
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    if min(seq) != 0 or max(seq) < 2:
        return False
    if not any(depth > 0 for depth in seq):
        return False
    if not any(s < f for s, f in zip(seq, free)):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, info["free_output"]):
        return False
    if _output_component_count(out) != len(objects):
        return False
    if not _same_area_after_rotation(info):
        return False
    return True


def _find_pair(
    grid_shape: tuple[int, int],
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    color_a: int,
    color_b: int,
) -> list[tuple[np.ndarray, int, int, int]] | None:
    a_positions = [
        pos
        for pos in _candidate_positions(shape_a, grid_shape)
        if pos[0] <= grid_shape[0] // 3 and pos[1] <= grid_shape[1] // 3 and (pos[0] == 1 or pos[1] == 1)
    ]
    b_positions = [
        pos
        for pos in _candidate_positions(shape_b, grid_shape)
        if pos[0] <= grid_shape[0] // 2 and pos[1] <= grid_shape[1] // 2
    ]
    random.shuffle(a_positions)
    random.shuffle(b_positions)

    for a_top, a_left in a_positions:
        for b_top, b_left in b_positions:
            if not _position_later_than((b_top, b_left), (a_top, a_left)):
                continue
            placements = [
                (shape_a, a_top, a_left, color_a),
                (shape_b, b_top, b_left, color_b),
            ]
            try:
                grid = _paint_objects(grid_shape, placements)
            except AssertionError:
                continue
            info = _analyze(grid)
            if len(info["objects"]) != 2:
                continue
            if info["sequential_depths"] == [1, 0] and info["free_depths"] == [1, 1]:
                return placements
    return None


def _find_third(
    grid_shape: tuple[int, int],
    base: list[tuple[np.ndarray, int, int, int]],
    shape_c: np.ndarray,
    color_c: int,
    target_depth: int,
) -> list[tuple[np.ndarray, int, int, int]] | None:
    base_last = (base[-1][1], base[-1][2])
    positions = [
        pos
        for pos in _candidate_positions(shape_c, grid_shape)
        if pos[0] >= grid_shape[0] // 2 - 1 and pos[1] >= grid_shape[1] // 2 - 1 and _position_later_than(pos, base_last)
    ]
    random.shuffle(positions)

    for top, left in positions:
        placements = base + [(shape_c, top, left, color_c)]
        try:
            grid = _paint_objects(grid_shape, placements)
            info = _analyze(grid)
        except AssertionError:
            continue
        if len(info["objects"]) != 3:
            continue
        if info["sequential_depths"][:2] != [1, 0]:
            continue
        if info["free_depths"][:2] != [1, 1]:
            continue
        if info["sequential_depths"][2] != target_depth or info["free_depths"][2] != target_depth:
            continue
        if _instance_is_valid(grid, 3):
            return placements
    return None


def _find_fourth(
    grid_shape: tuple[int, int],
    base: list[tuple[np.ndarray, int, int, int]],
    shape_d: np.ndarray,
    color_d: int,
    target_depth: int,
) -> list[tuple[np.ndarray, int, int, int]] | None:
    base_last = (base[-1][1], base[-1][2])
    positions = [
        pos
        for pos in _candidate_positions(shape_d, grid_shape)
        if _position_later_than(pos, base_last)
    ]
    random.shuffle(positions)

    for top, left in positions:
        placements = base + [(shape_d, top, left, color_d)]
        try:
            grid = _paint_objects(grid_shape, placements)
            info = _analyze(grid)
        except AssertionError:
            continue
        if len(info["objects"]) != 4:
            continue
        if info["sequential_depths"][:3] != [1, 0, target_depth]:
            continue
        if info["free_depths"][:3] != [1, 1, target_depth]:
            continue
        if info["sequential_depths"][3] > 2:
            continue
        if _instance_is_valid(grid, 4):
            return placements
    return None


def _sample_shapes(count: int) -> list[np.ndarray]:
    return [random.choice(SHAPE_LIBRARY) for _ in range(count)]


def _generate_attempt() -> np.ndarray | None:
    side = random.randint(*GRID_RANGE)
    grid_shape = (side, side)
    count = 4 if random.random() < 0.45 else 3
    colors = random.sample(PALETTE, count)
    target_depth = random.choice((2, 3))
    shapes = _sample_shapes(count)

    pair = _find_pair(grid_shape, shapes[0], shapes[1], colors[0], colors[1])
    if pair is None:
        return None

    triple = _find_third(grid_shape, pair, shapes[2], colors[2], target_depth)
    if triple is None:
        return None

    placements = triple
    if count == 4:
        placements = _find_fourth(grid_shape, triple, shapes[3], colors[3], target_depth)
        if placements is None:
            return None

    grid = _paint_objects(grid_shape, placements)
    if not _instance_is_valid(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid outline-depth rotation puzzle")
