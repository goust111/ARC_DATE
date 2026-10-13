from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

SHAPE_LIBRARY: dict[int, list[tuple[tuple[int, int], ...]]] = {
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
    5: [
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ],
}

GEN_GRID_RANGE = (19, 29)
GEN_BLUE_COUNT_RANGE = (2, 4)
GEN_GREEN_COUNT_RANGE = (1, 3)
GEN_GAP_RANGE = (2, 6)
GEN_MAX_ATTEMPTS = 400


def _normalize(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _dims(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[int, int]:
    norm = _normalize(coords)
    return max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    height, _width = _dims(norm)
    return _normalize(tuple((c, height - 1 - r) for r, c in norm))


def _flip_lr(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    _height, width = _dims(norm)
    return _normalize(tuple((r, width - 1 - c) for r, c in norm))


def _unique_orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []

    cur = _normalize(coords)
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        flipped = _flip_lr(cur)
        if flipped not in seen:
            seen.add(flipped)
            out.append(flipped)
        cur = _rot90(cur)
    return out


def _canonical_shape(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_unique_orientations(_normalize(tuple(coords))))


def _bbox(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shift(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r) + dr, int(c) + dc) for r, c in coords))


def _component_from_mask(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    visited = np.zeros_like(mask, dtype=bool)
    out: list[tuple[tuple[int, int], ...]] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            out.append(tuple(sorted(cells)))
    return out


def _find_components(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], ...]]:
    return _component_from_mask(np.asarray(grid, dtype=int) == int(color))


def _exposed(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    coord_set = set((int(r), int(c)) for r, c in coords)
    return [
        (int(r), int(c))
        for r, c in coords
        if (int(r) + dr, int(c) + dc) not in coord_set
    ]


def _straight_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        lo, hi = sorted((int(ac), int(bc)))
        return [(int(ar), c) for c in range(lo + 1, hi)]
    if ac == bc:
        lo, hi = sorted((int(ar), int(br)))
        return [(r, int(ac)) for r in range(lo + 1, hi)]
    raise AssertionError("line endpoints must share a row or column")


def _cells_clear(scene: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    height, width = scene.shape
    return all(0 <= r < height and 0 <= c < width and int(scene[r, c]) == BG for r, c in cells)


def _best_connection(
    template: tuple[tuple[int, int], ...],
    obj: tuple[tuple[int, int], ...],
    scene: np.ndarray,
) -> dict[str, object]:
    template_left = _exposed(template, (0, -1))
    template_right = _exposed(template, (0, 1))
    template_up = _exposed(template, (-1, 0))
    template_down = _exposed(template, (1, 0))

    obj_left = _exposed(obj, (0, -1))
    obj_right = _exposed(obj, (0, 1))
    obj_up = _exposed(obj, (-1, 0))
    obj_down = _exposed(obj, (1, 0))

    candidates: list[dict[str, object]] = []

    for orow, ocol in obj_right:
        for trow, tcol in template_left:
            if int(orow) == int(trow) and int(ocol) < int(tcol):
                cells = _straight_cells((orow, ocol), (trow, tcol))
                if _cells_clear(scene, cells):
                    candidates.append(
                        {
                            "axis": 0,
                            "gap": len(cells),
                            "cells": cells,
                            "obj_endpoint": (int(orow), int(ocol)),
                            "template_endpoint": (int(trow), int(tcol)),
                        }
                    )

    for orow, ocol in obj_left:
        for trow, tcol in template_right:
            if int(trow) == int(orow) and int(tcol) < int(ocol):
                cells = _straight_cells((trow, tcol), (orow, ocol))
                if _cells_clear(scene, cells):
                    candidates.append(
                        {
                            "axis": 0,
                            "gap": len(cells),
                            "cells": cells,
                            "obj_endpoint": (int(orow), int(ocol)),
                            "template_endpoint": (int(trow), int(tcol)),
                        }
                    )

    for orow, ocol in obj_down:
        for trow, tcol in template_up:
            if int(ocol) == int(tcol) and int(orow) < int(trow):
                cells = _straight_cells((orow, ocol), (trow, tcol))
                if _cells_clear(scene, cells):
                    candidates.append(
                        {
                            "axis": 1,
                            "gap": len(cells),
                            "cells": cells,
                            "obj_endpoint": (int(orow), int(ocol)),
                            "template_endpoint": (int(trow), int(tcol)),
                        }
                    )

    for orow, ocol in obj_up:
        for trow, tcol in template_down:
            if int(ocol) == int(tcol) and int(trow) < int(orow):
                cells = _straight_cells((trow, tcol), (orow, ocol))
                if _cells_clear(scene, cells):
                    candidates.append(
                        {
                            "axis": 1,
                            "gap": len(cells),
                            "cells": cells,
                            "obj_endpoint": (int(orow), int(ocol)),
                            "template_endpoint": (int(trow), int(tcol)),
                        }
                    )

    assert candidates, "each blue object must have a straight path to the template"
    candidates.sort(
        key=lambda item: (
            int(item["gap"]),
            int(item["axis"]),
            tuple(item["template_endpoint"]),
            tuple(item["obj_endpoint"]),
        )
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the red template and its blue shape matches, erase the green distractors, and connect each match back to the template.

    Input structure:
    - The background is black.
    - There is exactly one small red 4-connected template object near the middle of the grid.
    - Several other small 4-connected objects are scattered around it; blue ones have the same normalized shape as the template up to rotation or reflection, while green ones do not.

    Rule:
    - Keep the red template object and every blue object unchanged.
    - Erase every green object to black.
    - For each blue object, connect it to the red template with a straight yellow segment.
    - The segment must be horizontal or vertical: choose the shorter valid connection between exposed boundary pixels that share a row or a column, and color only the cells strictly between the two endpoints.

    Key insight:
    - The colors already encode which scattered objects match the template, and the remaining work is to find the nearest straight boundary-to-boundary link from each blue object back to the red anchor.
    """
    g = np.asarray(grid, dtype=int)
    red_components = _find_components(g, RED)
    assert len(red_components) == 1, "expected exactly one red template object"

    template = red_components[0]
    blue_components = sorted(_find_components(g, BLUE), key=_bbox)

    scene = g.copy()
    scene[scene == GREEN] = BG

    out = scene.copy()
    for blue in blue_components:
        connection = _best_connection(template, blue, scene)
        for r, c in connection["cells"]:
            out[int(r), int(c)] = YELLOW

    return out


def _paint(grid: np.ndarray, coords: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _mark_occupied(occupied: np.ndarray, coords: tuple[tuple[int, int], ...]) -> None:
    for r, c in coords:
        occupied[int(r), int(c)] = True


def _update_blocked(blocked: np.ndarray, coords: tuple[tuple[int, int], ...]) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for dr, dc in NEIGHBORS8 + ((0, 0),):
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < height and 0 <= nc < width:
                blocked[nr, nc] = True


def _can_place_object(
    coords: tuple[tuple[int, int], ...],
    blocked: np.ndarray,
    corridor_mask: np.ndarray,
) -> bool:
    height, width = blocked.shape
    for r, c in coords:
        if not (0 <= int(r) < height and 0 <= int(c) < width):
            return False
        if bool(blocked[int(r), int(c)]) or bool(corridor_mask[int(r), int(c)]):
            return False
    return True


def _reserve_corridor(corridor_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        corridor_mask[int(r), int(c)] = True


def _side_directions(side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "left":
        return (0, -1), (0, 1)
    if side == "right":
        return (0, 1), (0, -1)
    if side == "up":
        return (-1, 0), (1, 0)
    if side == "down":
        return (1, 0), (-1, 0)
    raise AssertionError(side)


def _attempt_blue_placement(
    shape: tuple[tuple[int, int], ...],
    side: str,
    template: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
    blocked: np.ndarray,
    corridor_mask: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int]]] | None:
    height, width = blocked.shape
    template_dir, object_dir = _side_directions(side)
    template_edges = _exposed(template, template_dir)
    object_edges = _exposed(shape, object_dir)

    choices: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for template_cell in template_edges:
        for object_cell in object_edges:
            for gap in range(GEN_GAP_RANGE[0], GEN_GAP_RANGE[1] + 1):
                choices.append((template_cell, object_cell, gap))
    random.shuffle(choices)

    for (tr, tc), (orow, ocol), gap in choices:
        if side == "left":
            top = int(tr) - int(orow)
            left = int(tc) - int(ocol) - gap - 1
            endpoint = (int(top + orow), int(left + ocol))
            cells = [(int(tr), c) for c in range(endpoint[1] + 1, int(tc))]
        elif side == "right":
            top = int(tr) - int(orow)
            left = int(tc) - int(ocol) + gap + 1
            endpoint = (int(top + orow), int(left + ocol))
            cells = [(int(tr), c) for c in range(int(tc) + 1, endpoint[1])]
        elif side == "up":
            top = int(tr) - int(orow) - gap - 1
            left = int(tc) - int(ocol)
            endpoint = (int(top + orow), int(left + ocol))
            cells = [(r, int(tc)) for r in range(endpoint[0] + 1, int(tr))]
        elif side == "down":
            top = int(tr) - int(orow) + gap + 1
            left = int(tc) - int(ocol)
            endpoint = (int(top + orow), int(left + ocol))
            cells = [(r, int(tc)) for r in range(int(tr) + 1, endpoint[0])]
        else:
            raise AssertionError(side)

        placed = _shift(shape, int(top), int(left))
        if not _can_place_object(placed, blocked, corridor_mask):
            continue
        if not all(0 <= r < height and 0 <= c < width for r, c in cells):
            continue
        if any(bool(corridor_mask[r, c]) or bool(occupied[r, c]) for r, c in cells):
            continue
        return placed, cells

    return None


def _random_free_location(
    shape: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
    blocked: np.ndarray,
    corridor_mask: np.ndarray,
    tries: int = 80,
) -> tuple[tuple[tuple[int, int], ...], int, int] | None:
    height, width = blocked.shape
    sh, sw = _dims(shape)
    if sh > height or sw > width:
        return None

    for _ in range(tries):
        top = random.randint(0, height - sh)
        left = random.randint(0, width - sw)
        placed = _shift(shape, top, left)
        if _can_place_object(placed, blocked, corridor_mask) and not any(bool(occupied[r, c]) for r, c in placed):
            return placed, top, left
    return None


def _choose_axes(blue_count: int) -> tuple[list[int], str]:
    roll = random.random()
    if roll < 0.25:
        return [0] * blue_count, "horizontal_only"
    if roll < 0.5:
        return [1] * blue_count, "vertical_only"
    axes = [0, 1]
    axes.extend(random.choice((0, 1)) for _ in range(blue_count - 2))
    random.shuffle(axes)
    return axes, "mixed"


def _choose_sides(axes: list[int]) -> list[str]:
    counts = {"left": 0, "right": 0, "up": 0, "down": 0}
    sides: list[str] = []
    for axis in axes:
        options = ("left", "right") if axis == 0 else ("up", "down")
        min_count = min(counts[name] for name in options)
        choices = [name for name in options if counts[name] == min_count]
        side = random.choice(choices)
        counts[side] += 1
        sides.append(side)
    return sides


def _place_template(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid.shape
    occupied = np.zeros_like(grid, dtype=bool)
    blocked = np.zeros_like(grid, dtype=bool)
    corridor_mask = np.zeros_like(grid, dtype=bool)

    sh, sw = _dims(shape)
    center_top = height // 2 - sh // 2 + random.randint(-1, 1)
    center_left = width // 2 - sw // 2 + random.randint(-1, 1)
    top = min(max(2, center_top), height - sh - 3)
    left = min(max(2, center_left), width - sw - 3)
    placed = _shift(shape, top, left)

    _paint(grid, placed, RED)
    _mark_occupied(occupied, placed)
    _update_blocked(blocked, placed)
    return placed, occupied, blocked, corridor_mask


def _validate_scene(grid: np.ndarray, mode: str) -> bool:
    red_components = _find_components(grid, RED)
    blue_components = _find_components(grid, BLUE)
    green_components = _find_components(grid, GREEN)
    if len(red_components) != 1 or not blue_components or not green_components:
        return False

    template_key = _canonical_shape(red_components[0])
    if len(red_components[0]) not in (3, 4, 5):
        return False
    if any(_canonical_shape(comp) != template_key for comp in blue_components):
        return False
    if any(_canonical_shape(comp) == template_key for comp in green_components):
        return False

    scene = grid.copy()
    scene[scene == GREEN] = BG
    axes = []
    for blue in blue_components:
        connection = _best_connection(red_components[0], blue, scene)
        if int(connection["gap"]) <= 0:
            return False
        axes.append(int(connection["axis"]))

    axis_set = set(axes)
    if mode == "horizontal_only" and axis_set != {0}:
        return False
    if mode == "vertical_only" and axis_set != {1}:
        return False
    if mode == "mixed" and axis_set != {0, 1}:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_GRID_RANGE)
    width = random.randint(*GEN_GRID_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    area = random.choice((3, 4, 5))
    template_family = random.choice(SHAPE_LIBRARY[area])
    template_shape = random.choice(_unique_orientations(template_family))

    template, occupied, blocked, corridor_mask = _place_template(grid, template_shape)

    blue_count = random.randint(*GEN_BLUE_COUNT_RANGE)
    green_count = random.randint(*GEN_GREEN_COUNT_RANGE)
    axes, mode = _choose_axes(blue_count)
    sides = _choose_sides(axes)

    blue_shapes: list[tuple[tuple[int, int], ...]] = []
    for _ in range(blue_count):
        blue_shapes.append(random.choice(_unique_orientations(template_family)))

    for shape, side in zip(blue_shapes, sides, strict=False):
        placed = _attempt_blue_placement(shape, side, template, occupied, blocked, corridor_mask)
        if placed is None:
            return None
        obj_coords, line_cells = placed
        _paint(grid, obj_coords, BLUE)
        _mark_occupied(occupied, obj_coords)
        _update_blocked(blocked, obj_coords)
        _reserve_corridor(corridor_mask, line_cells)

    other_families = [shape for shape in SHAPE_LIBRARY[area] if _canonical_shape(shape) != _canonical_shape(template_family)]
    for _ in range(green_count):
        family = random.choice(other_families)
        shape = random.choice(_unique_orientations(family))
        placed = _random_free_location(shape, occupied, blocked, corridor_mask)
        if placed is None:
            return None
        obj_coords, _top, _left = placed
        _paint(grid, obj_coords, GREEN)
        _mark_occupied(occupied, obj_coords)
        _update_blocked(blocked, obj_coords)

    if not _validate_scene(grid, mode):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
