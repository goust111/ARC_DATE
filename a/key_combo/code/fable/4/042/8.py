import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]

GEN_RETRIES = 400
GRID_SIZE_RANGE = (14, 24)
LANE_COUNT_CHOICES = [1, 2, 2, 3]
OBJECT_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
GAP_RANGE = (1, 4)
MARKER_DEPTH_RANGE = (1, 2)
THICKNESS_RANGE = (1, 3)
LENGTH_RANGE = (1, 4)
LANE_PLACEMENT_TRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(grid: np.ndarray, colors: Iterable[int]) -> list[dict]:
    color_set = set(int(color) for color in colors)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) not in color_set:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            anchor = (r0, c0)
            shape = frozenset((int(rr - r0), int(cc - c0)) for rr, cc in coords.tolist())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "anchor": anchor,
                    "shape": shape,
                }
            )
    return components


def _component_index_grid(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    index_grid = np.full(shape, -1, dtype=int)
    for idx, comp in enumerate(components):
        coords = comp["coords"]
        index_grid[coords[:, 0], coords[:, 1]] = idx
    return index_grid


def _steps_to_edge(coords: np.ndarray, shape: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr == -1:
        return int(coords[:, 0].min())
    if dr == 1:
        return int(shape[0] - 1 - coords[:, 0].max())
    if dc == -1:
        return int(coords[:, 1].min())
    return int(shape[1] - 1 - coords[:, 1].max())


def _axis_extent(comp: dict, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    if direction[0] != 0:
        return int(r1 - r0 + 1)
    return int(c1 - c0 + 1)


def _marker_launch(marker: dict, object_index: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    touched_ids = set()
    touched_dirs = set()
    h, w = object_index.shape
    for r, c in marker["coords"].tolist():
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            comp_id = int(object_index[nr, nc])
            if comp_id >= 0:
                touched_ids.add(comp_id)
                touched_dirs.add((dr, dc))
    if not touched_ids:
        return None
    assert len(touched_ids) == 1, "each marker must touch exactly one starter object"
    assert len(touched_dirs) == 1, "each marker must launch in one cardinal direction"
    return next(iter(touched_ids)), next(iter(touched_dirs))


def _lane_queue(
    starter_id: int,
    direction: tuple[int, int],
    components: list[dict],
    processed: set[int],
) -> list[int]:
    starter = components[starter_id]
    r0, c0, r1, c1 = starter["bbox"]
    shape = starter["shape"]
    dr, dc = direction
    queue = []
    for idx, comp in enumerate(components):
        if idx in processed or comp["shape"] != shape:
            continue
        rr0, cc0, rr1, cc1 = comp["bbox"]
        if dc != 0:
            if (rr0, rr1) != (r0, r1):
                continue
            if dc == 1 and cc0 < c0:
                continue
            if dc == -1 and cc0 > c0:
                continue
            queue.append(idx)
        else:
            if (cc0, cc1) != (c0, c1):
                continue
            if dr == 1 and rr0 < r0:
                continue
            if dr == -1 and rr0 > r0:
                continue
            queue.append(idx)
    if dc == 1:
        queue.sort(key=lambda idx: components[idx]["anchor"][1])
    elif dc == -1:
        queue.sort(key=lambda idx: components[idx]["anchor"][1], reverse=True)
    elif dr == 1:
        queue.sort(key=lambda idx: components[idx]["anchor"][0])
    else:
        queue.sort(key=lambda idx: components[idx]["anchor"][0], reverse=True)
    assert queue and queue[0] == starter_id
    return queue


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay each launched queue forward by filling the gaps between same-lane objects until the last object reaches the edge.

    Input structure:
    - The background is black.
    - Maroon marker blocks sit at one end of a horizontal or vertical lane and touch the first colored object head-on.
    - Each lane contains one or more separated objects of the same shape, aligned in the same row band or column band.

    Rule:
    - For every maroon marker, find the touching starter object and the direction pointing away from the marker.
    - Collect the full queue of same-shape objects in that lane, ordered along the launch direction.
    - Move each object straight ahead until it becomes adjacent to the next object's original position; move the final object straight to the grid edge; keep the maroon markers unchanged.

    Key insight:
    - The collision chain preserves object order: every object only consumes the empty gap in front of it, while the last object consumes the final gap to the boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    object_components = _find_components(g, [color for color in np.unique(g) if color not in (bg, MARKER)])
    marker_components = _find_components(g, [MARKER])
    object_index = _component_index_grid(g.shape, object_components)

    out = g.copy()
    processed: set[int] = set()
    placements: list[tuple[int, np.ndarray]] = []

    for marker in marker_components:
        launch = _marker_launch(marker, object_index)
        if launch is None:
            continue
        starter_id, direction = launch
        if starter_id in processed:
            continue
        queue = _lane_queue(starter_id, direction, object_components, processed)
        for obj_id in queue:
            coords = object_components[obj_id]["coords"]
            out[coords[:, 0], coords[:, 1]] = bg
        for i, obj_id in enumerate(queue):
            comp = object_components[obj_id]
            coords = comp["coords"]
            if i + 1 < len(queue):
                target_anchor = object_components[queue[i + 1]]["anchor"]
                source_anchor = comp["anchor"]
                extent = _axis_extent(comp, direction)
                offset = np.array(
                    [
                        target_anchor[0] - source_anchor[0] - direction[0] * extent,
                        target_anchor[1] - source_anchor[1] - direction[1] * extent,
                    ],
                    dtype=int,
                )
            else:
                steps = _steps_to_edge(coords, g.shape, direction)
                offset = np.array(direction, dtype=int) * steps
            placements.append((int(comp["color"]), coords + offset))
            processed.add(obj_id)

    for color, coords in placements:
        assert np.all(out[coords[:, 0], coords[:, 1]] == bg), "translated lane objects must not collide"
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _rect_cells(top: int, left: int, height: int, width: int) -> np.ndarray:
    rows = np.repeat(np.arange(top, top + height, dtype=int), width)
    cols = np.tile(np.arange(left, left + width, dtype=int), height)
    return np.column_stack((rows, cols))


def _reserve_ok(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_horizontal_lane(grid: np.ndarray, reserved: np.ndarray, direction: int) -> bool:
    h, w = grid.shape
    thickness = random.randint(*THICKNESS_RANGE)
    obj_w = random.randint(*LENGTH_RANGE)
    marker_w = random.randint(*MARKER_DEPTH_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    used = marker_w + count * obj_w + sum(gaps)
    if used + 1 > w:
        return False
    row = random.randint(0, h - thickness)
    if direction == 1:
        marker_left_max = w - used - 1
        marker_left = random.randint(0, marker_left_max)
        marker_box = (row, marker_left, row + thickness - 1, marker_left + marker_w - 1)
        protected_box = (row, marker_left, row + thickness - 1, w - 1)
        if not _reserve_ok(reserved, *protected_box):
            return False
        colors = random.sample(OBJECT_COLORS, count)
        grid[marker_box[0] : marker_box[2] + 1, marker_box[1] : marker_box[3] + 1] = MARKER
        col = marker_left + marker_w
        for idx in range(count):
            grid[row : row + thickness, col : col + obj_w] = colors[idx]
            col += obj_w
            if idx < len(gaps):
                col += gaps[idx]
        _reserve(reserved, *protected_box)
        return True

    marker_right_min = used
    marker_right = random.randint(marker_right_min, w - 1)
    marker_box = (row, marker_right - marker_w + 1, row + thickness - 1, marker_right)
    protected_box = (row, 0, row + thickness - 1, marker_right)
    if not _reserve_ok(reserved, *protected_box):
        return False
    colors = random.sample(OBJECT_COLORS, count)
    grid[marker_box[0] : marker_box[2] + 1, marker_box[1] : marker_box[3] + 1] = MARKER
    right_edge = marker_box[1] - 1
    for idx in range(count):
        left = right_edge - obj_w + 1
        grid[row : row + thickness, left : right_edge + 1] = colors[idx]
        right_edge = left - 1
        if idx < len(gaps):
            right_edge -= gaps[idx]
    _reserve(reserved, *protected_box)
    return True


def _place_vertical_lane(grid: np.ndarray, reserved: np.ndarray, direction: int) -> bool:
    h, w = grid.shape
    thickness = random.randint(*THICKNESS_RANGE)
    obj_h = random.randint(*LENGTH_RANGE)
    marker_h = random.randint(*MARKER_DEPTH_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    used = marker_h + count * obj_h + sum(gaps)
    if used + 1 > h:
        return False
    col = random.randint(0, w - thickness)
    if direction == 1:
        marker_top_max = h - used - 1
        marker_top = random.randint(0, marker_top_max)
        marker_box = (marker_top, col, marker_top + marker_h - 1, col + thickness - 1)
        protected_box = (marker_top, col, h - 1, col + thickness - 1)
        if not _reserve_ok(reserved, *protected_box):
            return False
        colors = random.sample(OBJECT_COLORS, count)
        grid[marker_box[0] : marker_box[2] + 1, marker_box[1] : marker_box[3] + 1] = MARKER
        row = marker_top + marker_h
        for idx in range(count):
            grid[row : row + obj_h, col : col + thickness] = colors[idx]
            row += obj_h
            if idx < len(gaps):
                row += gaps[idx]
        _reserve(reserved, *protected_box)
        return True

    marker_bottom_min = used
    marker_bottom = random.randint(marker_bottom_min, h - 1)
    marker_box = (marker_bottom - marker_h + 1, col, marker_bottom, col + thickness - 1)
    protected_box = (0, col, marker_bottom, col + thickness - 1)
    if not _reserve_ok(reserved, *protected_box):
        return False
    colors = random.sample(OBJECT_COLORS, count)
    grid[marker_box[0] : marker_box[2] + 1, marker_box[1] : marker_box[3] + 1] = MARKER
    bottom_edge = marker_box[0] - 1
    for idx in range(count):
        top = bottom_edge - obj_h + 1
        grid[top : bottom_edge + 1, col : col + thickness] = colors[idx]
        bottom_edge = top - 1
        if idx < len(gaps):
            bottom_edge -= gaps[idx]
    _reserve(reserved, *protected_box)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    lane_count = random.choice(LANE_COUNT_CHOICES)
    orientations = [random.choice(["h", "v"]) for _ in range(lane_count)]
    if lane_count >= 2 and all(ori == orientations[0] for ori in orientations):
        orientations[random.randrange(lane_count)] = "v" if orientations[0] == "h" else "h"

    for ori in orientations:
        placed = False
        for _ in range(LANE_PLACEMENT_TRIES):
            direction = random.choice([-1, 1])
            if ori == "h":
                placed = _place_horizontal_lane(grid, reserved, direction)
            else:
                placed = _place_vertical_lane(grid, reserved, direction)
            if placed:
                break
        if not placed:
            return None

    moved = main(grid)
    if np.array_equal(moved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
