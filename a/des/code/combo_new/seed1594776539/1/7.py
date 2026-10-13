from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(
                {
                    "coords": coords,
                    "bbox": (min_r, min_c, max_r, max_c),
                }
            )
    return comps


def _component_patch(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _axis_direction(patch: np.ndarray) -> str:
    vertical = np.array_equal(patch, np.fliplr(patch))
    horizontal = np.array_equal(patch, np.flipud(patch))
    assert vertical ^ horizontal, "each object must have exactly one symmetry axis"
    return "vertical" if vertical else "horizontal"


def _nearest_side(bbox: tuple[int, int, int, int], shape: tuple[int, int], axis: str) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, shape)

    if axis == "vertical":
        center_sum = c0 + c1
        assert center_sum != w - 1, "vertical object must be strictly closer to one side"
        return "left" if center_sum < w - 1 else "right"

    center_sum = r0 + r1
    assert center_sum != h - 1, "horizontal object must be strictly closer to one side"
    return "top" if center_sum < h - 1 else "bottom"


def _half_mask(local_shape: tuple[int, int], axis: str, side: str) -> np.ndarray:
    h, w = map(int, local_shape)
    rows, cols = np.indices((h, w))

    if axis == "vertical":
        if side == "left":
            return 2 * cols <= w - 1
        assert side == "right"
        return 2 * cols >= w - 1

    if side == "top":
        return 2 * rows <= h - 1
    assert side == "bottom"
    return 2 * rows >= h - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each symmetric object along its own axis and keep the half nearer the relevant grid edge.

    Input structure:
    - The background is black.
    - The grid contains 2-4 separated colored 4-connected objects.
    - Each object has exactly one symmetry axis: either vertical or horizontal, but not both.

    Rule:
    - For each object, crop it to its own bounding box and detect whether it is vertically or horizontally symmetric.
    - If the axis is vertical, compare the bounding-box center to the left and right grid edges; if the axis is horizontal, compare it to the top and bottom edges.
    - Keep only the half on the side closer to that relevant edge, plus any pixels lying on the symmetry axis itself, and clear the opposite half to black.

    Key insight:
    - The transformation combines two independent cues: first infer each object's unique symmetry axis, then use the object's position in the whole grid to decide which symmetric half survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for comp in _find_components(g):
        patch = _component_patch(g, comp)
        axis = _axis_direction(patch)
        side = _nearest_side(comp["bbox"], g.shape, axis)
        keep = _half_mask(patch.shape, axis, side) & (patch != BG)

        r0, c0, _r1, _c1 = comp["bbox"]
        rr, cc = np.nonzero(keep)
        out[r0 + rr, c0 + cc] = patch[rr, cc]

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 120
GRID_MIN = 14
GRID_MAX = 25
COLOR_POOL = tuple(range(1, 10))
OBJECT_COUNT_CHOICES = (2, 3, 4)
AXES = ("vertical", "horizontal")
SIZE_BY_COUNT = {
    2: ((4, 8), (4, 9)),
    3: ((4, 7), (4, 8)),
    4: ((3, 6), (4, 7)),
}
AREA_FRACTION_RANGE = (0.35, 0.68)
EARLY_STOP_PROB = 0.25


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size > 0 and cols.size > 0
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = map(int, mask.shape)
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _add_vertical_orbit(mask: np.ndarray, r: int, c: int) -> None:
    w = int(mask.shape[1])
    mask[r, c] = True
    mask[r, w - 1 - c] = True


def _orbit_cells(width: int, r: int, c: int) -> list[tuple[int, int]]:
    left = (r, c)
    right = (r, width - 1 - c)
    if left == right:
        return [left]
    return [left, right]


def _touches_mask(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, mask.shape)
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _make_vertical_mask(height_range: tuple[int, int], width_range: tuple[int, int]) -> np.ndarray | None:
    h = random.randint(*height_range)
    w = random.randint(*width_range)
    canon_w = (w + 1) // 2

    mask = np.zeros((h, w), dtype=bool)
    seed_r = random.randrange(h)
    seed_c = w // 2 if w % 2 == 1 else w // 2 - 1
    _add_vertical_orbit(mask, seed_r, seed_c)

    min_area = max(6, int(np.ceil(AREA_FRACTION_RANGE[0] * h * w)))
    max_area = max(min_area, int(np.floor(AREA_FRACTION_RANGE[1] * h * w)))
    target_area = random.randint(min_area, max_area)

    for _ in range(h * w * 4):
        current_area = int(mask.sum())
        if current_area >= target_area and current_area >= min_area and random.random() < EARLY_STOP_PROB:
            break

        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(canon_w):
                if mask[r, c]:
                    continue
                cells = _orbit_cells(w, r, c)
                if _touches_mask(mask, cells):
                    candidates.append((r, c))

        if not candidates:
            break

        r, c = random.choice(candidates)
        _add_vertical_orbit(mask, r, c)

    cropped = _crop_bool(mask)
    if cropped.shape[0] < 3 or cropped.shape[1] < 3:
        return None
    if int(cropped.sum()) < 6:
        return None
    if not _is_connected(cropped):
        return None
    if not np.array_equal(cropped, np.fliplr(cropped)):
        return None
    if np.array_equal(cropped, np.flipud(cropped)):
        return None
    return cropped


def _make_object_mask(axis: str, n_objects: int) -> np.ndarray | None:
    height_range, width_range = SIZE_BY_COUNT[n_objects]
    for _ in range(OBJECT_ATTEMPTS):
        vertical_mask = _make_vertical_mask(height_range, width_range)
        if vertical_mask is None:
            continue
        if axis == "vertical":
            return vertical_mask
        transposed = vertical_mask.T
        if np.array_equal(transposed, np.flipud(transposed)) and not np.array_equal(transposed, np.fliplr(transposed)):
            return transposed
    return None


def _side_choices(axis: str) -> tuple[str, str]:
    return ("left", "right") if axis == "vertical" else ("top", "bottom")


def _candidate_origins(mask: np.ndarray, grid_shape: tuple[int, int], axis: str, side: str) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    gh, gw = map(int, grid_shape)
    origins: list[tuple[int, int]] = []

    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            if axis == "vertical":
                center_sum = 2 * c0 + w - 1
                if side == "left" and center_sum >= gw - 1:
                    continue
                if side == "right" and center_sum <= gw - 1:
                    continue
            else:
                center_sum = 2 * r0 + h - 1
                if side == "top" and center_sum >= gh - 1:
                    continue
                if side == "bottom" and center_sum <= gh - 1:
                    continue
            origins.append((r0, c0))

    random.shuffle(origins)
    return origins


def _grid_bounds(objects: list[dict]) -> tuple[int, int] | None:
    heights = sorted((int(obj["mask"].shape[0]) for obj in objects), reverse=True)
    widths = sorted((int(obj["mask"].shape[1]) for obj in objects), reverse=True)

    min_h = max(GRID_MIN, sum(heights[: min(2, len(heights))]) + 6)
    min_w = max(GRID_MIN, sum(widths[: min(2, len(widths))]) + 6)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _place_objects(objects: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    gh, gw = map(int, shape)
    candidates = [
        _candidate_origins(obj["mask"], shape, obj["axis"], obj["side"])
        for obj in objects
    ]
    if any(not origins for origins in candidates):
        return None

    order = sorted(
        range(len(objects)),
        key=lambda i: (len(candidates[i]), -int(objects[i]["mask"].sum())),
    )
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    reserved = np.zeros((gh, gw), dtype=bool)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        obj_idx = order[pos]
        obj = objects[obj_idx]
        h, w = map(int, obj["mask"].shape)
        for r0, c0 in candidates[obj_idx]:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(gh, r0 + h + 1)
            cc1 = min(gw, c0 + w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            snapshot = reserved.copy()
            reserved[rr0:rr1, cc0:cc1] = True
            placements[obj_idx] = (r0, c0)
            if backtrack(pos + 1):
                return True
            reserved[:, :] = snapshot
            placements[obj_idx] = None

        return False

    if not backtrack(0):
        return None

    grid = np.zeros((gh, gw), dtype=int)
    for obj, placement in zip(objects, placements):
        assert placement is not None
        r0, c0 = placement
        mask = obj["mask"]
        h, w = map(int, mask.shape)
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = int(obj["color"])
        region = grid[r0 : r0 + h, c0 : c0 + w]
        region[mask] = patch[mask]
    return grid


def _scene_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = _find_components(grid)
    if not 2 <= len(comps) <= 4:
        return False

    for comp in comps:
        patch = _component_patch(grid, comp)
        try:
            axis = _axis_direction(patch)
            side = _nearest_side(comp["bbox"], grid.shape, axis)
        except AssertionError:
            return False
        keep = _half_mask(patch.shape, axis, side) & (patch != BG)
        if not np.any(keep):
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape


def _generate_scene() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, n_objects)

    objects = []
    for color in colors:
        axis = random.choice(AXES)
        side = random.choice(_side_choices(axis))
        mask = _make_object_mask(axis, n_objects)
        if mask is None:
            return None
        objects.append(
            {
                "mask": mask,
                "axis": axis,
                "side": side,
                "color": color,
            }
        )

    shape = _grid_bounds(objects)
    if shape is None:
        return None

    grid = _place_objects(objects, shape)
    if grid is None or not _scene_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
