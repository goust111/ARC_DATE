from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
OBJECT_COUNT_CHOICES = (2, 3, 4)
COMPONENT_COUNT_CHOICES = (1, 2, 3, 4)
CANDIDATES_PER_COUNT = 5
SLOT_SIZE_MIN = 1
SLOT_SIZE_MAX = 4
SLOT_GAP = 1
MAX_SCALED_DIM = 28
GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a nonempty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _scan_order_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid.ravel():
        color = int(value)
        if color == BACKGROUND or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _count_components(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            components += 1
            queue = deque([(r, c)])
            seen[r, c] = True

            while queue:
                x, y = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        queue.append((nx, ny))

    return components


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = _scan_order_colors(g)
    assert colors, "expected at least one foreground color"

    objects: list[dict[str, object]] = []
    for color in colors:
        mask = g == color
        box = _bbox(mask)
        assert box is not None, "foreground color must be present"
        r0, c0, r1, c1 = box
        local_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        component_count = _count_components(local_mask)
        assert component_count >= 1

        scaled_mask = _scale_mask(local_mask, component_count)
        scaled_h, scaled_w = map(int, scaled_mask.shape)

        objects.append(
            {
                "color": color,
                "bbox_r0": r0,
                "bbox_c0": c0,
                "bbox_h": int(local_mask.shape[0]),
                "bbox_w": int(local_mask.shape[1]),
                "bbox_area": int(local_mask.shape[0] * local_mask.shape[1]),
                "pixel_count": int(np.count_nonzero(local_mask)),
                "component_count": component_count,
                "mask": local_mask,
                "scaled_mask": scaled_mask,
                "scaled_h": scaled_h,
                "scaled_w": scaled_w,
                "scaled_bbox_area": int(scaled_h * scaled_w),
            }
        )

    return objects


def _sorted_objects(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        objects,
        key=lambda obj: (
            -int(obj["scaled_bbox_area"]),
            -int(obj["scaled_h"]),
            -int(obj["scaled_w"]),
            int(obj.get("bbox_r0", 0)),
            int(obj.get("bbox_c0", 0)),
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each monochrome object by its number of 4-connected components, then
    center-stack the scaled masks from largest scaled bbox area to smallest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _sorted_objects(_extract_objects(g))
    out_h = max(int(obj["scaled_h"]) for obj in objects)
    out_w = max(int(obj["scaled_w"]) for obj in objects)

    out = np.full((out_h, out_w), BACKGROUND, dtype=int)
    for obj in objects:
        scaled_mask = np.asarray(obj["scaled_mask"], dtype=bool)
        scaled_h = int(obj["scaled_h"])
        scaled_w = int(obj["scaled_w"])
        r0 = (out_h - scaled_h) // 2
        c0 = (out_w - scaled_w) // 2
        out[r0 : r0 + scaled_h, c0 : c0 + scaled_w][scaled_mask] = int(obj["color"])

    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    area_min = 1 if height * width <= 2 else 2
    target_area = int(_RNG.integers(area_min, height * width + 1))

    start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
    cells = {start}
    frontier = [start]

    while len(cells) < target_area:
        expandable: list[tuple[int, int, list[tuple[int, int]]]] = []
        for x, y in frontier:
            neighbors: list[tuple[int, int]] = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in cells:
                    neighbors.append((nx, ny))
            if neighbors:
                expandable.append((x, y, neighbors))

        if not expandable:
            break

        _, _, neighbors = expandable[int(_RNG.integers(len(expandable)))]
        pick = neighbors[int(_RNG.integers(len(neighbors)))]
        cells.add(pick)
        frontier.append(pick)

    mask = np.zeros((height, width), dtype=bool)
    for x, y in cells:
        mask[x, y] = True
    return _crop(mask)


def _make_object_mask(component_count: int) -> np.ndarray:
    layouts: list[tuple[int, int]] = [(1, component_count), (component_count, 1)]
    if component_count >= 2:
        layouts.append(((component_count + 1) // 2, 2))

    rows, cols = layouts[int(_RNG.integers(len(layouts)))]

    slot_masks = [
        _random_connected_mask(
            int(_RNG.integers(SLOT_SIZE_MIN, SLOT_SIZE_MAX + 1)),
            int(_RNG.integers(SLOT_SIZE_MIN, SLOT_SIZE_MAX + 1)),
        )
        for _ in range(component_count)
    ]

    slot_ids = _RNG.permutation(rows * cols)[:component_count]
    row_heights = [0] * rows
    col_widths = [0] * cols
    placements: list[tuple[np.ndarray, int, int]] = []

    for idx, slot_id in enumerate(slot_ids):
        rr, cc = divmod(int(slot_id), cols)
        mask = slot_masks[idx]
        h, w = map(int, mask.shape)
        row_heights[rr] = max(row_heights[rr], h)
        col_widths[cc] = max(col_widths[cc], w)
        placements.append((mask, rr, cc))

    canvas_h = sum(row_heights) + SLOT_GAP * (rows - 1)
    canvas_w = sum(col_widths) + SLOT_GAP * (cols - 1)
    canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

    row_starts: list[int] = []
    col_starts: list[int] = []

    cursor = 0
    for height in row_heights:
        row_starts.append(cursor)
        cursor += height + SLOT_GAP

    cursor = 0
    for width in col_widths:
        col_starts.append(cursor)
        cursor += width + SLOT_GAP

    for mask, rr, cc in placements:
        h, w = map(int, mask.shape)
        r0 = row_starts[rr] + int(_RNG.integers(row_heights[rr] - h + 1))
        c0 = col_starts[cc] + int(_RNG.integers(col_widths[cc] - w + 1))
        canvas[r0 : r0 + h, c0 : c0 + w] |= mask

    canvas = _crop(canvas)
    assert _count_components(canvas) == component_count
    return canvas


def _has_bbox_area_inversion(objects: list[dict[str, object]]) -> bool:
    for i, larger in enumerate(objects):
        for smaller in objects[i + 1 :]:
            if int(larger["bbox_area"]) < int(smaller["bbox_area"]):
                return True
    return False


def _sample_object_set() -> list[dict[str, object]] | None:
    object_count = int(_RNG.choice(np.array(OBJECT_COUNT_CHOICES)))
    component_counts = _RNG.choice(
        np.array(COMPONENT_COUNT_CHOICES), size=object_count, replace=False
    )

    candidate_lists: list[list[dict[str, object]]] = []
    for component_count in component_counts:
        candidates: list[dict[str, object]] = []
        for _ in range(CANDIDATES_PER_COUNT):
            mask = _make_object_mask(int(component_count))
            h, w = map(int, mask.shape)
            scaled_h = h * int(component_count)
            scaled_w = w * int(component_count)
            candidates.append(
                {
                    "component_count": int(component_count),
                    "mask": mask,
                    "bbox_h": h,
                    "bbox_w": w,
                    "bbox_area": int(h * w),
                    "pixel_count": int(np.count_nonzero(mask)),
                    "scaled_h": scaled_h,
                    "scaled_w": scaled_w,
                    "scaled_bbox_area": int(scaled_h * scaled_w),
                }
            )
        candidate_lists.append(candidates)

    for combo in itertools.product(*candidate_lists):
        objects = _sorted_objects(list(combo))

        scaled_areas = [int(obj["scaled_bbox_area"]) for obj in objects]
        if len(set(scaled_areas)) != len(objects):
            continue

        pixel_counts = [int(obj["pixel_count"]) for obj in objects]
        if len(set(pixel_counts)) != len(objects):
            continue

        max_scaled_h = int(objects[0]["scaled_h"])
        max_scaled_w = int(objects[0]["scaled_w"])
        if max_scaled_h > MAX_SCALED_DIM or max_scaled_w > MAX_SCALED_DIM:
            continue

        valid_chain = True
        for idx, obj in enumerate(objects):
            scaled_h = int(obj["scaled_h"])
            scaled_w = int(obj["scaled_w"])
            if scaled_h > max_scaled_h or scaled_w > max_scaled_w:
                valid_chain = False
                break
            if (max_scaled_h - scaled_h) % 2 != 0 or (max_scaled_w - scaled_w) % 2 != 0:
                valid_chain = False
                break
            if idx > 0:
                prev = objects[idx - 1]
                if int(prev["scaled_h"]) < scaled_h or int(prev["scaled_w"]) < scaled_w:
                    valid_chain = False
                    break
        if not valid_chain:
            continue

        if not _has_bbox_area_inversion(objects):
            continue

        return objects

    return None


def _place_objects(objects: list[dict[str, object]]) -> np.ndarray | None:
    colors = list(_RNG.choice(np.array(FOREGROUND_COLORS), size=len(objects), replace=False))
    order = _RNG.permutation(len(objects))

    if len(objects) == 2 and _RNG.random() < 0.5:
        groups = [list(order)]
    elif len(objects) == 1:
        groups = [list(order)]
    else:
        cut = int(_RNG.integers(1, len(objects)))
        groups = [list(order[:cut]), list(order[cut:])]
        groups = [group for group in groups if group]

    left = int(_RNG.integers(1, 4))
    right = int(_RNG.integers(1, 4))
    top = int(_RNG.integers(1, 4))
    bottom = int(_RNG.integers(1, 4))
    row_gap = int(_RNG.integers(1, 4))

    row_heights: list[int] = []
    row_widths: list[int] = []
    row_internal_gaps: list[list[int]] = []

    for group in groups:
        row_heights.append(max(int(objects[idx]["bbox_h"]) for idx in group))
        row_internal_gaps.append(
            [int(_RNG.integers(1, 4)) for _ in range(max(0, len(group) - 1))]
        )
        row_widths.append(
            sum(int(objects[idx]["bbox_w"]) for idx in group) + sum(row_internal_gaps[-1])
        )

    grid_h = top + sum(row_heights) + row_gap * (len(groups) - 1) + bottom
    grid_w = left + max(row_widths) + right
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    y = top
    for row_index, group in enumerate(groups):
        slack = grid_w - left - right - row_widths[row_index]
        x = left + int(_RNG.integers(slack + 1))
        for idx_in_group, obj_index in enumerate(group):
            obj = objects[obj_index]
            mask = np.asarray(obj["mask"], dtype=bool)
            h = int(obj["bbox_h"])
            w = int(obj["bbox_w"])
            row_offset = int(_RNG.integers(row_heights[row_index] - h + 1))
            r0 = y + row_offset
            c0 = x
            patch = grid[r0 : r0 + h, c0 : c0 + w]
            if np.any(patch[mask] != BACKGROUND):
                return None
            patch[mask] = int(colors[obj_index])
            if idx_in_group < len(group) - 1:
                x += w + row_internal_gaps[row_index][idx_in_group]
        y += row_heights[row_index] + row_gap

    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    objects = _sorted_objects(_extract_objects(grid))
    if not (2 <= len(objects) <= 4):
        return False

    pixel_counts = [int(obj["pixel_count"]) for obj in objects]
    if len(set(pixel_counts)) != len(pixel_counts):
        return False

    scaled_areas = [int(obj["scaled_bbox_area"]) for obj in objects]
    if len(set(scaled_areas)) != len(objects):
        return False

    max_scaled_h = int(objects[0]["scaled_h"])
    max_scaled_w = int(objects[0]["scaled_w"])
    for obj in objects:
        scaled_h = int(obj["scaled_h"])
        scaled_w = int(obj["scaled_w"])
        if scaled_h > max_scaled_h or scaled_w > max_scaled_w:
            return False
        if (max_scaled_h - scaled_h) % 2 != 0 or (max_scaled_w - scaled_w) % 2 != 0:
            return False

    if not _has_bbox_area_inversion(objects):
        return False

    main(grid)
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        objects = _sample_object_set()
        if objects is None:
            continue
        grid = _place_objects(objects)
        if grid is None:
            continue
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
