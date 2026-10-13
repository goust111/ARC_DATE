from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BACKGROUND = 0
GREY = 5
INPUT_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
)
WIDE_SHAPES = tuple(shape for shape in SHAPES if shape.shape[1] >= 2 and shape.shape[0] <= 3)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            objects.append({"color": color, "coords": arr, "bbox": bbox, "area": int(len(arr))})
    return objects


def _frequency_order(grid: np.ndarray, objects: list[dict[str, object]]) -> list[int]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    order_index = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == BACKGROUND:
                continue
            counts[color] += 1
            if color not in first_seen:
                first_seen[color] = order_index
                order_index += 1
    colors = {int(obj["color"]) for obj in objects}
    return sorted(colors, key=lambda color: (-counts[color], first_seen[color]))


def _drop_object(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    h = occupied.shape[0]
    shift = 0
    while True:
        next_rows = coords[:, 0] + shift + 1
        cols = coords[:, 1]
        if np.any(next_rows >= h) or np.any(occupied[next_rows, cols]):
            break
        shift += 1
    return np.column_stack((coords[:, 0] + shift, coords[:, 1]))


def _solve(grid: np.ndarray, color_order: list[int]) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _extract_objects(g)
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=int)
    placed: list[dict[str, object]] = []

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        by_color[int(obj["color"])].append(obj)

    for color in color_order:
        same_color = by_color[color]
        same_color.sort(key=lambda obj: (-int(obj["bbox"][2]), int(obj["bbox"][0]), int(obj["bbox"][1])))
        for obj in same_color:
            final_coords = _drop_object(np.asarray(obj["coords"], dtype=int), occupied)
            reached_bottom = bool(final_coords[:, 0].max() == h - 1)
            final_color = color if reached_bottom else GREY
            out[final_coords[:, 0], final_coords[:, 1]] = final_color
            occupied[final_coords[:, 0], final_coords[:, 1]] = True
            placed.append(
                {
                    "color": color,
                    "input_bbox": obj["bbox"],
                    "final_coords": final_coords,
                    "reached_bottom": reached_bottom,
                }
            )
    return out, placed


def _bottom_priority_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _extract_objects(g)
    occupied = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(objects, key=lambda obj: (-int(obj["bbox"][2]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    for obj in ordered:
        color = int(obj["color"])
        final_coords = _drop_object(np.asarray(obj["coords"], dtype=int), occupied)
        reached_bottom = bool(final_coords[:, 0].max() == h - 1)
        out[final_coords[:, 0], final_coords[:, 1]] = color if reached_bottom else GREY
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop every monochrome object downward, resolving collisions by global color frequency and greying any object that does not reach the floor.

    Input structure:
    - The grid has a black background with 4-6 monochrome 4-connected foreground objects.
    - Objects use 3-4 colors chosen from blue, red, green, and yellow, and some colors may label multiple separate objects.
    - Objects start in the upper or middle rows with empty space below, and at least some differently colored objects share column projections.

    Rule:
    - Count the total number of pixels of each non-black color in the input.
    - Process colors from highest total frequency to lowest; within one color, process that color's objects from lower original bottom row to higher.
    - For each object, clear the original foreground and slide that object's shape straight downward until its next step would cross the last row or overlap an already fixed object.
    - After every object is fixed, keep the original color for objects whose final bottommost pixel is on the last row; recolor every other settled object to grey.

    Key insight:
    - Collision priority is controlled by global color frequency, not by which object started lower, so a higher-frequency object can claim the floor first and leave a lower-frequency rival suspended above it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"
    color_order = _frequency_order(g, objects)
    out, _ = _solve(g, color_order)
    return out


def _shape_area(shape: np.ndarray) -> int:
    return int(shape.sum())


def _mask_coords(shape: np.ndarray) -> np.ndarray:
    return np.argwhere(shape != 0)


def _paint(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    coords = _mask_coords(shape)
    grid[top + coords[:, 0], left + coords[:, 1]] = color


def _can_place(grid: np.ndarray, shape: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    for dr, dc in _mask_coords(shape):
        rr = top + int(dr)
        cc = left + int(dc)
        if grid[rr, cc] != BACKGROUND:
            return False
        for nr in range(rr - gap, rr + gap + 1):
            for nc in range(cc - gap, cc + gap + 1):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BACKGROUND:
                    return False
    return True


def _column_overlap(obj_a: dict[str, object], obj_b: dict[str, object]) -> bool:
    cols_a = set(np.asarray(obj_a["coords"], dtype=int)[:, 1].tolist())
    cols_b = set(np.asarray(obj_b["coords"], dtype=int)[:, 1].tolist())
    return bool(cols_a & cols_b)


def _build_color_multiset(chosen_colors: list[int], object_count: int) -> list[int] | None:
    if len(chosen_colors) == 3:
        counts = [2, 1, 1]
    else:
        counts = [2, 1, 1, 1]
    remaining = object_count - sum(counts)
    if remaining < 0:
        return None
    for _ in range(remaining):
        counts[random.randrange(len(counts))] += 1
    color_list: list[int] = []
    for color, count in zip(chosen_colors, counts):
        color_list.extend([color] * count)
    return color_list


def _valid_color_totals(color_list: list[int], shapes: list[np.ndarray], dominant_color: int) -> bool:
    totals: dict[int, int] = defaultdict(int)
    for color, shape in zip(color_list, shapes):
        totals[color] += _shape_area(shape)
    return len(set(totals.values())) == len(totals) and max(totals, key=totals.get) == dominant_color


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(11, 14)
    width = random.randint(9, 12)
    color_count = random.randint(3, 4)
    chosen_colors = random.sample(list(INPUT_COLORS), color_count)
    dominant_color = chosen_colors[0]
    blocked_color = chosen_colors[1]
    object_count = random.randint(4, 6)

    color_list = _build_color_multiset(chosen_colors, object_count)
    if color_list is None or color_list.count(dominant_color) < 2:
        return None

    shapes = [random.choice(SHAPES).copy() for _ in color_list]
    dominant_indices = [i for i, color in enumerate(color_list) if color == dominant_color]
    blocked_indices = [i for i, color in enumerate(color_list) if color == blocked_color]
    dominant_idx = dominant_indices[0]
    blocked_idx = blocked_indices[0]
    shapes[dominant_idx] = random.choice(WIDE_SHAPES).copy()
    shapes[blocked_idx] = random.choice(WIDE_SHAPES).copy()
    if not _valid_color_totals(color_list, shapes, dominant_color):
        return None

    grid = np.zeros((height, width), dtype=int)
    dominant_shape = shapes[dominant_idx]
    blocked_shape = shapes[blocked_idx]

    dominant_left = random.randint(1, width - max(dominant_shape.shape[1], blocked_shape.shape[1]) - 1)
    blocked_left = dominant_left + random.randint(-(blocked_shape.shape[1] - 1), dominant_shape.shape[1] - 1)
    blocked_left = max(0, min(width - blocked_shape.shape[1], blocked_left))
    overlap = max(
        0,
        min(dominant_left + dominant_shape.shape[1], blocked_left + blocked_shape.shape[1])
        - max(dominant_left, blocked_left),
    )
    if overlap < 1:
        return None

    dominant_top = random.randint(0, max(0, height // 3 - 1))
    min_blocked_top = dominant_top + dominant_shape.shape[0] + 1
    max_blocked_top = min(height // 2, height - blocked_shape.shape[0] - 3)
    if min_blocked_top > max_blocked_top:
        return None
    blocked_top = random.randint(min_blocked_top, max_blocked_top)

    if not _can_place(grid, dominant_shape, dominant_top, dominant_left):
        return None
    _paint(grid, dominant_shape, dominant_top, dominant_left, dominant_color)
    if not _can_place(grid, blocked_shape, blocked_top, blocked_left):
        return None
    _paint(grid, blocked_shape, blocked_top, blocked_left, blocked_color)

    used_indices = {dominant_idx, blocked_idx}
    remaining_indices = list(range(len(color_list)))
    random.shuffle(remaining_indices)
    for idx in remaining_indices:
        if idx in used_indices:
            continue
        placed = False
        shape = shapes[idx]
        for _ in range(80):
            top = random.randint(0, max(0, height - shape.shape[0] - 3))
            left = random.randint(0, width - shape.shape[1])
            if _can_place(grid, shape, top, left):
                _paint(grid, shape, top, left, color_list[idx])
                placed = True
                break
        if not placed:
            return None

    objects = _extract_objects(grid)
    if len(objects) != len(color_list):
        return None
    if len({int(obj["color"]) for obj in objects}) != color_count:
        return None
    if max(int(obj["bbox"][2]) for obj in objects) >= height - 2:
        return None

    color_totals = Counter(map(int, grid[grid != BACKGROUND].tolist()))
    if len(set(color_totals.values())) != len(color_totals):
        return None

    out, placed = _solve(grid, _frequency_order(grid, objects))
    if np.array_equal(out, grid) or np.array_equal(out, _bottom_priority_output(grid)):
        return None

    reached_bottom = sum(int(item["reached_bottom"]) for item in placed)
    if reached_bottom == 0 or reached_bottom == len(placed):
        return None
    if not np.any(out == GREY):
        return None

    overlap_pairs = 0
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if int(obj_a["color"]) != int(obj_b["color"]) and _column_overlap(obj_a, obj_b):
                overlap_pairs += 1
    if overlap_pairs == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
