from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, is_contiguous as _gen_is_contiguous, paste as _gen_paste, random_free_location as _gen_random_free_location
import random as _gen_random


def _boundary_background(g: np.ndarray) -> int:
    boundary = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    colors, counts = np.unique(boundary, return_counts=True)
    assert colors.size > 0, "expected a nonempty grid boundary"

    max_count = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
    assert len(winners) == 1, "expected a unique boundary background color"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique smallest solid monochrome rectangle as the entire output grid.

    Input structure
    - The empty canvas color is the unique most common color on the grid boundary.
    - Non-background pixels form one or more 4-connected monochrome objects.
    - At least one non-background object is a filled axis-aligned rectangle.

    Rule
    - Find every monochrome non-background object whose pixels exactly fill its bounding box.
    - Select the unique filled rectangle with the smallest area.
    - Return a solid grid with the selected rectangle's height, width, and color.

    Key insight
    - Other objects are distractor rectangles or partial rectangles; the output is the
      smallest complete rectangular object, detached from its original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"

    bg = _boundary_background(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one non-background object"

    rectangles = []
    for obj in objects:
        box = obj["bbox"]
        height = bbox_h(box)
        width = bbox_w(box)
        if int(obj["area"]) == height * width:
            rectangles.append((int(obj["area"]), int(obj["color"]), height, width))

    assert rectangles, "expected at least one filled rectangle"
    min_area = min(area for area, _color, _height, _width in rectangles)
    smallest = [rect for rect in rectangles if rect[0] == min_area]
    assert len(smallest) == 1, "expected a unique smallest filled rectangle"

    _area, color, height, width = smallest[0]
    return np.full((height, width), color, dtype=int)

# --------------- generator (o5x) ---------------

# Canvas/object size bounds.
GEN_BACKGROUND = 0
GEN_GRID_MAX_DIM = 20
GEN_MIN_GRID_DIM_BY_OBJECT_COUNT = {2: 10, 3: 10, 4: 12, 5: 14}
GEN_SIZE_CAP_BY_OBJECT_COUNT = {2: 12, 3: 12, 4: 9, 5: 7}

# Counts, palette, and target bounds.
GEN_OBJECT_COUNT_RANGE = (2, 5)
GEN_RECT_NONRECT_COUNTS_BY_OBJECT_COUNT = {
    2: [(0, 1), (1, 0)],
    3: [(0, 2), (1, 1), (2, 0)],
    4: [(1, 2), (2, 1)],
    5: [(2, 2)],
}
GEN_COLOR_RANGE = (1, 9)
GEN_TARGET_HEIGHT_RANGE, GEN_TARGET_WIDTH_RANGE = (2, 3), (2, 4)

# Shape, placement, and retry filters.
GEN_DISTRACTOR_RECT_MIN_DIM, GEN_NONRECT_MIN_DIM = 2, 3
GEN_DISTRACTOR_AREA_RATIO_RANGE = (2, 7)
GEN_NONRECT_FILL_RANGE = (0.67, 0.95)
GEN_CONNECTIVITY = 4
GEN_TOUCHING_BORDER, GEN_INTERIOR_BORDER = 0, 1
GEN_EMBEDDED_NONRECT_PROB, GEN_ALLOW_BOUNDARY_TOUCH_PROB = 0.65, 0.7
GEN_RETRY_LIMIT = 300


def _gen_boundary_bg_ok(g: np.ndarray) -> bool:
    boundary = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    colors, counts = np.unique(boundary, return_counts=True)
    max_count = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
    return winners == [GEN_BACKGROUND]


def _gen_place_patch(canvas: np.ndarray, patch: np.ndarray, border: int) -> bool:
    try:
        r0, c0 = _gen_random_free_location(canvas, patch, transparent=GEN_BACKGROUND, border=border)
    except ValueError:
        return False
    _gen_paste(canvas, patch, r0, c0, transparent=GEN_BACKGROUND)
    return True


def _gen_rect_patch(height: int, width: int, color: int) -> np.ndarray:
    return np.full((int(height), int(width)), int(color), dtype=int)


def _gen_distractor_rect_patch(grid_h: int, grid_w: int, target_area: int, color: int, size_cap: int) -> np.ndarray | None:
    area_ratio_min, area_ratio_max = GEN_DISTRACTOR_AREA_RATIO_RANGE
    lo = area_ratio_min * int(target_area)
    hi = area_ratio_max * int(target_area)
    max_h, max_w = min(int(grid_h) - 1, int(size_cap)), min(int(grid_w) - 1, int(size_cap))
    candidates = [
        (h, w)
        for h in range(GEN_DISTRACTOR_RECT_MIN_DIM, max_h + 1)
        for w in range(GEN_DISTRACTOR_RECT_MIN_DIM, max_w + 1)
        if lo <= h * w <= hi
    ]
    if not candidates:
        return None
    h, w = _gen_random.choice(candidates)
    return _gen_rect_patch(h, w, color)


def _gen_notched_nonrect_patch(grid_h: int, grid_w: int, target_area: int, color: int, size_cap: int) -> np.ndarray | None:
    max_h, max_w = min(int(grid_h) - 1, int(size_cap)), min(int(grid_w) - 1, int(size_cap))
    fill_min, fill_max = GEN_NONRECT_FILL_RANGE
    candidates: list[tuple[int, int, str, int, int, int]] = []
    for h in range(GEN_NONRECT_MIN_DIM, max_h + 1):
        for w in range(GEN_NONRECT_MIN_DIM, max_w + 1):
            area = h * w
            for side in ("top", "bottom"):
                for length in range(1, w - 1):
                    for start in range(1, w - length):
                        for depth in range(1, h):
                            removed = length * depth
                            fill = (area - removed) / area
                            if fill_min <= fill <= fill_max and area - removed > target_area:
                                candidates.append((h, w, side, start, length, depth))
            for side in ("left", "right"):
                for length in range(1, h - 1):
                    for start in range(1, h - length):
                        for depth in range(1, w):
                            removed = length * depth
                            fill = (area - removed) / area
                            if fill_min <= fill <= fill_max and area - removed > target_area:
                                candidates.append((h, w, side, start, length, depth))

    if not candidates:
        return None

    h, w, side, start, length, depth = _gen_random.choice(candidates)
    patch = np.full((h, w), int(color), dtype=int)
    if side == "top":
        patch[:depth, start : start + length] = GEN_BACKGROUND
    elif side == "bottom":
        patch[h - depth :, start : start + length] = GEN_BACKGROUND
    elif side == "left":
        patch[start : start + length, :depth] = GEN_BACKGROUND
    else:
        patch[start : start + length, w - depth :] = GEN_BACKGROUND

    if not _gen_is_contiguous(patch, background=GEN_BACKGROUND, connectivity=GEN_CONNECTIVITY):
        return None
    return patch


def _gen_embedded_nonrect(
    canvas: np.ndarray, target_r: int, target_c: int, target_h: int, target_w: int, color: int, size_cap: int
) -> bool:
    grid_h, grid_w = canvas.shape
    target_area = int(target_h) * int(target_w)
    fill_min, fill_max = GEN_NONRECT_FILL_RANGE
    candidates: list[tuple[int, int, int, int]] = []
    for top in range(1, int(target_r) + 1):
        for bottom in range(1, int(grid_h) - int(target_r) - int(target_h) + 1):
            for left in range(1, int(target_c) + 1):
                for right in range(1, int(grid_w) - int(target_c) - int(target_w) + 1):
                    full_h = int(target_h) + top + bottom
                    full_w = int(target_w) + left + right
                    if full_h > int(size_cap) or full_w > int(size_cap):
                        continue
                    full_area = full_h * full_w
                    fill = (full_area - target_area) / full_area
                    if fill_min <= fill <= fill_max and full_area - target_area > target_area:
                        candidates.append((top, bottom, left, right))
    if not candidates:
        return False

    top, bottom, left, right = _gen_random.choice(candidates)
    full_h = int(target_h) + top + bottom
    full_w = int(target_w) + left + right
    patch = np.full((full_h, full_w), int(color), dtype=int)
    patch[top : top + int(target_h), left : left + int(target_w)] = GEN_BACKGROUND
    if not _gen_is_contiguous(patch, background=GEN_BACKGROUND, connectivity=GEN_CONNECTIVITY):
        return False

    r0 = int(target_r) - top
    c0 = int(target_c) - left
    if np.any((canvas[r0 : r0 + full_h, c0 : c0 + full_w] != GEN_BACKGROUND) & (patch != GEN_BACKGROUND)):
        return False
    _gen_paste(canvas, patch, r0, c0, transparent=GEN_BACKGROUND)
    return True


def _gen_sample_layout() -> tuple[int, int, int, int, int, int]:
    """Sample counts and canvas size; return object/rect/nonrect counts, H/W, size cap."""
    min_objects, max_objects = GEN_OBJECT_COUNT_RANGE
    object_count = _gen_random.randint(min_objects, max_objects)
    rect_count, nonrect_count = _gen_random.choice(
        GEN_RECT_NONRECT_COUNTS_BY_OBJECT_COUNT[object_count]
    )
    min_grid_dim = GEN_MIN_GRID_DIM_BY_OBJECT_COUNT[object_count]
    grid_h = _gen_random.randint(min_grid_dim, GEN_GRID_MAX_DIM)
    grid_w = _gen_random.randint(min_grid_dim, GEN_GRID_MAX_DIM)
    size_cap = GEN_SIZE_CAP_BY_OBJECT_COUNT[object_count]
    return object_count, rect_count, nonrect_count, grid_h, grid_w, size_cap


def _gen_build_distractor_patches(
    grid_h, grid_w, target_area, colors, color_i, rect_count, nonrect_count, size_cap
) -> list[np.ndarray] | None:
    """Build unplaced distractor patches in color order, or None if candidates fail."""
    patches: list[np.ndarray] = []
    for _ in range(rect_count):
        patch = _gen_distractor_rect_patch(grid_h, grid_w, target_area, colors[color_i], size_cap)
        if patch is None:
            return None
        patches.append(patch)
        color_i += 1
    for _ in range(nonrect_count):
        patch = _gen_notched_nonrect_patch(grid_h, grid_w, target_area, colors[color_i], size_cap)
        if patch is None:
            return None
        patches.append(patch)
        color_i += 1
    return patches


def _gen_place_distractor_patches(canvas: np.ndarray, patches: list[np.ndarray]) -> bool:
    """Shuffle and place each distractor, allowing boundary contact at the original rate."""
    _gen_random.shuffle(patches)
    for patch in patches:
        border = GEN_TOUCHING_BORDER if _gen_random.random() < GEN_ALLOW_BOUNDARY_TOUCH_PROB else GEN_INTERIOR_BORDER
        if not _gen_place_patch(canvas, patch, border):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count, rect_count, nonrect_count, grid_h, grid_w, size_cap = _gen_sample_layout()
    canvas = np.zeros((grid_h, grid_w), dtype=int)

    target_h = _gen_random.randint(*GEN_TARGET_HEIGHT_RANGE)
    target_w = _gen_random.randint(*GEN_TARGET_WIDTH_RANGE)
    target_area = target_h * target_w
    color_min, color_max = GEN_COLOR_RANGE
    colors = _gen_random.sample(range(color_min, color_max + 1), object_count)
    target_color = colors[0]

    target = _gen_rect_patch(target_h, target_w, target_color)
    try:
        target_r, target_c = _gen_random_free_location(canvas, target, transparent=GEN_BACKGROUND, border=GEN_INTERIOR_BORDER)
    except ValueError:
        return None
    _gen_paste(canvas, target, target_r, target_c, transparent=GEN_BACKGROUND)

    color_i = 1
    if nonrect_count and _gen_random.random() < GEN_EMBEDDED_NONRECT_PROB:
        if _gen_embedded_nonrect(canvas, target_r, target_c, target_h, target_w, colors[color_i], size_cap):
            nonrect_count -= 1
            color_i += 1

    patches = _gen_build_distractor_patches(grid_h, grid_w, target_area, colors, color_i, rect_count, nonrect_count, size_cap)
    if patches is None:
        return None
    if not _gen_place_distractor_patches(canvas, patches):
        return None

    if not _gen_boundary_bg_ok(canvas):
        return None

    expected = np.full((target_h, target_w), target_color, dtype=int)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
