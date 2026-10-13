from __future__ import annotations

import random

import numpy as np


def _color_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return (area, r0, c0, r1, c1, color) data except background."""
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    boxes: list[tuple[int, int, int, int, int, int]] = []
    for color in colors:
        rows, cols = np.where(g == color)
        assert rows.size > 0
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        boxes.append((area, r0, c0, r1, c1, color))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each overlapped solid rectangle and list its color from smallest area to largest.

    Input structure:
    - The background is black.
    - Several different colors each belong to one axis-aligned solid rectangle.
    - Rectangles may overlap, so only the topmost color is visible in shared cells.
    - Every rectangle keeps at least some visible pixels, and the visible row/column extremes match its true bounds.

    Rule:
    - For each non-black color, collect all visible pixels of that color.
    - Use the minimum and maximum visible row and column as that rectangle's full bounding box.
    - Compute each reconstructed rectangle's full area from its bounding-box height times width.
    - Sort the rectangle colors by increasing reconstructed area.
    - Output a single-column grid whose rows are those colors in sorted order.

    Key insight:
    - Occlusion changes which interior cells remain visible, but it does not change the true bounding-box extremes for any color, so the full rectangle area is recoverable from the visible pixels alone.
    """
    boxes = _color_bboxes(np.asarray(grid, dtype=int))
    boxes.sort(key=lambda item: item[:5])
    colors = [color for _area, _r0, _c0, _r1, _c1, color in boxes]
    return np.array(colors, dtype=int).reshape(-1, 1)


ARC_MAX = 30
GRID_MIN = 12
GRID_MAX = 24
RECT_COUNT_RANGE = (3, 5)
RECT_SIDE_RANGE = (3, 9)
GEN_RETRIES = 400
COLOR_POOL = tuple(range(1, 10))
PLACEMENT_MODE_WEIGHTS = (
    ("hub", 4),
    ("chain", 3),
    ("mixed", 3),
)


def _sample_grid_shape() -> tuple[int, int]:
    return random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)


def _sample_rect_sizes(count: int, height: int, width: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    areas: set[int] = set()
    side_hi = min(RECT_SIDE_RANGE[1], height - 1, width - 1)
    side_lo = min(RECT_SIDE_RANGE[0], side_hi)
    for _ in range(200):
        sizes.clear()
        areas.clear()
        while len(sizes) < count:
            rect_h = random.randint(side_lo, side_hi)
            rect_w = random.randint(side_lo, side_hi)
            area = rect_h * rect_w
            if area in areas:
                continue
            sizes.append((rect_h, rect_w))
            areas.add(area)
        if len(sizes) == count:
            random.shuffle(sizes)
            return sizes[:]
    raise RuntimeError("failed to sample unique rectangle areas")


def _random_point_in_rect(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return random.randint(r0, r1), random.randint(c0, c1)


def _rect_from_anchor(
    rect_h: int,
    rect_w: int,
    anchor_r: int,
    anchor_c: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    r_lo = max(0, anchor_r - rect_h + 1)
    r_hi = min(anchor_r, height - rect_h)
    c_lo = max(0, anchor_c - rect_w + 1)
    c_hi = min(anchor_c, width - rect_w)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _pairwise_overlap_count(rects: list[tuple[int, int, int, int]]) -> int:
    count = 0
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if _intersection_area(rect_a, rect_b) > 0:
                count += 1
    return count


def _sample_hub_rects(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int, int, int]] | None:
    anchor_r = random.randint(2, height - 3)
    anchor_c = random.randint(2, width - 3)
    rects: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in sizes:
        rect = _rect_from_anchor(rect_h, rect_w, anchor_r, anchor_c, height, width)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _sample_chain_rects(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    first_h, first_w = sizes[0]
    first_r0 = random.randint(0, height - first_h)
    first_c0 = random.randint(0, width - first_w)
    rects.append((first_r0, first_c0, first_r0 + first_h - 1, first_c0 + first_w - 1))
    for rect_h, rect_w in sizes[1:]:
        anchor_r, anchor_c = _random_point_in_rect(rects[-1])
        rect = _rect_from_anchor(rect_h, rect_w, anchor_r, anchor_c, height, width)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _sample_mixed_rects(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int, int, int]] | None:
    overlap_count = random.randint(2, min(3, len(sizes)))
    cluster = _sample_hub_rects(sizes[:overlap_count], height, width)
    if cluster is None:
        return None
    rects = cluster[:]
    for rect_h, rect_w in sizes[overlap_count:]:
        for _ in range(40):
            r0 = random.randint(0, height - rect_h)
            c0 = random.randint(0, width - rect_w)
            rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            overlaps_cluster = any(_intersection_area(rect, other) > 0 for other in cluster)
            if overlaps_cluster or any(_intersection_area(rect, other) > 0 for other in rects):
                continue
            rects.append(rect)
            break
        else:
            return None
    return rects


def _placement_mode() -> str:
    labels = [label for label, _weight in PLACEMENT_MODE_WEIGHTS]
    weights = [weight for _label, weight in PLACEMENT_MODE_WEIGHTS]
    return random.choices(labels, weights=weights, k=1)[0]


def _sample_rects(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int, int, int]] | None:
    mode = _placement_mode()
    if mode == "hub":
        return _sample_hub_rects(sizes, height, width)
    if mode == "chain":
        return _sample_chain_rects(sizes, height, width)
    return _sample_mixed_rects(sizes, height, width)


def _paint_scene(
    height: int,
    width: int,
    colors: list[int],
    rects: list[tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]
    return grid


def _visible_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _edge_visibility_ok(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        mask[r0, c0 : c1 + 1].any()
        and mask[r1, c0 : c1 + 1].any()
        and mask[r0 : r1 + 1, c0].any()
        and mask[r0 : r1 + 1, c1].any()
    )


def _scene_is_valid(
    grid: np.ndarray,
    colors: list[int],
    rects: list[tuple[int, int, int, int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > ARC_MAX:
        return False

    overlap_pairs = _pairwise_overlap_count(rects)
    if overlap_pairs < 1:
        return False

    if any(_contains(a, b) or _contains(b, a) for i, a in enumerate(rects) for b in rects[i + 1 :]):
        return False

    areas = [int((r1 - r0 + 1) * (c1 - c0 + 1)) for r0, c0, r1, c1 in rects]
    if len(areas) != len(set(areas)):
        return False

    any_occluded = False
    for color, bbox, full_area in zip(colors, rects, areas):
        visible_mask = grid == color
        visible_bbox = _visible_bbox(visible_mask)
        if visible_bbox is None or visible_bbox != bbox:
            return False
        if not _edge_visibility_ok(visible_mask, bbox):
            return False
        if int(visible_mask.sum()) < full_area:
            any_occluded = True

    if not any_occluded:
        return False

    non_bg = int(np.count_nonzero(grid))
    if non_bg >= int(grid.size * 0.8):
        return False

    ranked = main(grid)
    expected = [color for _area, _r0, _c0, _r1, _c1, color in sorted(_color_bboxes(grid), key=lambda item: item[:5])]
    return ranked.shape == (len(colors), 1) and expected == ranked[:, 0].tolist()


def _generate_case() -> tuple[np.ndarray, list[int], list[tuple[int, int, int, int]]]:
    height, width = _sample_grid_shape()
    count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(list(COLOR_POOL), count)
    sizes = _sample_rect_sizes(count, height, width)
    rects = _sample_rects(sizes, height, width)
    if rects is None:
        raise RuntimeError("failed to place rectangles")
    order = list(range(count))
    random.shuffle(order)
    grid = _paint_scene(height, width, colors, rects, order)
    return grid, colors, rects


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, colors, rects = _generate_case()
        except RuntimeError:
            continue
        if _scene_is_valid(grid, colors, rects):
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
