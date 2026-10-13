from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

GEN_RETRY_LIMIT = 500
GRID_SIDE_RANGE = (16, 29)
RECT_COUNT_CHOICES = (2, 3, 3, 4)
NOISE_GROUP_COUNT_CHOICES = (1, 2, 2, 3)
MIN_RECT_SIDE = 3
MAX_RECT_SIDE = 12
MAX_SHARED_BORDER = 6
RECT_TRIES = 250
POINT_TRIES = 250

BBox = tuple[int, int, int, int]
Point = tuple[int, int]


def _corner_bbox(coords: np.ndarray | list[Point]) -> BBox | None:
    pts = [tuple(map(int, point)) for point in coords]
    if len(pts) != 4 or len(set(pts)) != 4:
        return None

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None

    expected = {
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    }
    if set(pts) != expected:
        return None
    return rows[0], cols[0], rows[1], cols[1]


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _corners(bbox: BBox) -> list[Point]:
    r0, c0, r1, c1 = bbox
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _frame_cells(bbox: BBox) -> set[Point]:
    r0, c0, r1, c1 = bbox
    cells: set[Point] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _paint_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _chebyshev_distance(a: Point, b: Point) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _points_are_isolated(points: list[Point], occupied: set[Point]) -> bool:
    if len(set(points)) != len(points):
        return False

    for i, point in enumerate(points):
        for other in occupied:
            if _chebyshev_distance(point, other) <= 1:
                return False
        for other in points[:i]:
            if _chebyshev_distance(point, other) <= 1:
                return False
    return True


def _sample_rectangle(shape: tuple[int, int], *, favor_large: bool) -> BBox:
    height, width = shape
    max_h = min(MAX_RECT_SIDE, height)
    max_w = min(MAX_RECT_SIDE, width)

    min_h = MIN_RECT_SIDE
    min_w = MIN_RECT_SIDE
    if favor_large:
        min_h = min(max_h, max(MIN_RECT_SIDE, 6))
        min_w = min(max_w, max(MIN_RECT_SIDE, 6))

    rect_h = random.randint(min_h, max_h)
    rect_w = random.randint(min_w, max_w)
    r0 = random.randint(0, height - rect_h)
    c0 = random.randint(0, width - rect_w)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _sample_rectangles(
    shape: tuple[int, int], colors: list[int]
) -> tuple[list[dict[str, object]], set[Point]] | None:
    rectangles: list[dict[str, object]] = []
    occupied_points: set[Point] = set()
    areas: set[int] = set()
    border_use: dict[Point, int] = {}
    overlap_pairs = 0
    required_overlaps = 1 if len(colors) == 2 else random.choice((1, 1, 2))

    for index, color in enumerate(colors):
        need_overlap = index > 0 and (index == 1 or overlap_pairs < required_overlaps)
        accepted = False

        for _ in range(RECT_TRIES):
            bbox = _sample_rectangle(shape, favor_large=index == 0)
            area = _bbox_area(bbox)
            if area in areas:
                continue

            corners = _corners(bbox)
            if not _points_are_isolated(corners, occupied_points):
                continue

            border = _frame_cells(bbox)
            if any(border_use.get(cell, 0) >= 2 for cell in border):
                continue

            overlaps = [len(border & rect["border"]) for rect in rectangles]
            if any(count == 1 or count > MAX_SHARED_BORDER for count in overlaps):
                continue
            if need_overlap and not any(2 <= count <= MAX_SHARED_BORDER for count in overlaps):
                continue
            if (
                not need_overlap
                and overlap_pairs >= required_overlaps
                and any(count > 0 for count in overlaps)
                and random.random() < 0.75
            ):
                continue

            rectangles.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "border": border,
                }
            )
            areas.add(area)
            occupied_points.update(corners)
            for cell in border:
                border_use[cell] = border_use.get(cell, 0) + 1
            overlap_pairs += sum(count > 0 for count in overlaps)
            accepted = True
            break

        if not accepted:
            return None

    if overlap_pairs < required_overlaps:
        return None
    return rectangles, occupied_points


def _sample_isolated_points(
    shape: tuple[int, int], count: int, occupied: set[Point]
) -> list[Point] | None:
    height, width = shape
    for _ in range(POINT_TRIES):
        pts: list[Point] = []
        temp_occupied = set(occupied)
        for _ in range(count):
            placed = False
            for _ in range(POINT_TRIES):
                point = (random.randrange(height), random.randrange(width))
                if not _points_are_isolated([point], temp_occupied):
                    continue
                pts.append(point)
                temp_occupied.add(point)
                placed = True
                break
            if not placed:
                break
        if len(pts) == count:
            return pts
    return None


def _sample_noise_group(
    shape: tuple[int, int], occupied: set[Point], *, force_bad_quad: bool
) -> list[Point] | None:
    for _ in range(POINT_TRIES):
        if force_bad_quad:
            count = 4
        else:
            count = 4 if random.random() < 0.35 else random.randint(1, 3)
        pts = _sample_isolated_points(shape, count, occupied)
        if pts is None:
            continue
        if count == 4 and _corner_bbox(pts) is not None:
            continue
        return pts
    return None


def _render_input(
    shape: tuple[int, int],
    rectangles: list[dict[str, object]],
    noise_groups: list[tuple[int, list[Point]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for rect in rectangles:
        color = int(rect["color"])
        for r, c in _corners(rect["bbox"]):
            grid[r, c] = color
    for color, pts in noise_groups:
        for r, c in pts:
            grid[r, c] = int(color)
    return grid


def _render_output(shape: tuple[int, int], rectangles: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    ordered = sorted(
        rectangles,
        key=lambda rect: _bbox_area(rect["bbox"]),
        reverse=True,
    )
    for rect in ordered:
        _paint_frame(out, rect["bbox"], int(rect["color"]))
    return out


def _all_foreground_points_are_isolated(grid: np.ndarray) -> bool:
    points = [tuple(map(int, point)) for point in np.argwhere(grid != BLACK)]
    for i, point in enumerate(points):
        for other in points[:i]:
            if _chebyshev_distance(point, other) <= 1:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hollow rectangular frames from single-pixel corner generators while deleting noise.

    Input structure:
    - The grid has a black background and isolated non-black single pixels.
    - Some colors appear exactly four times and those four pixels may be the four corners of one axis-aligned rectangle.
    - Other colored pixels are distractors whose color counts or positions do not match a rectangle-corner pattern.

    Rule:
    - Discard every color that does not form exactly one four-corner set.
    - For each valid color, draw the complete 1-cell-thick hollow rectangle defined by its corner bbox.
    - If multiple reconstructed frames cover the same cell, keep the color of the frame with the smaller bbox area.

    Key insight:
    - The visible input pixels are only generators: a color is meaningful exactly when its four isolated dots encode a rectangle's corners, and overlap order is recovered from rectangle area rather than color id or scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BLACK, dtype=int)
    frames: list[tuple[int, int, BBox]] = []

    for color in sorted(int(value) for value in np.unique(g) if int(value) != BLACK):
        bbox = _corner_bbox(np.argwhere(g == color))
        if bbox is None:
            continue
        frames.append((_bbox_area(bbox), color, bbox))

    for _area, color, bbox in sorted(frames, key=lambda item: item[0], reverse=True):
        _paint_frame(out, bbox, color)
    return out


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    valid_count = random.choice(RECT_COUNT_CHOICES)
    noise_count = min(random.choice(NOISE_GROUP_COUNT_CHOICES), len(PALETTE) - valid_count)
    colors = random.sample(PALETTE, valid_count + noise_count)
    valid_colors = colors[:valid_count]
    noise_colors = colors[valid_count:]

    sampled = _sample_rectangles(shape, valid_colors)
    if sampled is None:
        return None
    rectangles, occupied = sampled

    noise_groups: list[tuple[int, list[Point]]] = []
    force_bad_quad = noise_count > 0 and random.random() < 0.5
    for index, color in enumerate(noise_colors):
        pts = _sample_noise_group(shape, occupied, force_bad_quad=force_bad_quad and index == 0)
        if pts is None:
            return None
        occupied.update(pts)
        noise_groups.append((int(color), pts))

    grid = _render_input(shape, rectangles, noise_groups)
    expected = _render_output(shape, rectangles)

    if not _all_foreground_points_are_isolated(grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
