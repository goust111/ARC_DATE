import random

import numpy as np


def _colors_in_raster_order(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in grid.reshape(-1):
        color = int(value)
        if color == 0 or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _bbox_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expanded_box(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _borders_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(_border_cells(a) & _border_cells(b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered pixels by color, draw each color's bbox frame, and overlay frames by area.
    Original marker pixels are restored after all frames are drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in _colors_in_raster_order(g):
        points = np.argwhere(g == color)
        if points.size == 0:
            continue
        bbox = _bbox_from_points(points)
        groups.append((color, _bbox_area(bbox), bbox))

    # Stable sort: ties keep raster-order discovery instead of using color ids.
    groups.sort(key=lambda item: -item[1])

    out = np.zeros_like(g)
    for color, _area, (r0, c0, r1, c1) in groups:
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    marker_mask = g != 0
    out[marker_mask] = g[marker_mask]
    return out


_GRID_H_RANGE = (12, 24)
_GRID_W_RANGE = (12, 24)
_COLOR_COUNT_RANGE = (2, 4)
_MARKER_COUNT_RANGE = (3, 6)
_MIN_BOX_SPAN = 4
_BOX_RETRIES = 250
_GEN_RETRIES = 400
_PALETTE = tuple(range(1, 10))


def _sample_span(limit: int) -> int:
    span = int(round(random.triangular(_MIN_BOX_SPAN, limit, min(limit, 7))))
    return max(_MIN_BOX_SPAN, min(limit, span))


def _sample_box(height: int, width: int) -> tuple[int, int, int, int]:
    box_h = _sample_span(height - 2)
    box_w = _sample_span(width - 2)
    top = random.randint(1, height - box_h - 1)
    left = random.randint(1, width - box_w - 1)
    return top, left, top + box_h - 1, left + box_w - 1


def _sample_markers(
    bbox: tuple[int, int, int, int], count: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    templates = [
        [(r0, c0), (r0, c1), (r1, c0)],
        [(r0, c0), (r0, c1), (r1, c1)],
        [(r0, c0), (r1, c0), (r1, c1)],
        [(r0, c1), (r1, c0), (r1, c1)],
    ]
    random.shuffle(templates)

    free_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occupied]
    if len(free_cells) < count:
        return None

    for base in templates:
        if any(cell in occupied for cell in base):
            continue
        markers = list(base)
        extras = [cell for cell in free_cells if cell not in markers]
        random.shuffle(extras)
        while len(markers) < count and extras:
            markers.append(extras.pop())
        if len(markers) == count:
            return markers
    return None


def _pairwise_border_overlap(boxes: list[tuple[int, int, int, int]]) -> int:
    overlaps = 0
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            if _borders_intersect(box_a, box_b):
                overlaps += 1
    return overlaps


def _valid_generated_grid(grid: np.ndarray, force_overlap: bool) -> bool:
    colors = _colors_in_raster_order(grid)
    if not (_COLOR_COUNT_RANGE[0] <= len(colors) <= _COLOR_COUNT_RANGE[1]):
        return False

    boxes: list[tuple[int, int, int, int]] = []
    areas: set[int] = set()
    for color in colors:
        points = np.argwhere(grid == color)
        if not (_MARKER_COUNT_RANGE[0] <= len(points) <= _MARKER_COUNT_RANGE[1]):
            return False
        bbox = _bbox_from_points(points)
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            return False
        area = _bbox_area(bbox)
        if area in areas:
            return False
        boxes.append(bbox)
        areas.add(area)

    overlap_pairs = _pairwise_border_overlap(boxes)
    if force_overlap and overlap_pairs == 0:
        return False
    if not force_overlap and overlap_pairs != 0:
        return False

    return not np.array_equal(main(grid), grid)


def _gen_attempt(force_overlap: bool) -> np.ndarray | None:
    height = random.randint(*_GRID_H_RANGE)
    width = random.randint(*_GRID_W_RANGE)
    color_count = random.randint(*_COLOR_COUNT_RANGE)
    colors = random.sample(_PALETTE, color_count)

    boxes: list[tuple[int, int, int, int]] = []
    areas: set[int] = set()

    for index in range(color_count):
        placed = None
        for _ in range(_BOX_RETRIES):
            candidate = _sample_box(height, width)
            area = _bbox_area(candidate)
            if area in areas:
                continue

            if index == 0:
                placed = candidate
                break

            if force_overlap and index == 1:
                if not _borders_intersect(candidate, boxes[0]):
                    continue
                placed = candidate
                break

            if any(_boxes_intersect(_expanded_box(candidate, 1), _expanded_box(other, 1)) for other in boxes):
                continue
            placed = candidate
            break

        if placed is None:
            return None
        boxes.append(placed)
        areas.add(_bbox_area(placed))

    occupied: set[tuple[int, int]] = set()
    grid = np.zeros((height, width), dtype=int)
    for color, bbox in zip(colors, boxes):
        marker_count = random.randint(*_MARKER_COUNT_RANGE)
        markers = _sample_markers(bbox, marker_count, occupied)
        if markers is None:
            return None
        for r, c in markers:
            grid[r, c] = color
            occupied.add((r, c))

    if not _valid_generated_grid(grid, force_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        force_overlap = random.random() < 0.5
        grid = _gen_attempt(force_overlap)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
