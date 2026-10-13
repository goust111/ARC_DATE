from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (12, 20)
BBOX_SIDE_RANGE = (4, 8)
POINT_COUNT_RANGE = (2, 6)
GROUP_GAP = 1
MAX_BOX_ATTEMPTS = 120
MAX_POINT_ATTEMPTS = 160
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _inflate(
    box: tuple[int, int, int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boundary_labels(
    cell: tuple[int, int],
    box: tuple[int, int, int, int],
) -> frozenset[str]:
    r, c = cell
    r0, c0, r1, c1 = box
    labels = set()
    if r == r0:
        labels.add("top")
    if r == r1:
        labels.add("bottom")
    if c == c0:
        labels.add("left")
    if c == c1:
        labels.add("right")
    return frozenset(labels)


def _can_add_point(
    cell: tuple[int, int],
    chosen: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    if cell in occupied:
        return False
    r, c = cell
    for rr, cc in chosen:
        if abs(r - rr) + abs(c - cc) == 1:
            return False
    return True


def _sample_points_in_box(
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    boundary_map = {cell: _boundary_labels(cell, box) for cell in all_cells}
    all_boundaries = {"top", "bottom", "left", "right"}

    for _ in range(MAX_POINT_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        covered: set[str] = set()

        while len(chosen) < count and covered != all_boundaries:
            best_gain = -1
            best_cells: list[tuple[int, int]] = []
            shuffled = all_cells[:]
            random.shuffle(shuffled)
            for cell in shuffled:
                if not _can_add_point(cell, chosen, occupied):
                    continue
                gain = len(boundary_map[cell] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_cells = [cell]
                elif gain == best_gain:
                    best_cells.append(cell)
            if best_gain <= 0:
                break
            pick = random.choice(best_cells)
            chosen.append(pick)
            covered |= set(boundary_map[pick])

        if covered != all_boundaries:
            continue

        shuffled = all_cells[:]
        random.shuffle(shuffled)
        for cell in shuffled:
            if len(chosen) == count:
                break
            if cell in chosen:
                continue
            if _can_add_point(cell, chosen, occupied):
                chosen.append(cell)

        if len(chosen) != count:
            continue
        if _bbox_from_points(chosen) == box:
            return chosen
    return None


def _sample_isolated_box(
    height: int,
    width: int,
    existing_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    max_h = min(BBOX_SIDE_RANGE[1], height)
    max_w = min(BBOX_SIDE_RANGE[1], width)
    min_h = BBOX_SIDE_RANGE[0]
    min_w = BBOX_SIDE_RANGE[0]
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(MAX_BOX_ATTEMPTS):
        box_h = random.randint(min_h, max_h)
        box_w = random.randint(min_w, max_w)
        r0 = random.randint(0, height - box_h)
        c0 = random.randint(0, width - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if any(_boxes_overlap(_inflate(box, GROUP_GAP), other) for other in existing_boxes):
            continue
        return box
    return None


def _sample_overlap_group(
    height: int,
    width: int,
    existing_boxes: list[tuple[int, int, int, int]],
    group_size: int,
) -> list[tuple[int, int, int, int]] | None:
    max_h = min(BBOX_SIDE_RANGE[1], height)
    max_w = min(BBOX_SIDE_RANGE[1], width)
    min_h = BBOX_SIDE_RANGE[0]
    min_w = BBOX_SIDE_RANGE[0]
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(MAX_BOX_ATTEMPTS):
        shared_r = random.randint(0, height - 1)
        shared_c = random.randint(0, width - 1)
        boxes: list[tuple[int, int, int, int]] = []
        seen = set()

        for _ in range(group_size):
            placed = False
            for _ in range(MAX_BOX_ATTEMPTS):
                box_h = random.randint(min_h, max_h)
                box_w = random.randint(min_w, max_w)
                r0_lo = max(0, shared_r - box_h + 1)
                r0_hi = min(shared_r, height - box_h)
                c0_lo = max(0, shared_c - box_w + 1)
                c0_hi = min(shared_c, width - box_w)
                if r0_lo > r0_hi or c0_lo > c0_hi:
                    continue
                r0 = random.randint(r0_lo, r0_hi)
                c0 = random.randint(c0_lo, c0_hi)
                box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if box in seen:
                    continue
                if any(_boxes_overlap(_inflate(box, GROUP_GAP), other) for other in existing_boxes):
                    continue
                boxes.append(box)
                seen.add(box)
                placed = True
                break
            if not placed:
                break

        if len(boxes) != group_size:
            continue
        if all(
            _boxes_overlap(boxes[i], boxes[j])
            for i in range(group_size)
            for j in range(i + 1, group_size)
        ):
            return boxes
    return None


def _choose_group_sizes(n_colors: int) -> list[int]:
    patterns = {
        3: [[2, 1]],
        4: [[2, 1, 1], [3, 1]],
        5: [[2, 1, 1, 1], [3, 1, 1], [2, 2, 1]],
    }
    return random.choice(patterns[n_colors])[:]


def _build_bbox_plan(
    height: int,
    width: int,
    n_colors: int,
) -> list[tuple[int, int, int, int]] | None:
    group_sizes = sorted(_choose_group_sizes(n_colors), reverse=True)
    boxes: list[tuple[int, int, int, int]] = []

    for size in group_sizes:
        if size == 1:
            box = _sample_isolated_box(height, width, boxes)
            if box is None:
                return None
            boxes.append(box)
        else:
            group = _sample_overlap_group(height, width, boxes, size)
            if group is None:
                return None
            boxes.extend(group)

    random.shuffle(boxes)
    return boxes


def _color_boxes_for_grid(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = list(range(len(boxes)))
    random.shuffle(order)
    for idx in order:
        box = boxes[idx]
        color = colors[idx]
        point_count = random.randint(*POINT_COUNT_RANGE)
        points = _sample_points_in_box(box, point_count, occupied)
        if points is None:
            return None
        for r, c in points:
            grid[r, c] = color
            occupied.add((r, c))
    return grid


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _validate_input(grid: np.ndarray) -> bool:
    colors = _foreground_colors(grid)
    if not (3 <= len(colors) <= 5):
        return False

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        points = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        if not (POINT_COUNT_RANGE[0] <= len(points) <= POINT_COUNT_RANGE[1]):
            return False
        for i, (r0, c0) in enumerate(points):
            for r1, c1 in points[i + 1 :]:
                if abs(r0 - r1) + abs(c0 - c1) == 1:
                    return False
        boxes[color] = _bbox_from_points(points)

    out = main(grid)
    out_colors = _foreground_colors(out)
    if not out_colors or len(out_colors) == len(colors):
        return False

    for color in colors:
        overlaps = any(
            other != color and _boxes_overlap(boxes[color], boxes[other])
            for other in colors
        )
        if color in out_colors and overlaps:
            return False
        if color not in out_colors and not overlaps:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only colors whose global bounding boxes do not overlap any other color's
    global bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = _foreground_colors(g)
    boxes = {
        color: _bbox_from_points([tuple(map(int, p)) for p in np.argwhere(g == color)])
        for color in colors
    }

    blocked: set[int] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if _boxes_overlap(boxes[color_a], boxes[color_b]):
                blocked.add(color_a)
                blocked.add(color_b)

    out = g.copy()
    for color in blocked:
        out[out == color] = BG
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        n_colors = random.randint(3, 5)
        colors = random.sample(COLOR_POOL, n_colors)

        boxes = _build_bbox_plan(height, width, n_colors)
        if boxes is None:
            continue
        grid = _color_boxes_for_grid((height, width), boxes, colors)
        if grid is None:
            continue
        if _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
