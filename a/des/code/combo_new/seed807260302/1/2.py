from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 500
MODES = ("separate", "pair", "triple")
GRID_SIDE_RANGE = (14, 30)
TRIPLE_GRID_SIDE_RANGE = (18, 30)
BOX_SIDE_RANGE = (4, 10)
POINT_COUNTS = tuple(range(3, 9))
MARGIN = 1
MIN_OVERLAP = 2


def _bbox_for_color(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(g == int(color))
    assert rows.size > 0, f"color {color} must appear"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _draw_border(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each color's bounding-box frame, then let the rarest overlapping color own shared areas.

    Input structure:
    - The background is black.
    - Every foreground color appears as scattered single-pixel markers.
    - Each foreground color has at least three markers whose rows and columns span a visible axis-aligned rectangle.

    Rule:
    - For each foreground color, find the minimal axis-aligned bounding box of its original markers.
    - Copy the input and draw a one-pixel-wide border around each box in that box's color.
    - For every cell lying inside two or more color boxes, recolor the whole shared region to the color with the fewest original markers.
    - In valid inputs, overlapping colors have a unique least frequent color.

    Key insight:
    The visible points define rectangular extents, but overlaps are resolved by counting the original markers, not by frame order or color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(c) for c in np.unique(g) if int(c) != BLACK]
    assert colors, "expected at least one foreground color"

    counts: dict[int, int] = {}
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        count = int(np.count_nonzero(g == color))
        assert count >= 3, "each foreground color needs at least three markers"
        box = _bbox_for_color(g, color)
        r0, c0, r1, c1 = box
        assert r0 < r1 and c0 < c1, "each color must span a visible rectangle"
        counts[color] = count
        boxes[color] = box

    out = g.copy()
    for color in colors:
        _draw_border(out, boxes[color], color)

    cover = np.zeros(g.shape, dtype=int)
    best_count = np.full(g.shape, 10_000, dtype=int)
    best_color = np.zeros(g.shape, dtype=int)
    best_tied = np.zeros(g.shape, dtype=bool)

    for color in colors:
        r0, c0, r1, c1 = boxes[color]
        region = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        cover[region] += 1

        count = counts[color]
        count_view = best_count[region]
        color_view = best_color[region]
        tied_view = best_tied[region]
        less = count < count_view
        equal = count == count_view
        color_view[less] = color
        count_view[less] = count
        tied_view[less] = False
        tied_view[equal] = True

    overlap = cover >= 2
    assert not bool(np.any(best_tied & overlap)), "overlapping boxes must have a unique least frequent color"
    out[overlap] = best_color[overlap]
    return out


# --------------- generator ---------------


def _random_box(height: int, width: int) -> tuple[int, int, int, int]:
    box_h = random.randint(*BOX_SIDE_RANGE)
    box_w = random.randint(*BOX_SIDE_RANGE)
    box_h = min(box_h, height - 2 * MARGIN)
    box_w = min(box_w, width - 2 * MARGIN)
    r0 = random.randint(MARGIN, height - MARGIN - box_h)
    c0 = random.randint(MARGIN, width - MARGIN - box_w)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _intersection_dims(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0, 0
    return r1 - r0 + 1, c1 - c0 + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 - gap <= br1 and ar1 + gap >= br0 and ac0 - gap <= bc1 and ac1 + gap >= bc0


def _add_nonoverlapping_box(
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> bool:
    for _ in range(120):
        box = _random_box(height, width)
        if all(not _boxes_overlap(box, other, gap=1) for other in boxes):
            boxes.append(box)
            return True
    return False


def _overlapping_box(
    base: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    br0, bc0, br1, bc1 = base
    for _ in range(120):
        box_h = random.randint(*BOX_SIDE_RANGE)
        box_w = random.randint(*BOX_SIDE_RANGE)
        box_h = min(box_h, height - 2 * MARGIN)
        box_w = min(box_w, width - 2 * MARGIN)

        r_min = max(MARGIN, br0 + MIN_OVERLAP - box_h)
        r_max = min(height - MARGIN - box_h, br1 - MIN_OVERLAP + 1)
        c_min = max(MARGIN, bc0 + MIN_OVERLAP - box_w)
        c_max = min(width - MARGIN - box_w, bc1 - MIN_OVERLAP + 1)
        if r_min > r_max or c_min > c_max:
            continue

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        inter_h, inter_w = _intersection_dims(base, box)
        if box != base and inter_h >= MIN_OVERLAP and inter_w >= MIN_OVERLAP:
            base_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
            box_area = box_h * box_w
            inter_area = inter_h * inter_w
            if base_area > inter_area and box_area > inter_area:
                return box
    return None


def _triple_boxes(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    overlap_h = random.randint(2, 4)
    overlap_w = random.randint(2, 4)
    r_min = MARGIN + 5
    r_max = height - MARGIN - overlap_h - 5
    c_min = MARGIN + 5
    c_max = width - MARGIN - overlap_w - 5
    if r_min > r_max or c_min > c_max:
        return None

    or0 = random.randint(r_min, r_max)
    oc0 = random.randint(c_min, c_max)
    or1 = or0 + overlap_h - 1
    oc1 = oc0 + overlap_w - 1

    caps = (
        or0 - MARGIN,
        height - MARGIN - 1 - or1,
        oc0 - MARGIN,
        width - MARGIN - 1 - oc1,
    )
    patterns = (
        (4, 1, 4, 1),
        (1, 4, 1, 4),
        (4, 1, 1, 4),
    )
    boxes: list[tuple[int, int, int, int]] = []
    for min_up, min_down, min_left, min_right in patterns:
        if caps[0] < min_up or caps[1] < min_down or caps[2] < min_left or caps[3] < min_right:
            return None
        up = random.randint(min_up, min(caps[0], min_up + 3))
        down = random.randint(min_down, min(caps[1], min_down + 3))
        left = random.randint(min_left, min(caps[2], min_left + 3))
        right = random.randint(min_right, min(caps[3], min_right + 3))
        boxes.append((or0 - up, oc0 - left, or1 + down, oc1 + right))

    return boxes if len(set(boxes)) == 3 else None


def _sample_boxes(mode: str) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    if mode == "triple":
        height = random.randint(*TRIPLE_GRID_SIDE_RANGE)
        width = random.randint(*TRIPLE_GRID_SIDE_RANGE)
        boxes = _triple_boxes(height, width)
        if boxes is None:
            return None
        target_count = random.randint(3, 4)
        while len(boxes) < target_count:
            if not _add_nonoverlapping_box(boxes, height, width):
                return None
        return height, width, boxes

    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    boxes = [_random_box(height, width)]

    if mode == "pair":
        second = _overlapping_box(boxes[0], height, width)
        if second is None:
            return None
        boxes.append(second)
        target_count = random.randint(2, 5)
    else:
        target_count = random.randint(2, 4)

    while len(boxes) < target_count:
        if not _add_nonoverlapping_box(boxes, height, width):
            return None
    return height, width, boxes


def _spread_candidates(
    candidates: list[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        cell
        for cell in candidates
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen)
    ]


def _points_for_box(
    box: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    diagonals = [
        ((r0, c0), (r1, c1)),
        ((r0, c1), (r1, c0)),
    ]
    random.shuffle(diagonals)

    chosen: set[tuple[int, int]] | None = None
    for diagonal in diagonals:
        if all(cell not in occupied for cell in diagonal):
            chosen = set(diagonal)
            break
    if chosen is None:
        return None

    candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occupied and (r, c) not in chosen
    ]

    while len(chosen) < count:
        if not candidates:
            return None
        spread = _spread_candidates(candidates, chosen)
        pool = spread if spread else candidates
        cell = random.choice(pool)
        chosen.add(cell)
        candidates.remove(cell)

    return chosen


def _coverage(grid_shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    cover = np.zeros(grid_shape, dtype=int)
    for r0, c0, r1, c1 in boxes:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _build_grid(
    height: int,
    width: int,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray | None:
    colors = random.sample(COLORS, len(boxes))
    counts = random.sample(POINT_COUNTS, len(boxes))
    order = list(range(len(boxes)))
    random.shuffle(order)

    grid = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for idx in order:
        pts = _points_for_box(boxes[idx], counts[idx], occupied)
        if pts is None:
            return None
        color = colors[idx]
        for r, c in pts:
            grid[r, c] = color
        occupied.update(pts)

    for idx, color in enumerate(colors):
        if _bbox_for_color(grid, color) != boxes[idx]:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    sampled = _sample_boxes(mode)
    if sampled is None:
        return None
    height, width, boxes = sampled

    grid = _build_grid(height, width, boxes)
    if grid is None:
        return None

    cover = _coverage(grid.shape, boxes)
    max_cover = int(cover.max())
    if mode == "separate" and max_cover != 1:
        return None
    if mode == "pair" and max_cover < 2:
        return None
    if mode == "triple" and max_cover < 3:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
