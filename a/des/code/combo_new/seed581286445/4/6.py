import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
RECT_COUNT_RANGE = (3, 5)
RECT_H_RANGE = (4, 9)
RECT_W_RANGE = (4, 9)
OVERLAP_PROB = 0.7
PLACEMENT_RETRIES = 120
GENERATE_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _color_bboxes(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue
        bboxes[color] = _bbox_from_coords(coords)
    return bboxes


def _draw_outline(shape: tuple[int, int], bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    return out


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _compatible(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> int | None:
    overlaps = 0
    for other in others:
        inter = _intersection(rect, other)
        if inter is None:
            continue
        inter_area = _rect_area(inter)
        if inter_area == _rect_area(rect) or inter_area == _rect_area(other):
            return None
        overlaps += 1
    return overlaps


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)


def _random_rect(rng: random.Random, h: int, w: int) -> tuple[int, int, int, int]:
    max_h = min(RECT_H_RANGE[1], h - 3)
    max_w = min(RECT_W_RANGE[1], w - 2)
    rect_h = rng.randint(RECT_H_RANGE[0], max_h)
    rect_w = rng.randint(RECT_W_RANGE[0], max_w)
    r0 = rng.randint(1, h - 2 - rect_h)
    c0 = rng.randint(1, w - 1 - rect_w)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _overlap_rect(
    rng: random.Random, h: int, w: int, anchor: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    max_h = min(RECT_H_RANGE[1], h - 3)
    max_w = min(RECT_W_RANGE[1], w - 2)
    rect_h = rng.randint(RECT_H_RANGE[0], max_h)
    rect_w = rng.randint(RECT_W_RANGE[0], max_w)

    anchor_h = anchor[2] - anchor[0] + 1
    anchor_w = anchor[3] - anchor[1] + 1
    overlap_h_max = min(anchor_h - 1, rect_h - 1)
    overlap_w_max = min(anchor_w - 1, rect_w - 1)
    if overlap_h_max < 2 or overlap_w_max < 2:
        return None

    overlap_h = rng.randint(2, overlap_h_max)
    overlap_w = rng.randint(2, overlap_w_max)
    ar0, ac0, ar1, ac1 = anchor
    pattern = rng.choice(("tl", "tr", "bl", "br"))

    if pattern == "tl":
        r1 = ar0 + overlap_h - 1
        c1 = ac0 + overlap_w - 1
        r0 = r1 - rect_h + 1
        c0 = c1 - rect_w + 1
    elif pattern == "tr":
        r1 = ar0 + overlap_h - 1
        c0 = ac1 - overlap_w + 1
        r0 = r1 - rect_h + 1
        c1 = c0 + rect_w - 1
    elif pattern == "bl":
        r0 = ar1 - overlap_h + 1
        c1 = ac0 + overlap_w - 1
        r1 = r0 + rect_h - 1
        c0 = c1 - rect_w + 1
    else:
        r0 = ar1 - overlap_h + 1
        c0 = ac1 - overlap_w + 1
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

    if r0 < 1 or c0 < 1 or r1 > h - 3 or c1 > w - 2:
        return None
    return r0, c0, r1, c1


def _overlap_stats(rects: list[tuple[int, int, int, int]]) -> tuple[int, list[int]]:
    degrees = [0] * len(rects)
    pairs = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection(rects[i], rects[j]) is None:
                continue
            pairs += 1
            degrees[i] += 1
            degrees[j] += 1
    return pairs, degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the selected overlapped rectangle and draw its full one-pixel outline.

    Input structure:
    - Black background fills the grid.
    - Several distinct-colored solid axis-aligned rectangles overlap, so earlier rectangles are partially hidden by later ones.
    - The bottom row contains one isolated marker pixel whose color matches one rectangle and does not touch any rectangle.

    Rule:
    - Read the marker pixel on the bottom row to obtain the target color, then remove that marker from consideration.
    - For each remaining non-black color, take the minimum and maximum visible row and column of that color as the full rectangle bounds.
    - Select the rectangle whose color matches the marker.
    - Return a same-sized black grid containing only that rectangle's complete single-cell-wide outline; the marker itself is not copied.

    Key insight:
    - Even when rectangles occlude one another, each color's visible pixels still expose the true row and column extremes, so the hidden full rectangle is recovered from its color bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_cols = np.flatnonzero(g[-1] != BG)
    assert len(marker_cols) == 1, "expected a single bottom-row marker pixel"
    marker_c = int(marker_cols[0])
    target_color = int(g[-1, marker_c])

    scene = g.copy()
    scene[-1, marker_c] = BG
    bboxes = _color_bboxes(scene)
    assert target_color in bboxes, "marker color must match a rectangle color"

    return _draw_outline(scene.shape, bboxes[target_color], target_color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    rect_count = rng.randint(*RECT_COUNT_RANGE)

    rects: list[tuple[int, int, int, int]] = []
    required_overlap = True
    for i in range(rect_count):
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            want_overlap = required_overlap or (i > 0 and rng.random() < OVERLAP_PROB)
            if i > 0 and want_overlap:
                anchor = rng.choice(rects)
                rect = _overlap_rect(rng, h, w, anchor)
                if rect is None:
                    continue
            else:
                rect = _random_rect(rng, h, w)

            if rect in rects:
                continue
            overlap_count = _compatible(rect, rects)
            if overlap_count is None:
                continue
            if want_overlap and i > 0 and overlap_count == 0:
                continue
            placed = rect
            required_overlap = required_overlap and overlap_count == 0
            break
        if placed is None:
            return None
        rects.append(placed)

    pair_count, degrees = _overlap_stats(rects)
    if pair_count == 0 or pair_count > rect_count + 1 or max(degrees) > 3:
        return None

    colors = rng.sample(PALETTE, rect_count)
    order = list(range(rect_count))
    rng.shuffle(order)

    grid = np.zeros((h, w), dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]

    visible_areas: dict[int, int] = {}
    for idx, color in enumerate(colors):
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            return None
        if _bbox_from_coords(coords) != rects[idx]:
            return None
        visible_areas[idx] = len(coords)

    occluded = [idx for idx in range(rect_count) if visible_areas[idx] < _rect_area(rects[idx])]
    exposed = [idx for idx in range(rect_count) if visible_areas[idx] == _rect_area(rects[idx])]
    if not occluded:
        return None

    if exposed and rng.random() < 0.5:
        target_idx = rng.choice(exposed)
    else:
        target_idx = rng.choice(occluded)

    marker_c = rng.randrange(w)
    grid[-1, marker_c] = colors[target_idx]

    if np.count_nonzero(grid[-1]) != 1:
        return None

    expected = _draw_outline(grid.shape, rects[target_idx], colors[target_idx])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GENERATE_RETRIES} attempts")
