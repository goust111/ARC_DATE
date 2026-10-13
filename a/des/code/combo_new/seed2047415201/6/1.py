from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_IDS = tuple(range(1, 10))
GENERATION_RETRIES = 300


def _foreground_colors_in_scan_order(g: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in g.reshape(-1):
        color = int(value)
        if color != BLACK and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn sparse same-color marker pixels into stacked solid rectangles.

    Input structure:
    - The background is Black.
    - Several non-black colors appear as sparse isolated marker pixels.
    - Each foreground color marks one axis-aligned rectangle by the bounding box
      of all pixels of that color.

    Rule:
    - For each foreground color, compute the bounding box of its pixels.
    - Sort these color rectangles by bounding-box area from largest to smallest.
    - On a new black canvas of the same size, paint each full bounding rectangle
      in that order, so later smaller rectangles cover earlier larger ones.

    Key insight:
    - The sparse pixels are only range markers; grouping by color reveals the
      nested rectangle stack hidden by the scattered points.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects: list[tuple[int, int, int, int, int, int]] = []
    for color in _foreground_colors_in_scan_order(g):
        pts = np.argwhere(g == int(color))
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        rects.append((area, r0, c0, r1, c1, int(color)))

    out = np.full_like(g, BLACK)
    for _area, r0, c0, r1, c1, color in sorted(
        rects, key=lambda item: (-item[0], item[1], item[2], item[3], item[4])
    ):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _sample_nested_boxes(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    min_outer_side = 2 * count
    if height < min_outer_side or width < min_outer_side:
        return None

    outer_h = random.randint(min_outer_side, height)
    outer_w = random.randint(min_outer_side, width)
    r0 = random.randint(0, height - outer_h)
    c0 = random.randint(0, width - outer_w)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    boxes = [(r0, c0, r1, c1)]

    for i in range(1, count):
        remaining = count - i
        cur_h = r1 - r0 + 1
        cur_w = c1 - c0 + 1
        min_next_side = 2 * remaining
        max_h_shrink = cur_h - min_next_side
        max_w_shrink = cur_w - min_next_side
        if max_h_shrink < 2 or max_w_shrink < 2:
            return None

        h_shrink = random.randint(2, max_h_shrink)
        w_shrink = random.randint(2, max_w_shrink)
        top = random.randint(1, h_shrink - 1)
        left = random.randint(1, w_shrink - 1)
        bottom = h_shrink - top
        right = w_shrink - left

        r0 += top
        r1 -= bottom
        c0 += left
        c1 -= right
        boxes.append((r0, c0, r1, c1))

    return boxes


def _dedupe(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        if point not in seen:
            seen.add(point)
            result.append(point)
    return result


def _sample_marker_points(
    box: tuple[int, int, int, int], occupied: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    pattern = random.choice(["diag_a", "diag_b", "sides", "corners"])
    if pattern == "diag_a":
        base = [(r0, c0), (r1, c1)]
    elif pattern == "diag_b":
        base = [(r0, c1), (r1, c0)]
    elif pattern == "corners":
        base = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    else:
        base = [
            (r0, random.randint(c0, c1)),
            (r1, random.randint(c0, c1)),
            (random.randint(r0, r1), c0),
            (random.randint(r0, r1), c1),
        ]

    base = _dedupe(base)
    if len(base) < 2:
        base = [(r0, c0), (r1, c1)]
    if any(point in occupied for point in base):
        return None

    candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occupied and (r, c) not in base
    ]
    max_count = min(6, len(base) + len(candidates))
    if max_count < len(base):
        return None
    target_count = random.randint(len(base), max_count)
    random.shuffle(candidates)
    return base + candidates[: target_count - len(base)]


def _render_rectangles(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    items = []
    for box, color in zip(boxes, colors):
        r0, c0, r1, c1 = box
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        items.append((area, r0, c0, r1, c1, int(color)))
    for _area, r0, c0, r1, c1, color in sorted(
        items, key=lambda item: (-item[0], item[1], item[2], item[3], item[4])
    ):
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 30)
    width = random.randint(10, 30)
    max_count = min(5, height // 2, width // 2, len(COLOR_IDS))
    if max_count < 3:
        return None
    count = random.randint(3, max_count)

    boxes = _sample_nested_boxes(height, width, count)
    if boxes is None:
        return None

    colors = random.sample(COLOR_IDS, count)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for box, color in reversed(list(zip(boxes, colors))):
        points = _sample_marker_points(box, occupied)
        if points is None:
            return None
        if not (2 <= len(points) <= 6):
            return None
        for r, c in points:
            grid[r, c] = int(color)
            occupied.add((r, c))

    expected = _render_rectangles((height, width), boxes, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
