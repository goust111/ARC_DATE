from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
INPUT_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


Rect = tuple[int, int, int, int]


def _corners(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[Rect]:
    rects: list[Rect] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 4, "each marker color must appear exactly four times"
        rows = sorted(set(int(r) for r in coords[:, 0]))
        cols = sorted(set(int(c) for c in coords[:, 1]))
        assert len(rows) == 2 and len(cols) == 2, "each color must mark rectangle corners"
        expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "same-colored markers must be the four corners of one axis-aligned rectangle"
        rects.append((rows[0], cols[0], rows[1], cols[1]))
    assert rects, "expected at least one implied rectangle"
    return rects


def _parity_mask(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] ^= True
    return mask


def _coverage_counts(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild corner-implied rectangles and color only the cells covered an odd number of times.

    Input structure:
    - A mostly black canvas contains several color groups of single-pixel corner markers.
    - Each non-black color appears exactly four times, at the four corners of one axis-aligned rectangle.
    - The solid rectangles themselves are absent, and different implied rectangles may overlap or nest.

    Rule:
    - Recover one filled rectangle from each same-colored corner quartet, including the border cells.
    - Count how many recovered rectangles cover each cell.
    - Output green on cells with odd coverage parity and black everywhere else; input marker colors are discarded.

    Key insight:
    - The visible pixels are only generators for hidden rectangles, and overlaps cancel by XOR rather than accumulating as a union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    rects = _extract_rectangles(g, bg)
    out = np.zeros_like(g)
    out[_parity_mask(g.shape, rects)] = GREEN
    return out


def _random_interval(limit: int, min_len: int, max_len: int) -> tuple[int, int]:
    length = random.randint(min_len, min(max_len, limit))
    start = random.randint(0, limit - length)
    return start, start + length - 1


def _intersection_area(a: Rect, b: Rect) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _valid_rect(rect: Rect, corner_cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return all(cell not in corner_cells for cell in _corners(rect))


def _sample_any_rect(height: int, width: int, corner_cells: set[tuple[int, int]]) -> Rect | None:
    for _ in range(80):
        rect = (*_random_interval(height, 3, max(3, min(11, height))), *_random_interval(width, 3, max(3, min(11, width))))
        rect = (rect[0], rect[2], rect[1], rect[3])
        if _valid_rect(rect, corner_cells):
            return rect
    return None


def _sample_disjoint_rect(
    height: int, width: int, corner_cells: set[tuple[int, int]], rects: list[Rect]
) -> Rect | None:
    for _ in range(120):
        rect = _sample_any_rect(height, width, corner_cells)
        if rect is None:
            return None
        if all(_intersection_area(rect, other) == 0 for other in rects):
            return rect
    return None


def _sample_overlap_pair(height: int, width: int) -> list[Rect] | None:
    min_h = min(11, height - 1)
    min_w = min(11, width - 1)
    if min_h < 4 or min_w < 4:
        return None

    for _ in range(150):
        a = _sample_any_rect(height, width, set())
        if a is None:
            return None
        for _ in range(80):
            b = _sample_any_rect(height, width, set(_corners(a)))
            if b is None:
                break
            inter = _intersection_area(a, b)
            if inter == 0:
                continue
            if a == b:
                continue
            if _contains(a, b) or _contains(b, a):
                continue
            if inter >= min(
                (a[2] - a[0] + 1) * (a[3] - a[1] + 1),
                (b[2] - b[0] + 1) * (b[3] - b[1] + 1),
            ):
                continue
            return [a, b]
    return None


def _sample_nested_pair(height: int, width: int) -> list[Rect] | None:
    for _ in range(150):
        outer_h = random.randint(6, min(14, height))
        outer_w = random.randint(6, min(14, width))
        r0 = random.randint(0, height - outer_h)
        c0 = random.randint(0, width - outer_w)
        outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

        inner_top = random.randint(r0 + 1, r0 + outer_h - 4)
        inner_left = random.randint(c0 + 1, c0 + outer_w - 4)
        inner_bottom = random.randint(inner_top + 2, r0 + outer_h - 2)
        inner_right = random.randint(inner_left + 2, c0 + outer_w - 2)
        inner = (inner_top, inner_left, inner_bottom, inner_right)
        if set(_corners(outer)) & set(_corners(inner)):
            continue
        return [outer, inner]
    return None


def _sample_extra_rect(
    height: int,
    width: int,
    corner_cells: set[tuple[int, int]],
    rects: list[Rect],
    allow_overlap: bool,
) -> Rect | None:
    for _ in range(120):
        rect = _sample_any_rect(height, width, corner_cells)
        if rect is None:
            return None
        overlaps = [_intersection_area(rect, other) for other in rects]
        if allow_overlap:
            if any(area > 0 for area in overlaps):
                if max(overlaps) <= 0.45 * min(
                    (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1),
                    min((r[2] - r[0] + 1) * (r[3] - r[1] + 1) for r in rects if _intersection_area(rect, r) > 0),
                ):
                    return rect
            elif random.random() < 0.35:
                return rect
        else:
            if all(area == 0 for area in overlaps):
                return rect
    return None


def _build_input(height: int, width: int, rects: list[Rect]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(INPUT_COLORS, len(rects))
    for color, rect in zip(colors, rects):
        for r, c in _corners(rect):
            grid[r, c] = color
    return grid


def _sample_rectangles(height: int, width: int, count: int) -> list[Rect] | None:
    mode = random.choices(["overlap", "nested", "disjoint"], weights=[0.45, 0.35, 0.20])[0]
    if mode == "overlap":
        rects = _sample_overlap_pair(height, width)
    elif mode == "nested":
        rects = _sample_nested_pair(height, width)
    else:
        first = _sample_any_rect(height, width, set())
        if first is None:
            return None
        second = _sample_disjoint_rect(height, width, set(_corners(first)), [first])
        rects = None if second is None else [first, second]
    if rects is None:
        return None

    while len(rects) < count:
        corner_cells = {cell for rect in rects for cell in _corners(rect)}
        if mode == "disjoint":
            rect = _sample_disjoint_rect(height, width, corner_cells, rects)
        else:
            rect = _sample_extra_rect(height, width, corner_cells, rects, allow_overlap=random.random() < 0.55)
            if rect is None:
                rect = _sample_disjoint_rect(height, width, corner_cells, rects)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)
    count = random.randint(2, min(6, len(INPUT_COLORS)))

    rects = _sample_rectangles(height, width, count)
    if rects is None:
        return None

    counts = _coverage_counts((height, width), rects)
    if not np.any(counts == 1):
        return None

    overlap_cells = int(np.sum(counts >= 2))
    if overlap_cells == 0 and count >= 3 and random.random() < 0.7:
        return None

    area = height * width
    parity_area = int(np.sum(counts % 2 == 1))
    if not (0.08 * area <= parity_area <= 0.62 * area):
        return None
    if overlap_cells > 0 and parity_area >= int(np.sum(counts > 0)):
        return None

    grid = _build_input(height, width, rects)
    expected = np.zeros_like(grid)
    expected[counts % 2 == 1] = GREEN
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
