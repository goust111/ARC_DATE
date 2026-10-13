import random
from typing import Iterable

import numpy as np


_PRIORITY_ORDER = [2, 1, 3, 4, 5, 6, 7, 8, 9]
_PRIORITY_RANK = {color: rank for rank, color in enumerate(_PRIORITY_ORDER)}
_GEN_RETRIES = 200


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    points: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        points.append((r0, c))
        if r1 != r0:
            points.append((r1, c))
    for r in range(r0 + 1, r1):
        points.append((r, c0))
        if c1 != c0:
            points.append((r, c1))
    return points


def _render_borders(
    shape: tuple[int, int],
    rectangles: Iterable[tuple[int, tuple[int, int, int, int]]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    best_rank = np.full(shape, -1, dtype=int)
    for color, bbox in rectangles:
        assert color in _PRIORITY_RANK, f"Unsupported foreground color: {color}"
        rank = _PRIORITY_RANK[color]
        for r, c in _border_points(bbox):
            if rank >= best_rank[r, c]:
                best_rank[r, c] = rank
                out[r, c] = color
    return out


def _input_points_for_bbox(
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if rng.random() < 0.5:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _apply_transform(grid: np.ndarray, rotations: int, transpose: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if transpose:
        out = out.T
    return np.array(out, copy=True)


def _local_chain_bboxes(
    rng: random.Random,
    pair_count: int,
) -> list[tuple[int, int, int, int]]:
    assert pair_count in (3, 4)

    box_h = rng.randint(6, 10)
    box_w = 1 if rng.random() < 0.35 else rng.randint(3, 6)

    upper_cross = rng.randint(1, max(1, box_h // 2 - 1))
    lower_cross = rng.randint(upper_cross + 2, box_h - 2)

    top_a = upper_cross if rng.random() < 0.25 else -rng.randint(2, 5)
    bottom_c = lower_cross if rng.random() < 0.25 else (box_h - 1) + rng.randint(2, 5)

    left_a = -rng.randint(2, 5)
    right_a = (box_w - 1) + rng.randint(2, 5)
    left_c = -rng.randint(2, 5)
    right_c = (box_w - 1) + rng.randint(5 if pair_count == 4 else 2, 8)

    rects = [
        (top_a, left_a, upper_cross, right_a),
        (0, 0, box_h - 1, box_w - 1),
        (lower_cross, left_c, bottom_c, right_c),
    ]

    if pair_count == 4:
        d_width = 1 if rng.random() < 0.6 else rng.randint(2, 3)
        d_height = rng.randint(3, 6)
        d_left_min = box_w + 2
        d_left_max = right_c - d_width - 1
        if d_left_min > d_left_max:
            raise ValueError("Cannot place fourth rectangle cleanly")
        d_left = rng.randint(d_left_min, d_left_max)
        rects.append((lower_cross, d_left, lower_cross + d_height - 1, d_left + d_width - 1))

    return rects


def _shift_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    min_r = min(r0 for r0, _, _, _ in bboxes)
    min_c = min(c0 for _, c0, _, _ in bboxes)
    max_r = max(r1 for _, _, r1, _ in bboxes)
    max_c = max(c1 for _, _, _, c1 in bboxes)

    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    if base_h > 24 or base_w > 24:
        raise ValueError("Base layout is too large")

    top = rng.randint(1, min(4, 30 - base_h))
    bottom = rng.randint(1, min(4, 30 - base_h - top))
    left = rng.randint(1, min(4, 30 - base_w))
    right = rng.randint(1, min(4, 30 - base_w - left))

    dr = top - min_r
    dc = left - min_c
    shifted = [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in bboxes]
    height = base_h + top + bottom
    width = base_w + left + right
    assert height <= 30 and width <= 30
    return shifted, (height, width)


def _overlap_count(
    shape: tuple[int, int],
    rectangles: list[tuple[int, tuple[int, int, int, int]]],
) -> int:
    cover = np.zeros(shape, dtype=int)
    for _color, bbox in rectangles:
        for r, c in _border_points(bbox):
            cover[r, c] += 1
    return int(np.count_nonzero(cover >= 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each same-colored pixel pair and draw only the border of its implied rectangle, resolving overlaps by a fixed visible color priority.

    Input structure:
    - The grid has a dominant background color, typically black.
    - Every non-background color appears in exactly two cells, and those two cells form one pair.
    - Each pair marks opposite corners of an axis-aligned rectangle; some rectangles may degenerate into a single horizontal or vertical segment.

    Rule:
    - For each non-background color, locate its two pixels and take the minimal axis-aligned rectangle having them as opposite corners.
    - Draw only that rectangle's border in the pair's own color; leave its interior unchanged.
    - If borders from different pairs overlap, keep the later color in the fixed order red < blue < green < yellow < grey < magenta < orange < azure < maroon.

    Key insight:
    - The colored pixels are not isolated dots; they are corner markers for invisible rectangles, and overlap cells are resolved by a consistent color-priority ladder rather than by draw order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "Expected at least one foreground pair"

    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"Color {color} must appear exactly twice"
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        rectangles.append((color, (int(r0), int(c0), int(r1), int(c1))))

    return _render_borders(g.shape, rectangles, bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    pair_count = rng.choice([3, 4])

    start = rng.randint(0, len(_PRIORITY_ORDER) - pair_count)
    colors = _PRIORITY_ORDER[start : start + pair_count]

    try:
        local_bboxes = _local_chain_bboxes(rng, pair_count)
        bboxes, shape = _shift_bboxes(local_bboxes, rng)
    except ValueError:
        return None

    rectangles = list(zip(colors, bboxes))
    if _overlap_count(shape, rectangles) < pair_count - 1:
        return None

    inp = np.full(shape, bg, dtype=int)
    for color, bbox in rectangles:
        p0, p1 = _input_points_for_bbox(bbox, rng)
        if p0 == p1 or inp[p0] != bg or inp[p1] != bg:
            return None
        inp[p0] = color
        inp[p1] = color

    expected = _render_borders(shape, rectangles, bg)

    rotations = rng.randrange(4)
    transpose = rng.random() < 0.5
    inp = _apply_transform(inp, rotations, transpose)
    expected = _apply_transform(expected, rotations, transpose)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
