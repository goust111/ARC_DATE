from __future__ import annotations

import itertools
import random

import numpy as np


SIDE_CYCLE = (1, 9, 7, 5, 3)
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDE_CYCLE)}
VALID_SIDES = (1, 3, 5, 7, 9)
FOREGROUND_COLORS = (1, 2, 3, 4, 5)
BACKGROUND = 0
MAX_ATTEMPTS = 400


def _bbox_from_center(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = center
    radius = side // 2
    return r - radius, c - radius, r + radius, c + radius


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _advance_side(side: int, steps: int) -> int:
    idx = SIDE_TO_INDEX[int(side)]
    return int(SIDE_CYCLE[(idx + int(steps)) % len(SIDE_CYCLE)])


def _paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    r0, c0, r1, c1 = _bbox_from_center(center, side)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _extract_squares(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, f"missing pixels for color {color}"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side_r = int(r1 - r0 + 1)
        side_c = int(c1 - c0 + 1)
        assert side_r == side_c, "each object must have a square bounding box"
        assert side_r in VALID_SIDES and side_r % 2 == 1, "side length must be an odd value in {1,3,5,7,9}"
        assert int(coords.shape[0]) == side_r * side_r, "each object must be a solid square"
        assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)), "square bbox must be solid"
        squares.append(
            {
                "color": int(color),
                "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
                "side": int(side_r),
                "area": int(coords.shape[0]),
            }
        )
    assert 3 <= len(squares) <= 5, "expected 3 to 5 solid odd-sided squares"
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank solid odd-sided squares by area and cycle each one by a rank-dependent number of size steps.

    Input structure:
    - The background is black (0).
    - The foreground contains 3 to 5 separated monochrome solid squares of distinct colors.
    - Every square has an odd side length from {1, 3, 5, 7, 9}, and the side lengths are all different.

    Rule:
    - Detect every solid square and record its color, center, and side length.
    - Sort the squares by descending area, with rank 0 assigned to the largest square.
    - For rank k, advance that square by k+1 steps in the side-length cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1.
    - Clear the canvas to black and redraw each square at the same center with its new side length and original color.

    Key insight:
    - The sort order is not the final output itself; it determines how many resize-cycle steps each square takes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    squares = _extract_squares(g)
    ranked = sorted(squares, key=lambda obj: obj["area"], reverse=True)

    out = np.full(g.shape, BACKGROUND, dtype=int)
    painted: list[tuple[int, int, int, int]] = []
    H, W = g.shape

    for rank, obj in enumerate(ranked):
        new_side = _advance_side(obj["side"], rank + 1)
        r0, c0, r1, c1 = _bbox_from_center(obj["center"], new_side)
        assert 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W, "transformed square must stay inside the grid"
        new_box = (r0, c0, r1, c1)
        assert all(_boxes_separate(new_box, other) for other in painted), "transformed squares must not overlap"
        _paint_square(out, obj["center"], new_side, obj["color"])
        painted.append(new_box)

    return out


def _layout_options(n: int) -> list[tuple[int, int]]:
    return [(rows, cols) for rows in (2, 3) for cols in (2, 3) if rows * cols >= n]


def _sample_centers(n: int) -> tuple[int, int, list[tuple[int, int]]]:
    rows, cols = random.choice(_layout_options(n))
    min_h = 9 * rows
    min_w = 9 * cols
    H = random.randint(min_h, min(30, min_h + 3))
    W = random.randint(min_w, min(30, min_w + 3))
    row_shift = random.randint(0, H - min_h)
    col_shift = random.randint(0, W - min_w)

    slots = [(row_shift + 4 + 9 * r, col_shift + 4 + 9 * c) for r in range(rows) for c in range(cols)]
    centers = random.sample(slots, n)
    return H, W, centers


def _sample_side_subset(n: int) -> list[int]:
    subsets = list(itertools.combinations(VALID_SIDES, n))
    weights = [1 + int(1 in subset) + int(9 in subset) for subset in subsets]
    return list(random.choices(subsets, weights=weights, k=1)[0])


def _build_input(centers: list[tuple[int, int]], sides: list[int], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for center, side, color in zip(centers, sides, colors):
        _paint_square(grid, center, side, color)
    return grid


def _all_reserved_boxes_clear(centers: list[tuple[int, int]]) -> bool:
    reserved = [_bbox_from_center(center, 9) for center in centers]
    for i, box_a in enumerate(reserved):
        for box_b in reserved[i + 1 :]:
            if not _boxes_separate(box_a, box_b):
                return False
    return True


def _expected_output_boxes(centers: list[tuple[int, int]], sides: list[int]) -> list[tuple[int, int, int, int]]:
    ranked_sides = sorted(sides, reverse=True)
    side_to_rank = {side: rank for rank, side in enumerate(ranked_sides)}
    boxes = []
    for center, side in zip(centers, sides):
        new_side = _advance_side(side, side_to_rank[side] + 1)
        boxes.append(_bbox_from_center(center, new_side))
    return boxes


def _valid_output_boxes(boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    H, W = shape
    for r0, c0, r1, c1 in boxes:
        if not (0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W):
            return False
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            if not _boxes_separate(box_a, box_b):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    H, W, centers = _sample_centers(n)
    if not _all_reserved_boxes_clear(centers):
        return None

    sides = _sample_side_subset(n)
    random.shuffle(sides)
    colors = random.sample(FOREGROUND_COLORS, n)

    expected_boxes = _expected_output_boxes(centers, sides)
    if not _valid_output_boxes(expected_boxes, (H, W)):
        return None

    grid = _build_input(centers, sides, colors, (H, W))
    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
