from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
BOX_HEIGHT_RANGE = (4, 9)
BOX_WIDTH_RANGE = (4, 9)
GENERATION_RETRIES = 400
ALL_COLORS = tuple(range(10))
ANGLE_STEPS = (1, 2, 3)  # 90, 180, 270 degrees counterclockwise


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_from_bbox(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    frame = {(r0, c) for c in range(c0, c1 + 1)}
    frame |= {(r1, c) for c in range(c0, c1 + 1)}
    frame |= {(r, c0) for r in range(r0, r1 + 1)}
    frame |= {(r, c1) for r in range(r0, r1 + 1)}
    return frame


def _rotate_offset(dr: int, dc: int, steps: int) -> Coord:
    for _ in range(steps % 4):
        dr, dc = -dc, dr
    return dr, dc


def _rotate_points(points: set[Coord], pivot: Coord, steps: int) -> set[Coord]:
    pr, pc = pivot
    rotated = set()
    for r, c in points:
        dr, dc = r - pr, c - pc
        nr_off, nc_off = _rotate_offset(dr, dc, steps)
        rotated.add((pr + nr_off, pc + nc_off))
    return rotated


def _all_in_bounds(shape: tuple[int, int], points: set[Coord]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in points)


def _sample_markers(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    if random.random() < 0.5:
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        return random.sample(corners, 3)

    corner_name = random.choice(("tl", "tr", "bl", "br"))
    edge_col = random.randint(c0 + 1, c1 - 1)
    edge_row = random.randint(r0 + 1, r1 - 1)
    corner_row = r0 if corner_name in ("tl", "tr") else r1
    corner_col = c0 if corner_name in ("tl", "bl") else c1
    other_row = r1 if corner_row == r0 else r0
    other_col = c1 if corner_col == c0 else c0
    return [(corner_row, corner_col), (other_row, edge_col), (edge_row, other_col)]


def _legal_rotations(frame: set[Coord], pivot: Coord, shape: tuple[int, int]) -> dict[int, set[Coord]]:
    legal: dict[int, set[Coord]] = {}
    for steps in ANGLE_STEPS:
        rotated = _rotate_points(frame, pivot, steps)
        if _all_in_bounds(shape, rotated):
            legal[steps] = rotated
    return legal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle from three markers, rotate it around the pivot, and draw the only in-bounds rotated frame.

    Input structure:
    - The grid has a uniform background, exactly three same-color marker pixels, and one different-color pivot pixel.
    - The three markers lie on the border of their shared bounding box, while the pivot lies outside that box.
    - The rectangle implied by the marker bounding box is not drawn in the input.

    Rule:
    - Compute the bounding box of the three marker pixels and treat its one-cell-thick border as the hidden rectangle.
    - Rotate that border around the pivot by 90, 180, and 270 degrees counterclockwise.
    - Exactly one rotated border stays fully inside the grid; draw it in the marker color, then preserve the original markers and pivot on top.

    Key insight:
    - The missing rectangle is defined only by sparse control points, and the correct rotation is inferred solely from the unique in-bounds placement constraint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    fg_coords = np.argwhere(g != bg)
    assert fg_coords.shape == (4, 2), "expected exactly three markers and one pivot"

    color_counts = Counter(map(int, g[g != bg]))
    assert sorted(color_counts.values()) == [1, 3], "expected one singleton pivot color and one triple marker color"
    marker_color = next(color for color, count in color_counts.items() if count == 3)
    pivot_color = next(color for color, count in color_counts.items() if count == 1)
    assert marker_color != pivot_color

    marker_coords = np.argwhere(g == marker_color)
    pivot_coords = np.argwhere(g == pivot_color)
    assert marker_coords.shape == (3, 2)
    assert pivot_coords.shape == (1, 2)

    markers = [tuple(map(int, rc)) for rc in marker_coords]
    pivot = tuple(map(int, pivot_coords[0]))

    box = _bbox_coords(marker_coords)
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1, "marker bbox must be a nondegenerate rectangle"
    assert not (r0 <= pivot[0] <= r1 and c0 <= pivot[1] <= c1), "pivot must lie outside the marker bbox"

    for r, c in markers:
        assert r in (r0, r1) or c in (c0, c1), "markers must lie on the rectangle border"

    frame = _frame_from_bbox(box)
    legal = _legal_rotations(frame, pivot, g.shape)
    assert len(legal) == 1, "expected exactly one in-bounds non-zero quarter-turn rotation"

    out = g.copy()
    rotated_frame = next(iter(legal.values()))
    for r, c in rotated_frame:
        out[r, c] = marker_color

    for r, c in markers:
        out[r, c] = marker_color
    pr, pc = pivot
    out[pr, pc] = pivot_color
    return out


def _gen_attempt(desired_steps: int) -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    box_h = random.randint(*BOX_HEIGHT_RANGE)
    box_w = random.randint(*BOX_WIDTH_RANGE)
    if box_h > height - 2 or box_w > width - 2:
        return None

    r0 = random.randint(1, height - box_h - 1)
    c0 = random.randint(1, width - box_w - 1)
    box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    frame = _frame_from_bbox(box)
    markers = _sample_markers(box)
    marker_set = set(markers)

    candidates: list[tuple[Coord, set[Coord]]] = []
    for pr in range(height):
        for pc in range(width):
            if r0 <= pr <= box[2] and c0 <= pc <= box[3]:
                continue
            pivot = (pr, pc)
            legal = _legal_rotations(frame, pivot, (height, width))
            if set(legal) != {desired_steps}:
                continue
            rotated = legal[desired_steps]
            if rotated.intersection(marker_set):
                continue
            if len(rotated - marker_set) < 6:
                continue
            candidates.append((pivot, rotated))

    if not candidates:
        return None

    marker_color, pivot_color, bg = random.sample(ALL_COLORS, 3)
    pivot, _rotated = random.choice(candidates)
    grid = np.full((height, width), bg, dtype=int)
    for r, c in markers:
        grid[r, c] = marker_color
    grid[pivot] = pivot_color

    assert tuple(np.argwhere(grid == marker_color).shape) == (3, 2)
    assert tuple(np.argwhere(grid == pivot_color).shape) == (1, 2)
    return grid


def generate_input() -> np.ndarray:
    desired_steps = random.choice(ANGLE_STEPS)
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(desired_steps)
        if grid is None:
            continue

        out = main(grid)
        assert isinstance(grid, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert grid.ndim == 2 and out.ndim == 2
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
