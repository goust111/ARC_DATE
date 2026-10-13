from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    length = max(abs(dr), abs(dc))
    assert length > 0
    assert abs(dr) in (0, length)
    assert abs(dc) in (0, length)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(length + 1)]


def _pair_and_geometry(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    bg = _dominant_color(grid)
    counts = Counter(map(int, grid[grid != bg]))
    repeated = [color for color, count in counts.items() if count == 2]
    singles = [color for color, count in counts.items() if count == 1]
    assert len(repeated) == 1, "expected exactly one repeated foreground color"
    assert len(singles) + 1 == len(counts), "all other foreground colors must appear once"

    pair_color = int(repeated[0])
    pair_coords = sorted(tuple(map(int, coord)) for coord in np.argwhere(grid == pair_color))
    assert len(pair_coords) == 2
    p, q = pair_coords

    dr = q[0] - p[0]
    dc = q[1] - p[1]
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    straight_len = abs(abs_dr - abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)

    if abs_dr > abs_dc:
        straight_vec = (sr * straight_len, 0)
    else:
        straight_vec = (0, sc * straight_len)
    diag_vec = (sr * diag_len, sc * diag_len)
    return pair_color, p, q, straight_vec, diag_vec


def _classify_point(
    point: tuple[int, int],
    origin: tuple[int, int],
    side_u: tuple[int, int],
    side_v: tuple[int, int],
) -> str:
    wr = point[0] - origin[0]
    wc = point[1] - origin[1]
    ur, uc = side_u
    vr, vc = side_v

    det = ur * vc - uc * vr
    assert det != 0
    s_num = wr * vc - wc * vr
    t_num = ur * wc - uc * wr
    if det < 0:
        det, s_num, t_num = -det, -s_num, -t_num

    if 0 <= s_num <= det and 0 <= t_num <= det:
        if s_num in (0, det) or t_num in (0, det):
            return "boundary"
        return "inside"
    return "outside"


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the invisible parallelogram from the repeated-color diagonal pair and use it to filter the scattered pixels.

    Input structure:
    - A black background contains scattered single-pixel foreground dots.
    - Exactly one foreground color appears twice; every other foreground color appears once.
    - The repeated-color pair differs in both row and column, and those two offsets have unequal magnitudes.

    Rule:
    - Treat the repeated-color pair as opposite corners of an implied parallelogram.
    - Decompose their offset into one 45-degree diagonal side and one axis-aligned side, then use those as the parallelogram's adjacent sides.
    - Do not draw that boundary. Keep dots on the boundary path unchanged, recolor strictly interior dots to the repeated color, erase exterior dots to the background, and preserve the two repeated-color corner dots.

    Key insight:
    - The crucial object is an unseen boundary inferred from the only duplicated color pair; that hidden parallelogram is used only as a classifier, not as an output shape to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    pair_color, origin, _opposite, side_u, side_v = _pair_and_geometry(g)

    out = np.full_like(g, bg)
    for r, c in np.argwhere(g != bg):
        point = (int(r), int(c))
        color = int(g[r, c])
        kind = _classify_point(point, origin, side_u, side_v)
        if color == pair_color:
            out[r, c] = pair_color
        elif kind == "boundary":
            out[r, c] = color
        elif kind == "inside":
            out[r, c] = pair_color
    return out


def _choose_lengths() -> tuple[int, int]:
    diag_len = random.randint(3, 6)
    straight_len = random.randint(2, 6)
    return diag_len, straight_len


def _sample_geometry() -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    diag_len, straight_len = _choose_lengths()
    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))
    axis_is_row = bool(random.getrandbits(1))

    if axis_is_row:
        side_u = (sr * straight_len, 0)
        side_v = (sr * diag_len, sc * diag_len)
    else:
        side_u = (0, sc * straight_len)
        side_v = (sr * diag_len, sc * diag_len)

    relative_corners = [(0, 0), side_u, side_v, (side_u[0] + side_v[0], side_u[1] + side_v[1])]
    rel_rows = [r for r, _ in relative_corners]
    rel_cols = [c for _, c in relative_corners]
    span_h = max(rel_rows) - min(rel_rows) + 1
    span_w = max(rel_cols) - min(rel_cols) + 1

    top_margin = random.randint(1, 6)
    bottom_margin = random.randint(1, 6)
    left_margin = random.randint(1, 6)
    right_margin = random.randint(1, 6)
    height = span_h + top_margin + bottom_margin
    width = span_w + left_margin + right_margin
    if height > 30 or width > 30:
        return _sample_geometry()

    origin = (top_margin - min(rel_rows), left_margin - min(rel_cols))
    opposite = (origin[0] + side_u[0] + side_v[0], origin[1] + side_u[1] + side_v[1])
    straight_corner = (origin[0] + side_u[0], origin[1] + side_u[1])
    diag_corner = (origin[0] + side_v[0], origin[1] + side_v[1])
    return (height, width), origin, opposite, straight_corner, diag_corner, side_u, side_v


def _bbox_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _all_classified_cells(
    shape: tuple[int, int],
    origin: tuple[int, int],
    side_u: tuple[int, int],
    side_v: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    height, width = shape
    classes = {"inside": [], "boundary": [], "outside": []}
    for r in range(height):
        for c in range(width):
            point = (r, c)
            classes[_classify_point(point, origin, side_u, side_v)].append(point)
    return classes


def _sample_points(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert len(candidates) >= count
    return random.sample(candidates, count)


def _build_expected_output(
    grid: np.ndarray,
    pair_color: int,
    interior_points: list[tuple[int, int]],
    boundary_points: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full_like(grid, BACKGROUND)
    for r, c in boundary_points:
        out[r, c] = int(grid[r, c])
    for r, c in interior_points:
        out[r, c] = pair_color
    for r, c in np.argwhere(grid == pair_color):
        out[int(r), int(c)] = pair_color
    return out


def _attempt_input() -> np.ndarray | None:
    shape, origin, opposite, straight_corner, diag_corner, side_u, side_v = _sample_geometry()
    all_corners = [origin, opposite, straight_corner, diag_corner]
    pair_color = random.choice(PALETTE)

    boundary_axis = sorted(
        (
            set(_segment_cells(origin, straight_corner))
            | set(_segment_cells(diag_corner, opposite))
        )
        - {origin, opposite, straight_corner, diag_corner}
    )
    boundary_diag = sorted(
        (
            set(_segment_cells(origin, diag_corner))
            | set(_segment_cells(straight_corner, opposite))
        )
        - {origin, opposite, straight_corner, diag_corner}
    )

    classes = _all_classified_cells(shape, origin, side_u, side_v)
    interior_all = classes["inside"]
    boundary_all = classes["boundary"]
    outside_all = classes["outside"]

    bbox = _bbox_points(all_corners)
    bbox_outside = [
        point
        for point in outside_all
        if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
    ]
    far_outside = [
        point
        for point in outside_all
        if point[0] < bbox[0] or point[0] > bbox[2] or point[1] < bbox[1] or point[1] > bbox[3]
    ]

    if not (boundary_axis and boundary_diag and interior_all and bbox_outside and far_outside):
        return None

    boundary_points = _sample_points(boundary_axis, 1) + _sample_points(boundary_diag, 1)
    boundary_pool = sorted(set(boundary_all) - set(all_corners) - set(boundary_points))
    extra_boundary_count = random.randint(0, min(1, len(boundary_pool)))
    if extra_boundary_count:
        boundary_points += _sample_points(boundary_pool, extra_boundary_count)

    interior_count = random.randint(2, min(3, len(interior_all)))
    interior_points = _sample_points(interior_all, interior_count)

    bbox_outside_count = random.randint(1, min(2, len(bbox_outside)))
    outside_points = _sample_points(bbox_outside, bbox_outside_count)
    outside_points += _sample_points(far_outside, 1)

    used_positions = set(all_corners) | set(boundary_points) | set(interior_points) | set(outside_points)
    if len(used_positions) != 4 + len(boundary_points) + len(interior_points) + len(outside_points):
        return None

    other_positions = boundary_points + interior_points + outside_points
    if len(other_positions) > len(PALETTE) - 1:
        return None

    other_colors = [color for color in PALETTE if color != pair_color]
    random.shuffle(other_colors)

    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[origin] = pair_color
    grid[opposite] = pair_color
    for point, color in zip(other_positions, other_colors):
        grid[point] = color

    expected = _build_expected_output(grid, pair_color, interior_points, boundary_points)

    counts = Counter(map(int, grid[grid != BACKGROUND]))
    if sorted(counts.values()) != [1] * (len(counts) - 1) + [2]:
        return None

    for point in boundary_points:
        if _classify_point(point, origin, side_u, side_v) != "boundary":
            return None
    for point in interior_points:
        if _classify_point(point, origin, side_u, side_v) != "inside":
            return None
    for point in outside_points:
        if _classify_point(point, origin, side_u, side_v) != "outside":
            return None

    if not any(bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3] for r, c in outside_points):
        return None
    if not any(expected[r, c] == int(grid[r, c]) and int(grid[r, c]) != pair_color for r, c in boundary_points):
        return None
    if not all(expected[r, c] == pair_color for r, c in interior_points):
        return None
    if not all(expected[r, c] == BACKGROUND for r, c in outside_points):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-parallelogram sample")
