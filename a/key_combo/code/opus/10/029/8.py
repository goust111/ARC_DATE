from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _blue_width(top_row: np.ndarray) -> int:
    width = 0
    for value in top_row.tolist():
        if int(value) != BLUE:
            break
        width += 1
    return int(width)


def _red_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.size > 0, "expected a red obstacle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED)), "red obstacle must be a filled rectangle"
    return r0, c0, r1, c1


def _segment_endpoints(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == TEAL)]
    assert len(coords) == 2, "expected exactly two teal cells"
    a, b = sorted(coords)
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    assert abs(dr) == 1 and abs(dc) == 1, "teal cells must form a diagonal segment"
    return a, b, (dr, dc)


def _trace_single_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    red_mask: np.ndarray,
    red_bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int]], int]:
    h, w = map(int, grid.shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    phase = 0
    bounces = 0
    seen: set[tuple[int, int, int, int]] = set()
    painted: list[tuple[int, int, int]] = []

    while _in_bounds((h, w), r, c):
        state = (int(r), int(c), int(dr), int(dc))
        if state in seen:
            break
        seen.add(state)

        if int(grid[r, c]) == BLACK:
            painted.append((int(r), int(c), int(phase)))

        nr, nc = int(r + dr), int(c + dc)
        if not _in_bounds((h, w), nr, nc):
            break

        if not bool(red_mask[nr, nc]):
            r, c = nr, nc
            continue

        flip_r = _in_bounds((h, w), r + dr, c) and bool(red_mask[r + dr, c])
        flip_c = _in_bounds((h, w), r, c + dc) and bool(red_mask[r, c + dc])

        if not flip_r and not flip_c:
            r0, c0, r1, c1 = red_bbox
            if nr in (r0, r1):
                flip_r = True
            if nc in (c0, c1):
                flip_c = True
            if not flip_r and not flip_c:
                break

        if flip_r:
            dr = -dr
        if flip_c:
            dc = -dc

        phase += 1
        bounces += 1
        r, c = int(r + dr), int(c + dc)

    return painted, int(bounces)


def _ray_analysis(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    red_mask = g == RED
    red_bbox = _red_bbox(g)
    a, b, delta = _segment_endpoints(g)

    rays = [
        (a, (-delta[0], -delta[1])),
        (b, delta),
    ]

    green_phase: dict[tuple[int, int], int] = {}
    bounce_count = 0
    for endpoint, direction in rays:
        start = (int(endpoint[0] + direction[0]), int(endpoint[1] + direction[1]))
        painted, bounces = _trace_single_ray(g, start, direction, red_mask, red_bbox)
        bounce_count += int(bounces)
        for r, c, phase in painted:
            key = (int(r), int(c))
            prev = green_phase.get(key)
            if prev is None or int(phase) < prev:
                green_phase[key] = int(phase)

    row_first_col: dict[int, int] = {}
    row_first_phase: dict[int, int] = {}
    for (r, c), phase in green_phase.items():
        if r not in row_first_col or int(c) < row_first_col[r]:
            row_first_col[r] = int(c)
            row_first_phase[r] = int(phase)
        elif int(c) == row_first_col[r]:
            row_first_phase[r] = min(int(row_first_phase[r]), int(phase))

    return {
        "blue_width": _blue_width(g[0]),
        "green_phase": green_phase,
        "row_first_col": row_first_col,
        "row_first_phase": row_first_phase,
        "bounce_count": int(bounce_count),
    }


def _expected_blue_width(row: int, natural_blue_width: int, width: int, first_green_col: int | None) -> int:
    natural = min(int(natural_blue_width) + int(row), int(width))
    if first_green_col is None:
        return int(natural)
    return int(min(natural, max(0, int(first_green_col))))


def _left_blue_prefix(row: np.ndarray) -> int:
    count = 0
    for value in row.tolist():
        if int(value) != BLUE:
            break
        count += 1
    return int(count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the top-left blue stack downward while a reflected diagonal ray clips each row at its first green cell.

    Input structure:
    - A black grid has a blue run on the top row starting at column 0, a two-cell teal diagonal segment, and one red filled rectangle.
    - The teal segment defines one diagonal axis, so its two endpoints launch rays outward in opposite 45-degree directions.
    - The red rectangle is the only reflector and is never overwritten.

    Rule:
    - Extend green rays from both teal endpoints along the diagonal line indicated by the segment.
    - Whenever the next diagonal step would enter the red rectangle, flip the blocked direction component and continue; only black traversed cells become green.
    - Then grow the blue top run downward so row i would naturally have width K+i, but stop just before the first green cell on that row if one exists.
    - Keep the original teal segment and the red rectangle unchanged.

    Key insight:
    - The task is the intersection of two independent computations: reflected ray tracing defines a per-row cutoff column, and row-by-row stacking fills blue only up to that moving boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    analysis = _ray_analysis(g)

    for r, c in analysis["green_phase"]:
        if int(out[r, c]) == BLACK:
            out[r, c] = GREEN

    k = int(analysis["blue_width"])
    h, w = map(int, g.shape)
    for r in range(1, h):
        first_green = analysis["row_first_col"].get(r)
        fill_width = _expected_blue_width(r, k, w, first_green)
        for c in range(fill_width):
            if int(out[r, c]) == BLACK:
                out[r, c] = BLUE

    return out


def _build_mode_left_face() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    k = random.choice((2, 3))
    max_d = min(4, w - 6)
    if max_d < k:
        return None
    d = random.randint(k, max_d)

    tr_min = 4
    tr_max = min(h - 6, w - d - 5)
    if tr_min > tr_max:
        return None
    tr = random.randint(tr_min, tr_max)
    tc = tr + d

    rw = random.randint(1, 2)
    rh = random.randint(3, min(5, h - 2))
    r_hit_min = tr + 3
    r_hit_max = min(h - 2, w - rw - 1 - d)
    if r_hit_min > r_hit_max:
        return None
    r_hit = random.randint(r_hit_min, r_hit_max)
    c_rect = d + r_hit

    r0_lo = max(1, r_hit - rh + 1)
    r0_hi = min(r_hit - 1, h - rh)
    if r0_lo > r0_hi:
        return None
    r0 = random.randint(r0_lo, r0_hi)

    grid = np.zeros((h, w), dtype=int)
    grid[0, :k] = BLUE
    grid[tr, tc] = TEAL
    grid[tr + 1, tc + 1] = TEAL
    grid[r0 : r0 + rh, c_rect : c_rect + rw] = RED
    return grid


def _build_mode_bottom_face() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    k = random.choice((2, 3))
    max_d = min(4, w - 7)
    if max_d < k:
        return None
    d = random.randint(k, max_d)

    tr_min = 4
    tr_max = min(h - 6, w - d - 5)
    if tr_min > tr_max:
        return None
    tr = random.randint(tr_min, tr_max)
    tc = tr + d

    rw = 2
    rh = random.randint(3, 5)
    c_rect_min = d + max(rh - 1, 2)
    c_rect_max = min(w - rw - 1, d + tr - 2)
    if c_rect_min > c_rect_max:
        return None
    c_rect = random.randint(c_rect_min, c_rect_max)
    r_hit = c_rect - d
    r0 = r_hit - rh + 1
    if r0 < 0:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :k] = BLUE
    grid[tr, tc] = TEAL
    grid[tr + 1, tc + 1] = TEAL
    grid[r0 : r0 + rh, c_rect : c_rect + rw] = RED
    return grid


def _semantic_ok(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    teal_coords = np.argwhere(inp == TEAL)
    if teal_coords.shape != (2, 2):
        return False
    if _blue_width(inp[0]) not in (2, 3):
        return False

    try:
        analysis = _ray_analysis(inp)
    except AssertionError:
        return False

    if analysis["bounce_count"] < 1:
        return False

    out = main(inp)
    k = int(analysis["blue_width"])
    h, w = map(int, inp.shape)

    clipped_rows = 0
    reflective_clipped_rows = 0
    ineffective_rows = 0
    no_green_rows = 0

    for r in range(h):
        first_green = analysis["row_first_col"].get(r)
        expected_prefix = k if r == 0 else _expected_blue_width(r, k, w, first_green)
        actual_prefix = _left_blue_prefix(out[r])
        if actual_prefix != expected_prefix:
            return False

        if r == 0:
            if not bool(np.all(out[0, :k] == BLUE)):
                return False
            continue

        natural = min(k + r, w)
        if first_green is None:
            no_green_rows += 1
            continue
        if first_green < natural:
            clipped_rows += 1
            if int(analysis["row_first_phase"][r]) > 0:
                reflective_clipped_rows += 1
        else:
            ineffective_rows += 1

    if clipped_rows == 0 or reflective_clipped_rows == 0:
        return False
    if ineffective_rows == 0 or no_green_rows == 0:
        return False

    if not bool(np.all(out[inp == RED] == RED)):
        return False
    if not bool(np.all(out[inp == TEAL] == TEAL)):
        return False

    return True


def generate_input() -> np.ndarray:
    builders = (_build_mode_left_face, _build_mode_bottom_face)
    for _ in range(MAX_ATTEMPTS):
        builder = random.choice(builders)
        grid = builder()
        if grid is not None and _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid dynamic-boundary ray sample")
