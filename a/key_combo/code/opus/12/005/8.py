from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GRID_MIN = 17
GRID_MAX = 29
STEP_CHOICES = (3, 4)
GEN_ATTEMPTS = 500
MIN_SPIRAL_EXTENSIONS = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _chebyshev_grid(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.maximum(np.abs(rows - int(center[0])), np.abs(cols - int(center[1])))


def _diagonal_mask(
    shape: tuple[int, int], center: tuple[int, int], diag_sign: int
) -> np.ndarray:
    rows, cols = np.indices(shape)
    return (cols - int(center[1])) == int(diag_sign) * (rows - int(center[0]))


def _parse_input(
    grid: np.ndarray,
) -> tuple[tuple[int, int], int, int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    red_coords = np.argwhere(g == RED)
    blue_coords = np.argwhere(g == BLUE)

    assert red_coords.shape == (3, 2), "expected exactly three red seeds"
    assert blue_coords.shape == (1, 2), "expected exactly one blue marker"

    red_coords = red_coords[np.argsort(red_coords[:, 0])]
    center = tuple(map(int, red_coords[1]))
    top = tuple(map(int, red_coords[0]))
    bottom = tuple(map(int, red_coords[2]))

    d_top = max(abs(top[0] - center[0]), abs(top[1] - center[1]))
    d_bottom = max(abs(bottom[0] - center[0]), abs(bottom[1] - center[1]))
    assert d_top == d_bottom and d_top > 0, "red seeds must be equally spaced"

    diag_sign = 1 if bottom[1] > center[1] else -1
    assert top == (center[0] - d_top, center[1] - diag_sign * d_top)
    assert bottom == (center[0] + d_top, center[1] + diag_sign * d_top)

    blue = tuple(map(int, blue_coords[0]))
    return center, int(d_top), int(diag_sign), blue


def _frame_mask(shape: tuple[int, int], center: tuple[int, int], d: int) -> np.ndarray:
    dist = _chebyshev_grid(shape, center)
    return (dist > 0) & (dist % int(d) == 0)


def _ring_mask(
    shape: tuple[int, int], center: tuple[int, int], d: int, ring_index: int
) -> np.ndarray:
    dist = _chebyshev_grid(shape, center)
    inner = int(ring_index) * int(d)
    outer = (int(ring_index) + 1) * int(d)
    return (dist > inner) & (dist < outer)


def _visible_frame_count(shape: tuple[int, int], center: tuple[int, int], d: int) -> int:
    max_dist = max(
        int(center[0]),
        int(shape[0] - 1 - center[0]),
        int(center[1]),
        int(shape[1] - 1 - center[1]),
    )
    return int(max_dist // int(d))


def _visible_ring_indices(
    shape: tuple[int, int], center: tuple[int, int], d: int, diag_sign: int
) -> list[int]:
    frame_count = _visible_frame_count(shape, center, d)
    diag = _diagonal_mask(shape, center, diag_sign)
    rings: list[int] = []
    for ring_index in range(1, frame_count):
        ring = _ring_mask(shape, center, d, ring_index) & ~diag
        if np.any(ring):
            rings.append(int(ring_index))
    return rings


def _paint_spiral(
    grid: np.ndarray, ring: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, int]:
    out = np.array(grid, copy=True)
    h, w = out.shape
    cur_r, cur_c = map(int, start)
    out[cur_r, cur_c] = BLUE

    step_len = 1
    dir_index = 0
    consecutive_failures = 0
    painted = 0

    while consecutive_failures < 4:
        for _ in range(2):
            dr, dc = DIRS[dir_index % 4]
            painted_this_dir = 0
            for _step in range(step_len):
                nr = cur_r + dr
                nc = cur_c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if not ring[nr, nc]:
                    break
                if int(out[nr, nc]) != BLACK:
                    break
                out[nr, nc] = BLUE
                cur_r, cur_c = int(nr), int(nc)
                painted += 1
                painted_this_dir += 1

            if painted_this_dir == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            dir_index += 1
            if consecutive_failures >= 4:
                break
        step_len += 1

    return out, int(painted)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw clipped concentric red square borders and fill the blue marker's ring with a constrained square spiral.

    Input structure:
    - The grid is mostly black and contains exactly three red seed pixels on one 45-degree diagonal.
    - The middle red seed is the common center, and the two outer seeds are equally spaced from it.
    - A single blue marker lies strictly inside one annular region between two adjacent concentric square borders, and not on the red diagonal.

    Rule:
    - Recover the center and spacing d from the three red seeds.
    - Paint red square borders centered at the middle seed for Chebyshev radii d, 2d, 3d, ... with automatic clipping at the grid boundary.
    - Find which ring between adjacent borders contains the blue marker.
    - Starting at that blue marker, draw a blue square spiral inside the ring using step lengths 1, 1, 2, 2, 3, 3, ... in up, right, down, left order; each direction truncates immediately when the next cell would hit a red border, the grid boundary, or an existing blue cell, and the process stops after four consecutive directions add no new blue cells.

    Key insight:
    - The three red seeds do not just mark a diagonal: they encode the center and spacing of a whole clipped concentric-square system, and the blue path is a standard square spiral only after being constrained to the single annular band selected by the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center, d, diag_sign, blue = _parse_input(g)
    dist = _chebyshev_grid(g.shape, center)

    blue_dist = int(dist[blue[0], blue[1]])
    assert blue_dist > d and blue_dist % d != 0, "blue marker must lie in a ring"
    assert not _diagonal_mask(g.shape, center, diag_sign)[blue[0], blue[1]]

    ring_index = int(blue_dist // d)
    ring = _ring_mask(g.shape, center, d, ring_index)

    out = np.array(g, copy=True)
    out[_frame_mask(g.shape, center, d)] = RED
    out, _ = _paint_spiral(out, ring, blue)
    return out


def _build_input(
    shape: tuple[int, int],
    center: tuple[int, int],
    d: int,
    diag_sign: int,
    blue: tuple[int, int],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    center_r, center_c = map(int, center)
    grid[center_r, center_c] = RED
    grid[center_r - d, center_c - diag_sign * d] = RED
    grid[center_r + d, center_c + diag_sign * d] = RED
    grid[blue[0], blue[1]] = BLUE
    return grid


def _sample_shape() -> tuple[int, int]:
    if random.random() < 0.45:
        side = random.randint(GRID_MIN, GRID_MAX)
        return int(side), int(side)
    return random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)


def _second_frame_fully_inside(
    shape: tuple[int, int], center: tuple[int, int], d: int
) -> bool:
    r, c = map(int, center)
    return (
        r - 2 * d >= 0
        and r + 2 * d < shape[0]
        and c - 2 * d >= 0
        and c + 2 * d < shape[1]
    )


def _candidate_centers(
    shape: tuple[int, int], d: int, prefer_clipped_second_frame: bool
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    matching: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []

    for r in range(d, h - d):
        for c in range(d, w - d):
            center = (int(r), int(c))
            if _visible_frame_count(shape, center, d) < 2:
                continue
            fallback.append(center)
            second_inside = _second_frame_fully_inside(shape, center, d)
            if prefer_clipped_second_frame == (not second_inside):
                matching.append(center)

    return matching if matching else fallback


def _pick_ring_index(
    shape: tuple[int, int], center: tuple[int, int], d: int, diag_sign: int
) -> int:
    rings = _visible_ring_indices(shape, center, d, diag_sign)
    assert rings
    if len(rings) == 1:
        return int(rings[0])
    emphasized = [rings[0], rings[-1], random.choice(rings)]
    return int(random.choice(emphasized))


def _ring_start_candidates(
    shape: tuple[int, int],
    center: tuple[int, int],
    d: int,
    diag_sign: int,
    ring_index: int,
) -> list[tuple[int, int]]:
    coords = _all_ring_candidates(shape, center, d, diag_sign, ring_index)
    random.shuffle(coords)

    dist = _chebyshev_grid(shape, center)
    inner_band = [rc for rc in coords if int(dist[rc[0], rc[1]]) == ring_index * d + 1]
    outer_band = [rc for rc in coords if int(dist[rc[0], rc[1]]) == (ring_index + 1) * d - 1]
    middle_band = [
        rc
        for rc in coords
        if ring_index * d + 1 < int(dist[rc[0], rc[1]]) < (ring_index + 1) * d - 1
    ]

    groups = [band for band in (inner_band, outer_band, middle_band) if band]
    assert groups
    group = random.choice(groups)
    return group


def _all_ring_candidates(
    shape: tuple[int, int],
    center: tuple[int, int],
    d: int,
    diag_sign: int,
    ring_index: int,
) -> list[tuple[int, int]]:
    dist = _chebyshev_grid(shape, center)
    ring = _ring_mask(shape, center, d, ring_index)
    diag = _diagonal_mask(shape, center, diag_sign)
    candidates = ring & ~diag
    return [tuple(map(int, rc)) for rc in np.argwhere(candidates)]


def _output_is_unique_for_blue_start(grid: np.ndarray) -> bool:
    center, d, diag_sign, blue = _parse_input(grid)
    expected = main(grid)
    shape = grid.shape

    for ring_index in _visible_ring_indices(shape, center, d, diag_sign):
        for other_blue in _all_ring_candidates(shape, center, d, diag_sign, ring_index):
            if other_blue == blue:
                continue
            other = _build_input(shape, center, d, diag_sign, other_blue)
            if np.array_equal(main(other), expected):
                return False
    return True


def _is_valid_instance(grid: np.ndarray) -> bool:
    center, d, diag_sign, blue = _parse_input(grid)
    shape = grid.shape

    if _visible_frame_count(shape, center, d) < 2:
        return False

    diag = _diagonal_mask(shape, center, diag_sign)
    if bool(diag[blue[0], blue[1]]):
        return False

    dist = _chebyshev_grid(shape, center)
    blue_dist = int(dist[blue[0], blue[1]])
    if blue_dist <= d or blue_dist % d == 0:
        return False

    ring_index = int(blue_dist // d)
    ring = _ring_mask(shape, center, d, ring_index)
    if not bool(ring[blue[0], blue[1]]):
        return False

    out = main(grid)
    blue_mask = out == BLUE
    if int(np.count_nonzero(blue_mask)) - 1 < MIN_SPIRAL_EXTENSIONS:
        return False
    if not np.all(blue_mask <= ring):
        return False
    if int(np.count_nonzero(out == RED)) <= 3:
        return False

    return _output_is_unique_for_blue_start(grid)


def _gen_attempt() -> np.ndarray | None:
    shape = _sample_shape()
    d = int(random.choice(STEP_CHOICES))
    diag_sign = random.choice((-1, 1))
    prefer_clipped = random.random() < 0.5

    centers = _candidate_centers(shape, d, prefer_clipped)
    if not centers:
        return None
    center = random.choice(centers)
    ring_index = _pick_ring_index(shape, center, d, diag_sign)

    blue_choices = _ring_start_candidates(shape, center, d, diag_sign, ring_index)
    if not blue_choices:
        return None

    random.shuffle(blue_choices)
    for blue in blue_choices[: min(len(blue_choices), 16)]:
        grid = _build_input(shape, center, d, diag_sign, blue)
        if _is_valid_instance(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
