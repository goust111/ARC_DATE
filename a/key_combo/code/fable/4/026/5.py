from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

GRID_SIDE_RANGE = (18, 30)
CENTER_MARGIN = 6
MAX_ATTEMPTS = 300

REQUIRED_LAPS = (1, 2, 3)
MIN_REQUIRED_LAP_CELLS = 4
PER_LAP_PICK_RANGE = (2, 5)
EXTRA_LAP_PICK_RANGE = (1, 2)
GAP_PICK_RANGE = (10, 18)
NOISE_PICK_RANGE = (2, 8)

Coord = tuple[int, int]


def _spiral_lap_map(shape: tuple[int, int], center: Coord) -> dict[Coord, int]:
    h, w = map(int, shape)
    r, c = map(int, center)
    lap_by_cell: dict[Coord, int] = {(r, c): 0}
    min_r = max_r = r
    min_c = max_c = c
    segment_index = 0

    while True:
        dr, dc = DIRS[segment_index % 4]
        step_length = 2 * (segment_index // 2 + 1)
        lap = segment_index // 4 + 1
        for _ in range(step_length):
            r += dr
            c += dc
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)
            if 0 <= r < h and 0 <= c < w and (r, c) not in lap_by_cell:
                lap_by_cell[(r, c)] = lap
        segment_index += 1
        if segment_index % 4 == 0 and min_r < 0 and max_r >= h and min_c < 0 and max_c >= w:
            return lap_by_cell


def _lap_buckets(shape: tuple[int, int], center: Coord) -> tuple[dict[Coord, int], dict[int, list[Coord]]]:
    lap_map = _spiral_lap_map(shape, center)
    buckets: dict[int, list[Coord]] = {}
    for cell, lap in lap_map.items():
        if lap == 0:
            continue
        buckets.setdefault(lap, []).append(cell)
    return lap_map, buckets


def _pick_count(pool_size: int, low: int, high: int) -> int:
    if pool_size <= 0:
        return 0
    if pool_size <= low:
        return pool_size
    return random.randint(low, min(high, pool_size))


def _gap_candidates(shape: tuple[int, int], path_cells: set[Coord], center: Coord) -> list[Coord]:
    h, w = map(int, shape)
    gaps: set[Coord] = set()
    for r, c in path_cells:
        for dr, dc in DIRS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) != center and (rr, cc) not in path_cells:
                gaps.add((rr, cc))
    return list(gaps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the gray dots that lie on the hidden sparse square spiral rooted at the red marker.

    Input structure:
    - A black background grid contains exactly one red marker pixel and many scattered gray pixels.
    - Starting at the red marker, an unpainted square spiral follows step lengths 2, 2, 4, 4, 6, 6, ... in fixed up, right, down, left order.
    - Gray dots may lie either on spiral cells or in the one-cell gaps between spiral arms.

    Rule:
    - Reconstruct the clipped spiral path inside the grid, counting each full up-right-down-left cycle as one lap.
    - Recolor every gray pixel on an odd-numbered lap to red and every gray pixel on an even-numbered lap to blue.
    - Leave gray pixels off the path, the original red marker, and all background cells unchanged.

    Key insight:
    - The visible output is determined by membership on an invisible sparse-arm spiral, not by distance from the marker; nearby gap pixels stay gray unless the spiral actually passes through them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_coords = np.argwhere(g == RED)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    center = tuple(map(int, marker_coords[0]))

    lap_by_cell = _spiral_lap_map(g.shape, center)
    out = g.copy()
    for r, c in np.argwhere(g == GRAY):
        lap = lap_by_cell.get((int(r), int(c)))
        if lap is None or lap == 0:
            continue
        out[r, c] = RED if lap % 2 == 1 else BLUE
    return out


def _sample_center(h: int, w: int) -> Coord:
    return (
        random.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1),
        random.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1),
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    center = _sample_center(h, w)
    lap_map, lap_buckets = _lap_buckets((h, w), center)

    if any(len(lap_buckets.get(lap, ())) < MIN_REQUIRED_LAP_CELLS for lap in REQUIRED_LAPS):
        return None

    visible_laps = sorted(lap_buckets)
    extra_laps = [lap for lap in visible_laps if lap > REQUIRED_LAPS[-1]]
    chosen_laps = list(REQUIRED_LAPS)
    extra_count = _pick_count(len(extra_laps), *EXTRA_LAP_PICK_RANGE)
    if extra_count:
        chosen_laps.extend(random.sample(extra_laps, extra_count))

    on_path: set[Coord] = set()
    for lap in chosen_laps:
        cells = lap_buckets[lap]
        count = _pick_count(len(cells), *PER_LAP_PICK_RANGE)
        on_path.update(random.sample(cells, count))

    path_cells = set(lap_map)
    gap_candidates = [cell for cell in _gap_candidates((h, w), path_cells, center) if cell not in on_path]
    gap_count = _pick_count(len(gap_candidates), *GAP_PICK_RANGE)
    off_path = set(random.sample(gap_candidates, gap_count))

    blocked = path_cells | off_path | {center}
    noise_candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    noise_count = _pick_count(len(noise_candidates), *NOISE_PICK_RANGE)
    off_path.update(random.sample(noise_candidates, noise_count))

    grid = np.full((h, w), BLACK, dtype=int)
    grid[center] = RED
    for r, c in on_path | off_path:
        grid[r, c] = GRAY

    out = main(grid)
    odd_hits = int(np.sum((grid == GRAY) & (out == RED)))
    even_hits = int(np.sum((grid == GRAY) & (out == BLUE)))
    unchanged = int(np.sum((grid == GRAY) & (out == GRAY)))
    if odd_hits < 3 or even_hits < 2 or unchanged < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-spiral input")
