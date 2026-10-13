from __future__ import annotations

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4
import random


Coord = tuple[int, int]


def _unique_marker_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g, return_counts=True)
    markers = [int(v) for v, n in zip(vals, counts) if int(n) == 1 and int(v) != bg]
    assert len(markers) == 1, "expected exactly one non-background marker pixel"
    return markers[0]


def _obstacle_color(g: np.ndarray, bg: int, marker: int) -> int:
    colors = {int(v) for v in np.unique(g)}
    obstacles = colors - {int(bg), int(marker)}
    assert len(obstacles) == 1, "expected one obstacle color besides bg and marker"
    return obstacles.pop()


def _adjacent_to_color(g: np.ndarray, pos: Coord, color: int) -> bool:
    return any(
        in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == int(color)
        for rr, cc in neighbors4(*pos)
    )


def _sightline_to_obstacle(
    g: np.ndarray, start: Coord, obstacle: int, bg: int
) -> tuple[Coord, Coord]:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = start

    for dr, dc in dirs:
        r, c = sr + dr, sc + dc
        previous_bg: Coord | None = None
        while in_bounds(g.shape, r, c):
            color = int(g[r, c])
            if color == obstacle:
                assert previous_bg is not None, "marker should not touch obstacle directly"
                hits.append(((dr, dc), previous_bg))
                break
            assert color == bg, "sightline should cross only background before obstacle"
            previous_bg = (r, c)
            r += dr
            c += dc

    assert len(hits) == 1, "expected one cardinal sightline from marker to obstacle"
    return hits[0]


def _walk_contour(g: np.ndarray, start: Coord, direction: Coord, obstacle: int, bg: int) -> Coord:
    dr, dc = direction
    if dr:
        turns = [(dr, -1), (dr, 1)]
    else:
        turns = [(-1, dc), (1, dc)]

    paths: list[tuple[int, Coord]] = []
    for tr, tc in turns:
        r, c = start
        length = 0
        while True:
            nr, nc = r + tr, c + tc
            if not in_bounds(g.shape, nr, nc):
                break
            if int(g[nr, nc]) != bg:
                break
            if not _adjacent_to_color(g, (nr, nc), obstacle):
                break
            r, c = nr, nc
            length += 1
        paths.append((length, (r, c)))

    max_len = max(length for length, _ in paths)
    endpoints = {pos for length, pos in paths if length == max_len}
    assert len(endpoints) == 1, "contour walk should have one structural endpoint"
    return endpoints.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the single marker to the endpoint reached by sliding along the obstacle contour.

    Input structure:
    - A dominant background color fills the canvas.
    - One single-pixel marker sits away from a connected obstacle region of the
      remaining color.
    - The marker has exactly one horizontal or vertical sightline to the obstacle.

    Rule:
    - Find the last background pixel before the obstacle along that sightline.
    - From that pixel, try the two diagonal turns that continue around the
      obstacle boundary.
    - Move the marker to the endpoint of the turn that remains 4-adjacent to the
      obstacle for the longest distance.

    Key insight:
    The marker is not placed by distance to the obstacle; it first points to the
    obstacle, then follows the visible outside contour until that contour segment ends.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker = _unique_marker_color(g, bg)
    obstacle = _obstacle_color(g, bg, marker)

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape == (1, 2), "expected exactly one marker pixel"
    marker_pos = tuple(map(int, marker_coords[0]))

    direction, contour_start = _sightline_to_obstacle(g, marker_pos, obstacle, bg)
    assert _adjacent_to_color(g, contour_start, obstacle), "contour must start next to obstacle"
    dest = _walk_contour(g, contour_start, direction, obstacle, bg)
    assert int(g[dest]) == bg, "destination should be background"

    out = g.copy()
    out[marker_pos] = bg
    out[dest] = marker
    return out

# --------------- generator (att2) ---------------

BG_COLOR = 7
GEN_RETRIES = 300

# Obstacle shape bounds.
OBSTACLE_HEIGHT_RANGE = (4, 8)
OBSTACLE_WIDTH_RANGE = (3, 8)
OBSTACLE_AREA_RANGE = (6, 33)
MAX_BOUNDARY_LEFT = 7
BOUNDARY_STEP_PROB = 0.55

# Marker sightline and contour-walk bounds.
WALK_LEN_RANGE = (1, 4)
SIGHTLINE_GAP_RANGE = (2, 5)
CANVAS_SIZE_RANGE = (6, 16)


def _gen_boundary(h: int, start_row: int, walk_len: int, start_left: int) -> list[int]:
    lefts = [0] * h
    lefts[start_row] = start_left

    for r in range(start_row - 1, -1, -1):
        if lefts[r + 1] == 0:
            lefts[r] = 0
        else:
            lefts[r] = lefts[r + 1] - random.randint(0, 1)

    for i in range(1, walk_len + 1):
        lefts[start_row + i] = start_left + i

    stop_row = start_row + walk_len + 1
    if stop_row < h:
        lefts[stop_row] = lefts[stop_row - 1]
        for r in range(stop_row + 1, h):
            if lefts[r - 1] < MAX_BOUNDARY_LEFT and random.random() < BOUNDARY_STEP_PROB:
                lefts[r] = lefts[r - 1] + 1
            else:
                lefts[r] = lefts[r - 1]

    return lefts


def _gen_transform(g: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(g, rotations)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    bg = BG_COLOR
    colors = [c for c in range(10) if c != bg]
    obstacle = random.choice(colors)
    marker = random.choice([c for c in colors if c != obstacle])

    h = random.randint(*OBSTACLE_HEIGHT_RANGE)
    walk_len = random.randint(WALK_LEN_RANGE[0], min(WALK_LEN_RANGE[1], h - 3))
    start_row = random.randint(1, h - walk_len - 2)
    gap = random.randint(*SIGHTLINE_GAP_RANGE)

    max_start_left = min(gap, MAX_BOUNDARY_LEFT - walk_len)
    if max_start_left < 1:
        return None
    start_left = random.randint(1, max_start_left)

    lefts = _gen_boundary(h, start_row, walk_len, start_left)
    min_left = min(lefts)
    max_left = max(lefts)
    sum_lefts = sum(lefts)

    w_min = max(
        OBSTACLE_WIDTH_RANGE[0],
        max_left + 1,
        min_left + 3,
        (OBSTACLE_AREA_RANGE[0] + sum_lefts + h - 1) // h,
    )
    w_max = min(OBSTACLE_WIDTH_RANGE[1], (OBSTACLE_AREA_RANGE[1] + sum_lefts) // h)
    if w_min > w_max:
        return None
    w = random.randint(w_min, w_max)
    area = h * w - sum_lefts
    if not (OBSTACLE_AREA_RANGE[0] <= area <= OBSTACLE_AREA_RANGE[1]):
        return None

    c0 = gap - start_left + 1
    min_cols = max(CANVAS_SIZE_RANGE[0], c0 + w)
    possible_cols = [
        cols
        for cols in range(min_cols, CANVAS_SIZE_RANGE[1] + 1)
        if max(CANVAS_SIZE_RANGE[0], h, (2 * area + 2 + cols - 1) // cols) <= CANVAS_SIZE_RANGE[1]
    ]
    if not possible_cols:
        return None
    cols = random.choice(possible_cols)
    min_rows = max(CANVAS_SIZE_RANGE[0], h, (2 * area + 2 + cols - 1) // cols)
    rows = random.randint(min_rows, CANVAS_SIZE_RANGE[1])
    r0 = random.randint(0, rows - h)

    out = np.full((rows, cols), bg, dtype=int)
    for rr, left in enumerate(lefts):
        out[r0 + rr, c0 + left : c0 + w] = obstacle

    contour_start = (r0 + start_row, gap)
    dest = (contour_start[0] + walk_len, contour_start[1] + walk_len)
    marker_pos = (contour_start[0], 0)

    if out[dest] != bg or out[marker_pos] != bg:
        return None

    inp = out.copy()
    inp[marker_pos] = marker
    out[dest] = marker

    rotations = random.randrange(4)
    mirror = random.random() < 0.5
    inp = _gen_transform(inp, rotations, mirror)
    out = _gen_transform(out, rotations, mirror)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 97c75046")
