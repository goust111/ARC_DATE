import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
LINE_COLORS = {BLUE, RED}
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 18)
LINE_LENGTH_RANGE = (4, 8)
MIN_EXTRA_EXTENSION = 3
GENERATION_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rotate_cw(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _distance_to_edge(shape: tuple[int, int], pivot: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pivot
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    if dc == 1:
        return w - 1 - c
    raise AssertionError("direction must be cardinal")


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pivot_coords = np.argwhere(g == GREY)
    assert pivot_coords.shape == (1, 2), "expected exactly one grey pivot"
    pivot = tuple(map(int, pivot_coords[0]))

    adjacent_dirs = []
    for dr, dc in CARDINALS:
        nr = pivot[0] + dr
        nc = pivot[1] + dc
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) in LINE_COLORS:
            adjacent_dirs.append((dr, dc))
    assert len(adjacent_dirs) == 1, "expected exactly one line direction from the pivot"
    direction = adjacent_dirs[0]

    coords: list[tuple[int, int]] = []
    colors: list[int] = []
    step = 1
    while True:
        r = pivot[0] + step * direction[0]
        c = pivot[1] + step * direction[1]
        if not _in_bounds(g.shape, r, c):
            break
        color = int(g[r, c])
        if color not in LINE_COLORS:
            break
        coords.append((r, c))
        colors.append(color)
        step += 1

    assert coords, "expected a non-empty line segment"
    assert len(coords) >= 2, "expected an alternating period-2 segment"

    c1 = colors[0]
    c2 = colors[1]
    assert c1 != c2, "expected two alternating colors"
    for i, color in enumerate(colors):
        expected = c1 if i % 2 == 0 else c2
        assert color == expected, "line colors must alternate with period 2"

    colored_mask = (g != BLACK) & (g != GREY)
    colored_coords = {tuple(map(int, rc)) for rc in np.argwhere(colored_mask)}
    assert colored_coords == set(coords), "unexpected extra colored pixels outside the segment"

    return {
        "pivot": pivot,
        "direction": direction,
        "length": len(coords),
        "c1": c1,
        "c2": c2,
        "coords": coords,
    }


def _expected_output_from_info(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in info["coords"]:
        out[r, c] = GREEN

    pivot = info["pivot"]
    c1 = info["c1"]
    c2 = info["c2"]
    rot_dr, rot_dc = _rotate_cw(*info["direction"])
    step = 1
    while True:
        r = pivot[0] + step * rot_dr
        c = pivot[1] + step * rot_dc
        if not _in_bounds(out.shape, r, c):
            break
        out[r, c] = c1 if step % 2 == 1 else c2
        step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the pivot-attached alternating segment clockwise, then continue its period to the new edge.

    Input structure:
    - The grid has a black background, exactly one grey pivot, and one straight
      horizontal or vertical segment attached to the pivot at one end.
    - The segment uses only blue and red, alternating with period 2 starting
      from the cell adjacent to the pivot.
    - No other non-black pixels are present.

    Rule:
    - Find the grey pivot and the unique adjacent colored cell to determine the
      segment direction away from the pivot.
    - Read the segment length and its two-color period in pivot-to-end order.
    - Recolor the original segment green.
    - Rotate the direction 90 degrees clockwise around the pivot and draw along
      that new ray all the way to the grid boundary, repeating the same period
      with the first rotated cell taking the original first color.

    Key insight:
    - The transformation depends on two linked pieces of information: the pivot
      changes the segment's orientation, and the segment's internal period must
      then be continued in the rotated coordinate system rather than merely
      copying the original cells.
    """
    info = _parse_instance(np.asarray(grid, dtype=int))
    return _expected_output_from_info(grid, info)


def _build_input(
    height: int,
    width: int,
    pivot: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    first_color: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[pivot] = GREY
    second_color = BLUE if first_color == RED else RED
    for step in range(1, length + 1):
        r = pivot[0] + step * direction[0]
        c = pivot[1] + step * direction[1]
        grid[r, c] = first_color if step % 2 == 1 else second_color
    return grid


def _is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        info = _parse_instance(grid)
    except AssertionError:
        return False

    if not (LINE_LENGTH_RANGE[0] <= info["length"] <= LINE_LENGTH_RANGE[1]):
        return False

    rotated_space = _distance_to_edge(grid.shape, info["pivot"], _rotate_cw(*info["direction"]))
    if rotated_space < info["length"] + MIN_EXTRA_EXTENSION:
        return False

    expected = _expected_output_from_info(grid, info)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    length = random.randint(*LINE_LENGTH_RANGE)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], length + 4), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(GRID_WIDTH_RANGE[0], length + 4), GRID_WIDTH_RANGE[1])
    direction = random.choice(CARDINALS)
    rotated = _rotate_cw(*direction)
    first_color = random.choice([BLUE, RED])

    feasible_pivots = []
    for r in range(height):
        for c in range(width):
            pivot = (r, c)
            if _distance_to_edge((height, width), pivot, direction) < length:
                continue
            if _distance_to_edge((height, width), pivot, rotated) < length + MIN_EXTRA_EXTENSION:
                continue
            feasible_pivots.append(pivot)

    if not feasible_pivots:
        return None

    pivot = random.choice(feasible_pivots)
    grid = _build_input(height, width, pivot, direction, length, first_color)

    if not _is_semantically_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
