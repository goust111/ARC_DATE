from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


def _single_coord(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.shape == (1, 2), "expected exactly one marker of the requested color"
    return int(coords[0, 0]), int(coords[0, 1])


def _sign(delta: int) -> int:
    return int((delta > 0) - (delta < 0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the green marker one cell along the straight line toward the yellow marker.

    Input structure
    - The grid background is the dominant color.
    - There is exactly one green marker pixel and exactly one yellow marker pixel.
    - The two markers lie on the same row, same column, or a 45-degree diagonal.

    Rule
    - Keep the yellow marker fixed.
    - Clear the original green marker cell to the background color.
    - Repaint the green marker one cell closer to the yellow marker along their shared line.

    Key insight
    - The yellow marker is the fixed destination, and the green marker advances by the normalized row/column direction vector toward it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    colors = set(map(int, np.unique(g)))
    assert colors == {int(bg), Color.GREEN, Color.YELLOW}, "expected only background, green, and yellow"

    source_r, source_c = _single_coord(g, Color.GREEN)
    target_r, target_c = _single_coord(g, Color.YELLOW)

    row_delta = target_r - source_r
    col_delta = target_c - source_c
    assert (row_delta, col_delta) != (0, 0), "green and yellow markers must be distinct"
    assert (
        row_delta == 0 or col_delta == 0 or abs(row_delta) == abs(col_delta)
    ), "markers must define a horizontal, vertical, or diagonal line"

    step_r = _sign(row_delta)
    step_c = _sign(col_delta)
    dest_r = source_r + step_r
    dest_c = source_c + step_c
    assert g[dest_r, dest_c] == bg, "green marker's next step must be background"

    out = g.copy()
    out[source_r, source_c] = bg
    out[dest_r, dest_c] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------
# Canvas shape and retry count.
GRID_SIZE_RANGE = (3, 12)
SQUARE_GRID_PROB = 0.8
MAX_GENERATION_ATTEMPTS = 300
# Marker line geometry.
BACKGROUND_COLOR, MIN_DISTANCE, MAX_DISTANCE = 0, 2, 7
DIRECTION_TYPE_COUNT = 3
HORIZONTAL_DIRECTION, VERTICAL_DIRECTION = 0, 1
STEP_CHOICES = (-1, 1)

def _rand_grid_size():
    """Return height and width with the original square-grid bias."""
    if np.random.random() < SQUARE_GRID_PROB:
        side = int(np.random.randint(*GRID_SIZE_RANGE))
        return side, side
    return int(np.random.randint(*GRID_SIZE_RANGE)), int(np.random.randint(*GRID_SIZE_RANGE))

def _rand_movement(height, width):
    """Return (step_r, step_c, distance), or None if the segment cannot fit."""
    direction_type = int(np.random.randint(DIRECTION_TYPE_COUNT))
    if direction_type == HORIZONTAL_DIRECTION:
        step_r, step_c, max_feasible_distance = 0, int(np.random.choice(STEP_CHOICES)), width - 1
    elif direction_type == VERTICAL_DIRECTION:
        step_r, step_c, max_feasible_distance = int(np.random.choice(STEP_CHOICES)), 0, height - 1
    else:
        step_r = int(np.random.choice(STEP_CHOICES))
        step_c = int(np.random.choice(STEP_CHOICES))
        max_feasible_distance = min(height, width) - 1
    max_distance = min(MAX_DISTANCE, max_feasible_distance)
    if max_distance < MIN_DISTANCE:
        return None
    distance = int(np.random.randint(MIN_DISTANCE, max_distance + 1))
    return step_r, step_c, distance

def _gen_attempt() -> np.ndarray | None:
    height, width = _rand_grid_size()
    movement = _rand_movement(height, width)
    if movement is None:
        return None
    step_r, step_c, distance = movement

    target_r_min = distance if step_r > 0 else 0
    target_r_max = height - distance if step_r < 0 else height
    target_c_min = distance if step_c > 0 else 0
    target_c_max = width - distance if step_c < 0 else width
    target_r = int(np.random.randint(target_r_min, target_r_max))
    target_c = int(np.random.randint(target_c_min, target_c_max))
    source_r = target_r - step_r * distance
    source_c = target_c - step_c * distance

    inp = np.zeros((height, width), dtype=int)
    inp[target_r, target_c] = Color.YELLOW
    inp[source_r, source_c] = Color.GREEN

    expected = inp.copy()
    expected[source_r, source_c] = BACKGROUND_COLOR
    expected[source_r + step_r, source_c + step_c] = Color.GREEN

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid dc433765 input")
