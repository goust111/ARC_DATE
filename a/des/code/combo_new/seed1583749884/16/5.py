import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

PANEL_SIZE = 5
DIRECTIONS = [
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1),  # up-left
]


def _panel_window(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    return grid[row - 2 : row + 3, col - 2 : col + 3]


def _is_clock_center(grid: np.ndarray, row: int, col: int) -> bool:
    height, width = grid.shape
    if not (2 <= row < height - 2 and 2 <= col < width - 2):
        return False
    if int(grid[row, col]) != RED:
        return False

    panel = _panel_window(grid, row, col)
    if int(np.count_nonzero(panel == RED)) != 1:
        return False

    yellow_count = int(np.count_nonzero(panel == YELLOW))
    if yellow_count not in (0, 2):
        return False

    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            if r == 2 and c == 2:
                continue
            value = int(panel[r, c])
            on_ring = max(abs(r - 2), abs(c - 2)) == 2
            allowed = {GREY, YELLOW} if on_ring else {BLACK, YELLOW}
            if value not in allowed:
                return False
    return True


def _find_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers = []
    for row, col in np.argwhere(grid == RED):
        row = int(row)
        col = int(col)
        if _is_clock_center(grid, row, col):
            centers.append((row, col))

    assert len(centers) == 4, f"expected 4 clock centers, found {len(centers)}"
    centers.sort(key=lambda rc: (rc[1], rc[0]))
    return centers


def _pointer_direction(panel: np.ndarray) -> int | None:
    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(panel == YELLOW)}
    if not yellow_cells:
        return None
    assert len(yellow_cells) == 2, "pointer must contain exactly two yellow cells"

    for direction_index, (dr, dc) in enumerate(DIRECTIONS):
        expected = {(2 + dr, 2 + dc), (2 + 2 * dr, 2 + 2 * dc)}
        if yellow_cells == expected:
            return direction_index
    raise AssertionError("yellow cells do not form a valid pointer")


def _draw_pointer(grid: np.ndarray, center: tuple[int, int], direction_index: int) -> None:
    row, col = center
    dr, dc = DIRECTIONS[direction_index]
    grid[row + dr, col + dc] = YELLOW
    grid[row + 2 * dr, col + 2 * dc] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the final clock panel by continuing a constant clockwise pointer rotation.

    Input structure:
    - The background is black.
    - Four equal square clock panels appear in one horizontal row, separated by grey columns.
    - Each panel has a red center pixel and a grey 5x5 square dial border.
    - The first three panels also contain a yellow 2-cell pointer in one of the 8 compass directions, while the fourth panel is missing its pointer.

    Rule:
    - Read the yellow pointer direction in each of the first three panels.
    - Infer the constant clockwise step size between consecutive panels, measured in 45-degree increments.
    - Rotate the third pointer by the same step once more and draw that yellow 2-cell pointer in the fourth panel.

    Key insight:
    - The four panels are animation frames of the same clock hand, so the missing hand is obtained by extrapolating a constant angular increment.
    """
    g = np.asarray(grid, dtype=int)
    centers = _find_centers(g)

    directions = []
    for row, col in centers:
        panel = _panel_window(g, row, col)
        directions.append(_pointer_direction(panel))

    assert all(direction is not None for direction in directions[:3]), "first three panels need pointers"
    assert directions[3] is None, "fourth panel should be missing its pointer"

    step_a = (directions[1] - directions[0]) % len(DIRECTIONS)
    step_b = (directions[2] - directions[1]) % len(DIRECTIONS)
    assert step_a == step_b, "pointer rotation step must be constant"

    out = g.copy()
    next_direction = (directions[2] + step_a) % len(DIRECTIONS)
    _draw_pointer(out, centers[3], next_direction)
    return out


def _blank_panel() -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BLACK, dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY
    panel[2, 2] = RED
    return panel


def _make_panel(direction_index: int | None) -> np.ndarray:
    panel = _blank_panel()
    if direction_index is not None:
        _draw_pointer(panel, (2, 2), direction_index)
    return panel


def _compose_strip(direction_sequence: list[int], separator_width: int) -> np.ndarray:
    strip_width = 4 * PANEL_SIZE + 3 * separator_width
    strip = np.full((PANEL_SIZE, strip_width), BLACK, dtype=int)

    for panel_index in range(4):
        start_col = panel_index * (PANEL_SIZE + separator_width)
        direction = direction_sequence[panel_index] if panel_index < 3 else None
        strip[:, start_col : start_col + PANEL_SIZE] = _make_panel(direction)
        if panel_index < 3:
            sep_start = start_col + PANEL_SIZE
            strip[:, sep_start : sep_start + separator_width] = GREY
    return strip


def generate_input() -> np.ndarray:
    start_direction = random.randrange(len(DIRECTIONS))
    step = random.randint(1, len(DIRECTIONS) - 1)
    directions = [(start_direction + i * step) % len(DIRECTIONS) for i in range(3)]

    separator_width = random.randint(1, 3)
    strip = _compose_strip(directions, separator_width)
    strip_height, strip_width = strip.shape

    total_height = random.randint(strip_height, 30)
    total_width = random.randint(strip_width, 30)
    top = random.randint(0, total_height - strip_height)
    left = random.randint(0, total_width - strip_width)

    grid = np.full((total_height, total_width), BLACK, dtype=int)
    grid[top : top + strip_height, left : left + strip_width] = strip
    return grid
