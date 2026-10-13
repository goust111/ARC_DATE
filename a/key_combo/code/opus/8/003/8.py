import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def _find_rectangle_and_control(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    control_cells = np.argwhere(grid == GREEN)
    assert control_cells.shape[0] == 1, "expected exactly one control pixel"
    control = tuple(map(int, control_cells[0]))

    rect_mask = (grid == BLUE) | (grid == GREEN)
    rect_cells = np.argwhere(rect_mask)
    assert rect_cells.size > 0, "expected a blue rectangle with one green control pixel"
    r0 = int(rect_cells[:, 0].min())
    c0 = int(rect_cells[:, 1].min())
    r1 = int(rect_cells[:, 0].max())
    c1 = int(rect_cells[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((patch == BLUE) | (patch == GREEN)), "rectangle must be solid blue except the control pixel"
    return (r0, c0, r1, c1), control


def _collect_exclusions(
    grid: np.ndarray, rect: tuple[int, int, int, int]
) -> tuple[set[int], set[int], set[int], set[int]]:
    r0, c0, r1, c1 = rect
    red_cells = np.argwhere(grid == RED)

    top_cols: set[int] = set()
    bottom_cols: set[int] = set()
    left_rows: set[int] = set()
    right_rows: set[int] = set()

    for rr, cc in red_cells:
        r = int(rr)
        c = int(cc)
        if c0 <= c <= c1 and r < r0:
            top_cols.add(c)
        elif c0 <= c <= c1 and r > r1:
            bottom_cols.add(c)
        elif r0 <= r <= r1 and c < c0:
            left_rows.add(r)
        elif r0 <= r <= r1 and c > c1:
            right_rows.add(r)

    return top_cols, bottom_cols, left_rows, right_rows


def _paint_vertical_ray(out: np.ndarray, start_r: int, step: int, col: int) -> None:
    r = start_r
    while 0 <= r < out.shape[0]:
        if int(out[r, col]) == RED:
            break
        out[r, col] = BLUE
        r += step


def _paint_horizontal_ray(out: np.ndarray, row: int, start_c: int, step: int) -> None:
    c = start_c
    while 0 <= c < out.shape[1]:
        if int(out[row, c]) == RED:
            break
        out[row, c] = BLUE
        c += step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a solid rectangle with parity-based square rings while the same side-band markers also block outward ray extensions.

    Input structure:
    - The grid is mostly black background.
    - A solid blue rectangle appears near the center, with exactly one green control pixel strictly inside it.
    - Red singleton markers appear only in the side bands directly above, below, left, or right of the rectangle.

    Rule:
    - Detect the blue rectangle, the green control pixel, and the red side-band markers.
    - Extend blue rays from the rectangle outward along every rectangle column and row: upward, downward, leftward, and rightward. A top marker blocks that top-side column, a bottom marker blocks that bottom-side column, a left marker blocks that left-side row, and a right marker blocks that right-side row. Any ray also stops just before a red marker cell.
    - Recolor every non-control rectangle cell by Chebyshev distance from the control pixel: odd distances become yellow and even distances become grey.
    - If a rectangle cell lies on any row blocked by a left/right marker or on any column blocked by a top/bottom marker, leave it blue instead of recoloring it.
    - Keep the green control pixel and all red markers unchanged.

    Key insight:
    - The same red markers simultaneously constrain two different spatial processes: they suppress selected exterior rays on their own side and also suppress interior ring filling on the corresponding rectangle rows or columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect, control = _find_rectangle_and_control(g)
    r0, c0, r1, c1 = rect
    ctrl_r, ctrl_c = control
    top_cols, bottom_cols, left_rows, right_rows = _collect_exclusions(g, rect)

    out = g.copy()

    for c in range(c0, c1 + 1):
        if c not in top_cols:
            _paint_vertical_ray(out, r0 - 1, -1, c)
        if c not in bottom_cols:
            _paint_vertical_ray(out, r1 + 1, 1, c)

    for r in range(r0, r1 + 1):
        if r not in left_rows:
            _paint_horizontal_ray(out, r, c0 - 1, -1)
        if r not in right_rows:
            _paint_horizontal_ray(out, r, c1 + 1, 1)

    blocked_rows = left_rows | right_rows
    blocked_cols = top_cols | bottom_cols
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) == (ctrl_r, ctrl_c):
                out[r, c] = GREEN
                continue
            if r in blocked_rows or c in blocked_cols:
                out[r, c] = BLUE
                continue
            distance = max(abs(r - ctrl_r), abs(c - ctrl_c))
            out[r, c] = YELLOW if distance % 2 == 1 else GREY

    return out


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _sample_layout() -> tuple[int, int, int, int, int, int]:
    rect_h = random.randint(6, 10)
    rect_w = random.randint(6, 10)

    top_gap = random.randint(3, 7)
    bottom_gap = _clamp(top_gap + random.randint(-2, 2), 3, 7)
    left_gap = random.randint(3, 7)
    right_gap = _clamp(left_gap + random.randint(-2, 2), 3, 7)

    height = top_gap + rect_h + bottom_gap
    width = left_gap + rect_w + right_gap
    assert height <= 30 and width <= 30
    return height, width, top_gap, left_gap, rect_h, rect_w


def _assign_side_sets(indices: list[int], first_label: str, second_label: str) -> tuple[set[int], set[int]]:
    first: set[int] = set()
    second: set[int] = set()
    for idx in indices:
        side = random.choice((first_label, second_label, "both"))
        if side in (first_label, "both"):
            first.add(idx)
        if side in (second_label, "both"):
            second.add(idx)
    return first, second


def generate_input() -> np.ndarray:
    height, width, r0, c0, rect_h, rect_w = _sample_layout()
    grid = np.zeros((height, width), dtype=int)

    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    ctrl_r = random.randint(r0 + 2, r1 - 2)
    ctrl_c = random.randint(c0 + 2, c1 - 2)
    grid[ctrl_r, ctrl_c] = GREEN

    vertical_count = random.randint(1, max(1, rect_w // 2))
    horizontal_count = random.randint(1, max(1, rect_h // 2))
    vertical_indices = random.sample(list(range(c0, c1 + 1)), vertical_count)
    horizontal_indices = random.sample(list(range(r0, r1 + 1)), horizontal_count)

    top_cols, bottom_cols = _assign_side_sets(vertical_indices, "top", "bottom")
    left_rows, right_rows = _assign_side_sets(horizontal_indices, "left", "right")

    for c in top_cols:
        offset = random.randint(1, min(3, r0))
        grid[r0 - offset, c] = RED
    for c in bottom_cols:
        offset = random.randint(1, min(3, height - 1 - r1))
        grid[r1 + offset, c] = RED
    for r in left_rows:
        offset = random.randint(1, min(3, c0))
        grid[r, c0 - offset] = RED
    for r in right_rows:
        offset = random.randint(1, min(3, width - 1 - c1))
        grid[r, c1 + offset] = RED

    assert grid.ndim == 2 and max(grid.shape) <= 30
    _ = main(grid)
    return grid
