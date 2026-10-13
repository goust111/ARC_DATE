import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

CONTROL_COLORS = (RED, BLUE, GREEN, YELLOW)
GEN_RETRIES = 300
HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (17, 23)  # Includes the control column.
CONTROL_COUNT_CHOICES = (2, 3)
CONTROL_MARKER_RANGE = (1, 4)
EXTRA_MARKER_RANGE = (1, 3)


def _find_cross(main_area: np.ndarray) -> tuple[int, int]:
    row_hits = np.where(np.all(main_area == AZURE, axis=1))[0]
    col_hits = np.where(np.all(main_area == AZURE, axis=0))[0]
    assert len(row_hits) == 1, "expected a unique azure horizontal axis"
    assert len(col_hits) == 1, "expected a unique azure vertical axis"
    return int(row_hits[0]), int(col_hits[0])


def _reflect(r: int, c: int, h_axis: int, v_axis: int, axis_kind: str) -> tuple[int, int]:
    if axis_kind == "horizontal":
        return 2 * h_axis - r, c
    return r, 2 * v_axis - c


def _non_cross_marker_positions(main_area: np.ndarray, h_axis: int, v_axis: int) -> dict[int, list[tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    h, w = main_area.shape
    for r in range(h):
        for c in range(w):
            if r == h_axis or c == v_axis:
                continue
            color = int(main_area[r, c])
            if color == BLACK:
                continue
            positions.setdefault(color, []).append((r, c))
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored scatter points according to the control column and then drop that column.

    Control colors above the azure horizontal axis use that horizontal axis as the mirror.
    Control colors below it use the azure vertical axis instead. Colors absent from the
    control column are erased from the main region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    control = g[:, 0]
    main_area = g[:, 1:]
    h_axis, v_axis = _find_cross(main_area)

    axis_by_color: dict[int, str] = {}
    for r, value in enumerate(control):
        color = int(value)
        if color == GREY:
            continue
        assert color in CONTROL_COLORS
        assert r != h_axis, "control markers cannot sit on the horizontal axis row"
        axis_by_color[color] = "horizontal" if r < h_axis else "vertical"

    marker_positions = _non_cross_marker_positions(main_area, h_axis, v_axis)
    out = main_area.copy()

    for color, cells in marker_positions.items():
        if color not in axis_by_color:
            for r, c in cells:
                out[r, c] = BLACK

    h, w = main_area.shape
    for color, axis_kind in axis_by_color.items():
        for r, c in marker_positions.get(color, []):
            rr, cc = _reflect(r, c, h_axis, v_axis, axis_kind)
            assert 0 <= rr < h and 0 <= cc < w
            assert rr != h_axis and cc != v_axis
            assert int(main_area[rr, cc]) == BLACK, "mirror destinations must be empty in the input"
            out[rr, cc] = color

    return out


def _all_non_cross_cells(h: int, w: int, h_axis: int, v_axis: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r != h_axis and c != v_axis]


def _controlled_candidates(
    h: int,
    w: int,
    h_axis: int,
    v_axis: int,
    axis_kind: str,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in _all_non_cross_cells(h, w, h_axis, v_axis):
        if (r, c) in occupied or (r, c) in reserved:
            continue
        rr, cc = _reflect(r, c, h_axis, v_axis, axis_kind)
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if rr == h_axis or cc == v_axis:
            continue
        if (rr, cc) in occupied or (rr, cc) in reserved:
            continue
        candidates.append((r, c))
    return candidates


def _free_cells(
    h: int,
    w: int,
    h_axis: int,
    v_axis: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, c in _all_non_cross_cells(h, w, h_axis, v_axis)
        if (r, c) not in occupied and (r, c) not in reserved
    ]


def _sample_control_rows(h: int, h_axis: int, colors: list[int]) -> tuple[dict[int, str], dict[int, int]]:
    n_above = random.randint(1, len(colors) - 1)
    above_colors = set(random.sample(colors, n_above))
    below_colors = [color for color in colors if color not in above_colors]

    above_rows = random.sample(list(range(h_axis)), len(above_colors))
    below_rows = random.sample(list(range(h_axis + 1, h)), len(below_colors))
    random.shuffle(above_rows)
    random.shuffle(below_rows)

    axis_by_color: dict[int, str] = {}
    row_by_color: dict[int, int] = {}
    for color, row in zip([c for c in colors if c in above_colors], above_rows):
        axis_by_color[color] = "horizontal"
        row_by_color[color] = int(row)
    for color, row in zip(below_colors, below_rows):
        axis_by_color[color] = "vertical"
        row_by_color[color] = int(row)
    return axis_by_color, row_by_color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w_full = random.randint(*WIDTH_RANGE)
    w_main = w_full - 1

    h_axis = random.randint(3, h - 4)
    v_axis = random.randint(2, w_main - 3)

    control_count = random.choice(CONTROL_COUNT_CHOICES)
    control_colors = random.sample(list(CONTROL_COLORS), control_count)
    extra_pool = [color for color in CONTROL_COLORS if color not in control_colors]
    if not extra_pool:
        return None

    axis_by_color, row_by_color = _sample_control_rows(h, h_axis, control_colors)
    if not ({axis_by_color[color] for color in control_colors} == {"horizontal", "vertical"}):
        return None

    positions_by_color: dict[int, list[tuple[int, int]]] = {color: [] for color in control_colors}
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    for color in random.sample(control_colors, len(control_colors)):
        count = random.randint(*CONTROL_MARKER_RANGE)
        for _ in range(count):
            candidates = _controlled_candidates(h, w_main, h_axis, v_axis, axis_by_color[color], occupied, reserved)
            if not candidates:
                return None
            cell = random.choice(candidates)
            occupied.add(cell)
            positions_by_color[color].append(cell)
            reserved.add(_reflect(cell[0], cell[1], h_axis, v_axis, axis_by_color[color]))

    extra_count = random.randint(1, len(extra_pool))
    extra_colors = random.sample(extra_pool, extra_count)
    for color in extra_colors:
        count = random.randint(*EXTRA_MARKER_RANGE)
        for _ in range(count):
            candidates = _free_cells(h, w_main, h_axis, v_axis, occupied, reserved)
            if not candidates:
                return None
            cell = random.choice(candidates)
            occupied.add(cell)
            positions_by_color.setdefault(color, []).append(cell)

    if not extra_colors:
        return None

    grid = np.full((h, w_full), BLACK, dtype=int)
    grid[:, 0] = GREY
    grid[h_axis, 1:] = AZURE
    grid[:, v_axis + 1] = AZURE

    for color, row in row_by_color.items():
        grid[row, 0] = color

    for color, cells in positions_by_color.items():
        for r, c in cells:
            grid[r, c + 1] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (h, w_main):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
