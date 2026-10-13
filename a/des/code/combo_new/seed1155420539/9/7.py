from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

RING_COLORS = (RED, ORANGE, YELLOW, GREEN, AZURE, BLUE, MAGENTA)
RING_INDEX = {color: idx for idx, color in enumerate(RING_COLORS)}
GENERATION_ATTEMPTS = 400


def _segment(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _make_hvh(
    top_row: int, bottom_row: int, left_col: int, mid_col: int, right_col: int
) -> list[tuple[int, int]]:
    return _polyline_cells(
        [(top_row, left_col), (top_row, mid_col), (bottom_row, mid_col), (bottom_row, right_col)]
    )


def _make_vhv(
    top_row: int, mid_row: int, bottom_row: int, left_col: int, right_col: int
) -> list[tuple[int, int]]:
    return _polyline_cells(
        [(top_row, left_col), (mid_row, left_col), (mid_row, right_col), (bottom_row, right_col)]
    )


def _mid_color(color_a: int, color_b: int) -> int:
    ia = RING_INDEX[color_a]
    ib = RING_INDEX[color_b]
    delta = (ib - ia) % len(RING_COLORS)
    if delta == 0:
        return color_a
    if delta == 2:
        return RING_COLORS[(ia + 1) % len(RING_COLORS)]
    if delta == 5:
        return RING_COLORS[(ib + 1) % len(RING_COLORS)]
    raise AssertionError("crossing colors must be separated by one ring midpoint")


def _is_line_color(value: int) -> bool:
    return value not in (BLACK, GREY)


def _crossing_pair(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    center = int(grid[r, c])
    if not _is_line_color(center):
        return None

    up = int(grid[r - 1, c])
    down = int(grid[r + 1, c])
    left = int(grid[r, c - 1])
    right = int(grid[r, c + 1])

    if not all(_is_line_color(value) for value in (up, down, left, right)):
        return None
    if up != down or left != right or up == left:
        return None
    if center not in (up, left):
        return None
    return up, left


def _find_crossings(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    crossings: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            pair = _crossing_pair(grid, r, c)
            if pair is not None:
                crossings[(r, c)] = pair
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each true crossing of the framed orthogonal polylines to the midpoint color of the two crossing line colors.

    Input structure:
    - The background is black and the outermost border is a one-cell grey frame.
    - The interior contains several one-pixel-wide orthogonal polylines; each polyline uses one color from the ring Red -> Orange -> Yellow -> Green -> Azure -> Blue -> Magenta -> Red.
    - When two polylines cross, the shared pixel still shows one of the two line colors, while the other line continues straight through it.

    Rule:
    - Inspect each interior line pixel and find the cells where a straight vertical continuation and a straight horizontal continuation overlap.
    - Read the two participating line colors from the opposite neighbors around that crossing.
    - Replace the crossing pixel by the unique midpoint color on the shorter color-ring arc between those two colors.
    - Leave all non-crossing line pixels, the black background, and the grey frame unchanged.

    Key insight:
    - A crossing is identified by reconstructing two straight line continuations through one pixel, and the new color comes from the hidden cyclic order of the seven line colors rather than from either visible line color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5
    assert np.all(g[0, :] == GREY) and np.all(g[-1, :] == GREY)
    assert np.all(g[:, 0] == GREY) and np.all(g[:, -1] == GREY)

    out = g.copy()
    for (r, c), (vertical_color, horizontal_color) in _find_crossings(g).items():
        out[r, c] = _mid_color(vertical_color, horizontal_color)
    return out


def _pick_ordered(total: int, count: int, margin: int = 0, min_gap: int = 1) -> list[int]:
    positions = list(range(margin, total - margin))
    if len(positions) < count:
        raise ValueError("not enough positions for the requested spacing")
    for _ in range(300):
        values = sorted(random.sample(positions, count))
        if all(b - a >= min_gap for a, b in zip(values, values[1:])):
            return values
    raise RuntimeError("failed to place ordered coordinates")


def _choose_pair_colors(used: set[int]) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    n = len(RING_COLORS)
    for i, color in enumerate(RING_COLORS):
        other = RING_COLORS[(i + 2) % n]
        if color in used or other in used:
            continue
        options.append((color, other))
        options.append((other, color))
    if not options:
        raise RuntimeError("no disjoint crossing color pair available")
    return random.choice(options)


def _choose_fan_colors(used: set[int]) -> tuple[int, int, int]:
    options: list[tuple[int, int, int]] = []
    n = len(RING_COLORS)
    for i, center in enumerate(RING_COLORS):
        branch_a = RING_COLORS[(i - 2) % n]
        branch_b = RING_COLORS[(i + 2) % n]
        if {center, branch_a, branch_b} & used:
            continue
        branches = [branch_a, branch_b]
        random.shuffle(branches)
        options.append((center, branches[0], branches[1]))
    if not options:
        raise RuntimeError("no disjoint fan colors available")
    return random.choice(options)


def _make_pair_box(
    row0: int, col0: int, box_h: int, box_w: int, colors: tuple[int, int]
) -> list[tuple[int, list[tuple[int, int]]]]:
    rows = _pick_ordered(box_h, 5, margin=1)
    cols = _pick_ordered(box_w, 5, margin=1)
    vertical_color, horizontal_color = colors
    return [
        (
            vertical_color,
            _make_vhv(row0 + rows[0], row0 + rows[2], row0 + rows[4], col0 + cols[0], col0 + cols[4]),
        ),
        (
            horizontal_color,
            _make_hvh(row0 + rows[1], row0 + rows[3], col0 + cols[1], col0 + cols[2], col0 + cols[3]),
        ),
    ]


def _make_fan_box(
    row0: int, col0: int, box_h: int, box_w: int, colors: tuple[int, int, int]
) -> list[tuple[int, list[tuple[int, int]]]]:
    rows = _pick_ordered(box_h, 7, margin=1)
    cols = _pick_ordered(box_w, 8, margin=1)
    center_color, branch_a, branch_b = colors
    return [
        (
            center_color,
            _make_vhv(row0 + rows[0], row0 + rows[3], row0 + rows[6], col0 + cols[0], col0 + cols[7]),
        ),
        (
            branch_a,
            _make_hvh(row0 + rows[1], row0 + rows[4], col0 + cols[1], col0 + cols[2], col0 + cols[3]),
        ),
        (
            branch_b,
            _make_hvh(row0 + rows[2], row0 + rows[5], col0 + cols[4], col0 + cols[5], col0 + cols[6]),
        ),
    ]


def _transform_lines(
    height: int, width: int, lines: list[tuple[int, list[tuple[int, int]]]]
) -> tuple[int, int, list[tuple[int, list[tuple[int, int]]]]]:
    transpose = random.random() < 0.5
    flip_rows = random.random() < 0.5
    flip_cols = random.random() < 0.5

    out_h, out_w = (width, height) if transpose else (height, width)
    transformed: list[tuple[int, list[tuple[int, int]]]] = []
    for color, cells in lines:
        new_cells: list[tuple[int, int]] = []
        for r, c in cells:
            if transpose:
                r, c = c, r
            if flip_rows:
                r = out_h - 1 - r
            if flip_cols:
                c = out_w - 1 - c
            new_cells.append((r, c))
        transformed.append((color, new_cells))
    return out_h, out_w, transformed


def _scenario_pair() -> tuple[int, int, list[tuple[int, list[tuple[int, int]]]], int]:
    height = random.randint(8, 24)
    width = random.randint(8, 24)
    lines = _make_pair_box(0, 0, height, width, _choose_pair_colors(set()))
    return height, width, lines, 1


def _scenario_fan() -> tuple[int, int, list[tuple[int, list[tuple[int, int]]]], int]:
    height = random.randint(9, 24)
    width = random.randint(10, 24)
    lines = _make_fan_box(0, 0, height, width, _choose_fan_colors(set()))
    return height, width, lines, 2


def _scenario_double() -> tuple[int, int, list[tuple[int, list[tuple[int, int]]]], int]:
    box_h1 = random.randint(8, 10)
    box_h2 = random.randint(8, 10)
    gap = random.randint(1, 3)
    height = box_h1 + gap + box_h2
    width = random.randint(8, 24)

    first_pair = _choose_pair_colors(set())
    second_pair = _choose_pair_colors(set(first_pair))

    lines = _make_pair_box(0, 0, box_h1, width, first_pair)
    lines.extend(_make_pair_box(box_h1 + gap, 0, box_h2, width, second_pair))
    return height, width, lines, 2


def _render_case(
    height: int, width: int, lines: list[tuple[int, list[tuple[int, int]]]]
) -> tuple[np.ndarray, np.ndarray, int] | None:
    occupancy: dict[tuple[int, int], list[int]] = {}
    for idx, (_color, cells) in enumerate(lines):
        if len(set(cells)) != len(cells):
            return None
        for r, c in cells:
            if not (0 <= r < height and 0 <= c < width):
                return None
            occupancy.setdefault((r, c), []).append(idx)

    input_grid = np.full((height + 2, width + 2), BLACK, dtype=int)
    output_grid = input_grid.copy()
    input_grid[0, :] = input_grid[-1, :] = GREY
    input_grid[:, 0] = input_grid[:, -1] = GREY
    output_grid[0, :] = output_grid[-1, :] = GREY
    output_grid[:, 0] = output_grid[:, -1] = GREY

    intersection_count = 0
    for (r, c), ids in occupancy.items():
        if len(ids) > 2:
            return None
        colors = [lines[idx][0] for idx in ids]
        if len(ids) == 1:
            visible = colors[0]
            mixed = colors[0]
        else:
            if colors[0] == colors[1]:
                return None
            try:
                mixed = _mid_color(colors[0], colors[1])
            except AssertionError:
                return None
            visible = random.choice(colors)
            intersection_count += 1

        input_grid[r + 1, c + 1] = visible
        output_grid[r + 1, c + 1] = mixed

    return input_grid, output_grid, intersection_count


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(("pair", "fan", "double"))
    if scenario == "pair":
        height, width, lines, expected_intersections = _scenario_pair()
    elif scenario == "fan":
        height, width, lines, expected_intersections = _scenario_fan()
    else:
        height, width, lines, expected_intersections = _scenario_double()

    height, width, lines = _transform_lines(height, width, lines)
    rendered = _render_case(height, width, lines)
    if rendered is None:
        return None

    inp, out, intersection_count = rendered
    if intersection_count != expected_intersections:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
