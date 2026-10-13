from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

COLOR_ORDER = [BLUE, GREEN, ORANGE, RED, YELLOW]
COLOR_DRAW_RANK = {color: rank for rank, color in enumerate(COLOR_ORDER)}

GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
PAIR_COUNT_RANGE = (2, 5)
LAYOUT_CHOICES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
SLOT_MIN_SIZE = 4
GEN_RETRIES = 300


def _foreground_colors_scan_order(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in grid.ravel():
        color = int(value)
        if color != BG and color not in seen:
            seen.add(color)
            ordered.append(color)
    return ordered


def _ordered_draw_colors(grid: np.ndarray) -> list[int]:
    colors = _foreground_colors_scan_order(grid)
    scan_rank = {color: i for i, color in enumerate(colors)}
    return sorted(
        colors,
        key=lambda color: (
            COLOR_DRAW_RANK.get(color, len(COLOR_DRAW_RANK) + scan_rank[color]),
            scan_rank[color],
        ),
    )


def _pair_points(box: tuple[int, int, int, int], orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    if orientation == "main":
        return (r0, c0), (r1, c1)
    if orientation == "anti":
        return (r0, c1), (r1, c0)
    raise AssertionError(f"unknown orientation: {orientation}")


def _output_cells_for_pair(
    box: tuple[int, int, int, int], orientation: str
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r, c0) for r in range(r0, r1 + 1))
    if orientation == "main":
        cells.add((r1, c1))
    elif orientation != "anti":
        raise AssertionError(f"unknown orientation: {orientation}")
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each same-color pixel duo and draw its corner-anchored row/column cross inside the spanning rectangle.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two singleton pixels.
    - The two pixels of a color lie on opposite corners of a non-degenerate axis-aligned rectangle.
    - No other foreground structure is present.

    Rule:
    - For each color, use the two pixels as opposite corners of their minimal axis-aligned rectangle.
    - Draw a horizontal segment in that color across the rectangle on the row of the upper pixel.
    - Draw a vertical segment in that color across the rectangle on the column of the leftmost pixel.
    - Keep the original endpoints, and resolve overlaps by drawing colors in Blue, Green, Orange, Red, Yellow order.

    Key insight:
    - The crossing position is not the rectangle center; it is built by combining the upper endpoint's row with the left endpoint's column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color in _ordered_draw_colors(g):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"

        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert r0 < r1 and c0 < c1, "paired pixels must define a non-degenerate rectangle"

        out[r0, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)

    return out


def _random_partition(total: int, parts: int, min_size: int) -> list[int] | None:
    if total < parts * min_size:
        return None
    sizes = [min_size] * parts
    for _ in range(total - parts * min_size):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _segment_bounds(sizes: list[int]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for size in sizes:
        end = start + size - 1
        bounds.append((start, end))
        start = end + 1
    return bounds


def _choose_layout(height: int, width: int, pair_count: int) -> tuple[int, int] | None:
    candidates = [
        (rows, cols)
        for rows, cols in LAYOUT_CHOICES
        if rows * cols >= pair_count and height >= rows * SLOT_MIN_SIZE and width >= cols * SLOT_MIN_SIZE
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _slot_inner_bounds(slot: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = slot
    if r1 - r0 + 1 >= 5:
        r0 += 1
        r1 -= 1
    if c1 - c0 + 1 >= 5:
        c0 += 1
        c1 -= 1
    if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
        return slot
    return r0, c0, r1, c1


def _sample_box_within(slot: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _slot_inner_bounds(slot)
    max_h = min(8, r1 - r0 + 1)
    max_w = min(8, c1 - c0 + 1)
    assert max_h >= 2 and max_w >= 2

    height = random.randint(2, max_h)
    width = random.randint(2, max_w)
    top = random.randint(r0, r1 - height + 1)
    left = random.randint(c0, c1 - width + 1)
    return top, left, top + height - 1, left + width - 1


def _orientation_schedule(pair_count: int) -> list[str]:
    orientations = ["main"] * (pair_count // 2) + ["anti"] * (pair_count // 2)
    if pair_count % 2 == 1:
        orientations.append(random.choice(["main", "anti"]))
    random.shuffle(orientations)
    return orientations


def _build_expected(specs: list[tuple[int, tuple[int, int, int, int], str]], shape: tuple[int, int]) -> np.ndarray:
    expected = np.zeros(shape, dtype=int)
    for color, box, orientation in specs:
        p0, p1 = _pair_points(box, orientation)
        expected[p0] = color
        expected[p1] = color
    for color, box, _orientation in sorted(specs, key=lambda item: COLOR_DRAW_RANK[item[0]]):
        r0, c0, r1, c1 = box
        expected[r0, c0 : c1 + 1] = color
        expected[r0 : r1 + 1, c0] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    layout = _choose_layout(height, width, pair_count)
    if layout is None:
        return None

    slot_rows, slot_cols = layout
    row_sizes = _random_partition(height, slot_rows, SLOT_MIN_SIZE)
    col_sizes = _random_partition(width, slot_cols, SLOT_MIN_SIZE)
    if row_sizes is None or col_sizes is None:
        return None

    row_bounds = _segment_bounds(row_sizes)
    col_bounds = _segment_bounds(col_sizes)
    all_slots = [
        (rr0, cc0, rr1, cc1)
        for rr0, rr1 in row_bounds
        for cc0, cc1 in col_bounds
    ]
    chosen_slots = random.sample(all_slots, pair_count)
    colors = random.sample(COLOR_ORDER, pair_count)
    orientations = _orientation_schedule(pair_count)

    specs: list[tuple[int, tuple[int, int, int, int], str]] = []
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    for color, slot, orientation in zip(colors, chosen_slots, orientations):
        box = _sample_box_within(slot)
        points = _pair_points(box, orientation)
        if points[0] in occupied_input or points[1] in occupied_input:
            return None

        output_cells = _output_cells_for_pair(box, orientation)
        if occupied_output & output_cells:
            return None

        occupied_input.update(points)
        occupied_output.update(output_cells)
        specs.append((int(color), box, orientation))

    inp = np.zeros((height, width), dtype=int)
    for color, box, orientation in specs:
        p0, p1 = _pair_points(box, orientation)
        inp[p0] = color
        inp[p1] = color

    for color in colors:
        coords = np.argwhere(inp == int(color))
        if coords.shape != (2, 2):
            return None

    expected = _build_expected(specs, inp.shape)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
