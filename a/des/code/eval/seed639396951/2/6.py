from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
ORANGE = 7
AZURE = 8
MAROON = 9

# --------------- solver ---------------


def _paint_axis_line(layer: np.ndarray, original_mask: np.ndarray, orient: str, axis: int, color: int) -> None:
    if orient == "vertical":
        for r in range(layer.shape[0]):
            if original_mask[r, axis]:
                continue
            prev = int(layer[r, axis])
            if prev == BG:
                layer[r, axis] = color
            elif prev != color and prev != ORANGE:
                layer[r, axis] = ORANGE
    elif orient == "horizontal":
        for c in range(layer.shape[1]):
            if original_mask[axis, c]:
                continue
            prev = int(layer[axis, c])
            if prev == BG:
                layer[axis, c] = color
            elif prev != color and prev != ORANGE:
                layer[axis, c] = ORANGE
    else:
        raise AssertionError(f"unknown orientation: {orient}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color singleton pixels, infer each pair's mirror axis, and draw that full axis line.

    Input structure
    - The background is black.
    - Every non-black color appears exactly twice.
    - Each pair is either on one row or on one column and is mirror-symmetric about a
      discrete vertical or horizontal axis.

    Rule
    - Same-row pairs imply a vertical axis through the midpoint column and draw an azure line.
    - Same-column pairs imply a horizontal axis through the midpoint row and draw a maroon line.
    - If an azure and a maroon axis cross, that cell becomes orange.
    - Original colored pixels stay unchanged; any axis skips those cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    original_mask = g != BG
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one colored pair"

    line_layer = np.zeros_like(g)
    height, width = g.shape

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()

        if r0 == r1:
            assert c0 != c1, "paired pixels must be distinct"
            col_sum = int(c0 + c1)
            assert col_sum % 2 == 0, "vertical mirror axis must land on a grid column"
            axis_c = col_sum // 2
            assert 0 <= axis_c < width
            _paint_axis_line(line_layer, original_mask, "vertical", axis_c, AZURE)
        elif c0 == c1:
            assert r0 != r1, "paired pixels must be distinct"
            row_sum = int(r0 + r1)
            assert row_sum % 2 == 0, "horizontal mirror axis must land on a grid row"
            axis_r = row_sum // 2
            assert 0 <= axis_r < height
            _paint_axis_line(line_layer, original_mask, "horizontal", axis_r, MAROON)
        else:
            raise AssertionError(f"color {color} does not form a row- or column-aligned mirror pair")

    out = g.copy()
    paint_mask = (~original_mask) & (line_layer != BG)
    out[paint_mask] = line_layer[paint_mask]
    return out


# --------------- generator ---------------

PAIR_COLORS = (1, 2, 3, 4, 5, 6)
GRID_HEIGHT_RANGE = (9, 17)
GRID_WIDTH_RANGE = (9, 17)
MAX_GENERATION_ATTEMPTS = 400
MODE_HORIZONTAL_ONLY = "horizontal_only"
MODE_VERTICAL_ONLY = "vertical_only"
MODE_MIXED = "mixed"
MODES = (MODE_HORIZONTAL_ONLY, MODE_VERTICAL_ONLY, MODE_MIXED)
AXIS_GAP = 2
PAIR_POSITION_GAP = 2


def _spaced_choices(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    combos = [
        combo
        for combo in itertools.combinations(values, count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
    ]
    if not combos:
        return None
    return list(random.choice(combos))


def _sample_counts(mode: str) -> tuple[int, int]:
    if mode == MODE_HORIZONTAL_ONLY:
        return random.randint(2, 5), 0
    if mode == MODE_VERTICAL_ONLY:
        return 0, random.randint(2, 5)
    total = random.randint(2, 5)
    horiz = random.randint(1, total - 1)
    return horiz, total - horiz


def _valid_distances(axis: int, limit: int, forbidden_positions: set[int]) -> list[int]:
    max_dist = min(axis, limit - 1 - axis)
    return [
        dist
        for dist in range(1, max_dist + 1)
        if axis - dist not in forbidden_positions and axis + dist not in forbidden_positions
    ]


def _mode_matches_expectation(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    if mode == MODE_HORIZONTAL_ONLY:
        return bool(np.any(out == AZURE) and not np.any(out == MAROON) and not np.any(out == ORANGE))
    if mode == MODE_VERTICAL_ONLY:
        return bool(np.any(out == MAROON) and not np.any(out == AZURE) and not np.any(out == ORANGE))
    return bool(np.any(out == AZURE) and np.any(out == MAROON) and np.any(out == ORANGE))


def _build_grid(mode: str) -> np.ndarray | None:
    horiz_count, vert_count = _sample_counts(mode)
    total_pairs = horiz_count + vert_count
    assert 1 <= total_pairs <= len(PAIR_COLORS)

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    vertical_axes = _spaced_choices(list(range(1, width - 1)), horiz_count, AXIS_GAP)
    horizontal_axes = _spaced_choices(list(range(1, height - 1)), vert_count, AXIS_GAP)
    if vertical_axes is None or horizontal_axes is None:
        return None

    available_rows = [r for r in range(height) if r not in set(horizontal_axes)]
    available_cols = [c for c in range(width) if c not in set(vertical_axes)]
    pair_rows = _spaced_choices(available_rows, horiz_count, PAIR_POSITION_GAP)
    pair_cols = _spaced_choices(available_cols, vert_count, PAIR_POSITION_GAP)
    if pair_rows is None or pair_cols is None:
        return None

    colors = random.sample(PAIR_COLORS, total_pairs)
    horiz_colors = colors[:horiz_count]
    vert_colors = colors[horiz_count:]

    specs = []
    specs.extend(
        {
            "kind": "same_row",
            "color": color,
            "row": row,
            "axis": axis,
        }
        for color, row, axis in zip(horiz_colors, pair_rows, vertical_axes)
    )
    specs.extend(
        {
            "kind": "same_col",
            "color": color,
            "col": col,
            "axis": axis,
        }
        for color, col, axis in zip(vert_colors, pair_cols, horizontal_axes)
    )
    random.shuffle(specs)

    grid = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    forbidden_cols = set(vertical_axes)
    forbidden_rows = set(horizontal_axes)

    for spec in specs:
        if spec["kind"] == "same_row":
            row = int(spec["row"])
            axis = int(spec["axis"])
            candidates = _valid_distances(axis, width, forbidden_cols)
            random.shuffle(candidates)
            placed = False
            for dist in candidates:
                left = axis - dist
                right = axis + dist
                cells = ((row, left), (row, right))
                if any(cell in occupied for cell in cells):
                    continue
                grid[row, left] = int(spec["color"])
                grid[row, right] = int(spec["color"])
                occupied.update(cells)
                placed = True
                break
            if not placed:
                return None
        else:
            col = int(spec["col"])
            axis = int(spec["axis"])
            candidates = _valid_distances(axis, height, forbidden_rows)
            random.shuffle(candidates)
            placed = False
            for dist in candidates:
                top = axis - dist
                bottom = axis + dist
                cells = ((top, col), (bottom, col))
                if any(cell in occupied for cell in cells):
                    continue
                grid[top, col] = int(spec["color"])
                grid[bottom, col] = int(spec["color"])
                occupied.update(cells)
                placed = True
                break
            if not placed:
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choice(MODES)
        grid = _build_grid(mode)
        if grid is None:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if _mode_matches_expectation(grid, out, mode):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
