from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, label_mask, object_boundary, object_interior


GREEN = int(Color.GREEN)
BLACK = int(Color.BLACK)
SOURCE_COLUMN = 0
COLOR_TO_TARGET_COLUMN = {
    int(Color.TEAL): 2,
    int(Color.RED): 3,
    int(Color.YELLOW): 4,
    int(Color.BLUE): 5,
}
SOURCE_COLORS = tuple(COLOR_TO_TARGET_COLUMN)

GRID_HEIGHT_RANGE = (11, 16)
GRID_WIDTH_RANGE = (9, 13)
SHAPE_COUNT_CHOICES = (1, 2, 2, 3)
PATCH_HEIGHT_RANGE = (4, 6)
PATCH_WIDTH_RANGE = (4, 6)
BOUNDARY_SIZE_RANGE = (10, 20)
SOURCE_COUNT_RANGE = (5, 8)
SHAPE_RETRY_LIMIT = 200
SOURCE_RETRY_LIMIT = 400
GEN_RETRY_LIMIT = 500


def _boundary_only_grid(grid: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(grid, dtype=int) == GREEN, GREEN, BLACK).astype(int)


def _enclosed_mask(grid: np.ndarray) -> np.ndarray:
    boundary_grid = _boundary_only_grid(grid)
    boundary_mask = object_boundary(boundary_grid, background=BLACK)
    interior_mask = object_interior(boundary_grid, background=BLACK)
    return interior_mask & ~boundary_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each left-edge color into its coded column and keep it only if the target cell lies inside a green-enclosed region.

    Input structure:
    - The grid has a black background with one or more internal green closed contours.
    - Column 0 contains colored source pixels in teal, red, yellow, and blue, with at most one source per row.
    - The green contours enclose black regions and never occupy column 0.

    Rule:
    - Detect the cells that are strictly inside the green closed contours, excluding the green boundary itself.
    - Map each colored source pixel in column 0 to its fixed target column on the same row: teal->2, red->3, yellow->4, blue->5.
    - Erase every source pixel from column 0, then draw the color at its target cell only when that target cell is inside an enclosed region.
    - Drop source pixels whose targets land outside all enclosed regions or on the green boundary, and leave all other cells unchanged.

    Key insight:
    - The color decides where a source tries to go, but the green topology decides whether that moved pixel survives there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] > max(COLOR_TO_TARGET_COLUMN.values())

    inside = _enclosed_mask(g)
    out = g.copy()
    for r in range(g.shape[0]):
        color = int(g[r, SOURCE_COLUMN])
        if color not in COLOR_TO_TARGET_COLUMN:
            continue
        out[r, SOURCE_COLUMN] = BLACK
        target_c = COLOR_TO_TARGET_COLUMN[color]
        if inside[r, target_c]:
            out[r, target_c] = color
    return out


def _random_row_intervals(height: int, width: int) -> list[tuple[int, int]] | None:
    rows: list[tuple[int, int]] = []
    left = random.randint(0, width - 3)
    right = random.randint(left + 2, width - 1)
    rows.append((left, right))

    for _ in range(1, height):
        prev_left, prev_right = rows[-1]
        options: list[tuple[int, int]] = []
        for next_left in range(max(0, prev_left - 1), min(width - 3, prev_left + 1) + 1):
            low_right = max(next_left + 2, prev_right - 1)
            high_right = min(width - 1, prev_right + 1)
            for next_right in range(low_right, high_right + 1):
                if max(next_left, prev_left) <= min(next_right, prev_right):
                    options.append((next_left, next_right))
        if not options:
            return None
        rows.append(random.choice(options))
    return rows


def _make_patch_from_rows(rows: list[tuple[int, int]], width: int) -> np.ndarray:
    patch = np.zeros((len(rows), width), dtype=int)
    for r, (left, right) in enumerate(rows):
        patch[r, left : right + 1] = 1
    return patch


def _generate_shape_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SHAPE_RETRY_LIMIT):
        height = random.randint(*PATCH_HEIGHT_RANGE)
        width = random.randint(*PATCH_WIDTH_RANGE)
        rows = _random_row_intervals(height, width)
        if rows is None:
            continue

        filled = _make_patch_from_rows(rows, width)
        if random.random() < 0.5:
            filled = np.flipud(filled)
        if random.random() < 0.5:
            filled = np.fliplr(filled)
        if random.random() < 0.3:
            filled = filled.T

        boundary = object_boundary(filled, background=0)
        interior = object_interior(filled, background=0) & ~boundary
        boundary_size = int(boundary.sum())
        if not (BOUNDARY_SIZE_RANGE[0] <= boundary_size <= BOUNDARY_SIZE_RANGE[1]):
            continue
        if int(interior.sum()) < 2:
            continue
        _labels, n_boundary_components = label_mask(boundary, connectivity=4)
        if n_boundary_components != 1:
            continue
        return boundary.astype(bool), interior.astype(bool)
    raise RuntimeError("failed to generate a closed contour patch")


def _place_shapes() -> tuple[np.ndarray, np.ndarray]:
    n_shapes = random.choice(SHAPE_COUNT_CHOICES)
    patches = [_generate_shape_patch() for _ in range(n_shapes)]
    heights = [patch[0].shape[0] for patch in patches]
    widths = [patch[0].shape[1] for patch in patches]

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(max(0, n_shapes - 1))]
    min_height = top_margin + bottom_margin + sum(heights) + sum(gaps)
    if min_height > GRID_HEIGHT_RANGE[1]:
        raise RuntimeError("shape stack is too tall")

    height = random.randint(max(GRID_HEIGHT_RANGE[0], min_height), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(GRID_WIDTH_RANGE[0], max(widths) + 3), GRID_WIDTH_RANGE[1])

    grid = np.zeros((height, width), dtype=int)
    intended_inside = np.zeros((height, width), dtype=bool)

    cursor = top_margin
    for i, (boundary_patch, interior_patch) in enumerate(patches):
        patch_h, patch_w = boundary_patch.shape
        max_left = min(3, width - patch_w - 1)
        if max_left < 1:
            raise RuntimeError("shape would touch the left border")
        left = random.randint(1, max_left)

        rows, cols = np.argwhere(boundary_patch).T
        grid[cursor + rows, left + cols] = GREEN
        rows, cols = np.argwhere(interior_patch).T
        intended_inside[cursor + rows, left + cols] = True

        cursor += patch_h
        if i < len(gaps):
            cursor += gaps[i]

    computed_inside = _enclosed_mask(grid)
    if not np.array_equal(computed_inside, intended_inside):
        raise RuntimeError("topology mismatch between intended and computed interiors")

    _green_labels, green_components = label_mask(grid == GREEN, connectivity=4)
    if green_components != n_shapes:
        raise RuntimeError("green contours merged unexpectedly")

    return grid, intended_inside


def _row_target_count(inside: np.ndarray, row: int) -> int:
    return sum(int(inside[row, target_c]) for target_c in COLOR_TO_TARGET_COLUMN.values())


def _candidate_pairs(inside: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    inside_pairs: list[tuple[int, int]] = []
    drop_pairs: list[tuple[int, int]] = []
    partial_inside_pairs: list[tuple[int, int]] = []
    partial_drop_pairs: list[tuple[int, int]] = []

    for row in range(inside.shape[0]):
        count = _row_target_count(inside, row)
        is_partial_row = 0 < count < len(COLOR_TO_TARGET_COLUMN)
        for color, target_c in COLOR_TO_TARGET_COLUMN.items():
            pair = (row, color)
            if inside[row, target_c]:
                inside_pairs.append(pair)
                if is_partial_row:
                    partial_inside_pairs.append(pair)
            else:
                drop_pairs.append(pair)
                if is_partial_row:
                    partial_drop_pairs.append(pair)
    return inside_pairs, drop_pairs, partial_inside_pairs, partial_drop_pairs


def _choose_pair(
    pairs: list[tuple[int, int]],
    used_rows: set[int],
    *,
    required_color: int | None = None,
    excluded_color: int | None = None,
) -> tuple[int, int] | None:
    options = [
        pair
        for pair in pairs
        if pair[0] not in used_rows
        and (required_color is None or pair[1] == required_color)
        and (excluded_color is None or pair[1] != excluded_color)
    ]
    if not options:
        return None
    return random.choice(options)


def _select_sources(inside: np.ndarray) -> list[tuple[int, int]]:
    inside_pairs, drop_pairs, partial_inside_pairs, partial_drop_pairs = _candidate_pairs(inside)
    partial_rows = {
        row
        for row in range(inside.shape[0])
        if 0 < _row_target_count(inside, row) < len(COLOR_TO_TARGET_COLUMN)
    }
    if len(partial_rows) < 2 or len(inside_pairs) < 2 or not drop_pairs:
        raise RuntimeError("not enough informative source targets")

    all_pairs = inside_pairs + drop_pairs
    for _ in range(SOURCE_RETRY_LIMIT):
        target_count = random.randint(*SOURCE_COUNT_RANGE)
        chosen: list[tuple[int, int]] = []
        used_rows: set[int] = set()

        first_inside = _choose_pair(partial_inside_pairs, used_rows)
        if first_inside is None:
            continue
        chosen.append(first_inside)
        used_rows.add(first_inside[0])

        partial_drop = _choose_pair(partial_drop_pairs, used_rows)
        if partial_drop is None:
            continue
        chosen.append(partial_drop)
        used_rows.add(partial_drop[0])

        second_inside = _choose_pair(inside_pairs, used_rows, excluded_color=first_inside[1])
        if second_inside is None:
            continue
        chosen.append(second_inside)
        used_rows.add(second_inside[0])

        present_colors = {color for _row, color in chosen}
        missing_colors = [color for color in SOURCE_COLORS if color not in present_colors]
        success = True
        for color in missing_colors:
            pair = _choose_pair(all_pairs, used_rows, required_color=color)
            if pair is None:
                success = False
                break
            chosen.append(pair)
            used_rows.add(pair[0])
        if not success:
            continue

        while len(chosen) < target_count:
            pair = _choose_pair(all_pairs, used_rows)
            if pair is None:
                success = False
                break
            chosen.append(pair)
            used_rows.add(pair[0])
        if not success:
            continue

        inside_count = sum(int(inside[row, COLOR_TO_TARGET_COLUMN[color]]) for row, color in chosen)
        drop_count = len(chosen) - inside_count
        inside_partial = sum(
            1
            for row, color in chosen
            if 0 < _row_target_count(inside, row) < len(COLOR_TO_TARGET_COLUMN)
            and inside[row, COLOR_TO_TARGET_COLUMN[color]]
        )
        drop_partial_count = sum(
            1
            for row, color in chosen
            if 0 < _row_target_count(inside, row) < len(COLOR_TO_TARGET_COLUMN)
            and not inside[row, COLOR_TO_TARGET_COLUMN[color]]
        )
        surviving_colors = {
            color for row, color in chosen if inside[row, COLOR_TO_TARGET_COLUMN[color]]
        }
        if inside_count < 2 or drop_count < 1:
            continue
        if inside_partial < 1 or drop_partial_count < 1:
            continue
        if len(surviving_colors) < 2:
            continue
        return chosen

    raise RuntimeError("failed to choose informative source pixels")


def _render_expected(grid: np.ndarray, inside: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for row in range(out.shape[0]):
        color = int(out[row, SOURCE_COLUMN])
        if color not in COLOR_TO_TARGET_COLUMN:
            continue
        out[row, SOURCE_COLUMN] = BLACK
        target_c = COLOR_TO_TARGET_COLUMN[color]
        if inside[row, target_c]:
            out[row, target_c] = color
    return out


def _validate_sample(grid: np.ndarray, inside: np.ndarray) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert np.all(grid[:, 0] != GREEN)
    assert not np.isin(grid[:, 1:], SOURCE_COLORS).any()

    computed_inside = _enclosed_mask(grid)
    assert np.array_equal(computed_inside, inside)

    green_labels, n_green = label_mask(grid == GREEN, connectivity=4)
    assert 1 <= n_green <= 3
    for label_id in range(1, n_green + 1):
        size = int(np.count_nonzero(green_labels == label_id))
        assert BOUNDARY_SIZE_RANGE[0] <= size <= BOUNDARY_SIZE_RANGE[1]

    sources = [(row, int(grid[row, SOURCE_COLUMN])) for row in range(grid.shape[0]) if int(grid[row, SOURCE_COLUMN]) in COLOR_TO_TARGET_COLUMN]
    assert SOURCE_COUNT_RANGE[0] <= len(sources) <= SOURCE_COUNT_RANGE[1]
    assert len({row for row, _color in sources}) == len(sources)
    assert set(color for _row, color in sources) == set(SOURCE_COLORS)

    inside_count = 0
    drop_count = 0
    inside_partial = 0
    drop_partial = 0
    for row, color in sources:
        target_c = COLOR_TO_TARGET_COLUMN[color]
        row_is_partial = 0 < _row_target_count(inside, row) < len(COLOR_TO_TARGET_COLUMN)
        if inside[row, target_c]:
            inside_count += 1
            if row_is_partial:
                inside_partial += 1
        else:
            drop_count += 1
            if row_is_partial:
                drop_partial += 1

    assert inside_count >= 2
    assert drop_count >= 1
    assert inside_partial >= 1
    assert drop_partial >= 1

    expected = _render_expected(grid, inside)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert not np.array_equal(expected, grid)
    assert np.count_nonzero(expected == GREEN) == np.count_nonzero(grid == GREEN)
    assert np.count_nonzero(expected[:, SOURCE_COLUMN]) == 0


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            grid, inside = _place_shapes()
            for row, color in _select_sources(inside):
                grid[row, SOURCE_COLUMN] = color
            _validate_sample(grid, inside)
            return grid
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle input")
