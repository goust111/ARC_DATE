from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
HEIGHT_RANGE = (9, 15)
HALF_WIDTH_RANGE = (8, 12)
COLOR_COUNT_RANGE = (3, 4)
TARGET_TOTAL_RANGE = (7, 12)
MAX_HALF_ATTEMPTS = 500
MAX_CASE_ATTEMPTS = 800


def _neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def _color_counter(grid: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in np.asarray(grid).ravel() if int(value) not in (BG, MARKER))


def _blank_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) != BG:
                continue
            neighbors = 0
            for nr, nc in _neighbors4(row, col):
                if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) not in (BG, MARKER):
                    neighbors += 1
            if neighbors >= 2:
                candidates.append((row, col))
    return candidates


def _selected_half(markers: np.ndarray, width: int) -> tuple[slice, slice]:
    half_width = width // 2
    cols = [int(col) for _row, col in markers]
    if all(col < half_width for col in cols):
        return slice(0, half_width), slice(half_width, width)
    if all(col >= half_width for col in cols):
        return slice(half_width, width), slice(0, half_width)
    raise AssertionError("marker pair must lie inside exactly one half")


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the marker-selected half, fill its one structural blank using the opposite half's color multiset, and erase the markers.

    Input structure:
    - The grid has black background, even width, and exactly two red marker pixels on one shared row inside one half.
    - The left and right halves contain scattered non-background content pixels drawn from the same non-red palette.
    - Each half has exactly one black content hole: a background cell that belongs inside the local structure and has at least two 4-neighbor content pixels.

    Rule:
    - Use the marker pair to choose which half is kept as the output crop; the opposite half is discarded except as a reference.
    - Count the non-background, non-red colors in both halves and define the completed multiset by taking the per-color maximum count across the two halves.
    - In the kept half, find the unique black cell with at least two 4-neighbor content pixels and fill it with the single color whose count is below the completed multiset.
    - Erase the red markers and return the kept half.

    Key insight:
    - The markers decide which half survives, but the discarded half still provides the missing color count needed to complete the surviving half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 0, "expected an even grid width"

    markers = np.argwhere(g == MARKER)
    assert markers.shape == (2, 2), "expected exactly two red markers"
    assert len({int(row) for row, _col in markers}) == 1, "markers must lie on one row"

    selected_cols, reference_cols = _selected_half(markers, width)
    selected = g[:, selected_cols].copy()
    reference = g[:, reference_cols]
    selected[selected == MARKER] = BG

    selected_counts = _color_counter(selected)
    reference_counts = _color_counter(reference)
    target_counts: Counter[int] = Counter()
    for color in set(selected_counts) | set(reference_counts):
        target_counts[color] = max(selected_counts[color], reference_counts[color])

    deficits = {color: target_counts[color] - selected_counts[color] for color in target_counts if target_counts[color] > selected_counts[color]}
    assert len(deficits) == 1, "kept half must be missing exactly one color count"
    fill_color, deficit = next(iter(deficits.items()))
    assert deficit == 1, "kept half must miss exactly one pixel"

    candidates = _blank_candidates(selected)
    assert len(candidates) == 1, "kept half must have one structural blank"
    fill_row, fill_col = candidates[0]
    selected[fill_row, fill_col] = int(fill_color)
    assert _color_counter(selected) == target_counts, "filled half must match the completed color multiset"
    return selected


def _sample_target_counts(colors: list[int]) -> dict[int, int]:
    total = random.randint(*TARGET_TOTAL_RANGE)
    total = max(total, len(colors))
    counts = [1] * len(colors)
    remaining = total - len(colors)
    while remaining > 0:
        idx = random.randrange(len(colors))
        if counts[idx] < 4:
            counts[idx] += 1
            remaining -= 1
    return {color: count for color, count in zip(colors, counts)}


def _candidate_cells_from_coords(height: int, width: int, cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in cells:
                continue
            neighbors = 0
            for nr, nc in _neighbors4(row, col):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in cells:
                    neighbors += 1
            if neighbors >= 2:
                candidates.append((row, col))
    return candidates


def _build_half(height: int, half_width: int, target_counts: dict[int, int], missing_color: int) -> dict[str, object] | None:
    target_total = int(sum(target_counts.values()))
    allowed = {(row, col) for row in range(2, height - 1) for col in range(1, half_width - 1)}
    allowed_list = sorted(allowed)
    if len(allowed_list) < target_total:
        return None

    for _ in range(MAX_HALF_ATTEMPTS):
        cells = {random.choice(allowed_list)}
        while len(cells) < target_total:
            frontier: list[tuple[int, int]] = []
            for row, col in cells:
                for nr, nc in _neighbors4(row, col):
                    if (nr, nc) in allowed and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_total:
            continue

        removable = []
        for cell in cells:
            if set(_candidate_cells_from_coords(height, half_width, cells - {cell})) == {cell}:
                removable.append(cell)
        if not removable:
            continue

        hole = random.choice(removable)
        observed = np.zeros((height, half_width), dtype=int)
        remaining_cells = list(cells - {hole})
        random.shuffle(remaining_cells)

        colors: list[int] = []
        for color, count in target_counts.items():
            need = count - (1 if color == missing_color else 0)
            colors.extend([int(color)] * int(need))
        random.shuffle(colors)
        assert len(colors) == len(remaining_cells)

        for (row, col), color in zip(remaining_cells, colors):
            observed[row, col] = int(color)

        return {
            "observed": observed,
            "hole": hole,
            "complete_cells": frozenset(cells),
            "observed_cells": frozenset(remaining_cells),
            "missing_color": int(missing_color),
        }
    return None


def _mirror_variants(coords: frozenset[tuple[int, int]], height: int, width: int) -> tuple[frozenset[tuple[int, int]], ...]:
    return (
        coords,
        frozenset((row, width - 1 - col) for row, col in coords),
        frozenset((height - 1 - row, col) for row, col in coords),
        frozenset((height - 1 - row, width - 1 - col) for row, col in coords),
    )


def _choose_marker_columns(half_width: int) -> tuple[int, int]:
    choices = list(range(1, half_width - 1))
    for _ in range(100):
        col0, col1 = sorted(random.sample(choices, 2))
        if col1 - col0 >= 2:
            return col0, col1
    return tuple(sorted(random.sample(choices, 2)))


def _expected_output(meta: dict[str, object]) -> np.ndarray:
    grid = np.asarray(meta["grid"], dtype=int)
    half_width = int(meta["half_width"])
    out = grid[:, :half_width].copy()
    out[out == MARKER] = BG
    hole_row, hole_col = meta["left"]["hole"]  # type: ignore[index]
    out[int(hole_row), int(hole_col)] = int(meta["left"]["missing_color"])  # type: ignore[index]
    return out


def _validate_case(meta: dict[str, object]) -> bool:
    grid = np.asarray(meta["grid"], dtype=int)
    height, width = map(int, grid.shape)
    half_width = int(meta["half_width"])
    target_counts = Counter({int(color): int(count) for color, count in meta["target_counts"].items()})  # type: ignore[union-attr]

    assert width == 2 * half_width and width <= 30 and height <= 30
    assert width > height and width % 2 == 0

    markers = np.argwhere(grid == MARKER)
    assert markers.shape == (2, 2)
    assert len({int(row) for row, _col in markers}) == 1
    assert all(int(col) < half_width for _row, col in markers)
    assert np.all(grid[1, :half_width] == BG), "marker row needs an empty spacer below it"

    left = grid[:, :half_width]
    right = grid[:, half_width:]
    assert set(_color_counter(left)) == set(target_counts)
    assert set(_color_counter(right)) == set(target_counts)

    left_counts = _color_counter(left)
    right_counts = _color_counter(right)
    left_missing = {color: target_counts[color] - left_counts[color] for color in target_counts if target_counts[color] > left_counts[color]}
    right_missing = {color: target_counts[color] - right_counts[color] for color in target_counts if target_counts[color] > right_counts[color]}
    assert len(left_missing) == len(right_missing) == 1
    assert next(iter(left_missing.values())) == next(iter(right_missing.values())) == 1
    assert next(iter(left_missing)) != next(iter(right_missing))

    left_hole = tuple(int(v) for v in meta["left"]["hole"])  # type: ignore[index]
    right_hole = tuple(int(v) for v in meta["right"]["hole"])  # type: ignore[index]
    assert set(_blank_candidates(left)) == {left_hole}
    assert set(_blank_candidates(right)) == {right_hole}

    left_complete = meta["left"]["complete_cells"]  # type: ignore[index]
    right_complete = meta["right"]["complete_cells"]  # type: ignore[index]
    assert all(left_complete != variant for variant in _mirror_variants(right_complete, height, half_width))

    expected = _expected_output(meta)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert _color_counter(solved) == target_counts
    return True


def _generate_case() -> dict[str, object] | None:
    height = random.randint(*HEIGHT_RANGE)
    half_width = random.randint(*HALF_WIDTH_RANGE)
    color_count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, color_count)

    target_counts = _sample_target_counts(colors)
    eligible_missing = [color for color, count in target_counts.items() if count >= 2]
    if len(eligible_missing) < 2:
        return None
    left_missing, right_missing = random.sample(eligible_missing, 2)

    left = _build_half(height, half_width, target_counts, left_missing)
    right = _build_half(height, half_width, target_counts, right_missing)
    if left is None or right is None:
        return None
    if any(left["complete_cells"] == variant for variant in _mirror_variants(right["complete_cells"], height, half_width)):
        return None

    grid = np.zeros((height, 2 * half_width), dtype=int)
    grid[:, :half_width] = left["observed"]
    grid[:, half_width:] = right["observed"]

    marker_col0, marker_col1 = _choose_marker_columns(half_width)
    grid[0, marker_col0] = MARKER
    grid[0, marker_col1] = MARKER

    meta = {
        "grid": grid,
        "half_width": half_width,
        "target_counts": target_counts,
        "left": left,
        "right": right,
    }
    return meta if _validate_case(meta) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_CASE_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return np.asarray(case["grid"], dtype=int)
    raise RuntimeError("failed to generate a valid marked-half multiset-completion input")
