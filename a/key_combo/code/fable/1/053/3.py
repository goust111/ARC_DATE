from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    arr = np.asarray(grid)
    if transform_id < 4:
        return np.rot90(arr, transform_id).copy()
    return np.fliplr(np.rot90(arr, transform_id - 4)).copy()


def _uniform_line_candidates(grid: np.ndarray) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == grid[r, 0]):
            candidates.append(("row", r, int(grid[r, 0])))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]):
            candidates.append(("col", c, int(grid[0, c])))
    return candidates


def _crop_non_color(region: np.ndarray, color: int) -> np.ndarray | None:
    box = _bbox(region != color)
    if box is None:
        return None
    return _crop_bbox(region, box)


def _split_by_separator(
    grid: np.ndarray, axis: str, index: int, separator_color: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if axis == "row":
        a = grid[:index, :]
        b = grid[index + 1 :, :]
    else:
        a = grid[:, :index]
        b = grid[:, index + 1 :]
    if a.size == 0 or b.size == 0:
        return None
    cropped_a = _crop_non_color(a, separator_color)
    cropped_b = _crop_non_color(b, separator_color)
    if cropped_a is None or cropped_b is None:
        return None
    return cropped_a, cropped_b


def _partition_ranges(length: int, breaks: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in breaks + [length - 1]:
        ranges.append((start, end))
        start = end + 1
    return ranges


def _recover_skeleton(main_region: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = main_region.shape
    row_breaks = []
    for r in range(h - 1):
        upper = main_region[r, :]
        lower = main_region[r + 1, :]
        visible = (upper != BLACK) & (lower != BLACK)
        if np.any(visible & (upper != lower)):
            row_breaks.append(r)

    col_breaks = []
    for c in range(w - 1):
        left = main_region[:, c]
        right = main_region[:, c + 1]
        visible = (left != BLACK) & (right != BLACK)
        if np.any(visible & (left != right)):
            col_breaks.append(c)

    row_ranges = _partition_ranges(h, row_breaks)
    col_ranges = _partition_ranges(w, col_breaks)

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            patch = main_region[r0 : r1 + 1, c0 : c1 + 1]
            visible = patch[patch != BLACK]
            if np.unique(visible).size > 1:
                raise ValueError("recovered rectangle is not monochromatic")

    return row_ranges, col_ranges


def _matching_transforms(
    key_region: np.ndarray,
    main_region: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> list[tuple[int, np.ndarray]]:
    matches: list[tuple[int, np.ndarray]] = []
    target_shape = (len(row_ranges), len(col_ranges))
    for transform_id in range(8):
        compressed = _transform(key_region, transform_id)
        if compressed.shape != target_shape:
            continue
        ok = True
        for i, (r0, r1) in enumerate(row_ranges):
            for j, (c0, c1) in enumerate(col_ranges):
                patch = main_region[r0 : r1 + 1, c0 : c1 + 1]
                visible = patch[patch != BLACK]
                if visible.size and np.any(visible != compressed[i, j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            matches.append((transform_id, compressed))
    return matches


def _render_mosaic(
    compressed: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((row_ranges[-1][1] + 1, col_ranges[-1][1] + 1), dtype=int)
    for i, (r0, r1) in enumerate(row_ranges):
        for j, (c0, c1) in enumerate(col_ranges):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(compressed[i, j])
    return out


def _solve_pair(main_region: np.ndarray, key_region: np.ndarray) -> np.ndarray:
    row_ranges, col_ranges = _recover_skeleton(main_region)
    matches = _matching_transforms(key_region, main_region, row_ranges, col_ranges)
    if len(matches) != 1:
        raise ValueError("key orientation is not uniquely determined")
    return _render_mosaic(matches[0][1], row_ranges, col_ranges)


def main(grid: np.ndarray) -> np.ndarray:
    """Use a rotated or flipped compressed partition key to repair black occlusions in a tiled mosaic.

    Input structure:
    - One side of the grid contains a large rectangular mosaic made of single-color rectangular partitions, with some pixels turned black as occlusions.
    - A full separator line splits off a smaller auxiliary region containing a compact color matrix.
    - That small matrix is the one-cell-per-partition compression of the full mosaic, but after one unknown dihedral transform.

    Rule:
    - Detect the separator line and isolate the large mosaic and the small key matrix.
    - Recover the mosaic row and column partition skeleton from visible cross-boundary color changes.
    - Among the eight dihedral orientations, keep the unique key orientation whose colors agree with every visible partition cell.
    - Fill every partition with its matched key color and return only the repaired large mosaic.

    Key insight:
    - The small matrix is not directly aligned with the mosaic; only the combination of recovered partition skeleton and unique orientation match makes the hidden colors deterministic.
    """
    g = np.asarray(grid, dtype=int)
    solutions: list[np.ndarray] = []

    for axis, index, color in _uniform_line_candidates(g):
        split = _split_by_separator(g, axis, index, color)
        if split is None:
            continue
        a, b = split
        for main_region, key_region in ((a, b), (b, a)):
            if main_region.size <= key_region.size:
                continue
            try:
                solutions.append(_solve_pair(main_region, key_region))
            except ValueError:
                continue

    if len(solutions) != 1:
        raise AssertionError("expected a unique valid separator and key orientation")
    return solutions[0]


def _ranges_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _expand_cells(
    cells: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((row_ranges[-1][1] + 1, col_ranges[-1][1] + 1), dtype=int)
    for i, (r0, r1) in enumerate(row_ranges):
        for j, (c0, c1) in enumerate(col_ranges):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(cells[i, j])
    return out


def _random_cells(n_rows: int, n_cols: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(3, min(6, n_rows * n_cols))
        palette = random.sample(PALETTE, n_colors)
        cells = np.array(
            [[random.choice(palette) for _ in range(n_cols)] for _ in range(n_rows)],
            dtype=int,
        )
        unique_colors, counts = np.unique(cells, return_counts=True)
        signatures = {
            (_transform(cells, transform_id).shape, _transform(cells, transform_id).tobytes())
            for transform_id in range(8)
        }
        if unique_colors.size < 3:
            continue
        if counts.max() < 2:
            continue
        if any(np.array_equal(cells[i, :], cells[i + 1, :]) for i in range(n_rows - 1)):
            continue
        if any(np.array_equal(cells[:, j], cells[:, j + 1]) for j in range(n_cols - 1)):
            continue
        if len(signatures) != 8:
            continue
        return cells
    raise RuntimeError("failed to generate a non-degenerate compressed key")


def _protected_mask(
    cells: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> np.ndarray:
    protected = np.zeros((row_ranges[-1][1] + 1, col_ranges[-1][1] + 1), dtype=bool)

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            rr = random.randint(r0, r1)
            cc = random.randint(c0, c1)
            protected[rr, cc] = True

    for i in range(cells.shape[0] - 1):
        options = [j for j in range(cells.shape[1]) if cells[i, j] != cells[i + 1, j]]
        j = random.choice(options)
        c0, c1 = col_ranges[j]
        cc = random.randint(c0, c1)
        protected[row_ranges[i][1], cc] = True
        protected[row_ranges[i + 1][0], cc] = True

    for j in range(cells.shape[1] - 1):
        options = [i for i in range(cells.shape[0]) if cells[i, j] != cells[i, j + 1]]
        i = random.choice(options)
        r0, r1 = row_ranges[i]
        rr = random.randint(r0, r1)
        protected[rr, col_ranges[j][1]] = True
        protected[rr, col_ranges[j + 1][0]] = True

    return protected


def _apply_occlusion(clean: np.ndarray, protected: np.ndarray) -> np.ndarray:
    visible = np.ones(clean.shape, dtype=bool)
    h, w = clean.shape

    for _ in range(random.randint(2, 4)):
        hole_h = random.randint(1, max(1, h // 2))
        hole_w = random.randint(1, max(1, w // 2))
        r0 = random.randint(0, h - hole_h)
        c0 = random.randint(0, w - hole_w)
        visible[r0 : r0 + hole_h, c0 : c0 + hole_w] &= protected[r0 : r0 + hole_h, c0 : c0 + hole_w]

    noise = np.random.random(clean.shape) < random.uniform(0.06, 0.18)
    visible[noise & ~protected] = False

    observed = clean.copy()
    observed[~visible] = BLACK
    return observed


def _layout_input(main_region: np.ndarray, key_region: np.ndarray, separator_color: int) -> tuple[np.ndarray, str]:
    h, w = main_region.shape
    kh, kw = key_region.shape
    orientations = []
    if kh <= h:
        orientations.append("vertical")
    if kw <= w:
        orientations.append("horizontal")
    if not orientations:
        raise ValueError("key cannot fit beside the mosaic")

    separator_axis = random.choice(orientations)
    if separator_axis == "vertical":
        grid = np.full((h, w + 1 + kw), separator_color, dtype=int)
        main_left = random.choice([True, False])
        top = random.randint(0, h - kh)
        if main_left:
            grid[:, :w] = main_region
            grid[top : top + kh, w + 1 :] = key_region
        else:
            grid[:, kw + 1 :] = main_region
            grid[top : top + kh, :kw] = key_region
    else:
        grid = np.full((h + 1 + kh, w), separator_color, dtype=int)
        main_top = random.choice([True, False])
        left = random.randint(0, w - kw)
        if main_top:
            grid[:h, :] = main_region
            grid[h + 1 :, left : left + kw] = key_region
        else:
            grid[kh + 1 :, :] = main_region
            grid[:kh, left : left + kw] = key_region

    return grid, separator_axis


def _boundary_witnesses(
    observed: np.ndarray,
    cells: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> bool:
    for i in range(cells.shape[0] - 1):
        found = False
        upper_row = row_ranges[i][1]
        lower_row = row_ranges[i + 1][0]
        for j in range(cells.shape[1]):
            if cells[i, j] == cells[i + 1, j]:
                continue
            c0, c1 = col_ranges[j]
            upper = observed[upper_row, c0 : c1 + 1]
            lower = observed[lower_row, c0 : c1 + 1]
            visible = (upper != BLACK) & (lower != BLACK)
            if np.any(visible & (upper != lower)):
                found = True
                break
        if not found:
            return False

    for j in range(cells.shape[1] - 1):
        found = False
        left_col = col_ranges[j][1]
        right_col = col_ranges[j + 1][0]
        for i in range(cells.shape[0]):
            if cells[i, j] == cells[i, j + 1]:
                continue
            r0, r1 = row_ranges[i]
            left = observed[r0 : r1 + 1, left_col]
            right = observed[r0 : r1 + 1, right_col]
            visible = (left != BLACK) & (right != BLACK)
            if np.any(visible & (left != right)):
                found = True
                break
        if not found:
            return False

    return True


def _count_hole_partitions(
    observed: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> int:
    total = 0
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            patch = observed[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(patch == BLACK):
                total += 1
    return total


def _every_partition_visible(
    observed: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> bool:
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            if not np.any(observed[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
                return False
    return True


def _build_instance() -> dict[str, object]:
    n_rows = random.randint(2, 5)
    n_cols = random.randint(2, 5)
    row_sizes = [random.randint(2, 4) for _ in range(n_rows)]
    col_sizes = [random.randint(2, 4) for _ in range(n_cols)]
    row_ranges = _ranges_from_sizes(row_sizes)
    col_ranges = _ranges_from_sizes(col_sizes)

    cells = _random_cells(n_rows, n_cols)
    clean = _expand_cells(cells, row_ranges, col_ranges)
    protected = _protected_mask(cells, row_ranges, col_ranges)
    observed = _apply_occlusion(clean, protected)

    transform_weights = [1, 2, 2, 2, 2, 2, 2, 2]
    applied_transform = random.choices(range(8), weights=transform_weights, k=1)[0]
    key_region = _transform(cells, applied_transform)

    separator_choices = [color for color in PALETTE if color not in set(map(int, cells.flat))]
    separator_color = random.choice(separator_choices)
    input_grid, separator_axis = _layout_input(observed, key_region, separator_color)

    return {
        "input": input_grid,
        "clean": clean,
        "observed": observed,
        "cells": cells,
        "key": key_region,
        "separator_color": separator_color,
        "separator_axis": separator_axis,
        "row_ranges": row_ranges,
        "col_ranges": col_ranges,
    }


def _validate_instance(instance: dict[str, object]) -> None:
    input_grid = np.asarray(instance["input"], dtype=int)
    clean = np.asarray(instance["clean"], dtype=int)
    observed = np.asarray(instance["observed"], dtype=int)
    cells = np.asarray(instance["cells"], dtype=int)
    key_region = np.asarray(instance["key"], dtype=int)
    separator_color = int(instance["separator_color"])
    row_ranges = instance["row_ranges"]
    col_ranges = instance["col_ranges"]
    assert isinstance(row_ranges, list) and isinstance(col_ranges, list)

    assert input_grid.ndim == 2 and max(input_grid.shape) <= 30
    assert clean.shape == observed.shape
    assert cells.shape == (len(row_ranges), len(col_ranges))
    assert np.any(observed == BLACK)
    assert _every_partition_visible(observed, row_ranges, col_ranges)
    assert _count_hole_partitions(observed, row_ranges, col_ranges) >= 2
    assert _boundary_witnesses(observed, cells, row_ranges, col_ranges)

    black_count = int(np.sum(observed == BLACK))
    assert black_count >= max(3, observed.size // 8)
    assert black_count <= (observed.size * 3) // 5

    recovered_rows, recovered_cols = _recover_skeleton(observed)
    assert recovered_rows == row_ranges
    assert recovered_cols == col_ranges

    matches = _matching_transforms(key_region, observed, row_ranges, col_ranges)
    assert len(matches) == 1
    assert np.array_equal(matches[0][1], cells)

    expected_axis = "col" if instance["separator_axis"] == "vertical" else "row"
    separator_lines = [
        (axis, idx, color)
        for axis, idx, color in _uniform_line_candidates(input_grid)
        if color == separator_color
    ]
    assert len(separator_lines) == 1
    assert separator_lines[0][0] == expected_axis

    solved = main(input_grid)
    assert np.array_equal(solved, clean)
    assert not np.array_equal(solved, observed)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            instance = _build_instance()
            _validate_instance(instance)
            return np.asarray(instance["input"], dtype=int)
        except (AssertionError, ValueError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid compressed-key tiling instance")
