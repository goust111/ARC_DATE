from __future__ import annotations

import random

import numpy as np


GREY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (10, 12)
WIDTH_RANGE = (8, 10)
MAX_GENERATION_ATTEMPTS = 500


def _is_uniform_row(row: np.ndarray) -> bool:
    return bool(np.all(row == row[0]))


def _recolor_uniform_rows(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r in range(out.shape[0]):
        if _is_uniform_row(out[r]):
            out[r, :] = GREY
    return out


def _compress_adjacent_identical_rows(grid: np.ndarray) -> np.ndarray:
    keep = [0]
    for r in range(1, grid.shape[0]):
        if not np.array_equal(grid[r], grid[r - 1]):
            keep.append(r)
    return grid[keep].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every uniform row to gray, then collapse consecutive identical rows.

    Input structure:
    - The input is a 2-D color grid whose rows are either fully uniform or contain
      multiple colors.
    - Several neighboring uniform rows may use different original colors, and some
      neighboring mixed rows may already be exact copies of one another.
    - No object extraction is needed: the row vectors themselves are the relevant units.

    Rule:
    - Inspect each row independently.
    - If every cell in a row is the same color, recolor the whole row to gray (5);
      otherwise leave the row unchanged.
    - After this recoloring step, scan top-to-bottom and keep only the first row of
      each consecutive run of identical rows.

    Key insight:
    - Recoloring must happen before compression, because differently colored uniform
      rows become identical gray rows and create new duplicate runs that were absent
      in the original input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2-D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "input grid must be non-empty"

    recolored = _recolor_uniform_rows(g)
    return _compress_adjacent_identical_rows(recolored)


def _random_mixed_row(width: int, used_rows: list[np.ndarray]) -> np.ndarray:
    for _ in range(200):
        n_segments = random.randint(2, min(4, width))
        cut_points = sorted(random.sample(range(1, width), n_segments - 1))
        colors = [random.choice(INPUT_COLORS)]
        while len(colors) < n_segments:
            choices = [color for color in INPUT_COLORS if color != colors[-1]]
            colors.append(random.choice(choices))

        row = np.empty(width, dtype=int)
        start = 0
        for color, end in zip(colors, cut_points + [width]):
            row[start:end] = color
            start = end

        if _is_uniform_row(row):
            continue
        if any(np.array_equal(row, other) for other in used_rows):
            continue
        return row
    raise RuntimeError("failed to sample a distinct mixed row")


def _sample_group_lengths(template: list[dict[str, int | str]]) -> list[int] | None:
    for _ in range(200):
        lengths = [random.randint(int(group["min_len"]), int(group["max_len"])) for group in template]
        total = sum(lengths)
        if HEIGHT_RANGE[0] <= total <= HEIGHT_RANGE[1]:
            return lengths
    return None


def _build_input_from_template(width: int, template: list[dict[str, int | str]]) -> np.ndarray | None:
    lengths = _sample_group_lengths(template)
    if lengths is None:
        return None

    rows: list[np.ndarray] = []
    used_mixed_rows: list[np.ndarray] = []

    for group, length in zip(template, lengths):
        kind = str(group["kind"])
        if kind == "uniform":
            colors = random.sample(INPUT_COLORS, length)
            for color in colors:
                rows.append(np.full(width, color, dtype=int))
        else:
            mixed_row = _random_mixed_row(width, used_mixed_rows)
            used_mixed_rows.append(mixed_row.copy())
            for _ in range(length):
                rows.append(mixed_row.copy())

    return np.stack(rows, axis=0)


def _count_uniform_diff_pairs(grid: np.ndarray) -> int:
    count = 0
    for r in range(grid.shape[0] - 1):
        row_a = grid[r]
        row_b = grid[r + 1]
        if _is_uniform_row(row_a) and _is_uniform_row(row_b) and int(row_a[0]) != int(row_b[0]):
            count += 1
    return count


def _count_identical_mixed_pairs(grid: np.ndarray) -> int:
    count = 0
    for r in range(grid.shape[0] - 1):
        row_a = grid[r]
        row_b = grid[r + 1]
        if not _is_uniform_row(row_a) and np.array_equal(row_a, row_b):
            count += 1
    return count


def _valid_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if not (HEIGHT_RANGE[0] <= grid.shape[0] <= HEIGHT_RANGE[1]):
        return False
    if not (WIDTH_RANGE[0] <= grid.shape[1] <= WIDTH_RANGE[1]):
        return False
    if _count_uniform_diff_pairs(grid) < 2:
        return False
    if _count_identical_mixed_pairs(grid) < 1:
        return False

    recolored = _recolor_uniform_rows(grid)
    out = main(grid)
    if out.shape[1] != grid.shape[1]:
        return False
    if out.shape[0] >= grid.shape[0]:
        return False
    if not np.array_equal(out, _compress_adjacent_identical_rows(recolored)):
        return False
    if np.any(recolored == GREY) and not np.any(out == GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    templates = [
        [
            {"kind": "uniform", "min_len": 2, "max_len": 3},
            {"kind": "mixed", "min_len": 2, "max_len": 3},
            {"kind": "uniform", "min_len": 2, "max_len": 3},
            {"kind": "mixed", "min_len": 1, "max_len": 2},
            {"kind": "uniform", "min_len": 1, "max_len": 2},
        ],
        [
            {"kind": "mixed", "min_len": 1, "max_len": 2},
            {"kind": "uniform", "min_len": 2, "max_len": 3},
            {"kind": "mixed", "min_len": 2, "max_len": 3},
            {"kind": "uniform", "min_len": 2, "max_len": 3},
            {"kind": "mixed", "min_len": 1, "max_len": 2},
        ],
    ]

    for _ in range(MAX_GENERATION_ATTEMPTS):
        width = random.randint(*WIDTH_RANGE)
        template = random.choice(templates)
        grid = _build_input_from_template(width, template)
        if grid is None:
            continue
        if _valid_generated_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
