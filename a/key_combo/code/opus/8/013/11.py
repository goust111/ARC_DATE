from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_TO_ROW_COLOR = {
    0: RED,
    1: YELLOW,
    2: GREEN,
}

_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _adjacent_dedup_rows(grid: np.ndarray) -> np.ndarray:
    rows = [grid[0].copy()]
    for r in range(1, grid.shape[0]):
        if not np.array_equal(grid[r], grid[r - 1]):
            rows.append(grid[r].copy())
    return np.stack(rows, axis=0)


def _adjacent_dedup_cols(grid: np.ndarray) -> np.ndarray:
    cols = [grid[:, 0].copy()]
    for c in range(1, grid.shape[1]):
        if not np.array_equal(grid[:, c], grid[:, c - 1]):
            cols.append(grid[:, c].copy())
    return np.stack(cols, axis=1)


def _marker_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for r in range(grid.shape[0]):
        marker_positions = np.flatnonzero(grid[r] == GREY)
        assert len(marker_positions) == 1, "each row must contain exactly one grey marker"
        col = int(marker_positions[0])
        assert col in MARKER_TO_ROW_COLOR, "marker must be in one of the first three columns"
        assert np.count_nonzero(grid[r] != BLACK) == 1, "only the grey marker may be non-background"
        cols.append(col)
    return cols


def _run_values(values: Iterable[int]) -> list[int]:
    runs: list[int] = []
    prev: int | None = None
    for value in values:
        iv = int(value)
        if prev is None or iv != prev:
            runs.append(iv)
            prev = iv
    return runs


def _output_signature(grid: np.ndarray) -> bytes:
    return grid.tobytes() + str(tuple(map(int, grid.shape))).encode()


def _validate_input_contract(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = map(int, grid.shape)
    assert 6 <= h <= 8
    assert 3 <= w <= 5
    assert set(np.unique(grid)).issubset({BLACK, GREY})

    cols = _marker_columns(grid)
    assert set(cols) == {0, 1, 2}, "all three marker columns must appear at least once"
    runs = _run_values(cols)
    assert len(runs) >= 3, "marker sequence must have at least three runs"
    assert len(runs) < len(cols), "at least one run must have length > 1"


def _expected_from_marker_columns(cols: list[int]) -> np.ndarray:
    run_cols = _run_values(cols)
    values = [MARKER_TO_ROW_COLOR[col] for col in run_cols]
    return np.array(values, dtype=int)[:, None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint rows from each marker's column position, then compress adjacent duplicate rows and columns.

    Input structure:
    - A small black grid contains exactly one grey marker pixel in every row.
    - All other cells are black, and every marker lies in column 0, 1, or 2.
    - Across the rows, the marker columns form a sequence with repeated runs and all three marker columns appear.

    Rule:
    - Read the grey marker column in each row.
    - Repaint the whole row by column position: column 0 -> red, column 1 -> yellow, column 2 -> green.
    - Scan rows from top to bottom and delete any row identical to the previous row.
    - Then scan columns from left to right and delete any column identical to the previous column.

    Key insight:
    - Repainting turns marker-column runs into repeated full rows, so row compression reveals the run structure and column compression collapses the uniform rows to one informative column.
    """
    g = np.asarray(grid, dtype=int)
    _validate_input_contract(g)

    recolored = np.zeros_like(g)
    for r, col in enumerate(_marker_columns(g)):
        recolored[r, :] = MARKER_TO_ROW_COLOR[col]

    dedup_rows = _adjacent_dedup_rows(recolored)
    out = _adjacent_dedup_cols(dedup_rows)

    expected = _expected_from_marker_columns(_marker_columns(g))
    assert np.array_equal(out, expected), "compression result must match the marker-column run sequence"
    return out


def _sample_run_sequence(length: int) -> list[int]:
    while True:
        seq = [random.randint(0, 2)]
        for _ in range(1, length):
            choices = [col for col in range(3) if col != seq[-1]]
            seq.append(random.choice(choices))
        if set(seq) == {0, 1, 2}:
            return seq


def _sample_run_lengths(total_rows: int, run_count: int) -> list[int]:
    lengths = [1] * run_count
    extra = total_rows - run_count
    while extra > 0:
        idx = random.randrange(run_count)
        lengths[idx] += 1
        extra -= 1
    assert max(lengths) >= 2
    return lengths


def _expand_runs(run_values: list[int], run_lengths: list[int]) -> list[int]:
    expanded: list[int] = []
    for value, length in zip(run_values, run_lengths):
        expanded.extend([int(value)] * int(length))
    return expanded


def _build_grid(marker_cols: list[int], width: int) -> np.ndarray:
    grid = np.full((len(marker_cols), width), BLACK, dtype=int)
    for r, col in enumerate(marker_cols):
        grid[r, int(col)] = GREY
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(6, 8)
        w = random.randint(3, 5)
        run_count = random.randint(3, h - 1)
        run_values = _sample_run_sequence(run_count)
        run_lengths = _sample_run_lengths(h, run_count)
        marker_cols = _expand_runs(run_values, run_lengths)
        grid = _build_grid(marker_cols, w)

        _validate_input_contract(grid)
        out = main(grid)

        expected = np.array([MARKER_TO_ROW_COLOR[col] for col in run_values], dtype=int)[:, None]
        assert np.array_equal(out, expected)
        assert out.shape == (run_count, 1)
        assert set(marker_cols) == {0, 1, 2}
        assert any(marker_cols[i] == marker_cols[i - 1] for i in range(1, len(marker_cols)))

        input_sig = _output_signature(grid)
        output_sig = _output_signature(out)
        if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(input_sig)
        _SEEN_OUTPUT_SIGS.add(output_sig)
        return grid

    raise RuntimeError("failed to generate a fresh non-degenerate input")
