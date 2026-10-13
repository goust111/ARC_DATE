from __future__ import annotations

import numpy as np


CONTROL_FILL = 5
MARKER_COLOR = 8
PAYLOAD_COLORS = np.array([1, 2, 3, 4, 6, 7, 9], dtype=int)
WIDTH_RANGE = (7, 13)  # inclusive low, exclusive high
PAYLOAD_ROW_RANGE = (4, 9)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _control_info(top_row: np.ndarray) -> tuple[int, int, int]:
    values, counts = np.unique(top_row, return_counts=True)
    assert len(values) == 2, "control row must contain filler and one marker color"

    singleton = values[counts == 1]
    repeated = values[counts == len(top_row) - 1]
    assert singleton.size == 1 and repeated.size == 1, "control row must have exactly one marker"

    marker = int(singleton[0])
    filler = int(repeated[0])
    marker_col = int(np.flatnonzero(top_row == marker)[0])
    return marker, filler, marker_col


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "expected a 2D grid with a control row"

    marker, _filler, shift = _control_info(g[0])
    payload = g[1:]
    out = np.empty_like(payload)
    trace: list[tuple[int, int, int]] = []

    for row_i, row in enumerate(payload):
        hits = np.flatnonzero(row == marker)
        assert hits.size == 1, f"payload row {row_i} must contain exactly one marker pixel"
        input_marker_col = int(hits[0])

        rolled = np.roll(row, shift)
        out[row_i] = rolled

        out_hits = np.flatnonzero(rolled == marker)
        assert out_hits.size == 1, "rolling must preserve a unique marker"
        output_marker_col = int(out_hits[0])
        trace.append((int(shift), input_marker_col, output_marker_col))
        shift = output_marker_col

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the top control row, then cyclically roll each payload row right using a cascaded marker-driven shift.

    Input structure:
    - The top row is a control row: one filler color spans the row except for a single cyan marker pixel.
    - Every lower row is a payload row made of visible colored stripes and contains exactly one cyan marker pixel.
    - The control row is not part of the final answer; only the payload rows are transformed.

    Rule:
    - Delete the top control row.
    - Roll the first payload row cyclically to the right by the control marker's zero-based column index.
    - For each later payload row, roll it cyclically to the right by the column index of the previous payload row's cyan marker after that previous row has already been rolled.
    - Return the stack of all rolled payload rows.

    Key insight:
    - The shift is not a single global amount: each output row passes a new command to the next row through the rolled marker position.
    """
    out, _trace = _solve_with_trace(grid)
    return out


def _sample_shift_sequence(width: int, n_rows: int) -> list[int] | None:
    force_zero = _RNG.random() < 0.35
    min_distinct = 3 if n_rows >= 4 and width >= 5 else 2

    for _ in range(200):
        candidate_count = min(width, max(min_distinct, 4 if width >= 6 else min_distinct))
        seed_values = [int(v) for v in _RNG.choice(width, size=candidate_count, replace=False)]
        seq = [seed_values[int(_RNG.integers(candidate_count))] for _ in range(n_rows)]

        if force_zero:
            seq[int(_RNG.integers(n_rows))] = 0

        if len(set(seq)) < min_distinct:
            continue
        if sum(seq[i] != seq[i - 1] for i in range(1, n_rows)) < 2:
            continue
        if not any(v != seq[0] for v in seq[1:]):
            continue
        return seq
    return None


def _build_base_cells(length: int, palette: np.ndarray) -> np.ndarray | None:
    for _ in range(150):
        cells: list[int] = []
        previous = -1
        while len(cells) < length:
            choices = palette[palette != previous]
            color = int(choices[int(_RNG.integers(len(choices)))])
            max_run = min(length - len(cells), 4)
            run_len = int(_RNG.integers(1, max_run + 1))
            cells.extend([color] * run_len)
            previous = color

        arr = np.asarray(cells, dtype=int)
        if len(set(map(int, arr))) < 3:
            continue
        run_count = 1 + sum(int(arr[i] != arr[i - 1]) for i in range(1, length))
        if run_count < 3:
            continue
        return arr
    return None


def _make_payload_row(width: int, marker_col: int) -> np.ndarray | None:
    for _ in range(150):
        palette_size = int(_RNG.integers(3, min(6, len(PAYLOAD_COLORS)) + 1))
        palette = _RNG.choice(PAYLOAD_COLORS, size=palette_size, replace=False)
        base = _build_base_cells(width - 1, np.asarray(palette, dtype=int))
        if base is None:
            continue

        row = np.empty(width, dtype=int)
        row[:marker_col] = base[:marker_col]
        row[marker_col] = MARKER_COLOR
        row[marker_col + 1 :] = base[marker_col:]

        if np.count_nonzero(row == MARKER_COLOR) != 1:
            continue
        if np.any(row == CONTROL_FILL):
            continue
        return row
    return None


def _uniform_shift_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker, _filler, shift = _control_info(g[0])
    payload = g[1:]
    out = np.empty_like(payload)
    for i, row in enumerate(payload):
        assert np.count_nonzero(row == marker) == 1
        out[i] = np.roll(row, shift)
    return out


def _self_marker_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker, _filler, _shift = _control_info(g[0])
    payload = g[1:]
    out = np.empty_like(payload)
    for i, row in enumerate(payload):
        marker_col = int(np.flatnonzero(row == marker)[0])
        out[i] = np.roll(row, marker_col)
    return out


def _prev_input_marker_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker, _filler, shift = _control_info(g[0])
    payload = g[1:]
    out = np.empty_like(payload)
    for i, row in enumerate(payload):
        out[i] = np.roll(row, shift)
        shift = int(np.flatnonzero(row == marker)[0])
    return out


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        out, trace = _solve_with_trace(grid)
    except AssertionError:
        return False

    payload = np.asarray(grid[1:], dtype=int)
    n_rows, width = map(int, payload.shape)
    applied = [step[0] for step in trace]
    input_markers = [step[1] for step in trace]
    output_markers = [step[2] for step in trace]

    if np.count_nonzero(grid == MARKER_COLOR) != n_rows + 1:
        return False
    if np.count_nonzero(grid == CONTROL_FILL) != width - 1:
        return False
    if len({tuple(map(int, row)) for row in payload}) != n_rows:
        return False
    if len({tuple(map(int, row)) for row in out}) != n_rows:
        return False
    if len(set(applied)) < (3 if n_rows >= 4 and width >= 5 else 2):
        return False
    if sum(not np.array_equal(payload[i], out[i]) for i in range(n_rows)) < 2:
        return False
    if sum(input_markers[i] != output_markers[i] for i in range(n_rows)) < 2:
        return False
    if not any(applied[i] != applied[0] for i in range(1, n_rows)):
        return False
    if np.array_equal(out, payload):
        return False
    if np.array_equal(out, _uniform_shift_output(grid)):
        return False
    if np.array_equal(out, _self_marker_output(grid)):
        return False
    if np.array_equal(out, _prev_input_marker_output(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = int(_RNG.integers(*WIDTH_RANGE))
    n_rows = int(_RNG.integers(*PAYLOAD_ROW_RANGE))

    shifts = _sample_shift_sequence(width, n_rows)
    if shifts is None:
        return None

    marker_cols = [int((shifts[i + 1] - shifts[i]) % width) for i in range(n_rows - 1)]
    marker_cols.append(int(_RNG.integers(width)))

    payload_rows = []
    seen_rows: set[tuple[int, ...]] = set()
    for marker_col in marker_cols:
        row = _make_payload_row(width, marker_col)
        if row is None:
            return None
        key = tuple(map(int, row))
        if key in seen_rows:
            return None
        seen_rows.add(key)
        payload_rows.append(row)

    grid = np.full((n_rows + 1, width), CONTROL_FILL, dtype=int)
    grid[0, shifts[0]] = MARKER_COLOR
    grid[1:] = np.asarray(payload_rows, dtype=int)

    if not _is_valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-roll input")
