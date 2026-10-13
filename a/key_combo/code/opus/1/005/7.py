from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

HEIGHT_RANGE = (15, 20)
WIDTH_RANGE = (18, 26)
ACTIVE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 600
MIN_FEATURE_GAP = 3
FEATURE_MARGIN = 2


def _separator_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GRAY)]
    assert len(candidates) == 1, "expected exactly one full-width gray separator"
    return int(candidates[0])


def _source_columns(grid: np.ndarray, sep: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:sep, c] == BLUE)
        if rows.size:
            sources.append((int(c), int(rows.max())))
    return sources


def _selected_count(yellow_count: int, step: int) -> int:
    return (int(yellow_count) + int(step) - 1) // int(step)


def _paint_active_column(out: np.ndarray, inp: np.ndarray, sep: int, col: int, source_row: int) -> None:
    gate_count = 0
    for r in range(source_row - 1, -1, -1):
        if int(inp[r, col]) == RED:
            gate_count += 1
        elif int(inp[r, col]) == BLACK:
            out[r, col] = GREEN

    assert gate_count >= 1, "each active channel must cross at least one gate"

    yellow_rows = [r for r in range(sep + 1, inp.shape[0]) if int(inp[r, col]) == YELLOW]
    count = _selected_count(len(yellow_rows), gate_count)

    lower = out[sep + 1 :, col]
    lower[lower == YELLOW] = BLACK
    if count > 0:
        lower[-count:] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each sourced channel upward through its gates, then use the crossed-gate count to compress every N-th yellow payload below the separator.

    Input structure:
    - The grid has black background and one full-width horizontal gray separator row.
    - Above the separator, columns may contain green pixels, red gate centers, and some source columns contain a blue pixel directly above the separator.
    - Below the separator, columns contain scattered yellow pixels on black.

    Rule:
    - For each column that contains a blue source, trace a vertical ray upward from that source to the top edge.
    - Recolor every black cell on that ray to green, and count how many red gate pixels the ray crosses; call that count N.
    - In the same column below the separator, sort the yellow pixels by row from top to bottom and keep items 1, 1+N, 1+2N, ...
    - Erase all original yellow pixels in that column, then place the kept yellow pixels as one contiguous stack against the bottom edge.
    - Columns without a blue source stay unchanged.

    Key insight:
    - The red gates do not just mark a channel; the number of crossed gates becomes that channel's own sampling interval for the yellow payload on the opposite side of the separator.
    """
    inp = np.asarray(grid, dtype=int)
    sep = _separator_row(inp)
    out = inp.copy()

    for col, source_row in _source_columns(inp, sep):
        _paint_active_column(out, inp, sep, col, source_row)

    return out


def _sample_spaced_columns(width: int, count: int, forbidden: set[int] | None = None) -> list[int] | None:
    blocked = forbidden or set()
    candidates = [c for c in range(FEATURE_MARGIN, width - FEATURE_MARGIN) if c not in blocked]
    if len(candidates) < count:
        return None

    for _ in range(300):
        cols = sorted(random.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= MIN_FEATURE_GAP for i in range(len(cols) - 1)):
            return cols
    return None


def _sample_gate_counts(active_count: int) -> list[int]:
    if active_count == 2:
        counts = random.sample([1, 2, 3], 2)
    elif active_count == 3:
        counts = [1, 2, 3]
        random.shuffle(counts)
    else:
        counts = [1, 2, 3, random.choice([1, 2, 3])]
        random.shuffle(counts)
    return counts


def _sample_rows(population: list[int], count: int, *, forbid_bottom_run: bool) -> list[int] | None:
    if count > len(population):
        return None

    bottom_run = population[-count:]
    for _ in range(300):
        rows = sorted(random.sample(population, count))
        if forbid_bottom_run and rows == bottom_run:
            continue
        return rows
    return None


def _active_yellow_count(lower_len: int, gate_count: int) -> int | None:
    choices = []
    for count in range(6, lower_len + 1):
        if _selected_count(count, gate_count) < 2:
            continue
        if gate_count > 1 and count % gate_count == 0:
            continue
        if gate_count == 1 and count == lower_len:
            continue
        choices.append(count)
    if not choices:
        return None
    return int(random.choice(choices))


def _inactive_yellow_count(lower_len: int) -> int:
    candidates = [count for count in range(4, lower_len + 1) if count < lower_len]
    return int(random.choice(candidates))


def _build_grid(spec: dict[str, object]) -> np.ndarray:
    height = int(spec["height"])
    width = int(spec["width"])
    sep = int(spec["sep"])
    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = GRAY

    active_cols = set(spec["active_cols"])
    gate_rows = spec["gate_rows"]
    green_rows = spec["green_rows"]
    yellow_rows = spec["yellow_rows"]

    for col in active_cols:
        grid[sep - 1, int(col)] = BLUE

    for col, rows in gate_rows.items():
        for row in rows:
            grid[int(row), int(col)] = RED

    for col, rows in green_rows.items():
        for row in rows:
            grid[int(row), int(col)] = GREEN

    for col, rows in yellow_rows.items():
        for row in rows:
            grid[int(row), int(col)] = YELLOW

    return grid


def _render_expected_from_spec(spec: dict[str, object]) -> np.ndarray:
    grid = _build_grid(spec)
    out = grid.copy()
    sep = int(spec["sep"])
    active_cols = list(spec["active_cols"])
    gate_rows = spec["gate_rows"]
    yellow_rows = spec["yellow_rows"]

    for col in active_cols:
        for row in range(sep - 2, -1, -1):
            if int(out[row, int(col)]) == BLACK:
                out[row, int(col)] = GREEN

        gate_count = len(gate_rows[int(col)])
        yellow_count = len(yellow_rows[int(col)])
        kept = _selected_count(yellow_count, gate_count)
        out[sep + 1 :, int(col)] = BLACK
        out[-kept:, int(col)] = YELLOW

    return out


def _column_gate_count(grid: np.ndarray, sep: int, col: int) -> int:
    return int(np.sum(grid[:sep, col] == RED))


def _column_yellow_rows(grid: np.ndarray, sep: int, col: int) -> list[int]:
    return [int(r) for r in np.flatnonzero(grid[sep + 1 :, col] == YELLOW) + sep + 1]


def _render_variant(
    grid: np.ndarray,
    extra_active_cols: set[int],
    count_fn,
) -> np.ndarray:
    sep = _separator_row(grid)
    out = grid.copy()
    source_cols = {col for col, _source_row in _source_columns(grid, sep)}
    active_cols = sorted(source_cols | extra_active_cols)

    for col in active_cols:
        if col in source_cols:
            _paint_active_column(out, grid, sep, col, sep - 1)

        yellow_rows = _column_yellow_rows(grid, sep, col)
        if not yellow_rows:
            continue

        gate_count = _column_gate_count(grid, sep, col)
        step = gate_count if gate_count > 0 else 1
        kept = int(count_fn(len(yellow_rows), step))
        lower = out[sep + 1 :, col]
        lower[lower == YELLOW] = BLACK
        if kept > 0:
            lower[-kept:] = YELLOW

    return out


def _wrong_all_yellow(grid: np.ndarray) -> np.ndarray:
    return _render_variant(grid, set(), lambda count, _step: count)


def _wrong_floor_selection(grid: np.ndarray) -> np.ndarray:
    return _render_variant(grid, set(), lambda count, step: count // step)


def _wrong_constant_two(grid: np.ndarray) -> np.ndarray:
    return _render_variant(grid, set(), lambda count, _step: _selected_count(count, 2))


def _wrong_extra_columns(grid: np.ndarray) -> np.ndarray:
    sep = _separator_row(grid)
    extras = {
        c
        for c in range(grid.shape[1])
        if _column_gate_count(grid, sep, c) > 0 or len(_column_yellow_rows(grid, sep, c)) >= 6
    }
    true_sources = {col for col, _source_row in _source_columns(grid, sep)}
    return _render_variant(grid, extras - true_sources, lambda count, step: _selected_count(count, step))


def _validate_case(spec: dict[str, object]) -> np.ndarray | None:
    grid = _build_grid(spec)
    expected = _render_expected_from_spec(spec)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(expected, _wrong_all_yellow(grid)):
        return None
    if np.array_equal(expected, _wrong_floor_selection(grid)):
        return None
    if np.array_equal(expected, _wrong_constant_two(grid)):
        return None
    if np.array_equal(expected, _wrong_extra_columns(grid)):
        return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    sep = height // 2
    lower_rows = list(range(sep + 1, height))
    lower_len = len(lower_rows)
    if lower_len < 7:
        return None

    active_count = random.randint(*ACTIVE_COUNT_RANGE)
    gate_only_count = random.choice([1, 1, 2])
    feature_cols = _sample_spaced_columns(width, active_count + gate_only_count)
    if feature_cols is None:
        return None
    random.shuffle(feature_cols)

    active_cols = sorted(feature_cols[:active_count])
    gate_only_cols = sorted(feature_cols[active_count:])

    gate_rows: dict[int, list[int]] = {}
    green_rows: dict[int, list[int]] = {}
    yellow_rows: dict[int, list[int]] = {}

    for col, gate_count in zip(active_cols, _sample_gate_counts(active_count)):
        rows = sorted(random.sample(list(range(sep - 1)), gate_count))
        gate_rows[col] = rows

        green_candidates = [r for r in range(sep - 1) if r not in rows]
        green_count = random.randint(0, min(3, len(green_candidates)))
        green_rows[col] = sorted(random.sample(green_candidates, green_count))

        yellow_count = _active_yellow_count(lower_len, gate_count)
        if yellow_count is None:
            return None
        rows = _sample_rows(lower_rows, yellow_count, forbid_bottom_run=(gate_count == 1))
        if rows is None:
            return None
        yellow_rows[col] = rows

    for col in gate_only_cols:
        gate_count = random.randint(1, 3)
        rows = sorted(random.sample(list(range(sep - 1)), gate_count))
        gate_rows[col] = rows

        green_candidates = [r for r in range(sep - 1) if r not in rows]
        green_count = random.randint(0, min(2, len(green_candidates)))
        green_rows[col] = sorted(random.sample(green_candidates, green_count))

        yellow_count = _inactive_yellow_count(lower_len)
        rows = _sample_rows(lower_rows, yellow_count, forbid_bottom_run=True)
        if rows is None:
            return None
        yellow_rows[col] = rows

    safe_green_forbidden = set()
    for col in feature_cols:
        safe_green_forbidden.update({col - 1, col, col + 1})
    safe_green_forbidden = {c for c in safe_green_forbidden if 0 <= c < width}
    pure_green_count = random.randint(0, 2)
    pure_green_cols = _sample_spaced_columns(width, pure_green_count, forbidden=safe_green_forbidden)
    if pure_green_cols is None and pure_green_count > 0:
        return None
    for col in pure_green_cols or []:
        green_count = random.randint(1, min(4, sep - 1))
        green_rows[col] = sorted(random.sample(list(range(sep - 1)), green_count))

    remaining = [c for c in range(width) if c not in yellow_rows]
    random.shuffle(remaining)
    plain_yellow_count = random.choice([1, 1, 2])
    plain_yellow_cols = remaining[:plain_yellow_count]
    if len(plain_yellow_cols) != plain_yellow_count:
        return None
    for col in plain_yellow_cols:
        yellow_count = _inactive_yellow_count(lower_len)
        rows = _sample_rows(lower_rows, yellow_count, forbid_bottom_run=True)
        if rows is None:
            return None
        yellow_rows[col] = rows

    spec = {
        "height": height,
        "width": width,
        "sep": sep,
        "active_cols": active_cols,
        "gate_rows": gate_rows,
        "green_rows": green_rows,
        "yellow_rows": yellow_rows,
    }
    return _validate_case(spec)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-gated compression puzzle")
