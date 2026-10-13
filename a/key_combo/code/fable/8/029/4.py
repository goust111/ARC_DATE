from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

TRUE_PERIODS = (3, 5, 7)
FALSE_PERIODS = (2, 3, 5, 7)
GRID_MIN = 13
GRID_MAX = 24
TRUE_LINE_COUNT_MIN = 3
TRUE_LINE_COUNT_MAX = 5
MAX_GENERATION_ATTEMPTS = 500


def _congruence_indices(length: int, period: int, phase: int) -> list[int]:
    return list(range(int(phase), int(length), int(period)))


def _axis_candidates(counts: np.ndarray) -> list[dict]:
    length = int(counts.shape[0])
    candidates: list[dict] = []
    for period in range(2, length):
        for phase in range(period):
            indices = _congruence_indices(length, period, phase)
            if len(indices) < 2:
                continue
            support = int(sum(int(counts[idx]) for idx in indices))
            occupied = int(sum(int(counts[idx]) > 0 for idx in indices))
            if occupied < 2:
                continue
            candidates.append(
                {
                    "period": int(period),
                    "phase": int(phase),
                    "indices": indices,
                    "support": support,
                    "occupied": occupied,
                }
            )
    return candidates


def _best_axis_class(counts: np.ndarray) -> dict:
    candidates = _axis_candidates(np.asarray(counts, dtype=int))
    if not candidates:
        raise ValueError("no supported congruence class found")
    best_support = max(candidate["support"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["support"] == best_support]
    if len(best) != 1:
        raise ValueError("best congruence class is not unique")
    return best[0]


def _render_output(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    row_idx = sorted(set(map(int, rows)))
    col_idx = sorted(set(map(int, cols)))
    if row_idx:
        out[row_idx, :] = BLUE
    if col_idx:
        out[:, col_idx] = BLUE
    if row_idx and col_idx:
        out[np.ix_(row_idx, col_idx)] = RED
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Recover the hidden periodic grid lines from the strongest row-beat and column-beat of blue evidence dots.

    Input structure:
    - A black background contains only sparse blue pixels.
    - Most blue pixels lie on a hidden family of equally spaced horizontal lines and a hidden family of equally spaced vertical lines.
    - A few blue pixels are impostor dots that do not belong to the true row beat or the true column beat.

    Rule:
    - Count blue-dot support for every periodic row congruence class and every periodic column congruence class.
    - In each axis, keep the unique support-maximizing congruence class; those row indices and column indices are the true grid lines.
    - Paint every true row and every true column blue, then recolor each true row/column intersection red; erase every other input dot.

    Key insight:
    - The real grid is not given explicitly: it is the only row rhythm and column rhythm with uniquely strongest multi-dot evidence, while stray dots fail to win the periodic vote.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert np.isin(grid, [BLACK, BLUE]).all(), "input should contain only black and blue"

    blue = grid == BLUE
    row_counts = blue.sum(axis=1)
    col_counts = blue.sum(axis=0)

    row_class = _best_axis_class(row_counts)
    col_class = _best_axis_class(col_counts)
    return _render_output(grid.shape, row_class["indices"], col_class["indices"])


def _axis_generation_options(length: int) -> list[tuple[int, int, list[int]]]:
    options: list[tuple[int, int, list[int]]] = []
    for period in TRUE_PERIODS:
        if period >= length:
            continue
        for phase in range(period):
            indices = _congruence_indices(length, period, phase)
            if TRUE_LINE_COUNT_MIN <= len(indices) <= TRUE_LINE_COUNT_MAX:
                options.append((int(period), int(phase), indices))
    return options


def _false_axis_options(length: int, true_indices: list[int]) -> list[tuple[int, int, list[int]]]:
    true_set = set(map(int, true_indices))
    options: list[tuple[int, int, list[int]]] = []
    for period in FALSE_PERIODS:
        if period >= length:
            continue
        for phase in range(period):
            indices = _congruence_indices(length, period, phase)
            usable = [idx for idx in indices if idx not in true_set]
            if len(usable) >= 2:
                options.append((int(period), int(phase), usable))
    return options


def _ordered_unique_choices(primary: list[int], secondary: list[int], k: int) -> list[int]:
    first = primary[:]
    second = [value for value in secondary if value not in set(primary)]
    random.shuffle(first)
    random.shuffle(second)
    merged = first + second
    return merged[: min(int(k), len(merged))]


def _add_intersection_evidence(
    points: set[tuple[int, int]],
    active_rows: list[int],
    active_cols: list[int],
) -> None:
    cols = active_cols[:]
    rows = active_rows[:]
    random.shuffle(cols)
    random.shuffle(rows)

    for idx, row in enumerate(active_rows):
        points.add((int(row), int(cols[idx % len(cols)])))
    for idx, col in enumerate(active_cols):
        points.add((int(rows[idx % len(rows)]), int(col)))

    fill_probability = random.uniform(0.15, 0.45)
    for row in active_rows:
        for col in active_cols:
            if random.random() < fill_probability:
                points.add((int(row), int(col)))


def _add_row_evidence(
    points: set[tuple[int, int]],
    active_rows: list[int],
    true_cols: list[int],
    false_cols: list[int],
    width: int,
) -> None:
    true_col_set = set(map(int, true_cols))
    fallback = [c for c in range(width) if c not in true_col_set]
    for row in active_rows:
        count = random.randint(1, min(3, len(fallback)))
        cols = _ordered_unique_choices(false_cols, fallback, count)
        for col in cols:
            points.add((int(row), int(col)))


def _add_col_evidence(
    points: set[tuple[int, int]],
    active_cols: list[int],
    true_rows: list[int],
    false_rows: list[int],
    height: int,
) -> None:
    true_row_set = set(map(int, true_rows))
    fallback = [r for r in range(height) if r not in true_row_set]
    for col in active_cols:
        count = random.randint(1, min(3, len(fallback)))
        rows = _ordered_unique_choices(false_rows, fallback, count)
        for row in rows:
            points.add((int(row), int(col)))


def _add_fake_points(
    points: set[tuple[int, int]],
    true_rows: list[int],
    true_cols: list[int],
    false_rows: list[int],
    false_cols: list[int],
    shape: tuple[int, int],
) -> None:
    height, width = shape
    true_row_set = set(map(int, true_rows))
    true_col_set = set(map(int, true_cols))

    structured_pool = [(r, c) for r in false_rows for c in false_cols]
    random.shuffle(structured_pool)
    for row, col in structured_pool[: random.randint(1, min(4, len(structured_pool)))]:
        points.add((int(row), int(col)))

    all_offbeat = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r not in true_row_set and c not in true_col_set
    ]
    random.shuffle(all_offbeat)
    extra_needed = random.randint(1, 4)
    for row, col in all_offbeat:
        if (row, col) in points:
            continue
        points.add((int(row), int(col)))
        extra_needed -= 1
        if extra_needed == 0:
            break


def _maybe_drop_supported_line(indices: list[int]) -> list[int]:
    kept = indices[:]
    if len(kept) >= 4 and random.random() < 0.4:
        kept.remove(random.choice(kept))
    return kept


def _build_case(height: int, width: int) -> tuple[np.ndarray, dict] | None:
    row_options = _axis_generation_options(height)
    col_options = _axis_generation_options(width)
    if not row_options or not col_options:
        return None

    row_period, row_phase, true_rows = random.choice(row_options)
    col_period, col_phase, true_cols = random.choice(col_options)

    false_row_options = _false_axis_options(height, true_rows)
    false_col_options = _false_axis_options(width, true_cols)
    if not false_row_options or not false_col_options:
        return None

    false_row_period, false_row_phase, false_rows = random.choice(false_row_options)
    false_col_period, false_col_phase, false_cols = random.choice(false_col_options)

    active_rows = _maybe_drop_supported_line(true_rows)
    active_cols = _maybe_drop_supported_line(true_cols)
    if len(active_rows) < 2 or len(active_cols) < 2:
        return None

    points: set[tuple[int, int]] = set()
    _add_intersection_evidence(points, active_rows, active_cols)
    _add_row_evidence(points, active_rows, true_cols, false_cols, width)
    _add_col_evidence(points, active_cols, true_rows, false_rows, height)
    _add_fake_points(points, true_rows, true_cols, false_rows, false_cols, (height, width))

    grid = np.zeros((height, width), dtype=int)
    for row, col in points:
        grid[int(row), int(col)] = BLUE

    meta = {
        "true_rows": list(map(int, true_rows)),
        "true_cols": list(map(int, true_cols)),
        "active_rows": list(map(int, active_rows)),
        "active_cols": list(map(int, active_cols)),
        "row_period": int(row_period),
        "row_phase": int(row_phase),
        "col_period": int(col_period),
        "col_phase": int(col_phase),
        "false_rows": list(map(int, false_rows)),
        "false_cols": list(map(int, false_cols)),
        "false_row_period": int(false_row_period),
        "false_row_phase": int(false_row_phase),
        "false_col_period": int(false_col_period),
        "false_col_phase": int(false_col_phase),
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.isin(grid, [BLACK, BLUE]).all():
        return False

    blue_points = {(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)}
    if len(blue_points) < 10:
        return False

    true_rows = set(meta["true_rows"])
    true_cols = set(meta["true_cols"])
    if not any(r in true_rows and c not in true_cols for r, c in blue_points):
        return False
    if not any(r not in true_rows and c in true_cols for r, c in blue_points):
        return False
    if not any(r not in true_rows and c not in true_cols for r, c in blue_points):
        return False

    row_counts = (grid == BLUE).sum(axis=1)
    col_counts = (grid == BLUE).sum(axis=0)
    try:
        inferred_rows = _best_axis_class(row_counts)
        inferred_cols = _best_axis_class(col_counts)
    except ValueError:
        return False

    if inferred_rows["indices"] != meta["true_rows"]:
        return False
    if inferred_cols["indices"] != meta["true_cols"]:
        return False

    expected = _render_output(grid.shape, meta["true_rows"], meta["true_cols"])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if actual.shape != grid.shape:
        return False
    if not np.any(actual == RED):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _generate_case(with_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        built = _build_case(height, width)
        if built is None:
            continue
        grid, meta = built
        if _validate_case(grid, meta):
            if with_meta:
                return grid, meta
            return grid
    raise RuntimeError("failed to generate a valid evidence-grid-lines instance")


def generate_input() -> np.ndarray:
    return _generate_case(with_meta=False)
