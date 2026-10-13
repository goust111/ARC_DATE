from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

PERIOD_MIN = 3
PERIOD_MAX = 7
MIN_EXPECTED_PER_ROW = 5
MIN_BBOX_HEIGHT = 3
MIN_BBOX_WIDTH = 5
MAX_BBOX_HEIGHT = 8
MAX_BBOX_WIDTH = 11
MAX_GENERATION_ATTEMPTS = 500


def _expected_cols(width: int, period: int, phase: int) -> list[int]:
    return list(range(int(phase), int(width), int(period)))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _candidate_stats(row: np.ndarray, period: int, phase: int) -> tuple[tuple[int, int, int], list[int]]:
    expected = _expected_cols(len(row), period, phase)
    support = int(sum(int(row[c] == BLUE) for c in expected))
    total_blue = int(np.count_nonzero(row == BLUE))
    extras = total_blue - support
    misses = len(expected) - support
    return (support, -extras, -misses), expected


def _infer_row_pattern(row: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    blue_cols = [int(c) for c in np.flatnonzero(row == BLUE)]
    assert blue_cols, "every occupied row must contain blue dots"

    best_score: tuple[int, int, int] | None = None
    best_rows: list[tuple[int, int, list[int]]] = []
    width = int(len(row))
    for period in range(2, width):
        for phase in range(period):
            score, expected = _candidate_stats(row, period, phase)
            if best_score is None or score > best_score:
                best_score = score
                best_rows = [(period, phase, expected)]
            elif score == best_score:
                best_rows.append((period, phase, expected))

    assert best_score is not None
    assert len(best_rows) == 1, "row period/phase must be uniquely inferable"

    period, phase, expected = best_rows[0]
    missing = [c for c in expected if int(row[c]) == BLACK]
    return int(period), int(phase), expected, missing


def _collect_missing_points(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int, list[int], list[int]]]]:
    points: list[tuple[int, int]] = []
    row_specs: dict[int, tuple[int, int, list[int], list[int]]] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        if not np.any(row == BLUE):
            continue
        period, phase, expected, missing = _infer_row_pattern(row)
        row_specs[int(r)] = (period, phase, expected, missing)
        points.extend((int(r), int(c)) for c in missing)
    return points, row_specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden figure formed by missing beats in row-wise blue dashed rhythms.

    Input structure:
    - A black canvas contains several horizontal blue dashed rows.
    - Each occupied row follows its own arithmetic rhythm: blue dots recur at one fixed phase and period across the row.
    - A few expected blue positions are missing and stay black; other black cells are ordinary gaps of the row rhythm.

    Rule:
    - For each occupied row, infer the unique period and phase that explain the majority rhythm of its blue dots.
    - Mark every column where that rhythm predicts a blue dot but the input has black instead.
    - Take the tight bounding box of all such missing positions and output that crop with missing cells colored red and all other cells black.

    Key insight:
    - The signal is not the visible blue dots but the holes in each row's periodic pattern; those absences form the real image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE}), "input must contain only black and blue"

    missing_points, _row_specs = _collect_missing_points(g)
    assert missing_points, "expected at least one missing beat"

    r0, c0, r1, c1 = _bbox(missing_points)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in missing_points:
        out[r - r0, c - c0] = RED
    return out


def _slots_in_window(expected: list[int], left: int, right: int) -> list[int]:
    return [c for c in expected if left <= c <= right]


def _feasible_row_options(width: int, left: int, right: int, required: set[int]) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for period in range(PERIOD_MIN, min(PERIOD_MAX, width - 1) + 1):
        for phase in range(period):
            expected = _expected_cols(width, period, phase)
            if len(expected) < MIN_EXPECTED_PER_ROW:
                continue
            slots = _slots_in_window(expected, left, right)
            if len(slots) < 2 or not required.issubset(slots):
                continue
            max_missing = min(3, len(slots), len(expected) - 4)
            if len(required) > max_missing:
                continue
            options.append(
                {
                    "period": period,
                    "phase": phase,
                    "expected": expected,
                    "slots": slots,
                    "max_missing": max_missing,
                }
            )
    return options


def _pick_anomalies(
    row_index: int,
    bbox_height: int,
    left: int,
    right: int,
    option: dict[str, object],
    mode: str,
    anchors: list[int],
    reverse_stair: bool,
) -> list[int]:
    slots = list(option["slots"])
    max_missing = int(option["max_missing"])
    required: set[int] = set()
    if row_index == 0:
        required.add(left)
    if row_index == bbox_height - 1:
        required.add(right)

    chosen = set(required)
    available = [c for c in slots if c not in chosen]

    if mode == "columns":
        for anchor in anchors:
            if anchor in available and random.random() < 0.7 and len(chosen) < max_missing:
                chosen.add(anchor)
    elif mode == "stair":
        frac = 0.0 if bbox_height == 1 else row_index / (bbox_height - 1)
        if reverse_stair:
            frac = 1.0 - frac
        target = left + round((right - left) * frac)
        if available and len(chosen) < max_missing:
            nearest = min(available, key=lambda c: (abs(c - target), c))
            chosen.add(nearest)
    else:
        if available and len(chosen) < max_missing:
            extra = random.randint(0, min(2, len(available), max_missing - len(chosen)))
            chosen.update(random.sample(available, extra))

    remaining = [c for c in available if c not in chosen]
    if mode != "scatter" and remaining and len(chosen) < max_missing and random.random() < 0.35:
        chosen.add(random.choice(remaining))

    if row_index in (0, bbox_height - 1) and not chosen:
        chosen.add(random.choice(slots))
    return sorted(chosen)


def _choose_extra_rows(height: int, bbox_rows: list[int], extra_count: int) -> list[int]:
    remaining = [r for r in range(height) if r not in bbox_rows]
    if extra_count > len(remaining):
        return []
    return sorted(random.sample(remaining, extra_count))


def _render_expected(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in points:
        out[r - r0, c - c0] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 29)
    width = random.randint(18, 30)
    bbox_height = random.randint(MIN_BBOX_HEIGHT, min(MAX_BBOX_HEIGHT, height - 2))
    bbox_width = random.randint(MIN_BBOX_WIDTH, min(MAX_BBOX_WIDTH, width - 2))
    top = random.randint(0, height - bbox_height)
    left = random.randint(0, width - bbox_width)
    right = left + bbox_width - 1
    bbox_rows = list(range(top, top + bbox_height))

    mode = random.choice(["scatter", "columns", "stair"])
    inner_cols = list(range(left + 1, right))
    anchor_count = random.randint(1, min(3, max(1, len(inner_cols))))
    anchors = random.sample(inner_cols if inner_cols else list(range(left, right + 1)), anchor_count)
    reverse_stair = random.choice([False, True])

    row_payloads: dict[int, dict[str, object]] = {}
    missing_points: list[tuple[int, int]] = []
    empty_interior_used = False

    for i, r in enumerate(bbox_rows):
        required = set()
        if i == 0:
            required.add(left)
        if i == bbox_height - 1:
            required.add(right)
        options = _feasible_row_options(width, left, right, required)
        if not options:
            return None
        option = random.choice(options)
        anomalies = _pick_anomalies(i, bbox_height, left, right, option, mode, anchors, reverse_stair)

        if 0 < i < bbox_height - 1 and not anomalies:
            empty_interior_used = True
        if len(anomalies) > int(option["max_missing"]):
            return None

        row_payloads[r] = {
            "period": int(option["period"]),
            "phase": int(option["phase"]),
            "expected": list(option["expected"]),
            "missing": anomalies,
        }
        missing_points.extend((r, c) for c in anomalies)

    if len(missing_points) < max(4, bbox_height):
        return None
    if len(missing_points) >= bbox_height * bbox_width:
        return None
    if not empty_interior_used and bbox_height >= 5 and random.random() < 0.35:
        return None

    if _bbox(missing_points) != (top, left, top + bbox_height - 1, right):
        return None

    extra_count = random.randint(1, min(4, height - bbox_height))
    extra_rows = _choose_extra_rows(height, bbox_rows, extra_count)
    for r in extra_rows:
        options = _feasible_row_options(width, left, right, set())
        if not options:
            return None
        option = random.choice(options)
        row_payloads[r] = {
            "period": int(option["period"]),
            "phase": int(option["phase"]),
            "expected": list(option["expected"]),
            "missing": [],
        }

    grid = np.full((height, width), BLACK, dtype=int)
    for r, spec in row_payloads.items():
        missing = set(spec["missing"])
        visible = [c for c in spec["expected"] if c not in missing]
        if len(visible) < 4:
            return None
        grid[int(r), visible] = BLUE

    expected = _render_expected(missing_points)
    try:
        got = main(grid)
        if not np.array_equal(got, expected):
            return None
    except AssertionError:
        return None

    inferred_points, inferred_rows = _collect_missing_points(grid)
    if sorted(inferred_points) != sorted(missing_points):
        return None
    for r, spec in row_payloads.items():
        period, phase, _expected_cols_list, missing = inferred_rows[int(r)]
        if period != int(spec["period"]) or phase != int(spec["phase"]):
            return None
        if sorted(missing) != sorted(spec["missing"]):
            return None

    periods = {int(spec["period"]) for spec in row_payloads.values()}
    if len(periods) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dashed-anomaly ARC input")
