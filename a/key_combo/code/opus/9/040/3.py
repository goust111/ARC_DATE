from __future__ import annotations

import numpy as np


LEFT_W = 10
RIGHT_W = 10
DIVIDER_COLOR = 5
BG = 0
STRIPE_COLORS = (1, 2, 3)
GEN_HEIGHT_RANGE = (8, 13)  # inclusive-exclusive
GEN_RETRIES = 400


def _minimal_period(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=int).reshape(-1)
    width = int(arr.size)
    assert width > 0
    for period in range(1, width + 1):
        if np.array_equal(arr, arr[np.arange(width) % period]):
            return period
    raise AssertionError("period not found")


def _binary_pattern_library() -> dict[int, list[tuple[int, ...]]]:
    library: dict[int, list[tuple[int, ...]]] = {}
    for period in range(2, 5):
        patterns: list[tuple[int, ...]] = []
        for mask in range(1 << (period - 1)):
            bits = (1,) + tuple((mask >> bit) & 1 for bit in range(period - 1))
            if 0 not in bits:
                continue
            if _minimal_period(np.array(bits, dtype=int)) == period:
                patterns.append(bits)
        library[period] = sorted(set(patterns))
    return library


PATTERN_LIBRARY = _binary_pattern_library()


def _find_divider(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(w)
        if np.all(grid[:, c] == DIVIDER_COLOR) and c > 0 and w - c - 1 == c
    ]
    assert len(candidates) == 1, "expected one centered grey divider"
    return int(candidates[0])


def _render_pattern_row(color: int, pattern: tuple[int, ...], width: int) -> np.ndarray:
    base = np.array(pattern, dtype=int)
    tiled = base[np.arange(width) % len(pattern)]
    return np.where(tiled == 1, int(color), BG)


def _extract_source_patterns(left: np.ndarray) -> dict[int, dict[str, object]]:
    info: dict[int, dict[str, object]] = {}
    for color in STRIPE_COLORS:
        rows = np.where(np.any(left == color, axis=1))[0]
        assert len(rows) == 1, f"expected exactly one source stripe row for color {color}"
        row_index = int(rows[0])
        row = left[row_index]
        assert set(map(int, np.unique(row))) <= {BG, color}
        period = _minimal_period(row)
        assert 2 <= period <= 4, f"unexpected period {period}"
        pattern = tuple(int(v) for v in row[:period])
        assert pattern[0] == color, "pattern phase must start with the stripe color"
        assert np.array_equal(row, np.array(pattern, dtype=int)[np.arange(left.shape[1]) % period])
        info[color] = {"row": row_index, "period": period, "pattern": pattern}
    assert len(info) == len(STRIPE_COLORS)
    return info


def _extract_seeds(right: np.ndarray) -> dict[int, tuple[int, int]]:
    coords = np.argwhere(right != BG)
    assert len(coords) == len(STRIPE_COLORS), "expected exactly three seeds"
    seeds: dict[int, tuple[int, int]] = {}
    for r, c in coords:
        color = int(right[r, c])
        assert color in STRIPE_COLORS, f"unexpected seed color {color}"
        assert color not in seeds, f"duplicate seed color {color}"
        seeds[color] = (int(r), int(c))

    rows = [pos[0] for pos in seeds.values()]
    cols = [pos[1] for pos in seeds.values()]
    assert len(set(rows)) == len(rows), "seed rows must be distinct"
    assert len(set(cols)) == len(cols), "seed columns must be distinct"
    return seeds


def _phase_fill(pattern: tuple[int, ...], seed_local_col: int, width: int) -> np.ndarray:
    period = len(pattern)
    out = np.zeros(width, dtype=int)
    for local_c in range(width):
        out[local_c] = int(pattern[(local_c - seed_local_col) % period])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed on the target panel into the matched periodic stripe from the source panel, using the seed column as the phase origin.

    Input structure:
    - The grid is split into two equal-width 10-column panels by a full-height gray divider column.
    - The left panel has a black background and exactly one horizontal periodic stripe for each of blue, red, and green.
    - The right panel has a black background and exactly one isolated seed pixel of each of those three colors.

    Rule:
    - Keep the left panel and the gray divider unchanged.
    - For each seed color C on the right, find the left-panel row whose stripe uses color C.
    - Read that row's shortest horizontal period and its period sequence, then redraw the full right-panel row with the seed's local column as phase zero.

    Key insight:
    - The seed color chooses which source stripe to copy, while the seed column independently fixes where phase 0 of that stripe lands on the target row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.shape[1] == RIGHT_W and right.shape[1] == RIGHT_W, "expected 10-column panels"

    source = _extract_source_patterns(left)
    seeds = _extract_seeds(right)

    out = g.copy()
    for color, (row, local_c) in seeds.items():
        pattern = tuple(int(v) for v in source[color]["pattern"])
        filled = _phase_fill(pattern, local_c, RIGHT_W)
        out[row, divider + 1 :] = filled

    return out


def _sample_spaced_rows(height: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(height))
    np.random.shuffle(candidates)
    chosen: list[int] = []
    for row in candidates:
        if all(abs(row - other) >= min_gap for other in chosen):
            chosen.append(int(row))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _sample_target_rows(height: int, source_rows: dict[int, int]) -> dict[int, int] | None:
    for _ in range(120):
        rows = _sample_spaced_rows(height, len(STRIPE_COLORS), min_gap=2)
        if rows is None:
            continue
        colors = list(STRIPE_COLORS)
        np.random.shuffle(colors)
        assignment = {color: int(row) for color, row in zip(colors, rows)}
        if all(assignment[color] != source_rows[color] for color in STRIPE_COLORS):
            return assignment
    return None


def _choose_patterns() -> dict[int, tuple[int, ...]]:
    for _ in range(120):
        colors = list(STRIPE_COLORS)
        np.random.shuffle(colors)

        if np.random.random() < 0.5:
            shared_period = int(np.random.choice([3, 4]))
            shared_colors = colors[:2]
            shared_patterns = PATTERN_LIBRARY[shared_period]
            shared_idx = np.random.choice(len(shared_patterns), size=2, replace=False)
            other_color = colors[2]
            other_period = int(
                np.random.choice([period for period in (2, 3, 4) if period != shared_period])
            )

            assignment = {
                shared_colors[0]: shared_patterns[int(shared_idx[0])],
                shared_colors[1]: shared_patterns[int(shared_idx[1])],
                other_color: PATTERN_LIBRARY[other_period][
                    int(np.random.randint(len(PATTERN_LIBRARY[other_period])))
                ],
            }
        else:
            periods = [2, 3, 4]
            np.random.shuffle(periods)
            assignment = {
                color: PATTERN_LIBRARY[period][int(np.random.randint(len(PATTERN_LIBRARY[period])))]
                for color, period in zip(colors, periods)
            }

        if len({len(pattern) for pattern in assignment.values()}) < 2:
            continue
        if not any(sum(pattern) >= 2 for pattern in assignment.values()):
            continue
        return assignment

    raise RuntimeError("failed to choose non-degenerate patterns")


def _build_instance(height: int) -> np.ndarray:
    grid = np.zeros((height, LEFT_W + 1 + RIGHT_W), dtype=int)
    grid[:, LEFT_W] = DIVIDER_COLOR

    patterns = _choose_patterns()
    source_rows_list = _sample_spaced_rows(height, len(STRIPE_COLORS), min_gap=2)
    assert source_rows_list is not None
    source_colors = list(STRIPE_COLORS)
    np.random.shuffle(source_colors)
    source_rows = {color: int(row) for color, row in zip(source_colors, source_rows_list)}

    target_rows = _sample_target_rows(height, source_rows)
    assert target_rows is not None

    local_cols = np.random.choice(RIGHT_W, size=len(STRIPE_COLORS), replace=False)
    seeds = {color: (target_rows[color], int(local_cols[i])) for i, color in enumerate(STRIPE_COLORS)}

    for color in STRIPE_COLORS:
        row = _render_pattern_row(color, patterns[color], LEFT_W)
        grid[source_rows[color], :LEFT_W] = row
        seed_r, seed_c = seeds[color]
        grid[seed_r, LEFT_W + 1 + seed_c] = color

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] == LEFT_W + 1 + RIGHT_W
    assert max(g.shape) <= 30

    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    source = _extract_source_patterns(left)
    seeds = _extract_seeds(right)
    out = main(g)

    periods = {color: int(source[color]["period"]) for color in STRIPE_COLORS}
    patterns = {color: tuple(int(v) for v in source[color]["pattern"]) for color in STRIPE_COLORS}

    assert len(set(periods.values())) >= 2, "at least two distinct periods are required"
    assert any(sum(1 for v in patterns[color] if v != BG) >= 2 for color in STRIPE_COLORS), (
        "at least one stripe must use a multi-hit period pattern"
    )

    same_period_pairs = [
        (a, b)
        for i, a in enumerate(STRIPE_COLORS)
        for b in STRIPE_COLORS[i + 1 :]
        if periods[a] == periods[b]
    ]
    if same_period_pairs:
        assert any(patterns[a] != patterns[b] for a, b in same_period_pairs), (
            "same-period stripes must not all share the same internal pattern"
        )

    phase_sensitive = False
    for color in STRIPE_COLORS:
        row, local_c = seeds[color]
        pattern = patterns[color]
        expected_local = _phase_fill(pattern, local_c, RIGHT_W)
        expected_global = np.where(expected_local != BG, color, BG)
        assert np.array_equal(out[row, divider + 1 :], expected_global)

        left_anchored = _render_pattern_row(color, tuple(1 if v == color else 0 for v in pattern), RIGHT_W)
        if not np.array_equal(expected_global, left_anchored):
            phase_sensitive = True

        assert row != int(source[color]["row"]), "seed row must differ from its source stripe row"

    assert phase_sensitive, "at least one seed must force a non-left-anchored phase"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = int(np.random.randint(*GEN_HEIGHT_RANGE))
        try:
            grid = _build_instance(height)
            _validate_instance(grid)
        except (AssertionError, RuntimeError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid cross-panel periodic-stripe instance")
