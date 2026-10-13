from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
MAGENTA = 6
COUNT_COLORS = (BLUE, GREEN, YELLOW)

GRID_H_RANGE = (23, 30)
GRID_W_RANGE = (15, 19)
ROW_COUNT_CHOICES = (3, 4)
ROW_MARGIN = 5
ROW_GAP_MIN = 6
RAY_MAX_LEN = 5
MAX_GENERATION_ATTEMPTS = 400


def _extract_specs(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    specs: list[dict[str, object]] = []

    for r in range(height):
        row = g[r]
        control_cols = np.flatnonzero(row == MAGENTA)
        if control_cols.size == 0:
            assert not np.any((row != BLACK) & (row != MAGENTA)), "non-black rows must contain a magenta control"
            continue

        assert control_cols.size == 1, "each active row must contain exactly one magenta control pixel"
        c = int(control_cols[0])
        left_cols = np.flatnonzero((row[:c] != BLACK) & (row[:c] != MAGENTA)).astype(int)
        right_cols = (c + 1 + np.flatnonzero((row[c + 1 :] != BLACK) & (row[c + 1 :] != MAGENTA))).astype(int)
        assert left_cols.size > 0 and right_cols.size > 0, "each side must contribute at least one count pixel"

        colors = {int(row[col]) for col in np.concatenate([left_cols, right_cols])}
        assert len(colors) == 1, "all count pixels on a row must share one color"
        color = colors.pop()

        specs.append(
            {
                "row": r,
                "col": c,
                "color": color,
                "left_count": int(left_cols.size),
                "right_count": int(right_cols.size),
                "left_cols": tuple(int(x) for x in left_cols.tolist()),
                "right_cols": tuple(int(x) for x in right_cols.tolist()),
                "height": height,
                "width": width,
            }
        )

    return specs


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    height = int(shape[0])

    for spec in specs:
        r = int(spec["row"])
        c = int(spec["col"])
        color = int(spec["color"])
        left_count = int(spec["left_count"])
        right_count = int(spec["right_count"])

        assert r - left_count >= 0 and r + right_count < height, "rays must stay within the grid"
        for dist in range(1, left_count + 1):
            if dist % 2 == 1:
                out[r - dist, c] = color
        for dist in range(1, right_count + 1):
            if dist % 2 == 1:
                out[r + dist, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate horizontal side counts around each magenta control into vertical dashed rays.

    Input structure:
    - The grid uses black as background, with 3 or 4 sparse active rows.
    - Each active row has exactly one magenta control pixel plus same-color count pixels on both its left and right sides.
    - Count pixels on one row all share one color, while different rows may use blue, green, or yellow.

    Rule:
    - For each active row, count non-black non-magenta pixels strictly left of the control as L and strictly right of it as R.
    - Use the count-pixel color C from that row to draw a vertical ray upward for L cells and downward for R cells in the control column.
    - Paint only odd distances from the control (1, 3, 5, ...) and leave even distances black to form dashed rays.
    - Erase the original horizontal row content entirely, including the magenta control pixel itself.

    Key insight:
    - The row encodes an instruction rather than an object to preserve: left and right horizontal counts are rotated ninety degrees into upward and downward dashed lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    specs = _extract_specs(g)
    return _render_output(g.shape, specs)


def _row_choices(height: int, count: int) -> list[tuple[int, ...]]:
    rows = range(ROW_MARGIN, height - ROW_MARGIN)
    return [combo for combo in itertools.combinations(rows, count) if min(np.diff(combo)) >= ROW_GAP_MIN]


def _ordered_pairs() -> list[tuple[int, int]]:
    return [(left, right) for left in range(1, RAY_MAX_LEN + 1) for right in range(1, RAY_MAX_LEN + 1) if left != right]


def _sample_length_specs(count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    odd_lengths = [1, 3, 5]
    even_lengths = [2, 4]
    a = int(rng.choice(odd_lengths))
    b = int(rng.choice(even_lengths))
    if rng.random() < 0.5:
        a, b = b, a

    specs = [(a, b), (b, a)]
    pool = [pair for pair in _ordered_pairs() if pair not in specs]
    rng.shuffle(pool)
    while len(specs) < count:
        specs.append(pool[len(specs) - 2])
    rng.shuffle(specs)
    return specs


def _sample_side_columns(
    control_col: int, count: int, width: int, side: str, rng: np.random.Generator
) -> tuple[int, ...]:
    if side == "left":
        candidates = list(range(control_col))
        distance = lambda col: control_col - col
    else:
        candidates = list(range(control_col + 1, width))
        distance = lambda col: col - control_col

    assert len(candidates) >= count
    if len(candidates) == count:
        return tuple(candidates)

    prefer_spread = rng.random() < 0.7
    for _ in range(50):
        picked = sorted(int(x) for x in rng.choice(candidates, size=count, replace=False).tolist())
        dists = [distance(col) for col in picked]
        contiguous = dists == list(range(1, count + 1))
        if prefer_spread and (max(dists) > count or not contiguous):
            return tuple(picked)
        if not prefer_spread:
            return tuple(picked)
    return tuple(sorted(int(x) for x in rng.choice(candidates, size=count, replace=False).tolist()))


def _paint_instruction_row(
    grid: np.ndarray,
    row: int,
    col: int,
    color: int,
    left_count: int,
    right_count: int,
    rng: np.random.Generator,
) -> None:
    grid[row, col] = MAGENTA
    for cc in _sample_side_columns(col, left_count, grid.shape[1], "left", rng):
        grid[row, cc] = color
    for cc in _sample_side_columns(col, right_count, grid.shape[1], "right", rng):
        grid[row, cc] = color


def _has_spread_side(spec: dict[str, object]) -> bool:
    c = int(spec["col"])
    left_cols = tuple(int(x) for x in spec["left_cols"])
    right_cols = tuple(int(x) for x in spec["right_cols"])
    left_count = int(spec["left_count"])
    right_count = int(spec["right_count"])
    left_span = max((c - col for col in left_cols), default=0)
    right_span = max((col - c for col in right_cols), default=0)
    return left_span > left_count or right_span > right_count


def _semantic_sample_ok(grid: np.ndarray) -> bool:
    specs = _extract_specs(grid)
    if len(specs) not in ROW_COUNT_CHOICES:
        return False

    rows = [int(spec["row"]) for spec in specs]
    cols = [int(spec["col"]) for spec in specs]
    if len(set(cols)) != len(cols):
        return False
    if any(r < ROW_MARGIN or r >= grid.shape[0] - ROW_MARGIN for r in rows):
        return False
    if rows != sorted(rows):
        return False
    if any(b - a < ROW_GAP_MIN for a, b in zip(rows, rows[1:])):
        return False

    has_left_shorter = any(int(spec["left_count"]) < int(spec["right_count"]) for spec in specs)
    has_right_shorter = any(int(spec["left_count"]) > int(spec["right_count"]) for spec in specs)
    if not (has_left_shorter and has_right_shorter):
        return False

    if not any(_has_spread_side(spec) for spec in specs):
        return False

    pairs = [(int(spec["left_count"]), int(spec["right_count"])) for spec in specs]
    totals = {}
    has_same_total_different_split = False
    for pair in pairs:
        total = pair[0] + pair[1]
        if total in totals and totals[total] != pair:
            has_same_total_different_split = True
            break
        totals[total] = pair
    if not has_same_total_different_split:
        return False

    lengths = [length for pair in pairs for length in pair]
    if not any(length % 2 == 0 for length in lengths):
        return False
    if not any(length % 2 == 1 for length in lengths):
        return False

    for index, spec in enumerate(specs):
        row = int(spec["row"])
        left_count = int(spec["left_count"])
        right_count = int(spec["right_count"])
        prev_row = rows[index - 1] if index > 0 else None
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        if prev_row is not None and left_count >= row - prev_row:
            return False
        if next_row is not None and right_count >= next_row - row:
            return False

    expected = _render_output(grid.shape, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    nonzero_output = np.argwhere(actual != BLACK)
    for rr_raw, cc_raw in nonzero_output:
        rr = int(rr_raw)
        cc = int(cc_raw)
        matches = [spec for spec in specs if int(spec["col"]) == cc]
        if len(matches) != 1:
            return False
        spec = matches[0]
        dist = abs(rr - int(spec["row"]))
        if dist % 2 == 0 or dist == 0:
            return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_H_RANGE))
    count = int(rng.choice(ROW_COUNT_CHOICES))
    row_sets = _row_choices(height, count)
    if not row_sets:
        return None

    width = int(rng.integers(*GRID_W_RANGE))
    control_candidates = list(range(RAY_MAX_LEN, width - RAY_MAX_LEN))
    if len(control_candidates) < count:
        return None

    rows = tuple(int(x) for x in row_sets[int(rng.integers(0, len(row_sets)))])
    cols = sorted(int(x) for x in rng.choice(control_candidates, size=count, replace=False).tolist())
    rng.shuffle(cols)

    length_specs = _sample_length_specs(count, rng)
    colors = [int(x) for x in rng.choice(np.array(COUNT_COLORS), size=count, replace=True).tolist()]
    if len(set(colors)) == 1:
        colors[int(rng.integers(0, count))] = COUNT_COLORS[int(rng.integers(0, len(COUNT_COLORS)))]

    grid = np.zeros((height, width), dtype=int)
    for row, col, color, (left_count, right_count) in zip(rows, cols, colors, length_specs, strict=True):
        if row - left_count < 0 or row + right_count >= height:
            return None
        _paint_instruction_row(grid, row, col, color, left_count, right_count, rng)

    if not _semantic_sample_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
