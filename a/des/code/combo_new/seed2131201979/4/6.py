from __future__ import annotations

import numpy as np


BG = 0
YELLOW = 4
COLOR_POOL = np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype=int)
_RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _pair_bounds(values: np.ndarray) -> tuple[int, int]:
    a, b = map(int, values.tolist())
    return (a, b) if a < b else (b, a)


def _trace_segment(
    layer: np.ndarray,
    mixed: np.ndarray,
    *,
    axis: str,
    fixed: int,
    start: int,
    end: int,
    color: int,
) -> None:
    if axis == "h":
        for c in range(int(start) + 1, int(end)):
            if int(layer[fixed, c]) == -1:
                layer[fixed, c] = int(color)
            elif int(layer[fixed, c]) != int(color):
                mixed[fixed, c] = True
        return

    for r in range(int(start) + 1, int(end)):
        if int(layer[r, fixed]) == -1:
            layer[r, fixed] = int(color)
        elif int(layer[r, fixed]) != int(color):
            mixed[r, fixed] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-colored pixel pair with a straight ray span whose crossings turn yellow.

    Input structure:
    - The background is black.
    - Each non-background color appears exactly twice.
    - The two pixels of a given color share one row or one column, and every cell between them is background.

    Rule:
    - For each non-background color, locate its two pixels and determine whether they form a horizontal or vertical pair.
    - From both pixels, grow rays toward the partner; together they fill exactly the background cells between the pair.
    - Cells traversed by only one color become that color, while cells traversed by different colors become yellow.
    - The original colored endpoint pixels remain unchanged.

    Key insight:
    - The two matching pixels of a color define both the ray direction and its exact length, so the task reduces to drawing the full segment between each pair and then resolving overlaps by a yellow crossing rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    layer = np.full(g.shape, -1, dtype=int)
    mixed = np.zeros(g.shape, dtype=bool)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert len(coords) == 2, "each non-background color must appear exactly twice"

        (r0, c0), (r1, c1) = coords.tolist()
        if int(r0) == int(r1):
            left, right = _pair_bounds(coords[:, 1])
            assert np.all(g[int(r0), int(left) + 1 : int(right)] == BG), "horizontal partner gap must be empty"
            _trace_segment(
                layer,
                mixed,
                axis="h",
                fixed=int(r0),
                start=int(left),
                end=int(right),
                color=int(color),
            )
        else:
            assert int(c0) == int(c1), "paired pixels must share a row or a column"
            top, bottom = _pair_bounds(coords[:, 0])
            assert np.all(g[int(top) + 1 : int(bottom), int(c0)] == BG), "vertical partner gap must be empty"
            _trace_segment(
                layer,
                mixed,
                axis="v",
                fixed=int(c0),
                start=int(top),
                end=int(bottom),
                color=int(color),
            )

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) != BG or int(layer[r, c]) == -1:
                continue
            out[r, c] = YELLOW if bool(mixed[r, c]) else int(layer[r, c])

    return out


def _sample_distinct(pool: range, count: int) -> list[int] | None:
    values = np.array(list(pool), dtype=int)
    if count < 0 or len(values) < count:
        return None
    return sorted(map(int, _RNG.choice(values, size=count, replace=False)))


def _sample_pair_in_range(lo: int, hi: int, min_gap: int = 2) -> tuple[int, int] | None:
    if hi - lo < min_gap:
        return None
    left = int(_RNG.integers(lo, hi - min_gap + 1))
    right = int(_RNG.integers(left + min_gap, hi + 1))
    return int(left), int(right)


def _sample_pair_covering(anchor: int, lo: int, hi: int) -> tuple[int, int] | None:
    if anchor - 1 < lo or anchor + 1 > hi:
        return None
    left = int(_RNG.integers(lo, anchor))
    right = int(_RNG.integers(anchor + 1, hi + 1))
    return int(left), int(right)


def _sample_pair_covering_many(anchors: list[int], lo: int, hi: int) -> tuple[int, int] | None:
    first = min(anchors)
    last = max(anchors)
    if first - 1 < lo or last + 1 > hi:
        return None
    left = int(_RNG.integers(lo, first))
    right = int(_RNG.integers(last + 1, hi + 1))
    return int(left), int(right)


def _count_crossings(segments: list[tuple[str, int, int, int]]) -> int:
    horizontals = [seg for seg in segments if seg[0] == "h"]
    verticals = [seg for seg in segments if seg[0] == "v"]
    total = 0
    for _, row, left, right in horizontals:
        for _, col, top, bottom in verticals:
            if int(left) < int(col) < int(right) and int(top) < int(row) < int(bottom):
                total += 1
    return int(total)


def _render_input(shape: tuple[int, int], segments: list[tuple[int, str, int, int, int]]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    for color, axis, fixed, start, end in segments:
        if axis == "h":
            if int(grid[fixed, start]) != BG or int(grid[fixed, end]) != BG:
                return None
            grid[fixed, start] = int(color)
            grid[fixed, end] = int(color)
        else:
            if int(grid[start, fixed]) != BG or int(grid[end, fixed]) != BG:
                return None
            grid[start, fixed] = int(color)
            grid[end, fixed] = int(color)
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    for color in (int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == int(color))
        if len(coords) != 2:
            return False
        (r0, c0), (r1, c1) = coords.tolist()
        if int(r0) == int(r1):
            left, right = _pair_bounds(coords[:, 1])
            if np.any(grid[int(r0), int(left) + 1 : int(right)] != BG):
                return False
        elif int(c0) == int(c1):
            top, bottom = _pair_bounds(coords[:, 0])
            if np.any(grid[int(top) + 1 : int(bottom), int(c0)] != BG):
                return False
        else:
            return False
    return True


def _build_none_segments(shape: tuple[int, int]) -> list[tuple[str, int, int, int]] | None:
    h, w = shape
    split = h // 2
    top_rows = range(1, split - 1)
    bottom_rows = range(split + 1, h - 1)
    if len(list(top_rows)) < 1 or len(list(bottom_rows)) < 1:
        return None

    h_count = int(_RNG.integers(1, min(3, len(list(top_rows))) + 1))
    v_count = int(_RNG.integers(1, min(3, w - 2) + 1))
    rows = _sample_distinct(top_rows, h_count)
    cols = _sample_distinct(range(1, w - 1), v_count)
    if rows is None or cols is None:
        return None

    segments: list[tuple[str, int, int, int]] = []
    for row in rows:
        pair = _sample_pair_in_range(0, w - 1)
        if pair is None:
            return None
        segments.append(("h", int(row), int(pair[0]), int(pair[1])))
    for col in cols:
        pair = _sample_pair_in_range(split + 1, h - 1)
        if pair is None:
            return None
        segments.append(("v", int(col), int(pair[0]), int(pair[1])))
    return segments


def _build_single_segments(shape: tuple[int, int]) -> list[tuple[str, int, int, int]] | None:
    h, w = shape
    mid = h // 2
    base_rows = list(range(2, mid))
    extra_rows = list(range(mid + 1, h - 1))
    if not base_rows:
        return None

    base_row = int(_RNG.choice(np.array(base_rows, dtype=int)))
    cross_col = int(_RNG.integers(3, w - 3))

    base_pair = _sample_pair_covering(cross_col, 0, w - 1)
    if base_pair is None:
        return None

    top_choices = list(range(0, base_row))
    bottom_choices = list(range(base_row + 1, mid + 1))
    if not top_choices or not bottom_choices:
        return None

    top = int(_RNG.choice(np.array(top_choices, dtype=int)))
    bottom = int(_RNG.choice(np.array(bottom_choices, dtype=int)))

    segments: list[tuple[str, int, int, int]] = [
        ("h", int(base_row), int(base_pair[0]), int(base_pair[1])),
        ("v", int(cross_col), int(top), int(bottom)),
    ]

    max_extras = min(2, len(extra_rows))
    extra_count = int(_RNG.integers(0, max_extras + 1))
    chosen_extra_rows = _sample_distinct(range(mid + 1, h - 1), extra_count)
    if chosen_extra_rows is None:
        return None

    for row in chosen_extra_rows:
        if bool(_RNG.integers(0, 2) == 0):
            pair = _sample_pair_in_range(0, cross_col - 1)
        else:
            pair = _sample_pair_in_range(cross_col + 1, w - 1)
        if pair is None:
            return None
        segments.append(("h", int(row), int(pair[0]), int(pair[1])))

    return segments


def _build_multi_segments(shape: tuple[int, int]) -> list[tuple[str, int, int, int]] | None:
    h, w = shape
    h_count = int(_RNG.integers(2, 4))
    v_count = int(_RNG.integers(2, 4))

    rows = _sample_distinct(range(2, h - 2), h_count)
    cols = _sample_distinct(range(2, w - 2), v_count)
    if rows is None or cols is None:
        return None

    segments: list[tuple[str, int, int, int]] = []
    for row in rows:
        pair = _sample_pair_covering_many(cols, 0, w - 1)
        if pair is None:
            return None
        segments.append(("h", int(row), int(pair[0]), int(pair[1])))
    for col in cols:
        pair = _sample_pair_covering_many(rows, 0, h - 1)
        if pair is None:
            return None
        segments.append(("v", int(col), int(pair[0]), int(pair[1])))
    return segments


def _colorize_segments(skeleton: list[tuple[str, int, int, int]]) -> list[tuple[int, str, int, int, int]] | None:
    if len(skeleton) > len(COLOR_POOL):
        return None
    colors = _RNG.choice(COLOR_POOL, size=len(skeleton), replace=False)
    return [
        (int(color), axis, int(fixed), int(start), int(end))
        for color, (axis, fixed, start, end) in zip(colors.tolist(), skeleton)
    ]


def _gen_attempt() -> np.ndarray | None:
    scenario = int(_RNG.integers(0, 3))
    h = int(_RNG.integers(10, 21))
    w = int(_RNG.integers(10, 21))
    shape = (h, w)

    if scenario == 0:
        skeleton = _build_none_segments(shape)
    elif scenario == 1:
        skeleton = _build_single_segments(shape)
    else:
        skeleton = _build_multi_segments(shape)
    if skeleton is None:
        return None

    expected_crossings = _count_crossings(skeleton)
    if scenario == 0 and expected_crossings != 0:
        return None
    if scenario == 1 and expected_crossings != 1:
        return None
    if scenario == 2 and expected_crossings < 4:
        return None

    colored = _colorize_segments(skeleton)
    if colored is None:
        return None

    grid = _render_input(shape, colored)
    if grid is None:
        return None

    if bool(_RNG.integers(0, 2) == 0):
        grid = grid.T.copy()

    if not _valid_input(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    yellow_count = int(np.count_nonzero(out == YELLOW))
    if yellow_count != expected_crossings:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
