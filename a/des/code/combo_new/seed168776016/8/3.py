import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
YELLOW = 4
COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_RANGE = (10, 20)
TOTAL_STRIPE_RANGE = (3, 5)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _detect_horizontal_stripes(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    height, width = grid.shape
    min_span = max(4, (width + 1) // 2)
    stripes: list[dict[str, int]] = []
    for row in range(height):
        cols = np.flatnonzero(grid[row] != bg)
        long_runs = [run for run in _contiguous_runs(cols) if run[1] - run[0] + 1 >= min_span]
        if not long_runs:
            continue
        assert len(long_runs) == 1, "expected one long horizontal band per stripe row"
        c0, c1 = long_runs[0]
        stripes.append({"row": row, "c0": c0, "c1": c1})
    return stripes


def _detect_vertical_stripes(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    height, width = grid.shape
    min_span = max(4, (height + 1) // 2)
    stripes: list[dict[str, int]] = []
    for col in range(width):
        rows = np.flatnonzero(grid[:, col] != bg)
        long_runs = [run for run in _contiguous_runs(rows) if run[1] - run[0] + 1 >= min_span]
        if not long_runs:
            continue
        assert len(long_runs) == 1, "expected one long vertical band per stripe column"
        r0, r1 = long_runs[0]
        stripes.append({"col": col, "r0": r0, "r1": r1})
    return stripes


def _plain_positions(
    start: int,
    end: int,
    blocked: Iterable[int],
) -> list[int]:
    blocked_set = set(blocked)
    return [idx for idx in range(start, end + 1) if idx not in blocked_set]


def _decode_horizontal_stripes(
    grid: np.ndarray,
    bg: int,
    verticals: list[dict[str, int]],
) -> list[dict[str, int]]:
    stripes = []
    for stripe in _detect_horizontal_stripes(grid, bg):
        row, c0, c1 = stripe["row"], stripe["c0"], stripe["c1"]
        crossing_cols = [
            vertical["col"]
            for vertical in verticals
            if vertical["r0"] <= row <= vertical["r1"] and c0 <= vertical["col"] <= c1
        ]
        plain_cols = _plain_positions(c0, c1, crossing_cols)
        assert plain_cols, "horizontal stripe must have non-intersection cells"

        values = grid[row, plain_cols]
        priority = int(np.count_nonzero(values == YELLOW))
        assert 1 <= priority <= 5, "marker count must encode a priority from 1 to 5"

        colors = sorted({int(v) for v in values if int(v) not in {bg, YELLOW}})
        assert len(colors) == 1, "horizontal stripe must expose one main color off intersections"
        stripes.append({"row": row, "c0": c0, "c1": c1, "priority": priority, "color": colors[0]})
    return stripes


def _decode_vertical_stripes(
    grid: np.ndarray,
    bg: int,
    horizontals: list[dict[str, int]],
) -> list[dict[str, int]]:
    stripes = []
    for stripe in _detect_vertical_stripes(grid, bg):
        col, r0, r1 = stripe["col"], stripe["r0"], stripe["r1"]
        crossing_rows = [
            horizontal["row"]
            for horizontal in horizontals
            if horizontal["c0"] <= col <= horizontal["c1"] and r0 <= horizontal["row"] <= r1
        ]
        plain_rows = _plain_positions(r0, r1, crossing_rows)
        assert plain_rows, "vertical stripe must have non-intersection cells"

        values = grid[plain_rows, col]
        priority = int(np.count_nonzero(values == YELLOW))
        assert 1 <= priority <= 5, "marker count must encode a priority from 1 to 5"

        colors = sorted({int(v) for v in values if int(v) not in {bg, YELLOW}})
        assert len(colors) == 1, "vertical stripe must expose one main color off intersections"
        stripes.append({"col": col, "r0": r0, "r1": r1, "priority": priority, "color": colors[0]})
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore woven horizontal and vertical color bands using yellow marker counts as layer priorities.

    Input structure:
    - The background is black.
    - Several one-pixel-wide horizontal and vertical colored bands extend across
      most of the grid width or height.
    - Each band has one visible main color plus a few yellow marker pixels on
      non-intersection cells.
    - Colors shown at crossings are unreliable because either crossing band may
      appear on top in the input.

    Rule:
    - Identify each horizontal and vertical band from its long contiguous run.
    - For every band, count its yellow markers to get its priority and recover
      the band's main color from non-yellow, non-intersection cells.
    - Repaint every marker cell back to that main color.
    - At each crossing, keep the color of the higher-priority band; all other
      band cells stay in their own main color and the background stays black.

    Key insight:
    - The yellow pixels are not decoration: their count encodes each band's
      stacking order, while only non-intersection cells reveal the true base color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    horizontals_raw = _detect_horizontal_stripes(g, bg)
    verticals_raw = _detect_vertical_stripes(g, bg)
    assert horizontals_raw and verticals_raw, "expected both horizontal and vertical bands"

    horizontals = _decode_horizontal_stripes(g, bg, verticals_raw)
    verticals = _decode_vertical_stripes(g, bg, horizontals_raw)

    priorities = [stripe["priority"] for stripe in horizontals + verticals]
    assert len(priorities) == len(set(priorities)), "band priorities must be unique"

    out = np.full(g.shape, bg, dtype=int)
    for stripe in horizontals:
        out[stripe["row"], stripe["c0"] : stripe["c1"] + 1] = stripe["color"]
    for stripe in verticals:
        out[stripe["r0"] : stripe["r1"] + 1, stripe["col"]] = stripe["color"]

    for horizontal in horizontals:
        row = horizontal["row"]
        for vertical in verticals:
            col = vertical["col"]
            if horizontal["c0"] <= col <= horizontal["c1"] and vertical["r0"] <= row <= vertical["r1"]:
                if horizontal["priority"] > vertical["priority"]:
                    out[row, col] = horizontal["color"]
                else:
                    out[row, col] = vertical["color"]

    return out


def _sample_non_adjacent_positions(length: int, count: int) -> list[int]:
    assert length >= 2 * count - 1
    base = sorted(random.sample(range(length - count + 1), count))
    return [base_i + i for i, base_i in enumerate(base)]


def _mixed_priorities(horizontal_count: int, vertical_count: int) -> tuple[list[int], list[int]] | None:
    total = horizontal_count + vertical_count
    sampled = random.sample(range(1, 6), total)
    for _ in range(60):
        random.shuffle(sampled)
        horizontal = sampled[:horizontal_count]
        vertical = sampled[horizontal_count:]
        has_horizontal_win = any(h > v for h in horizontal for v in vertical)
        has_vertical_win = any(h < v for h in horizontal for v in vertical)
        if has_horizontal_win and has_vertical_win:
            return horizontal[:], vertical[:]
    return None


def _render_expected(
    height: int,
    width: int,
    horizontal_specs: list[dict[str, object]],
    vertical_specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((height, width), BG_COLOR, dtype=int)
    for stripe in horizontal_specs:
        out[int(stripe["row"]), :] = int(stripe["color"])
    for stripe in vertical_specs:
        out[:, int(stripe["col"])] = int(stripe["color"])

    for horizontal in horizontal_specs:
        row = int(horizontal["row"])
        for vertical in vertical_specs:
            col = int(vertical["col"])
            if int(horizontal["priority"]) > int(vertical["priority"]):
                out[row, col] = int(horizontal["color"])
            else:
                out[row, col] = int(vertical["color"])
    return out


def _render_input(
    height: int,
    width: int,
    horizontal_specs: list[dict[str, object]],
    vertical_specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    horizontal_rows = {int(spec["row"]) for spec in horizontal_specs}
    vertical_cols = {int(spec["col"]) for spec in vertical_specs}

    for stripe in horizontal_specs:
        grid[int(stripe["row"]), :] = int(stripe["color"])
    for stripe in vertical_specs:
        grid[:, int(stripe["col"])] = int(stripe["color"])

    for horizontal in horizontal_specs:
        row = int(horizontal["row"])
        for vertical in vertical_specs:
            col = int(vertical["col"])
            grid[row, col] = random.choice([int(horizontal["color"]), int(vertical["color"])])

    for stripe in horizontal_specs:
        row = int(stripe["row"])
        plain_cols = [c for c in range(width) if c not in vertical_cols]
        for col in random.sample(plain_cols, int(stripe["priority"])):
            grid[row, col] = YELLOW

    for stripe in vertical_specs:
        col = int(stripe["col"])
        plain_rows = [r for r in range(height) if r not in horizontal_rows]
        for row in random.sample(plain_rows, int(stripe["priority"])):
            grid[row, col] = YELLOW

    return grid


def _gen_attempt() -> np.ndarray | None:
    total = random.randint(*TOTAL_STRIPE_RANGE)
    horizontal_count = random.randint(1, total - 1)
    vertical_count = total - horizontal_count

    priorities = _mixed_priorities(horizontal_count, vertical_count)
    if priorities is None:
        return None
    horizontal_priorities, vertical_priorities = priorities

    min_height = max(
        GRID_SIZE_RANGE[0],
        2 * horizontal_count - 1,
        horizontal_count + max(vertical_priorities) + 2,
    )
    min_width = max(
        GRID_SIZE_RANGE[0],
        2 * vertical_count - 1,
        vertical_count + max(horizontal_priorities) + 2,
    )
    if min_height > GRID_SIZE_RANGE[1] or min_width > GRID_SIZE_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_SIZE_RANGE[1])
    width = random.randint(min_width, GRID_SIZE_RANGE[1])

    rows = _sample_non_adjacent_positions(height, horizontal_count)
    cols = _sample_non_adjacent_positions(width, vertical_count)
    colors = random.sample(COLOR_POOL, total)

    horizontal_specs = [
        {"row": row, "color": color, "priority": priority}
        for row, color, priority in zip(rows, colors[:horizontal_count], horizontal_priorities)
    ]
    vertical_specs = [
        {"col": col, "color": color, "priority": priority}
        for col, color, priority in zip(cols, colors[horizontal_count:], vertical_priorities)
    ]

    grid = _render_input(height, width, horizontal_specs, vertical_specs)
    expected = _render_expected(height, width, horizontal_specs, vertical_specs)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-band input")
