from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FLOOR_COLOR = 8
SAND_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GEN_HEIGHT_RANGE = (16, 26)
GEN_WIDTH_RANGE = (10, 24)
GEN_PERIOD_RANGE = (3, 6)
GEN_FLOOR_COUNT_RANGE = (3, 5)
GEN_EXTRA_GRAINS_RANGE = (1, 4)
GEN_ATTEMPTS = 400


def _sand_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BACKGROUND) & (grid != FLOOR_COLOR)


def _infer_floor_rows(grid: np.ndarray) -> tuple[list[int], int, int]:
    rows = np.flatnonzero(np.any(grid == FLOOR_COLOR, axis=1)).astype(int).tolist()
    assert len(rows) >= 2, "need at least two visible floor rows to infer the period"

    diffs = np.diff(rows)
    period = int(diffs[0])
    assert period >= 2
    assert bool(np.all(diffs == period)), "floor rows must form an arithmetic progression"

    phase = rows[0] % period
    assert rows == list(range(rows[0], rows[-1] + 1, period))
    for row in rows:
        count = int(np.count_nonzero(grid[row] == FLOOR_COLOR))
        assert 1 <= count < grid.shape[1], "each floor row must be visible but incomplete"
    return rows, period, phase


def _intervals(height: int, floor_rows: list[int]) -> list[dict[str, int | None | str]]:
    start = 0
    out: list[dict[str, int | None | str]] = []
    for idx, stop in enumerate(floor_rows):
        out.append(
            {
                "kind": "top" if idx == 0 else "middle",
                "start": start,
                "stop": stop,
                "base": stop - 1,
                "support_row": stop,
            }
        )
        start = stop + 1
    out.append(
        {
            "kind": "bottom",
            "start": start,
            "stop": height,
            "base": height - 1,
            "support_row": None,
        }
    )
    return out


def _settle_plan(grid: np.ndarray) -> tuple[list[int], int, int, list[dict[str, int | None | str]], list[dict[str, int | str | bool | None]]]:
    g = np.asarray(grid, dtype=int)
    floor_rows, period, phase = _infer_floor_rows(g)
    intervals = _intervals(g.shape[0], floor_rows)

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(_sand_mask(g)):
        idx = int(np.searchsorted(floor_rows, int(r), side="right"))
        groups.setdefault((int(c), idx), []).append((int(r), int(g[r, c])))

    plan: list[dict[str, int | str | bool | None]] = []
    for (col, idx), grains in groups.items():
        interval = intervals[idx]
        start = int(interval["start"])
        stop = int(interval["stop"])
        base = int(interval["base"])
        support_row = interval["support_row"]
        ordered = sorted(grains, key=lambda item: item[0], reverse=True)
        assert len(ordered) <= stop - start

        for offset, (start_row, color) in enumerate(ordered):
            landing_row = base - offset
            assert start <= landing_row < stop
            plan.append(
                {
                    "col": col,
                    "color": color,
                    "start_row": start_row,
                    "landing_row": landing_row,
                    "kind": str(interval["kind"]),
                    "support_row": None if support_row is None else int(support_row),
                    "support_visible": bool(support_row is not None and g[int(support_row), col] == FLOOR_COLOR),
                    "stack_size": len(ordered),
                }
            )
    return floor_rows, period, phase, intervals, plan


def _oracle_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[_sand_mask(out)] = BACKGROUND

    _floor_rows, _period, _phase, _intervals, plan = _settle_plan(g)
    for item in plan:
        out[int(item["landing_row"]), int(item["col"])] = int(item["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Settle every colored sand grain onto the nearest inferred periodic floor row below it.

    Input structure
    - The grid has black background, cyan floor traces, and non-cyan sand pixels.
    - Cyan pixels appear only on several equally spaced horizontal rows, and each such row is only partially drawn.
    - Sand pixels are never on a floor row.

    Rule
    - Infer the common row step and phase from the visible cyan rows, then treat each of those rows as a full horizontal wall across every column.
    - Process sand grains from lower initial rows to higher initial rows.
    - Each grain falls straight down in its own column, stopping one cell above the nearest floor row below; if no floor lies below, it stops on the bottom edge.
    - If that resting cell is already occupied by an earlier settled grain in the same column, keep stacking upward by one cell at a time.
    - Preserve the visible cyan traces exactly as drawn.

    Key insight
    - The support is mostly invisible horizontally: a grain can stop above an empty-looking column only because a periodic floor row is present there even when that column's cyan trace was never drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    floor_rows, _period, _phase = _infer_floor_rows(g)
    row_array = np.asarray(floor_rows, dtype=int)
    floor_row_set = set(floor_rows)

    out = g.copy()
    sand = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(_sand_mask(g))]
    out[_sand_mask(out)] = BACKGROUND

    occupied: set[tuple[int, int]] = set()
    for start_row, col, color in sorted(sand, key=lambda item: (-item[0], item[1], item[2])):
        idx = int(np.searchsorted(row_array, start_row, side="right"))
        base = floor_rows[idx] - 1 if idx < len(floor_rows) else g.shape[0] - 1
        landing_row = base
        while landing_row >= 0 and (landing_row in floor_row_set or (landing_row, col) in occupied):
            landing_row -= 1
        assert landing_row >= 0
        occupied.add((landing_row, col))
        out[landing_row, col] = color

    return out


def _random_trace_cols(width: int) -> set[int]:
    cols: set[int] = set()
    n_segments = random.randint(1, min(3, max(1, width // 5 + 1)))
    max_seg_len = max(1, width // 3)
    for _ in range(n_segments):
        seg_len = random.randint(1, max_seg_len)
        c0 = random.randint(0, width - seg_len)
        cols.update(range(c0, c0 + seg_len))

    while len(cols) >= width:
        cols.remove(random.choice(tuple(cols)))
    if not cols:
        cols.add(random.randrange(width))
    return cols


def _choose_floor_spec(height: int) -> tuple[list[int], int] | None:
    for _ in range(100):
        period = random.randint(*GEN_PERIOD_RANGE)
        max_floors = min(GEN_FLOOR_COUNT_RANGE[1], (height - 5) // period + 1)
        if max_floors < GEN_FLOOR_COUNT_RANGE[0]:
            continue

        n_floors = random.randint(GEN_FLOOR_COUNT_RANGE[0], max_floors)
        top_min = 3 if period == 3 else 2
        top_max = height - 3 - period * (n_floors - 1)
        if top_max < top_min:
            continue

        top = random.randint(top_min, top_max)
        floor_rows = [top + period * idx for idx in range(n_floors)]
        if floor_rows[-1] > height - 3:
            continue
        return floor_rows, period
    return None


def _pick_column(candidates: list[int], used: set[int]) -> int | None:
    fresh = [col for col in candidates if col not in used]
    pool = fresh if fresh else candidates
    if not pool:
        return None
    return random.choice(pool)


def _available_rows(interval: dict[str, int | None | str], used_cells: set[tuple[int, int]], col: int) -> list[int]:
    start = int(interval["start"])
    stop = int(interval["stop"])
    return [row for row in range(start, stop) if (row, col) not in used_cells]


def _add_grain(
    grains: list[tuple[int, int, int]],
    used_cells: set[tuple[int, int]],
    row: int,
    col: int,
    color: int,
) -> None:
    assert (row, col) not in used_cells
    used_cells.add((row, col))
    grains.append((row, col, color))


def _special_stack_rows(interval: dict[str, int | None | str]) -> tuple[int, int] | None:
    start = int(interval["start"])
    base = int(interval["base"])
    if base - start < 2:
        return None
    return start, base


def _scene_features(grid: np.ndarray) -> dict[str, bool]:
    _floor_rows, _period, _phase, _intervals, plan = _settle_plan(grid)
    features = {
        "top": False,
        "middle": False,
        "bottom": False,
        "stack": False,
        "invisible_support": False,
        "moved": False,
    }
    for item in plan:
        kind = str(item["kind"])
        features[kind] = True
        features["stack"] |= int(item["stack_size"]) >= 2
        features["invisible_support"] |= item["support_row"] is not None and not bool(item["support_visible"])
        features["moved"] |= int(item["landing_row"]) != int(item["start_row"])
    return features


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)

        floor_spec = _choose_floor_spec(height)
        if floor_spec is None:
            continue
        floor_rows, _period = floor_spec
        intervals = _intervals(height, floor_rows)
        top_interval = intervals[0]
        middle_intervals = [interval for interval in intervals if interval["kind"] == "middle"]
        bottom_interval = intervals[-1]

        grid = np.zeros((height, width), dtype=int)
        trace_cols: dict[int, set[int]] = {}
        for row in floor_rows:
            cols = _random_trace_cols(width)
            trace_cols[row] = cols
            for col in cols:
                grid[row, col] = FLOOR_COLOR

        stack_choices = [interval for interval in [top_interval, *middle_intervals] if _special_stack_rows(interval) is not None]
        if not stack_choices:
            continue
        stack_interval = random.choice(stack_choices)
        stack_support = int(stack_interval["support_row"])
        stack_blank_cols = [col for col in range(width) if grid[stack_support, col] != FLOOR_COLOR]
        if not stack_blank_cols:
            continue

        used_start_cells: set[tuple[int, int]] = set()
        used_special_cols: set[int] = set()
        grains: list[tuple[int, int, int]] = []

        stack_col = _pick_column(stack_blank_cols, used_special_cols)
        if stack_col is None:
            continue
        used_special_cols.add(stack_col)
        stack_rows = _special_stack_rows(stack_interval)
        assert stack_rows is not None
        stack_colors = random.sample(SAND_COLORS, 2)
        _add_grain(grains, used_start_cells, stack_rows[0], stack_col, stack_colors[0])
        _add_grain(grains, used_start_cells, stack_rows[1], stack_col, stack_colors[1])

        if stack_interval["kind"] != "top":
            top_support = floor_rows[0]
            top_candidates = list(range(width))
            top_col = _pick_column(top_candidates, used_special_cols)
            if top_col is None:
                continue
            used_special_cols.add(top_col)
            top_row = int(top_interval["start"])
            _add_grain(grains, used_start_cells, top_row, top_col, random.choice(SAND_COLORS))

        if stack_interval["kind"] != "middle":
            middle_interval = random.choice(middle_intervals)
            middle_col = _pick_column(list(range(width)), used_special_cols)
            if middle_col is None:
                continue
            used_special_cols.add(middle_col)
            middle_row = int(middle_interval["start"])
            _add_grain(grains, used_start_cells, middle_row, middle_col, random.choice(SAND_COLORS))

        bottom_col = _pick_column(list(range(width)), used_special_cols)
        if bottom_col is None:
            continue
        used_special_cols.add(bottom_col)
        bottom_row = int(bottom_interval["start"])
        _add_grain(grains, used_start_cells, bottom_row, bottom_col, random.choice(SAND_COLORS))

        n_extra = random.randint(*GEN_EXTRA_GRAINS_RANGE)
        for _extra_idx in range(n_extra):
            interval = random.choice(intervals)
            col = random.randrange(width)
            rows = _available_rows(interval, used_start_cells, col)
            if not rows:
                continue
            row = random.choice(rows)
            _add_grain(grains, used_start_cells, row, col, random.choice(SAND_COLORS))

        for row, col, color in grains:
            grid[row, col] = color

        if not np.array_equal(main(grid), _oracle_main(grid)):
            continue

        features = _scene_features(grid)
        if not all(features[key] for key in ("top", "middle", "bottom", "stack", "invisible_support", "moved")):
            continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue
        assert grid.ndim == 2 and out.ndim == 2
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
        return grid

    raise RuntimeError("failed to generate a valid invisible periodic floor scene")
