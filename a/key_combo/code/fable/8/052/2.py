import random

import numpy as np


BG = 0
BAR_COLOR = 8
CYCLE_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _marker_columns(start_col: int, count: int) -> list[int]:
    cols = [int(start_col)]
    gap = 1
    while len(cols) < count:
        cols.append(cols[-1] + gap)
        gap += 1
    return cols


def _extend_target(last_col: int, visible_count: int, steps: int) -> int:
    col = int(last_col)
    gap = int(visible_count)
    for _ in range(int(steps)):
        col += gap
        gap += 1
    return col


def _minimal_period(colors: list[int]) -> int:
    for period in range(1, len(colors) + 1):
        if all(colors[i] == colors[i % period] for i in range(len(colors))):
            return period
    raise AssertionError("no period found")


def _find_marker_row(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        fg = row[row != bg]
        if len(fg) >= 4 and len(set(map(int, fg))) >= 2:
            candidates.append(r)
    assert len(candidates) == 1, f"expected one marker row, got {candidates}"
    return int(candidates[0])


def _find_bar(grid: np.ndarray, bg: int, marker_row: int) -> tuple[int, np.ndarray]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rows = np.sort(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(cols) != 1:
            continue
        if len(rows) < 2:
            continue
        if not np.all(np.diff(rows) == 1):
            continue
        if marker_row not in rows:
            continue
        candidates.append((int(color), rows))
    assert len(candidates) == 1, f"expected one bar color, got {candidates}"
    color, rows = candidates[0]
    return color, rows.astype(int)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    marker_row = _find_marker_row(g, bg)
    bar_color, bar_rows = _find_bar(g, bg, marker_row)
    bar_col = int(np.argwhere(g == bar_color)[0, 1])

    marker_cols = [int(c) for c in np.flatnonzero(g[marker_row] != bg) if int(c) != bar_col]
    assert len(marker_cols) >= 4, "need at least four visible markers"
    diffs = np.diff(marker_cols)
    expected_diffs = np.arange(1, len(marker_cols), dtype=int)
    assert np.array_equal(diffs, expected_diffs), f"bad gap sequence: {diffs} vs {expected_diffs}"

    marker_colors = [int(g[marker_row, c]) for c in marker_cols]
    period = _minimal_period(marker_colors)
    assert period < len(marker_colors), "visible colors must expose a repeated cycle"
    assert len(marker_colors) >= 2 * period, "cycle must be visible for at least two repeats"
    cycle = marker_colors[:period]

    return {
        "bg": bg,
        "marker_row": marker_row,
        "marker_cols": marker_cols,
        "marker_colors": marker_colors,
        "cycle": cycle,
        "cycle_len": period,
        "bar_color": bar_color,
        "bar_rows": bar_rows,
        "bar_col": bar_col,
        "height": int(len(bar_rows)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the vertical bar to the hidden next triangular slot and recolor it by continuing the visible cycle.

    Input structure:
    - A black background contains one horizontal marker row and one monochrome vertical bar.
    - The marker pixels lie on that row at columns whose left-to-right gaps are 1, 2, 3, ...
    - The marker colors repeat a fixed cycle, while the vertical bar uses a different color and crosses the marker row.

    Rule:
    - Keep every marker pixel unchanged.
    - Let the bar height be h. Starting from the last visible marker column, extend the increasing-gap sequence by h more terms to get the destination column.
    - Erase the source bar, repaint the same vertical segment at the destination column, and recolor it to the cycle color reached by advancing h more marker steps beyond the last visible marker.

    Key insight:
    - The bar height is the only step count, so both the hidden docking column and the new bar color must be inferred by continuing two separate sequences from the visible marker row.
    """
    scene = _parse_scene(grid)
    g = np.asarray(grid, dtype=int)

    marker_cols = scene["marker_cols"]
    cycle = scene["cycle"]
    steps = scene["height"]
    target_col = _extend_target(marker_cols[-1], len(marker_cols), steps)
    target_color = int(cycle[(len(marker_cols) - 1 + steps) % len(cycle)])

    assert marker_cols[-1] < target_col < scene["bar_col"], "bar must start to the right of the hidden dock"
    assert scene["bar_color"] not in cycle, "bar color must differ from marker cycle"

    out = g.copy()
    out[scene["bar_rows"], scene["bar_col"]] = scene["bg"]
    out[scene["bar_rows"], target_col] = target_color
    return out


def _source_gap(target_col: int, height: int, cycle_len: int, start_col: int) -> int:
    preferred = height + cycle_len - 1 + (start_col % 2)
    return max(1, min(preferred, 29 - target_col))


def _build_pair(params: dict) -> tuple[np.ndarray, np.ndarray]:
    height = params["grid_h"]
    width = params["grid_w"]
    row = params["marker_row"]
    marker_cols = params["marker_cols"]
    marker_colors = params["marker_colors"]
    target_col = params["target_col"]
    source_col = params["source_col"]
    bar_rows = params["bar_rows"]
    target_color = params["target_color"]

    out = np.full((height, width), BG, dtype=int)
    for col, color in zip(marker_cols, marker_colors):
        out[row, col] = color
    out[bar_rows, target_col] = target_color

    inp = out.copy()
    inp[bar_rows, target_col] = BG
    inp[bar_rows, source_col] = BAR_COLOR
    return inp, out


def _sample_params() -> dict | None:
    cycle_len = random.choice((2, 3))
    marker_count = 4 if cycle_len == 2 else 6
    step_choices = (2, 3, 4) if cycle_len == 2 else (2,)
    steps = random.choice(step_choices)
    cycle = random.sample(list(CYCLE_POOL), cycle_len)
    marker_colors = [int(cycle[i % cycle_len]) for i in range(marker_count)]

    valid_starts = []
    for start_col in range(0, 30):
        marker_cols = _marker_columns(start_col, marker_count)
        if marker_cols[-1] >= 30:
            break
        target_col = _extend_target(marker_cols[-1], marker_count, steps)
        if target_col >= 30:
            continue
        gap = _source_gap(target_col, steps, cycle_len, start_col)
        source_col = target_col + gap
        if source_col >= 30:
            continue
        valid_starts.append((start_col, marker_cols, target_col, source_col))
    if not valid_starts:
        return None

    start_col, marker_cols, target_col, source_col = random.choice(valid_starts)
    up_extent = random.randint(0, steps - 1)
    down_extent = steps - 1 - up_extent
    min_h = max(steps + 3, 7)
    grid_h = random.randint(min_h, 12)
    row_min = up_extent + 1
    row_max = grid_h - down_extent - 2
    if row_min > row_max:
        return None
    marker_row = random.randint(row_min, row_max)
    bar_rows = np.arange(marker_row - up_extent, marker_row + down_extent + 1, dtype=int)
    target_color = int(cycle[(marker_count - 1 + steps) % cycle_len])

    return {
        "cycle_len": cycle_len,
        "cycle": cycle,
        "marker_count": marker_count,
        "steps": steps,
        "start_col": start_col,
        "marker_cols": marker_cols,
        "marker_colors": marker_colors,
        "target_col": target_col,
        "source_col": source_col,
        "grid_h": grid_h,
        "grid_w": source_col + 1,
        "marker_row": marker_row,
        "bar_rows": bar_rows,
        "target_color": target_color,
    }


def _validate_generated(inp: np.ndarray, expected: np.ndarray, params: dict) -> bool:
    try:
        scene = _parse_scene(inp)
        produced = main(inp)
    except Exception:
        return False

    if not np.array_equal(produced, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    if scene["cycle_len"] != params["cycle_len"]:
        return False
    if scene["height"] != params["steps"]:
        return False
    if scene["bar_col"] != params["source_col"]:
        return False
    if scene["marker_row"] != params["marker_row"]:
        return False
    if scene["marker_cols"] != params["marker_cols"]:
        return False

    target_col = _extend_target(scene["marker_cols"][-1], len(scene["marker_cols"]), scene["height"])
    target_color = int(scene["cycle"][(len(scene["marker_cols"]) - 1 + scene["height"]) % scene["cycle_len"]])
    if target_col != params["target_col"]:
        return False
    if target_color != params["target_color"]:
        return False
    if params["target_col"] in params["marker_cols"] or params["source_col"] in params["marker_cols"]:
        return False
    if not (params["marker_cols"][-1] < params["target_col"] < params["source_col"]):
        return False
    if BAR_COLOR in params["cycle"]:
        return False
    if params["target_color"] == BAR_COLOR:
        return False

    wrong_target = _extend_target(scene["marker_cols"][-1], len(scene["marker_cols"]), scene["height"] - 1)
    if wrong_target == target_col:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    params = _sample_params()
    if params is None:
        return None
    inp, expected = _build_pair(params)
    if not _validate_generated(inp, expected, params):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
