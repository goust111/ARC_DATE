from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


LEGEND_COLORS = (Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
BAR_COLORS = (Color.BLUE, Color.GREEN)
MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    """Mirror arc_utils.get_bg: background is the most common grid color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _extract_legend(row: np.ndarray) -> list[int]:
    legend: list[int] = []
    seen: set[int] = set()
    c = 0
    while c < row.shape[0] and int(row[c]) in LEGEND_COLORS:
        color = int(row[c])
        assert color not in seen, "legend colors must be distinct"
        seen.add(color)
        legend.append(color)
        c += 1
    assert len(legend) >= 5, "legend must cover every possible detour cost index"
    assert np.all(row[c:] == Color.BLACK), "legend row must be a left-aligned color prefix"
    return legend


def _find_bars(grid: np.ndarray, start_c: int, legend_row: int) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    for r in range(1, legend_row):
        c = 0
        while c < w:
            val = int(grid[r, c])
            if val in BAR_COLORS:
                c0 = c
                while c < w and int(grid[r, c]) == val:
                    c += 1
                c1 = c - 1
                assert c1 - c0 + 1 >= 3, "bars must be at least 3 cells wide"
                assert c0 <= start_c <= c1, "every bar must block the path column"
                bars.append({"row": r, "c0": c0, "c1": c1, "color": val})
            else:
                assert val == Color.BLACK, "only obstacles may appear above the legend"
                c += 1
    bars.sort(key=lambda bar: bar["row"])
    assert 2 <= len(bars) <= 4, "expected 2-4 obstacle bars"
    for i, bar in enumerate(bars):
        if bar["color"] == Color.BLUE:
            assert bar["c0"] >= 1, "blue detours need one free column on the left"
        else:
            assert bar["c1"] + 1 < w, "green detours need one free column on the right"
        if i:
            assert bar["row"] - bars[i - 1]["row"] >= 3, "bars must leave room for separate detours"
    return bars


def _parse(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "background must be black"

    starts = np.argwhere(g == Color.RED)
    assert starts.shape == (1, 2), "input must contain exactly one red start marker"
    start_r, start_c = map(int, starts[0])
    assert start_r == 0, "the start marker must be on the top border"

    legend_row = g.shape[0] - 1
    legend = _extract_legend(g[legend_row])
    assert np.all(g[0, :start_c] == Color.BLACK) and np.all(g[0, start_c + 1 :] == Color.BLACK), (
        "top row may only contain the red start marker"
    )

    bars = _find_bars(g, start_c, legend_row)
    return g, start_c, legend_row, legend, bars


def _paint_path_cell(out: np.ndarray, base: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = base.shape
    assert 0 <= r < h and 0 <= c < w
    assert int(base[r, c]) in (Color.BLACK, Color.RED), "trace may only overwrite background/start cells"
    out[r, c] = int(color)


def _bar_side_column(bar: dict[str, int]) -> int:
    if bar["color"] == Color.BLUE:
        return bar["c0"] - 1
    return bar["c1"] + 1


def _bar_cost(start_c: int, bar: dict[str, int]) -> int:
    side_c = _bar_side_column(bar)
    horizontal = abs(side_c - start_c)
    steps = horizontal + 2 + horizontal
    return steps - 2


def _render_trace(
    base: np.ndarray,
    start_c: int,
    legend_row: int,
    legend: list[int],
    bars: list[dict[str, int]],
) -> np.ndarray:
    out = base.copy()
    current_row = 0
    for bar in bars:
        row = bar["row"]
        for rr in range(current_row + 1, row):
            _paint_path_cell(out, base, rr, start_c, Color.RED)

        side_c = _bar_side_column(bar)
        cost = _bar_cost(start_c, bar)
        assert 0 <= cost < len(legend), "legend must contain the detour-cost index"
        detour_color = legend[cost]

        step = 1 if side_c > start_c else -1
        for cc in range(start_c + step, side_c + step, step):
            _paint_path_cell(out, base, row - 1, cc, detour_color)
        for rr in range(row, row + 2):
            assert rr < legend_row, "detour may not touch the legend row"
            _paint_path_cell(out, base, rr, side_c, detour_color)
        back_step = 1 if start_c > side_c else -1
        for cc in range(side_c + back_step, start_c + back_step, back_step):
            _paint_path_cell(out, base, row + 1, cc, detour_color)

        current_row = row + 1

    for rr in range(current_row + 1, legend_row):
        _paint_path_cell(out, base, rr, start_c, Color.RED)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a downward path whose detour segments are recolored by legend indices equal to their extra travel cost.

    Input structure:
    - A black background contains one red marker on the top border.
    - The interior has 2-4 horizontal one-cell-thick blue/green obstacle bars, and every bar crosses the marker column.
    - The bottom border is a legend row: a left-aligned prefix of distinct yellow/magenta/orange/teal/maroon pixels followed by black.

    Rule:
    - Start from the red border marker and move straight downward, stopping one row above the legend.
    - When a blue bar blocks the lane, detour around its left end; when a green bar blocks the lane, detour around its right end; after passing below the bar, return to the original column.
    - For each detour, compute the extra travel cost as detour step count minus vertical row displacement, use that value as a 0-based legend index, and paint only the non-straight detour pixels with the indexed legend color.
    - Keep the straight vertical path pixels red and preserve the obstacles, legend row, and background.

    Key insight:
    - Obstacle color selects the routing side, but the detour geometry determines a cost that must then be decoded through the bottom legend to color that local route segment.
    """
    g, start_c, legend_row, legend, bars = _parse(grid)
    return _render_trace(g, start_c, legend_row, legend, bars)


def _row_choices(height: int, n_bars: int) -> list[int]:
    rows: list[int] = []
    next_min = 2
    for remain in range(n_bars, 0, -1):
        max_row = height - 3 - 3 * (remain - 1)
        if next_min > max_row:
            return []
        rows.append(random.randint(next_min, max_row))
        next_min = rows[-1] + 3
    return rows


def _build_bar(row: int, start_c: int, width: int, color: int, detour_span: int) -> dict[str, int]:
    if color == Color.BLUE:
        c0 = start_c - detour_span + 1
        c1 = c0 + width - 1
    else:
        c1 = start_c + detour_span - 1
        c0 = c1 - width + 1
    return {"row": row, "c0": c0, "c1": c1, "color": color}


def _combo_plan(n_bars: int) -> list[tuple[int, int]]:
    all_combos = [
        (Color.BLUE, 1),
        (Color.BLUE, 2),
        (Color.GREEN, 1),
        (Color.GREEN, 2),
    ]
    if n_bars == 2:
        plans = [
            [(Color.BLUE, 1), (Color.BLUE, 2)],
            [(Color.GREEN, 1), (Color.GREEN, 2)],
            [(Color.BLUE, 1), (Color.GREEN, 1)],
            [(Color.BLUE, 2), (Color.GREEN, 2)],
        ]
        plan = random.choice(plans)
    elif n_bars == 3:
        drop = random.randrange(len(all_combos))
        plan = [combo for i, combo in enumerate(all_combos) if i != drop]
    else:
        plan = all_combos[:]
    random.shuffle(plan)
    return plan


def _metadata_expected(grid: np.ndarray, meta: dict) -> np.ndarray:
    return _render_trace(
        np.asarray(grid, dtype=int),
        meta["start_c"],
        meta["legend_row"],
        list(meta["legend"]),
        [dict(bar) for bar in meta["bars"]],
    )


def _alt_bar(bar: dict[str, int], start_c: int, width_limit: int) -> dict[str, int]:
    if bar["color"] == Color.BLUE:
        old_span = start_c - bar["c0"] + 1
        new_span = 1 if old_span == 2 else 2
        new_c0 = start_c - new_span + 1
        min_width = max(3, start_c - new_c0 + 1)
        max_width = min(7, width_limit - new_c0)
        width = min(max(bar["c1"] - bar["c0"] + 1, min_width), max_width)
        return {"row": bar["row"], "c0": new_c0, "c1": new_c0 + width - 1, "color": bar["color"]}

    old_span = bar["c1"] - start_c + 1
    new_span = 1 if old_span == 2 else 2
    new_c1 = start_c + new_span - 1
    min_width = max(3, new_c1 - start_c + 1)
    max_width = min(7, new_c1 + 1)
    width = min(max(bar["c1"] - bar["c0"] + 1, min_width), max_width)
    return {"row": bar["row"], "c0": new_c1 - width + 1, "c1": new_c1, "color": bar["color"]}


def _build_alt_variant(grid: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    alt = np.asarray(grid, dtype=int).copy()
    bars = [dict(bar) for bar in meta["bars"]]
    idx = 0
    old_bar = bars[idx]
    alt[old_bar["row"], old_bar["c0"] : old_bar["c1"] + 1] = Color.BLACK
    new_bar = _alt_bar(old_bar, meta["start_c"], alt.shape[1])
    alt[new_bar["row"], new_bar["c0"] : new_bar["c1"] + 1] = new_bar["color"]
    bars[idx] = new_bar
    alt_meta = dict(meta)
    alt_meta["bars"] = bars
    return alt, alt_meta


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    expected = _metadata_expected(grid, meta)
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(grid, solved):
        return False

    bars = meta["bars"]
    detour_costs = [_bar_cost(meta["start_c"], bar) for bar in bars]
    detour_colors = [meta["legend"][cost] for cost in detour_costs]
    if len(set(detour_costs)) < 2:
        return False
    if len(set(detour_colors)) < 2:
        return False

    colors = [bar["color"] for bar in bars]
    spans = [1 if cost == 2 else 2 for cost in detour_costs]
    if len(bars) >= 3:
        if len(set(colors)) != 2 or len(set(spans)) != 2:
            return False
        if len(set(zip(colors, spans))) < 3:
            return False
    else:
        same_color = colors[0] == colors[1]
        same_span = spans[0] == spans[1]
        if same_color == same_span:
            return False

    if not np.any((solved != grid) & (solved == Color.RED)):
        return False
    if not any(np.any(solved == color) for color in detour_colors):
        return False

    current_row = 0
    for bar, cost in zip(bars, detour_costs, strict=False):
        row = bar["row"]
        side_c = _bar_side_column(bar)
        detour_color = meta["legend"][cost]
        for rr in range(current_row + 1, row):
            if int(solved[rr, meta["start_c"]]) != Color.RED:
                return False
        step = 1 if side_c > meta["start_c"] else -1
        for cc in range(meta["start_c"] + step, side_c + step, step):
            if int(solved[row - 1, cc]) != detour_color:
                return False
        for rr in range(row, row + 2):
            if int(solved[rr, side_c]) != detour_color:
                return False
        back_step = 1 if meta["start_c"] > side_c else -1
        for cc in range(side_c + back_step, meta["start_c"] + back_step, back_step):
            if int(solved[row + 1, cc]) != detour_color:
                return False
        current_row = row + 1
    for rr in range(current_row + 1, meta["legend_row"]):
        if int(solved[rr, meta["start_c"]]) != Color.RED:
            return False

    alt_grid, alt_meta = _build_alt_variant(grid, meta)
    alt_expected = _metadata_expected(alt_grid, alt_meta)
    try:
        alt_solved = main(alt_grid)
    except Exception:
        return False
    if not np.array_equal(alt_solved, alt_expected):
        return False
    if np.array_equal(alt_solved, solved):
        return False

    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    n_bars = random.choices([2, 3, 4], weights=[1, 3, 2], k=1)[0]
    height = random.randint(max(15, 3 * n_bars + 2), 24)
    width = random.randint(12, 20)
    start_c = random.randint(5, width - 6)
    legend = list(LEGEND_COLORS)
    random.shuffle(legend)

    rows = _row_choices(height, n_bars)
    if len(rows) != n_bars:
        return None

    bars: list[dict[str, int]] = []
    for row, (color, detour_span) in zip(rows, _combo_plan(n_bars), strict=False):
        width_bar = random.randint(3, 7)
        bar = _build_bar(row, start_c, width_bar, color, detour_span)
        if not (0 <= bar["c0"] <= bar["c1"] < width):
            return None
        if not (bar["c0"] <= start_c <= bar["c1"]):
            return None
        bars.append(bar)

    grid = np.zeros((height, width), dtype=int)
    grid[0, start_c] = Color.RED
    grid[height - 1, : len(legend)] = np.array(legend, dtype=int)
    for bar in bars:
        grid[bar["row"], bar["c0"] : bar["c1"] + 1] = bar["color"]

    meta = {
        "start_c": start_c,
        "legend_row": height - 1,
        "legend": tuple(legend),
        "bars": bars,
    }
    if not _semantic_checks(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            grid, _meta = attempt
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
