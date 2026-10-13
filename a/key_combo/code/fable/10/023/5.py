from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_10_023_5",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BLACK = Color.BLACK
BLUE = Color.BLUE
GREEN = Color.GREEN
MOTIF_PALETTE = (
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (12, 22)
MOTIF_LEN_RANGE = (2, 5)
HIT_COUNT_RANGE = (2, 5)
SCENERY_COUNT_RANGE = (0, 2)
BAR_LEN_RANGE = (2, 6)
GEN_RETRIES = 400


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    bg = get_bg(g)
    assert bg == BLACK, "expected black background"

    top_cols = np.flatnonzero(g[0] != bg)
    assert top_cols.size == 1, "expected a single top motif column"
    motif_col = int(top_cols[0])

    motif_colors: list[int] = []
    motif_rows = set()
    row = 0
    while row < h and int(g[row, motif_col]) != bg:
        non_bg_cols = np.flatnonzero(g[row] != bg)
        assert non_bg_cols.size == 1 and int(non_bg_cols[0]) == motif_col, "top motif rows must contain only the motif pixel"
        motif_colors.append(int(g[row, motif_col]))
        motif_rows.add(row)
        row += 1

    assert motif_colors, "expected a non-empty motif"
    motif_len = len(motif_colors)
    assert len(set(motif_colors)) >= 2, "motif must use at least two colors"

    motif_coords = {(r, motif_col) for r in range(motif_len)}
    non_bg_coords = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    obstacle_coords = non_bg_coords - motif_coords

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    obstacles = []
    component_at: dict[tuple[int, int], dict] = {}
    for comp in components:
        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        if coords <= motif_coords:
            continue
        assert not (coords & motif_coords), "motif cells may not belong to obstacle components"
        color = int(comp["color"])
        assert color in (BLUE, GREEN), "obstacles must be blue or green"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r0 == r1, "obstacles must be horizontal"
        assert c0 > 0 and c1 < w - 1, "obstacles may not touch the side borders"
        assert int(comp["area"]) == c1 - c0 + 1, "obstacles must be solid bars"
        assert g[r0, c0 - 1] == bg and g[r0, c1 + 1] == bg, "outside cells next to each bar end must be background"
        if r0 + 1 < h:
            assert g[r0 + 1, c0 - 1] == bg and g[r0 + 1, c1 + 1] == bg, "cells below the outside lanes must be background"
        for rr, cc in coords:
            component_at[(rr, cc)] = {
                "row": r0,
                "start": c0,
                "end": c1,
                "color": color,
            }
        obstacles.append(
            {
                "row": r0,
                "start": c0,
                "end": c1,
                "color": color,
            }
        )

    assert obstacle_coords == set(component_at), "all non-motif foreground cells must belong to obstacles"

    rows = sorted(bar["row"] for bar in obstacles)
    assert len(rows) == len(set(rows)), "obstacles must lie on distinct rows"

    candidates = []
    for c in range(w):
        r = 0
        while r < h and int(g[r, c]) != bg:
            if np.count_nonzero(g[r] != bg) != 1:
                break
            r += 1
        if r > 0 and all(int(g[rr, c]) != bg for rr in range(r)):
            candidates.append((c, r))
    assert candidates == [(motif_col, motif_len)], "motif column must be uniquely identifiable"

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "motif_col": motif_col,
        "motif_len": motif_len,
        "motif_colors": motif_colors,
        "obstacles": obstacles,
        "component_at": component_at,
    }


def _trace_path(parsed: dict) -> tuple[list[tuple[int, int, int]], list[dict]]:
    h = int(parsed["h"])
    motif_len = int(parsed["motif_len"])
    motif_colors = list(parsed["motif_colors"])
    lane = int(parsed["motif_col"])
    component_at = dict(parsed["component_at"])

    placements: list[tuple[int, int, int]] = []
    events: list[dict] = []
    phase = 0
    for row in range(motif_len, h):
        event = {
            "row": row,
            "entry_lane": lane,
            "bar": None,
            "exit_lane": lane,
        }
        bar = component_at.get((row, lane))
        if bar is not None:
            if int(bar["color"]) == BLUE:
                lane = int(bar["start"]) - 1
            else:
                lane = int(bar["end"]) + 1
            event["bar"] = dict(bar)
            event["exit_lane"] = lane
        placements.append((row, lane, motif_colors[phase]))
        events.append(event)
        phase = (phase + 1) % motif_len

    return placements, events


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the top color motif downward, changing lanes around blue and green obstacle bars.

    Input structure:
    - The grid has a black background and exactly one colored motif column touching the top edge.
    - The motif occupies a contiguous run of cells in one column, with only background below it until lower obstacles appear.
    - Lower in the grid are one-cell-thick horizontal blue and green bars on distinct rows, never touching the side borders.

    Rule:
    - Start directly below the motif in the same column and continue writing motif colors cyclically one row at a time.
    - Before writing each next row, inspect the cell straight below the current lane.
    - If that cell belongs to a blue bar, jump to the column immediately left of the bar's left end; if it belongs to a green bar, jump to the column immediately right of the bar's right end.
    - Write the next motif color in the resulting lane, keep all bars unchanged, and continue until the bottom row.

    Key insight:
    - The fill phase depends only on the motif cycle, while lane changes depend on the full obstacle bar geometry: color picks the side, but the bar endpoints determine the actual new lane.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()
    for row, col, color in _trace_path(parsed)[0]:
        assert int(out[row, col]) == BLACK, "train path may only paint background cells"
        out[row, col] = int(color)
    return out


def _sample_spaced_rows(start: int, end: int, count: int) -> list[int]:
    span = end - start + 1
    offset = 1
    effective = span - offset * (count - 1)
    assert effective >= count
    base = sorted(random.sample(range(effective), count))
    return [start + value + offset * idx for idx, value in enumerate(base)]


def _sample_motif(length: int) -> list[int]:
    color_count = random.randint(2, min(4, length, len(MOTIF_PALETTE)))
    palette = random.sample(list(MOTIF_PALETTE), color_count)
    motif = [random.choice(palette) for _ in range(length)]
    if len(set(motif)) < 2:
        motif[-1] = next(color for color in palette if color != motif[0])
    return motif


def _feasible_exits(entry_lane: int, sign: int, width: int) -> list[int]:
    exits: list[int] = []
    if sign < 0:
        for exit_lane in range(0, entry_lane):
            left = exit_lane + 1
            low_right = max(entry_lane, left + 1)
            high_right = min(width - 2, left + BAR_LEN_RANGE[1] - 1)
            if low_right <= high_right:
                exits.append(exit_lane)
    else:
        for exit_lane in range(entry_lane + 1, width):
            right = exit_lane - 1
            low_left = max(1, right - (BAR_LEN_RANGE[1] - 1))
            high_left = min(entry_lane, right - 1)
            if low_left <= high_left:
                exits.append(exit_lane)
    return exits


def _build_hit_bars(hit_rows: list[int], width: int, motif_col: int) -> list[dict] | None:
    lane = motif_col
    bars: list[dict] = []
    seen_signs: set[int] = set()
    used_big_shift = False
    used_extension = False

    for idx, row in enumerate(hit_rows):
        remaining = len(hit_rows) - idx
        feasible: dict[int, list[int]] = {}
        for sign in (-1, 1):
            exits = _feasible_exits(lane, sign, width)
            if exits:
                feasible[sign] = exits
        if not feasible:
            return None

        missing = [sign for sign in (-1, 1) if sign not in seen_signs]
        if len(missing) == remaining:
            sign_choices = [sign for sign in missing if sign in feasible]
            if not sign_choices:
                return None
        else:
            sign_choices = list(feasible)
        sign = random.choice(sign_choices)

        exit_choices = feasible[sign]
        if not used_big_shift:
            big_choices = [exit_lane for exit_lane in exit_choices if abs(exit_lane - lane) > 1]
            if big_choices and (idx == len(hit_rows) - 1 or random.random() < 0.6):
                exit_choices = big_choices
        exit_lane = random.choice(exit_choices)

        if sign < 0:
            start = exit_lane + 1
            low_end = max(lane, start + 1)
            high_end = min(width - 2, start + BAR_LEN_RANGE[1] - 1)
            end_choices = list(range(low_end, high_end + 1))
            if not used_extension:
                ext_choices = [end for end in end_choices if end > lane]
                if ext_choices and (idx == len(hit_rows) - 1 or random.random() < 0.6):
                    end_choices = ext_choices
            end = random.choice(end_choices)
            start_col, end_col = start, end
            color = BLUE
            extension = end_col - lane
        else:
            end = exit_lane - 1
            low_start = max(1, end - (BAR_LEN_RANGE[1] - 1))
            high_start = min(lane, end - 1)
            start_choices = list(range(low_start, high_start + 1))
            if not used_extension:
                ext_choices = [start for start in start_choices if start < lane]
                if ext_choices and (idx == len(hit_rows) - 1 or random.random() < 0.6):
                    start_choices = ext_choices
            start = random.choice(start_choices)
            start_col, end_col = start, end
            color = GREEN
            extension = lane - start_col

        bars.append(
            {
                "row": row,
                "start": start_col,
                "end": end_col,
                "color": color,
                "entry_lane": lane,
                "exit_lane": exit_lane,
                "hit": True,
            }
        )
        seen_signs.add(sign)
        used_big_shift |= abs(exit_lane - lane) > 1
        used_extension |= extension > 0
        lane = exit_lane

    if seen_signs != {-1, 1} or not used_big_shift or not used_extension:
        return None
    return bars


def _lane_before_each_row(height: int, motif_len: int, motif_col: int, hit_bars: list[dict]) -> dict[int, int]:
    hit_by_row = {int(bar["row"]): bar for bar in hit_bars}
    lane = motif_col
    lane_before: dict[int, int] = {}
    for row in range(motif_len, height):
        lane_before[row] = lane
        if row in hit_by_row:
            lane = int(hit_by_row[row]["exit_lane"])
    return lane_before


def _sample_scenery_bar(row: int, lane: int, width: int) -> dict | None:
    candidates = []
    for length in range(BAR_LEN_RANGE[0], min(BAR_LEN_RANGE[1], width - 2) + 1):
        for start in range(1, width - length):
            end = start + length - 1
            if start <= lane <= end:
                continue
            distance = min(abs(lane - start), abs(lane - end))
            candidates.append((distance, start, end))
    if not candidates:
        return None

    close = [item for item in candidates if item[0] <= 2]
    distance, start, end = random.choice(close or candidates)
    _ = distance
    return {
        "row": row,
        "start": start,
        "end": end,
        "color": random.choice((BLUE, GREEN)),
        "entry_lane": lane,
        "exit_lane": lane,
        "hit": False,
    }


def _render_grid(height: int, width: int, motif_col: int, motif: list[int], bars: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row, color in enumerate(motif):
        grid[row, motif_col] = int(color)
    for bar in bars:
        grid[int(bar["row"]), int(bar["start"]) : int(bar["end"]) + 1] = int(bar["color"])
    return grid


def _expected_from_plan(height: int, width: int, motif_col: int, motif: list[int], bars: list[dict]) -> np.ndarray:
    out = _render_grid(height, width, motif_col, motif, bars)
    motif_len = len(motif)
    hit_by_row = {int(bar["row"]): bar for bar in bars if bar["hit"]}
    lane = motif_col
    phase = 0
    for row in range(motif_len, height):
        if row in hit_by_row:
            lane = int(hit_by_row[row]["exit_lane"])
        out[row, lane] = int(motif[phase])
        phase = (phase + 1) % motif_len
    return out


def _validate_case(grid: np.ndarray, motif_col: int, motif: list[int], bars: list[dict]) -> None:
    height, width = grid.shape
    hit_bars = [bar for bar in bars if bar["hit"]]
    scenery_bars = [bar for bar in bars if not bar["hit"]]

    lane_before = _lane_before_each_row(height, len(motif), motif_col, hit_bars)
    for bar in hit_bars:
        row = int(bar["row"])
        entry_lane = int(bar["entry_lane"])
        exit_lane = int(bar["exit_lane"])
        assert lane_before[row] == entry_lane
        assert int(bar["start"]) <= entry_lane <= int(bar["end"])
        if int(bar["color"]) == BLUE:
            assert exit_lane == int(bar["start"]) - 1
        else:
            assert exit_lane == int(bar["end"]) + 1
    for bar in scenery_bars:
        row = int(bar["row"])
        entry_lane = lane_before[row]
        assert not (int(bar["start"]) <= entry_lane <= int(bar["end"])), "scenery bar may not lie on the active lane"

    expected = _expected_from_plan(height, width, motif_col, motif, bars)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the explicit plan"

    parsed = _parse_grid(grid)
    placements, events = _trace_path(parsed)
    painted = {(row, col): color for row, col, color in placements}
    assert len(painted) == height - len(motif)
    assert len({col for _, col, _ in placements}) >= 2, "detours must change lanes"
    assert len(set(color for _, _, color in placements)) >= 2, "cycle phase must be visible in the output"
    assert any(event["bar"] is not None for event in events), "at least one obstacle must be hit"
    assert any(event["bar"] is not None and abs(event["exit_lane"] - event["entry_lane"]) > 1 for event in events), "at least one hit must require a multi-column lane jump"
    assert any(event["bar"] is not None and not (event["entry_lane"] in (event["bar"]["start"], event["bar"]["end"])) for event in events), "at least one hit must depend on a full bar, not just a single adjacent block"


def _gen_attempt() -> np.ndarray | None:
    motif_len = random.randint(*MOTIF_LEN_RANGE)
    hit_count = random.randint(*HIT_COUNT_RANGE)
    scenery_count = random.randint(*SCENERY_COUNT_RANGE)
    height_min = max(GRID_H_RANGE[0], motif_len + 2 * (hit_count + scenery_count) + 1)
    if height_min > GRID_H_RANGE[1]:
        return None

    height = random.randint(height_min, GRID_H_RANGE[1])
    width = random.randint(*GRID_W_RANGE)
    motif_col = random.randint(2, width - 3)
    motif = _sample_motif(motif_len)

    total_bar_count = hit_count + scenery_count
    rows = _sample_spaced_rows(motif_len, height - 2, total_bar_count)
    hit_rows = sorted(random.sample(rows, hit_count))
    scenery_rows = sorted(row for row in rows if row not in set(hit_rows))

    hit_bars = _build_hit_bars(hit_rows, width, motif_col)
    if hit_bars is None:
        return None

    lane_before = _lane_before_each_row(height, motif_len, motif_col, hit_bars)
    scenery_bars = []
    for row in scenery_rows:
        bar = _sample_scenery_bar(row, lane_before[row], width)
        if bar is None:
            return None
        scenery_bars.append(bar)

    bars = sorted(hit_bars + scenery_bars, key=lambda bar: int(bar["row"]))
    grid = _render_grid(height, width, motif_col, motif, bars)

    try:
        _validate_case(grid, motif_col, motif, bars)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
