from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
WATER_COLOR = 1
GEN_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 800
_USE_PARSED_PAIR = object()


def _top_terrain_rows(grid: np.ndarray, terrain: int) -> np.ndarray:
    starts = []
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == terrain)
        assert rows.size > 0, "each column must contain terrain"
        starts.append(int(rows[0]))
    return np.array(starts, dtype=int)


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 4 <= h <= 30 and 5 <= w <= 30

    bg = int(Counter(map(int, g[0])).most_common(1)[0][0])
    terrain = int(Counter(map(int, g[-1])).most_common(1)[0][0])
    assert bg != terrain

    colors = sorted(set(map(int, g.ravel())))
    assert len(colors) == 3, "expected background, terrain, and one spring color"
    spring_colors = [color for color in colors if color not in {bg, terrain}]
    assert len(spring_colors) == 1
    spring_color = spring_colors[0]

    spring_cells = np.argwhere(g == spring_color)
    assert spring_cells.shape[0] == 1, "expected exactly one spring cell"
    spring = tuple(map(int, spring_cells[0]))

    terrain_top = _top_terrain_rows(g, terrain)

    hidden = np.zeros_like(g, dtype=bool)
    for c in range(w):
        hidden[terrain_top[c] :, c] = g[terrain_top[c] :, c] == bg

    hidden_coords = np.argwhere(hidden)
    assert hidden_coords.size > 0, "expected one hidden channel"
    hidden_rows = np.unique(hidden_coords[:, 0])
    assert hidden_rows.size == 1, "channel must be one row high"
    channel_row = int(hidden_rows[0])
    hidden_cols = np.flatnonzero(hidden[channel_row])
    assert hidden_cols.size > 0 and np.all(np.diff(hidden_cols) == 1), "channel must be contiguous"

    ridges = [0]
    for c in range(1, w - 1):
        if terrain_top[c] < terrain_top[c - 1] and terrain_top[c] < terrain_top[c + 1]:
            ridges.append(c)
    ridges.append(w - 1)
    assert len(ridges) >= 4, "expected at least three basins"

    basins = []
    for left, right in zip(ridges, ridges[1:]):
        assert right - left >= 2, "each basin needs an interior"
        segment = terrain_top[left : right + 1]
        depth = int(segment.max())
        valley_cols = left + np.flatnonzero(segment == depth)
        first_valley = int(valley_cols[0])
        last_valley = int(valley_cols[-1])
        assert left < first_valley <= last_valley < right
        assert np.all(np.diff(terrain_top[left : first_valley + 1]) > 0)
        assert np.all(np.diff(terrain_top[last_valley : right + 1]) < 0)
        valley = int(valley_cols[len(valley_cols) // 2])
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "valley": int(valley),
                "left_rim": int(terrain_top[left]),
                "right_rim": int(terrain_top[right]),
                "depth": depth,
            }
        )

    channel_pair = [i for i, basin in enumerate(basins) if basin["depth"] == channel_row + 1]
    assert len(channel_pair) == 2, "channel must connect exactly two equal-depth basins"
    assert channel_pair[1] == channel_pair[0] + 1, "connected basins must be adjacent"
    pair = (int(channel_pair[0]), int(channel_pair[1]))
    middle_ridge = basins[pair[0]]["right"]
    assert middle_ridge == basins[pair[1]]["left"]
    assert int(hidden_cols[0]) <= middle_ridge <= int(hidden_cols[-1])

    start_basin = None
    for idx, basin in enumerate(basins):
        if basin["left"] < spring[1] < basin["right"]:
            start_basin = idx
            break
    assert start_basin is not None, "spring must drop into a basin or slope"

    return {
        "grid": g,
        "bg": bg,
        "terrain": terrain,
        "spring_color": spring_color,
        "spring": spring,
        "terrain_top": terrain_top,
        "hidden": hidden,
        "channel_row": channel_row,
        "basins": basins,
        "pair": pair,
        "start_basin": int(start_basin),
    }


def _fill_basin(out: np.ndarray, terrain_top: np.ndarray, basin: dict, level: int, water: int) -> None:
    for c in range(basin["left"] + 1, basin["right"]):
        top = int(terrain_top[c])
        if top > level:
            out[level:top, c] = water


def _stage_members(basin_idx: int, pair: tuple[int, int] | None) -> tuple[int, ...]:
    if pair is not None and basin_idx in pair:
        return pair
    return (int(basin_idx),)


def _simulate(
    parsed: dict, *, pair_override: tuple[int, int] | None | object = _USE_PARSED_PAIR
) -> tuple[np.ndarray, list[dict]]:
    pair = parsed["pair"] if pair_override is _USE_PARSED_PAIR else pair_override
    out = parsed["grid"].copy()
    terrain_top = parsed["terrain_top"]
    basins = parsed["basins"]
    current = int(parsed["start_basin"])

    trace: list[dict] = []
    seen_stages: set[tuple[int, ...]] = set()
    while True:
        members = _stage_members(current, pair)
        assert members not in seen_stages, "overflow loop is not allowed"
        seen_stages.add(members)

        left_idx = members[0]
        right_idx = members[-1]
        left_rim = basins[left_idx]["left_rim"]
        right_rim = basins[right_idx]["right_rim"]
        if left_rim > right_rim:
            level = int(left_rim)
            spill_side = "left"
        else:
            level = int(right_rim)
            spill_side = "right"

        for basin_idx in members:
            _fill_basin(out, terrain_top, basins[basin_idx], level, WATER_COLOR)

        if pair is not None and members == pair:
            out[parsed["hidden"]] = WATER_COLOR

        trace.append({"members": members, "level": level, "side": spill_side})

        if spill_side == "left":
            next_basin = left_idx - 1
            if next_basin < 0:
                break
        else:
            next_basin = right_idx + 1
            if next_basin >= len(basins):
                break
        current = int(next_basin)

    out[parsed["spring"]] = parsed["spring_color"]
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every basin touched by the spring-and-channel water system up to each stage's spill level.

    Input structure:
    - The lower part of the grid is a single-color terrain profile made of unit-step slopes, creating several basins.
    - Exactly one spring marker sits above the terrain, and one one-row horizontal background tunnel is buried inside the terrain.
    - The tunnel links the equal-depth bottoms of two adjacent basins, while all other below-surface cells remain solid terrain.

    Rule:
    - Drop the spring vertically to its column's terrain surface; that slope belongs to one basin, so the water starts there.
    - A normal basin fills alone up to its lower rim; the tunnel-linked pair instead fills together at a shared level and also fills the buried tunnel cells.
    - When the current filled stage reaches its unique lowest outer rim, it spills to the next basin on that side; transient downhill flow is ignored, and the process repeats until the water exits the grid.

    Key insight:
    - Surface overflows are sequential, but the buried tunnel turns one adjacent basin pair into a single communicating reservoir whose overflow rim must be chosen globally across both outer rims.
    """
    parsed = _parse_grid(grid)
    out, _trace = _simulate(parsed)
    return out


def _build_profile(
    start_height: int, descents: list[int], ascents: list[int]
) -> tuple[np.ndarray, list[int], list[int]]:
    values = [int(start_height)]
    ridge_heights = [int(start_height)]
    valley_heights: list[int] = []
    current = int(start_height)
    for basin_idx in range(len(descents)):
        for _ in range(descents[basin_idx]):
            current += 1
            values.append(current)
        valley_heights.append(int(current))
        for _ in range(ascents[basin_idx]):
            current -= 1
            values.append(current)
        ridge_heights.append(int(current))
    return np.array(values, dtype=int), ridge_heights, valley_heights


def _render_input(
    profile: np.ndarray,
    pair_idx: int,
    terrain: int,
    spring_color: int,
    spring_pos: tuple[int, int],
) -> np.ndarray:
    h = int(profile.max()) + 4
    w = int(profile.size)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for c, top in enumerate(profile):
        grid[int(top) :, c] = terrain

    ridges = [0]
    for c in range(1, w - 1):
        if profile[c] < profile[c - 1] and profile[c] < profile[c + 1]:
            ridges.append(c)
    ridges.append(w - 1)
    basins = []
    for left, right in zip(ridges, ridges[1:]):
        valley = left + int(np.argmax(profile[left : right + 1]))
        basins.append({"left": left, "right": right, "valley": valley, "depth": int(profile[valley])})

    left_basin = basins[pair_idx]
    right_basin = basins[pair_idx + 1]
    channel_row = left_basin["depth"] - 1
    grid[channel_row, left_basin["valley"] : right_basin["valley"] + 1] = BG_COLOR

    sr, sc = spring_pos
    grid[int(sr), int(sc)] = int(spring_color)
    return grid


def _simulate_without_pair(parsed: dict) -> tuple[np.ndarray, list[dict]]:
    return _simulate(parsed, pair_override=None)


def _semantic_checks(grid: np.ndarray) -> bool:
    parsed = _parse_grid(grid)
    out, trace = _simulate(parsed)
    pair = parsed["pair"]
    no_pair_out, no_pair_trace = _simulate_without_pair(parsed)

    if len(trace) < 2:
        return False
    if not any(stage["members"] == pair for stage in trace):
        return False
    if np.array_equal(out, no_pair_out):
        return False
    if not np.any((out != no_pair_out) & ~parsed["hidden"]):
        return False
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == WATER_COLOR) < 8:
        return False
    if len({tuple(stage["members"]) for stage in trace}) != len(trace):
        return False
    if len(no_pair_trace) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    basin_count = random.choice([3, 4])
    pair_idx = random.randrange(basin_count - 1)

    start_height = random.randint(2, 5)
    descents = [random.randint(2, 4) for _ in range(basin_count)]
    ascents = [random.randint(2, 4) for _ in range(basin_count)]
    descents[pair_idx + 1] = ascents[pair_idx]

    profile, ridge_heights, valley_heights = _build_profile(start_height, descents, ascents)
    width = int(profile.size)
    if not (16 <= width <= 28):
        return None
    if profile.max() + 4 > 30:
        return None
    if min(ridge_heights) < 1:
        return None
    if len(set(ridge_heights)) != len(ridge_heights):
        return None

    pair_depth = valley_heights[pair_idx]
    if valley_heights[pair_idx + 1] != pair_depth:
        return None
    if sum(depth == pair_depth for depth in valley_heights) != 2:
        return None

    terrain, spring_color = random.sample(GEN_COLOR_POOL, 2)

    ridges = [0]
    for c in range(1, width - 1):
        if profile[c] < profile[c - 1] and profile[c] < profile[c + 1]:
            ridges.append(c)
    ridges.append(width - 1)
    basins = []
    for left, right in zip(ridges, ridges[1:]):
        valley = left + int(np.argmax(profile[left : right + 1]))
        basins.append({"left": left, "right": right, "valley": valley})
    if len(basins) != basin_count:
        return None

    start_modes = []
    if pair_idx > 0:
        start_modes.append("left")
    if pair_idx + 2 < basin_count:
        start_modes.append("right")
    start_modes.append("pair")
    random.shuffle(start_modes)

    for mode in start_modes:
        if mode == "left":
            basin_choices = list(range(0, pair_idx))
        elif mode == "right":
            basin_choices = list(range(pair_idx + 2, basin_count))
        else:
            basin_choices = [pair_idx, pair_idx + 1]
        random.shuffle(basin_choices)

        for basin_idx in basin_choices:
            basin = basins[basin_idx]
            cols = list(range(basin["left"] + 1, basin["right"]))
            if len(cols) < 2:
                continue
            slope_cols = [c for c in cols if c != basin["valley"]]
            if random.random() < 0.7 and slope_cols:
                spring_col = random.choice(slope_cols)
            else:
                spring_col = random.choice(cols)
            spring_row = random.randint(0, max(0, int(profile[spring_col]) - 3))
            grid = _render_input(profile, pair_idx, terrain, spring_color, (spring_row, spring_col))
            try:
                if _semantic_checks(grid):
                    return grid
            except AssertionError:
                continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-channel basin input")
