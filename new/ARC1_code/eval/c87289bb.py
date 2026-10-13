import numpy as np

from arc_utils import colors_present, get_bg
import itertools
import random


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if len(cols) == 0:
        return runs
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
        else:
            runs.append((start, prev))
            start = prev = col
    runs.append((start, prev))
    return runs


def _nearest_side(col: int, left: int, right: int, width: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    assert candidates
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the top colored columns downward, routing blocked columns around horizontal obstacle segments.

    Input structure
    The background is the dominant color.  Row 0 contains a single non-background
    source color forming vertical columns at the top.  The remaining foreground
    color appears only as one row of horizontal obstacle segments below those
    columns.

    Rule
    Copy the input, then extend every source column to the bottom of the grid.
    If a source column hits an obstacle segment, paint a horizontal run on the
    row just above the obstacle from the source column to the nearest outside
    side of that segment, then continue vertically downward in that outside
    column.  Exact midpoint ties detour to the right side.

    Key insight
    The red segments do not move or change color; they only redirect the falling
    teal columns to the closest free side before the columns continue downward.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    bg = get_bg(g)

    top_colors = [color for color in colors_present(g[0], exclude={bg})]
    assert len(top_colors) == 1
    source_color = top_colors[0]

    obstacle_colors = colors_present(g, exclude={bg, source_color})
    assert len(obstacle_colors) == 1
    obstacle_color = obstacle_colors[0]

    obstacle_rows = np.flatnonzero(np.any(g == obstacle_color, axis=1))
    assert len(obstacle_rows) == 1
    obstacle_row = int(obstacle_rows[0])
    assert obstacle_row > 0

    source_cols = np.flatnonzero(g[0] == source_color)
    assert len(source_cols) > 0

    obstacle_runs = _contiguous_runs(np.flatnonzero(g[obstacle_row] == obstacle_color))
    assert obstacle_runs

    run_by_col: dict[int, tuple[int, int]] = {}
    for left, right in obstacle_runs:
        for col in range(left, right + 1):
            run_by_col[col] = (left, right)

    out = g.copy()
    for col in source_cols:
        col = int(col)
        if col not in run_by_col:
            assert not np.any(g[:, col] == obstacle_color)
            out[:, col] = source_color
            continue

        left, right = run_by_col[col]
        side_col = _nearest_side(col, left, right, w)
        c0, c1 = sorted((col, side_col))
        assert np.all(g[obstacle_row - 1, c0 : c1 + 1] != obstacle_color)
        assert np.all(g[obstacle_row:, side_col] != obstacle_color)
        out[obstacle_row - 1, c0 : c1 + 1] = source_color
        out[obstacle_row:, side_col] = source_color

    out[g == obstacle_color] = obstacle_color
    return out

# --------------- generator (o5x) ---------------

# Colors used by the task.
GEN_BG_COLOR, GEN_SOURCE_COLOR, GEN_OBSTACLE_COLOR = 0, 8, 2

# Canvas and vertical layout bounds.
GEN_HEIGHT_MIN, GEN_HEIGHT_MAX = 9, 10
GEN_WIDTH_MIN, GEN_WIDTH_MAX = 10, 15
GEN_OBSTACLE_ROW_MIN, GEN_OBSTACLE_ROW_MAX_DIVISOR = 3, 2
GEN_SOURCE_CLEAR_GAP_MIN, GEN_SOURCE_CLEAR_GAP_MAX = 0, 1

# Source column and detour constraints.
GEN_SOURCE_COUNT_MIN, GEN_SOURCE_COUNT_MAX = 3, 6
GEN_SOURCE_SPACING_MIN, GEN_SOURCE_SPACING_MAX = 2, 4
GEN_DETOUR_COUNT_MIN = 1

# Obstacle segment constraints.
GEN_OBSTACLE_COUNTS = (1, 2)
GEN_OBSTACLE_LENGTH_MIN, GEN_OBSTACLE_LENGTH_MAX = 2, 4
GEN_OBSTACLE_EDGE_PAD, GEN_OBSTACLE_SEGMENT_GAP = 1, 1

# Generation retry budget.
GEN_RETRY_LIMIT = 300


def _gen_obstacle_sets(width):
    runs = []
    for length in range(GEN_OBSTACLE_LENGTH_MIN, GEN_OBSTACLE_LENGTH_MAX + 1):
        stop = width - length - GEN_OBSTACLE_EDGE_PAD + 1
        for left in range(GEN_OBSTACLE_EDGE_PAD, stop):
            runs.append((left, left + length - 1))
    by_key = {}

    for run in runs:
        by_key.setdefault((1, (run[1] - run[0] + 1,)), []).append((run,))

    for first in runs:
        for second in runs:
            if first[1] + GEN_OBSTACLE_SEGMENT_GAP + 1 <= second[0]:
                lengths = (first[1] - first[0] + 1, second[1] - second[0] + 1)
                by_key.setdefault((2, lengths), []).append((first, second))

    choices = []
    for count in GEN_OBSTACLE_COUNTS:
        length_keys = [key for key in by_key if key[0] == count]
        if not length_keys:
            continue
        key = random.choice(length_keys)
        choices.append(random.choice(by_key[key]))
    return random.choice(choices)


def _gen_nearest_side(col, left, right, width):
    candidates = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _gen_side_entries(width, runs):
    entries = []
    for run_i, (left, right) in enumerate(runs):
        for dest in (left - 1, right + 1):
            source_options = [
                col
                for col in range(left, right + 1)
                if _gen_nearest_side(col, left, right, width) == dest
            ]
            if source_options:
                entries.append((dest, run_i, tuple(source_options)))
    return entries


def _gen_sources_well_spaced(source_cols):
    cols = sorted(source_cols)
    return all(
        GEN_SOURCE_SPACING_MIN <= cols[i + 1] - cols[i] <= GEN_SOURCE_SPACING_MAX
        for i in range(len(cols) - 1)
    )


def _gen_paths_clear(source_cols, detours):
    source_set = set(source_cols)
    for src, dst in detours:
        lo, hi = sorted((src, dst))
        crossed = source_set.intersection(range(lo, hi + 1)) - {src}
        if crossed:
            return False
    return True


def _gen_mapping_scenarios(width, runs):
    blocked = {col for left, right in runs for col in range(left, right + 1)}
    side_entries = _gen_side_entries(width, runs)
    side_cols = {entry[0] for entry in side_entries}
    through_cols = [col for col in range(width) if col not in blocked and col not in side_cols]
    scenarios_by_key = {}

    for count in range(GEN_SOURCE_COUNT_MIN, GEN_SOURCE_COUNT_MAX + 1):
        max_detours = min(count, len(side_entries))
        for detour_count in range(GEN_DETOUR_COUNT_MIN, max_detours + 1):
            through_count = count - detour_count
            if through_count > len(through_cols):
                continue
            for side_combo in itertools.combinations(side_entries, detour_count):
                detour_dests = [entry[0] for entry in side_combo]
                if len(set(detour_dests)) != len(detour_dests):
                    continue
                source_products = [entry[2] for entry in side_combo]
                for detour_sources in itertools.product(*source_products):
                    if len(set(detour_sources)) != len(detour_sources):
                        continue
                    detours = tuple(zip(detour_sources, detour_dests))
                    for through_combo in itertools.combinations(through_cols, through_count):
                        dest_cols = tuple(detour_dests) + tuple(through_combo)
                        if len(set(dest_cols)) != len(dest_cols):
                            continue
                        source_cols = tuple(detour_sources) + tuple(through_combo)
                        if len(set(source_cols)) != len(source_cols):
                            continue
                        if not _gen_paths_clear(source_cols, detours):
                            continue
                        scenarios_by_key.setdefault((count, detour_count), []).append(
                            (source_cols, dest_cols, detours)
                        )
    return scenarios_by_key


def _gen_choose_mapping(scenarios_by_key):
    if not scenarios_by_key:
        return None

    spaced_by_key = {
        key: [item for item in scenarios if _gen_sources_well_spaced(item[0])]
        for key, scenarios in scenarios_by_key.items()
    }
    spaced_by_key = {key: scenarios for key, scenarios in spaced_by_key.items() if scenarios}
    scenario_pool = spaced_by_key or scenarios_by_key

    counts = sorted({key[0] for key in scenario_pool})
    count = random.choice(counts)
    detour_counts = sorted(key[1] for key in scenario_pool if key[0] == count)
    detour_count = random.choice(detour_counts)
    scenarios = scenario_pool[(count, detour_count)]
    return random.choice(scenarios)


def _gen_paint_obstacles(grid, obstacle_row, runs):
    for left, right in runs:
        grid[obstacle_row, left : right + 1] = GEN_OBSTACLE_COLOR


def _gen_build_output(height, width, obstacle_row, runs, source_cols, dest_cols, detours):
    out = np.full((height, width), GEN_BG_COLOR, dtype=int)

    for src, dst in zip(source_cols, dest_cols):
        if src == dst:
            out[:, dst] = GEN_SOURCE_COLOR
        else:
            out[:obstacle_row, src] = GEN_SOURCE_COLOR
            lo, hi = sorted((src, dst))
            out[obstacle_row - 1, lo : hi + 1] = GEN_SOURCE_COLOR
            out[obstacle_row:, dst] = GEN_SOURCE_COLOR

    _gen_paint_obstacles(out, obstacle_row, runs)
    return out


def _gen_sample_layout():
    height = random.randint(GEN_HEIGHT_MIN, GEN_HEIGHT_MAX)
    width = random.randint(GEN_WIDTH_MIN, GEN_WIDTH_MAX)
    obstacle_row = random.randint(GEN_OBSTACLE_ROW_MIN, height // GEN_OBSTACLE_ROW_MAX_DIVISOR)
    clear_gap = random.randint(GEN_SOURCE_CLEAR_GAP_MIN, GEN_SOURCE_CLEAR_GAP_MAX)
    source_bottom = obstacle_row - 1 - clear_gap
    return height, width, obstacle_row, source_bottom


def _gen_attempt():
    height, width, obstacle_row, source_bottom = _gen_sample_layout()
    runs = _gen_obstacle_sets(width)
    mapping = _gen_choose_mapping(_gen_mapping_scenarios(width, runs))
    if mapping is None:
        return None
    source_cols, dest_cols, detours = mapping

    expected = _gen_build_output(height, width, obstacle_row, runs, source_cols, dest_cols, detours)
    inp = np.full((height, width), GEN_BG_COLOR, dtype=int)
    for col in source_cols:
        inp[: source_bottom + 1, col] = GEN_SOURCE_COLOR
    _gen_paint_obstacles(inp, obstacle_row, runs)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c87289bb input")
