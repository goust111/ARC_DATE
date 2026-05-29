# Combo variant of 28a6681f (index 7)
# combo_spec: D7/2+D9/1
# Name: 带孔洞壁面+连通器原理

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete layout and behavior trace
#    The fused task keeps the original 10x10 bottom-anchored basin vocabulary, but fixes the
#    geometry to a three-basin chain separated by two thin vertical dividers. Each divider has
#    exactly one one-pixel hole below its crest. Before mirroring, the layout is:
#    - basin A: cols 0..3 (wide source basin),
#    - divider 1: col 4 with a hole at row h1,
#    - basin B: cols 5..6,
#    - divider 2: col 7 with a hole at row h2,
#    - basin C: cols 8..9.
#    Small trace on a profile with h1=6 and h2=5:
#    - Original local-gravity logic would keep source water stacked in basin A until individual
#      pixels can slide through a hole; after that drop, it still cannot "rise" inside basin C.
#    - Fused logic instead treats a hole as a connection threshold. Once basin A reaches row 6,
#      A and B become one communicating system and must share a common level. If that shared level
#      also reaches row 5, basin C joins too, and all three basins finish with one flat top row.
#    - The visible difference is that water appears in later basins at the same final height as the
#      source basin even though that requires a down-then-up redistribution through lower side holes.
#
# b) Complexity calibration
#    Three basins are enough to make staged activation visible without turning the scene noisy.
#    The generator only varies a small palette of wall tops and hole heights, and only samples a
#    small set of exact final waterlines. That gives repeated support across examples: humans see
#    the same motif of "wide source basin + two holed dividers + shared final level" rather than a
#    one-off arrangement.
#
# c) Implicit-assumption audit for main()
#    - The original `settle()` assumes motion is monotone downward with local horizontal sliding.
#      That breaks here because communicating vessels require water to descend through a hole and
#      then occupy higher cells in a neighboring basin. The local simulation is therefore replaced.
#    - The old output-side support check still largely survives because the fused generator keeps
#      narrow basin columns bounded by walls or grid edges; final water still occupies complete
#      bottom-up layers, so every water cell remains supported from below.
#    - The solver now assumes basins can be detected from the open runs on row 8 and that adjacent
#      runs are separated by exactly one divider column. The generator enforces this fixed geometry.
#    - The solver also assumes the total water amount corresponds to a whole communicating level,
#      not a partial top row. `generate_input()` samples only exact row-capacity counts.
#
# d) Placement legality
#    Divider columns are fixed at 4 and 7 before optional mirroring, so nothing can overlap or go
#    out of bounds. Hole rows are always strictly between the wall top and the floor, preserving a
#    visible solid wall above and below each hole. Source water is packed only inside the wide edge
#    basin, where every cell is supported by water or by the floor.
#
# e) Visual observability
#    The intended clue is visible in two steps:
#    - the input shows two clear wall holes below the wall tops,
#    - the output shows all three basins ending with the same top water row.
#    Middle and right basins start empty, so their later water cannot be explained by simply keeping
#    the original local settling behavior. Equal final waterlines across separated basins point to
#    communicating-vessel rebalancing, and the hole heights explain when that rebalancing becomes
#    possible.
#
# f) Visual cleanliness
#    Only four visual categories appear: black background, blue water, and three obstacle colors.
#    There are no guides, labels, or overlapping symbolic markers. Each divider is one column wide,
#    so the hole reads as a literal missing wall pixel rather than a color code.
#
# g) Change inferability
#    Colors do not carry any behavioral meaning. The rule is recoverable from geometry alone:
#    basin width, divider height, and the explicit hole positions determine the final redistribution.
#    Because the same shared-level pattern repeats over several hole-height profiles and mirrored
#    copies, humans can infer "hole-triggered communicating basins" from a small example set.
#
# New solver assumptions audited against the generator
#    - `main()` detects basin runs from row 8; every generated basin has at least one open cell there.
#    - Every pair of neighboring basin runs has exactly one separator column; the generator fixes
#      this with columns 4 and 7 before optional horizontal mirroring.
#    - All input water starts in one wide source basin, so the eventual communicating interval is a
#      single contiguous basin range containing that source basin.
#    - The sampled water amount always matches a complete lowest-row fill of the final connected
#      span. This guarantees a unique candidate interval and avoids ambiguous partial top rows.


def _group_rows(cells: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    by_row: dict[int, list[tuple[int, int]]] = {}
    for r, c in cells:
        by_row.setdefault(int(r), []).append((int(r), int(c)))
    for row_cells in by_row.values():
        row_cells.sort()
    return by_row


def _fill_lowest_rows(
    cells: list[tuple[int, int]], water_count: int, require_exact_level: bool
) -> set[tuple[int, int]]:
    by_row = _group_rows(cells)
    filled: set[tuple[int, int]] = set()
    remaining = int(water_count)
    for r in sorted(by_row, reverse=True):
        row_cells = by_row[r]
        if remaining >= len(row_cells):
            filled.update(row_cells)
            remaining -= len(row_cells)
            continue
        if remaining == 0:
            break
        if require_exact_level:
            raise AssertionError("water amount does not match a complete communicating level")
        filled.update(row_cells[:remaining])
        remaining = 0
        break
    if remaining != 0:
        raise AssertionError("water amount exceeds cavity capacity")
    return filled


def _top_water_row(filled: set[tuple[int, int]], default: int) -> int:
    return int(min((r for r, _ in filled), default=default))


def _find_basin_runs(g: np.ndarray, bg: int, water: int) -> list[tuple[int, int]]:
    open_row = [bool(int(g[g.shape[0] - 2, c]) in (int(bg), int(water))) for c in range(g.shape[1])]
    runs: list[tuple[int, int]] = []
    c = 0
    while c < len(open_row):
        if not open_row[c]:
            c += 1
            continue
        start = c
        while c + 1 < len(open_row) and open_row[c + 1]:
            c += 1
        runs.append((int(start), int(c)))
        c += 1
    return runs


def _separator_threshold(
    g: np.ndarray, left_run: tuple[int, int], right_run: tuple[int, int], bg: int, water: int
) -> int:
    H = g.shape[0]
    assert int(right_run[0]) - int(left_run[1]) == 2, "adjacent basins must be separated by one divider column"
    sep = int(left_run[1] + 1)
    pass_rows = []
    for r in range(H - 1):
        if (
            int(g[r, sep]) in (int(bg), int(water))
            and int(g[r, int(left_run[1])]) in (int(bg), int(water))
            and int(g[r, int(right_run[0])]) in (int(bg), int(water))
        ):
            pass_rows.append(int(r))
    if not pass_rows:
        raise AssertionError("neighboring basins have no visible connection row")
    return int(max(pass_rows))


def _collect_interval_cells(
    g: np.ndarray, left_col: int, right_col: int, bg: int, water: int
) -> list[tuple[int, int]]:
    H = g.shape[0]
    cells: list[tuple[int, int]] = []
    for r in range(H - 1):
        for c in range(int(left_col), int(right_col) + 1):
            if int(g[r, c]) in (int(bg), int(water)):
                cells.append((int(r), int(c)))
    return cells


def _choose_equilibrium_fill(
    g: np.ndarray, require_exact_level: bool
) -> tuple[set[tuple[int, int]], tuple[int, int], list[tuple[int, int]], list[int]]:
    H, _W = g.shape
    bg = 0
    water = 1
    total_water = int(np.count_nonzero(g == int(water)))
    if total_water == 0:
        return set(), (0, -1), [], []

    runs = _find_basin_runs(g, bg, water)
    if not runs:
        raise AssertionError("no basin runs detected")
    thresholds = [
        _separator_threshold(g, runs[i], runs[i + 1], bg, water) for i in range(len(runs) - 1)
    ]

    water_basin_indices = [
        i for i, (start, end) in enumerate(runs) if bool(np.any(g[: H - 1, start : end + 1] == int(water)))
    ]
    if not water_basin_indices:
        raise AssertionError("water must start inside at least one basin")
    source_lo = int(min(water_basin_indices))
    source_hi = int(max(water_basin_indices))
    if require_exact_level:
        expected_span = source_hi - source_lo + 1
        assert expected_span == len(set(water_basin_indices)), "water-bearing basins must form one contiguous block"

    candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
    for lo in range(source_lo, -1, -1):
        for hi in range(source_hi, len(runs)):
            cells = _collect_interval_cells(g, runs[lo][0], runs[hi][1], bg, water)
            try:
                filled = _fill_lowest_rows(cells, total_water, require_exact_level=require_exact_level)
            except AssertionError:
                continue
            top = _top_water_row(filled, default=H - 1)
            if any(top > thresholds[k] for k in range(lo, hi)):
                continue
            if lo > 0 and top <= thresholds[lo - 1]:
                continue
            if hi < len(runs) - 1 and top <= thresholds[hi]:
                continue
            candidates.append((int(lo), int(hi), filled))

    if require_exact_level:
        assert len(candidates) == 1, "equilibrium interval must be uniquely determined"
    if not candidates:
        raise AssertionError("no valid communicating-basin equilibrium found")

    lo, hi, filled = max(candidates, key=lambda item: (item[1] - item[0], -item[0]))
    return filled, (int(lo), int(hi)), runs, thresholds


def _water_top_in_run(g: np.ndarray, run: tuple[int, int]) -> int | None:
    rows = np.argwhere(g[:, int(run[0]) : int(run[1]) + 1] == 1)
    if len(rows) == 0:
        return None
    return int(rows[:, 0].min())


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Rebalance water across hole-linked basins until every activated basin shares one level.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors draw bottom-anchored basin floors and thin divider walls; each divider may have
      a one-cell hole below its top, so some neighboring basins connect early through the wall.
    - Water begins as a supported pile in one basin while neighboring basins can start empty.

    Rule:
    - Detect the basin runs and the deepest visible crossing row on each divider column. A hole gives
      a lower crossing row than ordinary wall-top overflow, so it activates the connection earlier.
    - Count the total water in the contiguous basin block that contains the source water.
    - Choose the unique basin interval whose shared lowest-row fill reaches every internal divider
      threshold inside that interval, but does not rise high enough to activate the next divider
      outside it.
    - Refill that full connected span from the bottom upward, producing one common waterline across
      every basin that became connected through its hole.

    Key insight: The decisive quantity is not where each pixel can slide locally, but which divider
    holes are submerged and therefore which basin span must behave as one communicating vessel.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1
    initial_water = int(np.count_nonzero(g == int(water)))

    if check_input:
        assert (H, W) == (10, 10), "this fused variant expects a 10x10 grid"
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
        runs = _find_basin_runs(g, bg, water)
        assert 2 <= len(runs) <= 3, "expected a short chain of adjacent bottom basins"
        for i in range(len(runs) - 1):
            assert int(runs[i + 1][0]) - int(runs[i][1]) == 2, "basins must be separated by one divider column"

    filled, _interval, _runs, _thresholds = _choose_equilibrium_fill(g, require_exact_level=check_input)

    g[g == int(water)] = int(bg)
    for r, c in filled:
        g[int(r), int(c)] = int(water)

    if check_output:
        assert int(np.count_nonzero(g == int(water))) == initial_water, "water count must be conserved"
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "output water must be supported"

        expected_filled, interval, runs, thresholds = _choose_equilibrium_fill(g, require_exact_level=True)
        actual_filled = {(int(r), int(c)) for r, c in np.argwhere(g == int(water))}
        assert actual_filled == expected_filled, "output must already be at the communicating equilibrium"

        top_rows = [_water_top_in_run(g, run) for run in runs[int(interval[0]) : int(interval[1]) + 1]]
        assert all(top is not None for top in top_rows), "every active basin must contain water"
        shared_top = int(top_rows[0])
        assert all(int(top) == shared_top for top in top_rows), "active basins must share one top water row"
        for k in range(int(interval[0]), int(interval[1])):
            assert shared_top <= int(thresholds[k]), "internal hole threshold must be submerged in the output"

    return g


_PROFILES = (
    {"w1_top": 3, "hole1": 6, "w2_top": 2, "hole2": 5},
    {"w1_top": 2, "hole1": 6, "w2_top": 3, "hole2": 6},
    {"w1_top": 3, "hole1": 7, "w2_top": 2, "hole2": 6},
    {"w1_top": 2, "hole1": 7, "w2_top": 3, "hole2": 7},
    {"w1_top": 4, "hole1": 7, "w2_top": 3, "hole2": 6},
)
_OBSTACLE_COLORS = tuple(range(2, 10))


def _build_terrain(profile: dict[str, int], colors: tuple[int, int, int]) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    left_color, mid_color, right_color = colors

    g[9, 0:5] = int(left_color)
    g[9, 5:8] = int(mid_color)
    g[9, 8:10] = int(right_color)

    for r in range(int(profile["w1_top"]), 9):
        if r != int(profile["hole1"]):
            g[r, 4] = int(left_color)
    for r in range(int(profile["w2_top"]), 9):
        if r != int(profile["hole2"]):
            g[r, 7] = int(mid_color)

    return g


def _source_capacity(base: np.ndarray, source_run: tuple[int, int]) -> int:
    return int(np.count_nonzero(np.isin(base[:9, int(source_run[0]) : int(source_run[1]) + 1], (0, 1))))


def _full_span_options(base: np.ndarray) -> list[tuple[int, int]]:
    runs = _find_basin_runs(base, 0, 1)
    assert len(runs) == 3, "generator expects three basins"
    thresholds = [_separator_threshold(base, runs[i], runs[i + 1], 0, 1) for i in range(2)]
    min_threshold = int(min(thresholds))
    cells = _collect_interval_cells(base, runs[0][0], runs[-1][1], 0, 1)
    source_capacity = _source_capacity(base, runs[0])

    options: list[tuple[int, int]] = []
    for level in range(min_threshold, -1, -1):
        water_count = sum(1 for r, _c in cells if int(r) >= int(level))
        if water_count > source_capacity:
            continue
        filled = _fill_lowest_rows(cells, water_count, require_exact_level=True)
        if _top_water_row(filled, default=9) != int(level):
            continue
        options.append((int(level), int(water_count)))
    return options


def _make_output(base: np.ndarray, water_count: int) -> np.ndarray:
    runs = _find_basin_runs(base, 0, 1)
    cells = _collect_interval_cells(base, runs[0][0], runs[-1][1], 0, 1)
    out = np.array(base, dtype=int).copy()
    for r, c in _fill_lowest_rows(cells, water_count, require_exact_level=True):
        out[int(r), int(c)] = 1
    return out


def _make_input(base: np.ndarray, source_run: tuple[int, int], water_count: int) -> np.ndarray:
    inp = np.array(base, dtype=int).copy()
    source_slots: list[tuple[int, int]] = []
    for r in range(8, -1, -1):
        for c in range(int(source_run[0]), int(source_run[1]) + 1):
            source_slots.append((int(r), int(c)))
    if water_count > len(source_slots):
        raise AssertionError("sampled water exceeds source-basin capacity")
    for r, c in source_slots[: int(water_count)]:
        inp[int(r), int(c)] = 1
    return inp


def _variant_visible(base: np.ndarray, output: np.ndarray) -> bool:
    runs = _find_basin_runs(base, 0, 1)
    thresholds = [_separator_threshold(base, runs[i], runs[i + 1], 0, 1) for i in range(2)]
    tops = [_water_top_in_run(output, run) for run in runs]
    if any(top is None for top in tops):
        return False
    shared_top = int(tops[0])
    if any(int(top) != shared_top for top in tops):
        return False
    if shared_top > min(thresholds):
        return False
    return bool(int(output[int(thresholds[0]), int(runs[0][1] + 1)]) == 1 and int(output[int(thresholds[1]), int(runs[1][1] + 1)]) == 1)


def generate_input() -> np.ndarray:
    for _ in range(100):
        profile = random.choice(_PROFILES)
        colors = tuple(random.sample(_OBSTACLE_COLORS, 3))
        base = _build_terrain(profile, colors)
        runs = _find_basin_runs(base, 0, 1)
        options = _full_span_options(base)
        if not options:
            continue

        _level, water_count = random.choice(options)
        output = _make_output(base, water_count)
        input_grid = _make_input(base, runs[0], water_count)

        if random.random() < 0.5:
            input_grid = np.fliplr(input_grid).copy()
            output = np.fliplr(output).copy()
            base = np.fliplr(base).copy()

        if not _variant_visible(base, output):
            continue

        try:
            solved = main(input_grid)
            stable_output = main(output)
        except AssertionError:
            continue
        if not np.array_equal(solved, output):
            continue
        if not np.array_equal(stable_output, output):
            continue
        if np.array_equal(input_grid, output):
            continue

        return input_grid

    raise AssertionError("failed to generate a valid holed communicating-basin example")
