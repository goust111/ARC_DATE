from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

NW_SE = "nw_se"
NE_SW = "ne_sw"

STEP_OPTIONS = [
    (0, 2),
    (0, 3),
    (0, 4),
    (2, 0),
    (3, 0),
    (4, 0),
    (2, 2),
    (3, 3),
    (4, 4),
    (2, -2),
    (3, -3),
    (4, -4),
]

GREEN_PATTERNS = {
    NW_SE: [
        [(-2, -2), (-1, -1)],
        [(1, 1), (2, 2)],
        [(-2, -1), (-1, 0), (0, 1)],
        [(0, -1), (1, 0), (2, 1)],
        [(-1, 0), (0, 1), (1, 2), (0, -2), (1, -1), (2, 0)],
        [(-2, -1), (-1, 0), (0, 1), (1, -1), (2, 0)],
    ],
    NE_SW: [
        [(-2, 2), (-1, 1)],
        [(1, -1), (2, -2)],
        [(-2, 1), (-1, 0), (0, -1)],
        [(0, 1), (1, 0), (2, -1)],
        [(-1, 0), (0, -1), (1, -2), (0, 2), (1, 1), (2, 0)],
        [(-2, 1), (-1, 0), (0, -1), (1, 1), (2, 0)],
    ],
}

MAX_GENERATION_ATTEMPTS = 800
GROUP_PLACEMENT_ATTEMPTS = 220


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _sort_row_major(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _valid_marker_step(dr: int, dc: int) -> bool:
    if dr == 0 and dc > 0:
        return True
    if dc == 0 and dr > 0:
        return True
    return dr > 0 and abs(dc) == dr


def _green_orientation(grid: np.ndarray, r: int, c: int) -> str:
    assert int(grid[r, c]) == GREEN

    for rr, cc in _neighbors4(r, c):
        if _in_bounds(grid.shape, rr, cc):
            assert int(grid[rr, cc]) != GREEN

    has_nw_se = (
        _in_bounds(grid.shape, r - 1, c - 1) and int(grid[r - 1, c - 1]) == GREEN
    ) or (
        _in_bounds(grid.shape, r + 1, c + 1) and int(grid[r + 1, c + 1]) == GREEN
    )
    has_ne_sw = (
        _in_bounds(grid.shape, r - 1, c + 1) and int(grid[r - 1, c + 1]) == GREEN
    ) or (
        _in_bounds(grid.shape, r + 1, c - 1) and int(grid[r + 1, c - 1]) == GREEN
    )
    assert has_nw_se ^ has_ne_sw
    return NW_SE if has_nw_se else NE_SW


def _find_marker_groups(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    blue_coords = _sort_row_major([tuple(x) for x in np.argwhere(grid == BLUE)])
    assert len(blue_coords) in (6, 9)
    blue_set = set(blue_coords)

    candidates: set[tuple[tuple[int, int], ...]] = set()
    for combo in combinations(blue_coords, 3):
        pts = _sort_row_major(list(combo))
        dr1 = pts[1][0] - pts[0][0]
        dc1 = pts[1][1] - pts[0][1]
        dr2 = pts[2][0] - pts[1][0]
        dc2 = pts[2][1] - pts[1][1]
        if (dr1, dc1) != (dr2, dc2):
            continue
        if not _valid_marker_step(dr1, dc1):
            continue
        prev_pt = (pts[0][0] - dr1, pts[0][1] - dc1)
        next_pt = (pts[2][0] + dr1, pts[2][1] + dc1)
        if prev_pt in blue_set or next_pt in blue_set:
            continue
        candidates.add(tuple(pts))

    group_count = len(blue_coords) // 3
    partitions: list[list[list[tuple[int, int]]]] = []
    candidate_list = sorted(candidates)
    for pick in combinations(candidate_list, group_count):
        used = [cell for group in pick for cell in group]
        if len(set(used)) != len(blue_coords):
            continue
        if set(used) != blue_set:
            continue
        partitions.append([list(group) for group in pick])
    assert len(partitions) == 1
    return sorted(partitions[0], key=lambda group: group[0])


def _continuation_cells(
    shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = step
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _parse_instance(grid: np.ndarray) -> tuple[list[dict], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG
    assert max(g.shape) <= 30

    groups = _find_marker_groups(g)
    assert len(groups) in (2, 3)

    green_coords = [tuple(x) for x in np.argwhere(g == GREEN)]
    assert green_coords
    green_axis = {(r, c): _green_orientation(g, int(r), int(c)) for r, c in green_coords}
    axis_counts = Counter(green_axis.values())
    assert set(axis_counts) == {NW_SE, NE_SW}
    assert axis_counts[NW_SE] != axis_counts[NE_SW]
    majority_axis = NW_SE if axis_counts[NW_SE] > axis_counts[NE_SW] else NE_SW

    group_infos: list[dict] = []
    claimed_neighborhood_greens: set[tuple[int, int]] = set()
    all_paths: set[tuple[int, int]] = set()

    for group in groups:
        step = (group[1][0] - group[0][0], group[1][1] - group[0][1])
        assert step == (group[2][0] - group[1][0], group[2][1] - group[1][1])

        middle = group[1]
        nearby = [
            (r, c)
            for r, c in green_coords
            if max(abs(r - middle[0]), abs(c - middle[1])) == 1
        ]
        assert 1 <= len(nearby) <= 3
        assert claimed_neighborhood_greens.isdisjoint(nearby)
        claimed_neighborhood_greens.update(nearby)

        local_axes = {green_axis[cell] for cell in nearby}
        assert len(local_axes) == 1
        local_axis = next(iter(local_axes))

        if local_axis == NE_SW:
            draw_step = step
            start = (group[-1][0] + step[0], group[-1][1] + step[1])
        else:
            draw_step = (-step[0], -step[1])
            start = (group[0][0] - step[0], group[0][1] - step[1])

        path = _continuation_cells(g.shape, start, draw_step)
        assert len(path) >= 2
        for cell in path:
            assert int(g[cell]) == BG
            assert cell not in all_paths
            all_paths.add(cell)

        group_infos.append(
            {
                "markers": group,
                "middle": middle,
                "step": step,
                "axis": local_axis,
                "path": path,
            }
        )

    assert {info["axis"] for info in group_infos} == {NW_SE, NE_SW}
    return group_infos, majority_axis


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each blue marker line in the direction encoded by nearby green diagonals, coloring by the global majority diagonal orientation.

    Input structure:
    - A black background contains 2 or 3 groups of exactly three blue markers.
    - Each blue group is equally spaced on a horizontal, vertical, or 45-degree diagonal line.
    - The middle marker of each group has 1 to 3 nearby green diagonal-segment pixels in its 8-neighborhood.

    Rule:
    - Row-major order each blue triple and use the common difference as its step vector.
    - Read the green orientation around the middle marker: NE/SW-oriented green pixels make the line continue from the last marker along +step, while NW/SE-oriented green pixels make it continue from the first marker along -step.
    - Count all green pixels globally by diagonal orientation; color continuations of the majority orientation teal and the minority orientation red.
    - Draw one pixel per step until the next step would leave the grid, preserving the original blue and green pixels.

    Key insight:
    - The same green diagonal pixels have two independent roles: locally they choose each line's extension direction, and globally they choose the color via majority-versus-minority orientation counts.
    """
    g = np.asarray(grid, dtype=int)
    group_infos, majority_axis = _parse_instance(g)

    out = g.copy()
    for info in group_infos:
        color = TEAL if info["axis"] == majority_axis else RED
        for r, c in info["path"]:
            out[r, c] = color
    return out


def _pattern_neighborhood_count(pattern: list[tuple[int, int]]) -> int:
    return sum(1 for dr, dc in pattern if max(abs(dr), abs(dc)) == 1)


def _pattern_total_count(pattern: list[tuple[int, int]]) -> int:
    return len(pattern)


def _build_group_geometry(
    shape: tuple[int, int],
    start: tuple[int, int],
    step: tuple[int, int],
    axis: str,
    pattern: list[tuple[int, int]],
) -> dict | None:
    markers = [
        start,
        (start[0] + step[0], start[1] + step[1]),
        (start[0] + 2 * step[0], start[1] + 2 * step[1]),
    ]
    if not all(_in_bounds(shape, r, c) for r, c in markers):
        return None

    middle = markers[1]
    greens = [(middle[0] + dr, middle[1] + dc) for dr, dc in pattern]
    if not all(_in_bounds(shape, r, c) for r, c in greens):
        return None
    if any(cell in markers for cell in greens):
        return None

    if axis == NE_SW:
        path_start = (markers[-1][0] + step[0], markers[-1][1] + step[1])
        path_step = step
    else:
        path_start = (markers[0][0] - step[0], markers[0][1] - step[1])
        path_step = (-step[0], -step[1])
    path = _continuation_cells(shape, path_start, path_step)
    if len(path) < 2:
        return None

    return {
        "markers": markers,
        "middle": middle,
        "greens": greens,
        "axis": axis,
        "path": path,
        "step": step,
        "pattern": pattern,
    }


def _sample_group_orientations(rng: np.random.Generator, majority_axis: str, count: int) -> list[str]:
    minority_axis = NE_SW if majority_axis == NW_SE else NW_SE
    if count == 2:
        axes = [majority_axis, minority_axis]
    else:
        mismatch = bool(rng.integers(0, 2))
        repeated_axis = minority_axis if mismatch else majority_axis
        single_axis = majority_axis if mismatch else minority_axis
        axes = [repeated_axis, repeated_axis, single_axis]
    rng.shuffle(axes)
    return axes


def _sample_group_specs(rng: np.random.Generator) -> list[dict] | None:
    group_count = int(rng.integers(2, 4))
    majority_axis = NW_SE if int(rng.integers(0, 2)) == 0 else NE_SW
    axes = _sample_group_orientations(rng, majority_axis, group_count)

    specs = []
    for axis in axes:
        pattern = list(GREEN_PATTERNS[axis][int(rng.integers(0, len(GREEN_PATTERNS[axis])))] )
        specs.append(
            {
                "axis": axis,
                "pattern": pattern,
                "green_total": _pattern_total_count(pattern),
            }
        )

    totals = Counter(spec["axis"] for spec in specs)
    pixel_totals = Counter()
    for spec in specs:
        pixel_totals[spec["axis"]] += spec["green_total"]

    if set(pixel_totals) != {NW_SE, NE_SW}:
        return None
    if pixel_totals[NW_SE] == pixel_totals[NE_SW]:
        return None
    if (NW_SE if pixel_totals[NW_SE] > pixel_totals[NE_SW] else NE_SW) != majority_axis:
        return None

    if group_count == 3 and totals[NW_SE] != totals[NE_SW]:
        group_majority = NW_SE if totals[NW_SE] > totals[NE_SW] else NE_SW
        if int(rng.integers(0, 2)) == 0 and group_majority == majority_axis:
            return None

    return specs


def _can_place_group(
    group: dict,
    occupied_input: set[tuple[int, int]],
    occupied_green: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    middles: list[tuple[int, int]],
) -> bool:
    group_input = set(group["markers"]) | set(group["greens"])
    if group_input & occupied_input:
        return False
    if group_input & reserved_paths:
        return False
    if set(group["path"]) & occupied_input:
        return False
    if set(group["path"]) & reserved_paths:
        return False

    for middle in middles:
        if max(abs(group["middle"][0] - middle[0]), abs(group["middle"][1] - middle[1])) < 4:
            return False

    for r, c in group["greens"]:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in occupied_green:
                return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied_green:
                    return False

    return True


def _place_groups(shape: tuple[int, int], specs: list[dict], rng: np.random.Generator) -> list[dict] | None:
    placed: list[dict] = []
    occupied_input: set[tuple[int, int]] = set()
    occupied_green: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    middles: list[tuple[int, int]] = []

    order = list(specs)
    rng.shuffle(order)
    order.sort(key=lambda spec: (-spec["green_total"], -_pattern_neighborhood_count(spec["pattern"])))

    for spec in order:
        success = False
        for _ in range(GROUP_PLACEMENT_ATTEMPTS):
            step = STEP_OPTIONS[int(rng.integers(0, len(STEP_OPTIONS)))]
            start = (
                int(rng.integers(0, shape[0])),
                int(rng.integers(0, shape[1])),
            )
            group = _build_group_geometry(shape, start, step, spec["axis"], spec["pattern"])
            if group is None:
                continue
            if not _can_place_group(group, occupied_input, occupied_green, reserved_paths, middles):
                continue

            placed.append(group)
            occupied_input |= set(group["markers"])
            occupied_input |= set(group["greens"])
            occupied_green |= set(group["greens"])
            reserved_paths |= set(group["path"])
            middles.append(group["middle"])
            success = True
            break
        if not success:
            return None

    return placed


def _validate_generated_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert _get_bg(g) == BG

    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, BLUE, GREEN}
    group_infos, majority_axis = _parse_instance(g)
    assert len(group_infos) in (2, 3)

    axis_counts = Counter(info["axis"] for info in group_infos)
    assert set(axis_counts) == {NW_SE, NE_SW}

    output = main(g)
    assert output.shape == g.shape
    assert np.array_equal(output[g == BLUE], g[g == BLUE])
    assert np.array_equal(output[g == GREEN], g[g == GREEN])
    assert np.any(output == RED)
    assert np.any(output == TEAL)

    for info in group_infos:
        expected_color = TEAL if info["axis"] == majority_axis else RED
        for cell in info["path"]:
            assert int(output[cell]) == expected_color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (
            int(rng.integers(14, 19)),
            int(rng.integers(14, 19)),
        )
        specs = _sample_group_specs(rng)
        if specs is None:
            continue
        groups = _place_groups(shape, specs, rng)
        if groups is None:
            continue

        grid = np.zeros(shape, dtype=int)
        for group in groups:
            for r, c in group["markers"]:
                grid[r, c] = BLUE
            for r, c in group["greens"]:
                grid[r, c] = GREEN

        try:
            _validate_generated_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
