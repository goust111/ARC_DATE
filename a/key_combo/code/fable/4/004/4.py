from __future__ import annotations

from collections import defaultdict

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
CYAN = 8

COOL_COLORS = {BLUE, CYAN}
WARM_COLORS = {RED, YELLOW}
MOVING_COLORS = COOL_COLORS | WARM_COLORS | {GREEN}

ROWS = "rows"
COLS = "cols"
GENERATION_ATTEMPTS = 500


def _find_edge_bars(grid: np.ndarray) -> tuple[str, dict[int, int]]:
    height, width = map(int, grid.shape)
    assert height >= 5 and width >= 5, "grid needs room between opposite edge bars"

    top = int(grid[0, 0])
    bottom = int(grid[height - 1, 0])
    row_ok = (
        np.all(grid[0, :] == top)
        and np.all(grid[height - 1, :] == bottom)
        and {top, bottom} == {BLUE, RED}
    )

    left = int(grid[0, 0])
    right = int(grid[0, width - 1])
    col_ok = (
        np.all(grid[:, 0] == left)
        and np.all(grid[:, width - 1] == right)
        and {left, right} == {BLUE, RED}
    )

    assert row_ok != col_ok, "expected exactly one pair of opposite blue/red edge bars"
    if row_ok:
        return ROWS, {top: 0, bottom: height - 1}
    return COLS, {left: 0, right: width - 1}


def _to_grid_coord(orientation: str, axis_pos: int, track: int) -> tuple[int, int]:
    if orientation == ROWS:
        return int(axis_pos), int(track)
    return int(track), int(axis_pos)


def _axis_and_track(orientation: str, r: int, c: int) -> tuple[int, int]:
    if orientation == ROWS:
        return int(r), int(c)
    return int(c), int(r)


def _scatter_points(grid: np.ndarray, orientation: str) -> list[dict[str, int]]:
    height, width = map(int, grid.shape)
    points: list[dict[str, int]] = []

    for r0, c0 in np.argwhere(grid != BACKGROUND):
        r, c = int(r0), int(c0)
        if orientation == ROWS and r in (0, height - 1):
            continue
        if orientation == COLS and c in (0, width - 1):
            continue

        color = int(grid[r, c])
        assert color in MOVING_COLORS, "scattered pixels must be blue, cyan, red, yellow, or green"
        axis_pos, track = _axis_and_track(orientation, r, c)
        points.append({"r": r, "c": c, "axis": axis_pos, "track": track, "color": color})

    assert points, "expected at least one scattered pixel between the bars"
    return points


def _nearer_bar(axis_pos: int, bars: dict[int, int]) -> int:
    blue_dist = abs(int(axis_pos) - int(bars[BLUE]))
    red_dist = abs(int(axis_pos) - int(bars[RED]))
    assert blue_dist != red_dist, "green fallback points may not be equidistant from both bars"
    return BLUE if blue_dist < red_dist else RED


def _target_bar_color(color: int, axis_pos: int, bars: dict[int, int]) -> int:
    color = int(color)
    if color in COOL_COLORS:
        return BLUE
    if color in WARM_COLORS:
        return RED
    assert color == GREEN
    return _nearer_bar(axis_pos, bars)


def _solve_grid(
    grid: np.ndarray,
    *,
    target_mode: str = "correct",
    reverse_stack_order: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    orientation, bars = _find_edge_bars(g)
    axis_len = g.shape[0] if orientation == ROWS else g.shape[1]
    points = _scatter_points(g, orientation)

    out = g.copy()
    for point in points:
        out[point["r"], point["c"]] = BACKGROUND

    groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for point in points:
        if target_mode == "correct":
            target = _target_bar_color(point["color"], point["axis"], bars)
        elif target_mode == "nearest":
            target = _nearer_bar(point["axis"], bars)
        elif target_mode == "green_to_blue":
            target = BLUE if point["color"] == GREEN else _target_bar_color(point["color"], point["axis"], bars)
        else:
            raise ValueError(f"unknown target mode: {target_mode}")
        groups[(target, point["track"])].append(point)

    for (target, track), group in groups.items():
        bar_pos = int(bars[target])
        assert bar_pos in (0, axis_len - 1), "bars must lie on opposite grid edges"
        step = 1 if bar_pos == 0 else -1
        group = sorted(
            group,
            key=lambda point: abs(point["axis"] - bar_pos),
            reverse=bool(reverse_stack_order),
        )

        for offset, point in enumerate(group, start=1):
            axis_pos = bar_pos + step * offset
            assert 0 <= axis_pos < axis_len, "stack would run outside the grid"
            r, c = _to_grid_coord(orientation, axis_pos, track)
            assert int(out[r, c]) == BACKGROUND, "two stacks collided"
            out[r, c] = int(point["color"])

    return out.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide scattered pixels home to their color-affiliated edge bar and stack them by original distance.

    Input structure:
    - A black-background grid has exactly two complete opposite edge bars: one blue and one red.
    - The bars are either the top/bottom full rows or the left/right full columns.
    - Between the bars are isolated single pixels colored blue, cyan, red, yellow, or green.
    - Blue/cyan belong to the blue bar, red/yellow belong to the red bar, and green has no color family.

    Rule:
    - Clear every scattered pixel from its original position while leaving both edge bars fixed.
    - Send blue/cyan pixels to the blue bar and red/yellow pixels to the red bar, regardless of distance.
    - Send each green pixel to the nearer bar along the bar-to-bar axis; green pixels are never equidistant.
    - Within one track and one target bar, place pixels next to the bar from nearest original distance to farthest.

    Key insight:
    - The attraction rule is layered: color family chooses first, and distance is only the fallback for green.
    """
    return _solve_grid(grid)


def _positions_closer_to(bars: dict[int, int], target: int, axis_len: int) -> list[int]:
    other = RED if target == BLUE else BLUE
    return [
        pos
        for pos in range(1, axis_len - 1)
        if abs(pos - bars[target]) < abs(pos - bars[other])
    ]


def _non_equidistant_positions(bars: dict[int, int], axis_len: int) -> list[int]:
    return [
        pos
        for pos in range(1, axis_len - 1)
        if abs(pos - bars[BLUE]) != abs(pos - bars[RED])
    ]


def _sample_spaced_tracks(rng: np.random.Generator, track_len: int, count: int) -> list[int] | None:
    candidates = list(range(1, track_len - 1))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if all(abs(value - other) >= 2 for other in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return chosen
    return None


def _attempt_add_point(
    grid: np.ndarray,
    orientation: str,
    axis_pos: int,
    track: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = _to_grid_coord(orientation, axis_pos, track)
    if int(grid[r, c]) != BACKGROUND:
        return False
    if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in occupied):
        return False
    grid[r, c] = int(color)
    occupied.add((r, c))
    return True


def _add_required_point(
    rng: np.random.Generator,
    grid: np.ndarray,
    orientation: str,
    positions: list[int],
    track: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    choices = positions.copy()
    rng.shuffle(choices)
    for axis_pos in choices:
        if _attempt_add_point(grid, orientation, axis_pos, track, color, occupied):
            return True
    return False


def _wrong_nearest_output(grid: np.ndarray) -> np.ndarray | None:
    try:
        return _solve_grid(grid, target_mode="nearest")
    except AssertionError:
        return None


def _wrong_green_fixed_output(grid: np.ndarray) -> np.ndarray | None:
    try:
        return _solve_grid(grid, target_mode="green_to_blue")
    except AssertionError:
        return None


def _wrong_reverse_stack_output(grid: np.ndarray) -> np.ndarray | None:
    try:
        return _solve_grid(grid, reverse_stack_order=True)
    except AssertionError:
        return None


def _semantic_requirements(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        orientation, bars = _find_edge_bars(grid)
        points = _scatter_points(grid, orientation)
    except AssertionError:
        return False

    has_cool_against_distance = False
    has_warm_against_distance = False
    green_targets: set[int] = set()
    groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)

    for point in points:
        nearest = _nearer_bar(point["axis"], bars)
        target = _target_bar_color(point["color"], point["axis"], bars)
        if point["color"] in COOL_COLORS and nearest == RED:
            has_cool_against_distance = True
        if point["color"] in WARM_COLORS and nearest == BLUE:
            has_warm_against_distance = True
        if point["color"] == GREEN:
            green_targets.add(target)
        groups[(target, point["track"])].append(point)

    has_visible_stack = any(
        len(group) >= 2 and len({int(point["color"]) for point in group}) >= 2
        for group in groups.values()
    )

    nearest_out = _wrong_nearest_output(grid)
    green_fixed_out = _wrong_green_fixed_output(grid)
    reverse_out = _wrong_reverse_stack_output(grid)
    if nearest_out is None or green_fixed_out is None or reverse_out is None:
        return False

    return (
        has_cool_against_distance
        and has_warm_against_distance
        and green_targets == {BLUE, RED}
        and has_visible_stack
        and not np.array_equal(expected, nearest_out)
        and not np.array_equal(expected, green_fixed_out)
        and not np.array_equal(expected, reverse_out)
        and not np.array_equal(expected, grid)
    )


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = ROWS if int(rng.integers(2)) == 0 else COLS
    axis_len = int(rng.integers(11, 23))
    track_len = int(rng.integers(12, 25))
    height, width = (axis_len, track_len) if orientation == ROWS else (track_len, axis_len)

    low_color, high_color = (BLUE, RED) if int(rng.integers(2)) == 0 else (RED, BLUE)
    bars = {low_color: 0, high_color: axis_len - 1}

    grid = np.full((height, width), BACKGROUND, dtype=int)
    if orientation == ROWS:
        grid[0, :] = low_color
        grid[height - 1, :] = high_color
    else:
        grid[:, 0] = low_color
        grid[:, width - 1] = high_color

    blue_side = _positions_closer_to(bars, BLUE, axis_len)
    red_side = _positions_closer_to(bars, RED, axis_len)
    if not blue_side or not red_side:
        return None

    tracks = _sample_spaced_tracks(rng, track_len, 5)
    if tracks is None:
        return None

    occupied: set[tuple[int, int]] = set()
    if not _add_required_point(
        rng, grid, orientation, red_side, tracks[0], int(rng.choice([BLUE, CYAN])), occupied
    ):
        return None
    if not _add_required_point(
        rng, grid, orientation, blue_side, tracks[1], int(rng.choice([RED, YELLOW])), occupied
    ):
        return None
    if not _add_required_point(rng, grid, orientation, blue_side, tracks[2], GREEN, occupied):
        return None
    if not _add_required_point(rng, grid, orientation, red_side, tracks[3], GREEN, occupied):
        return None

    stack_target = BLUE if int(rng.integers(2)) == 0 else RED
    stack_near = _positions_closer_to(bars, stack_target, axis_len)
    stack_far = _positions_closer_to(bars, RED if stack_target == BLUE else BLUE, axis_len)
    stack_colors = [BLUE, CYAN] if stack_target == BLUE else [RED, YELLOW]
    if int(rng.integers(2)) == 1:
        stack_colors.reverse()

    near_choices = stack_near.copy()
    far_choices = stack_far.copy()
    rng.shuffle(near_choices)
    rng.shuffle(far_choices)
    placed_stack = False
    for near_axis in near_choices:
        for far_axis in far_choices:
            if abs(near_axis - far_axis) < 2:
                continue
            trial = grid.copy()
            trial_occ = set(occupied)
            if not _attempt_add_point(trial, orientation, near_axis, tracks[4], stack_colors[0], trial_occ):
                continue
            if not _attempt_add_point(trial, orientation, far_axis, tracks[4], stack_colors[1], trial_occ):
                continue
            grid = trial
            occupied = trial_occ
            placed_stack = True
            break
        if placed_stack:
            break
    if not placed_stack:
        return None

    all_axis_positions = _non_equidistant_positions(bars, axis_len)
    extra_count = int(rng.integers(2, 7))
    for _ in range(extra_count):
        for _try in range(100):
            color = int(rng.choice([BLUE, CYAN, RED, YELLOW, GREEN]))
            axis_pos = int(rng.choice(all_axis_positions))
            track = int(rng.integers(1, track_len - 1))
            if _attempt_add_point(grid, orientation, axis_pos, track, color, occupied):
                break

    try:
        expected = _solve_grid(grid)
    except AssertionError:
        return None

    if not _semantic_requirements(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_attempt(np.random.default_rng())
        if grid is not None:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid layered homing input")
