import random
from typing import Callable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

UPPER = "upper"
LOWER = "lower"

DetourRule = Callable[[int, str], int]


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    return int(rows[0])


def _find_marker(grid: np.ndarray, divider: int) -> tuple[int, int]:
    starts = np.argwhere(grid == RED)
    assert starts.shape == (1, 2), "expected exactly one red marker"
    r, c = map(int, starts[0])
    assert c == 0 and r != divider, "red marker must be on the left border and off the divider"
    return r, c


def _extract_obstacles(grid: np.ndarray, divider: int) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    h, w = grid.shape
    obstacles: list[dict] = []
    obstacle_at: dict[tuple[int, int], dict] = {}

    for c in range(w):
        r = 0
        while r < h:
            color = int(grid[r, c])
            if color not in (BLUE, GREEN):
                r += 1
                continue

            r0 = r
            while r + 1 < h and int(grid[r + 1, c]) == color:
                r += 1
            r1 = r

            assert r1 > r0, "obstacle bars must have length >= 2"
            assert r1 < divider or r0 > divider, "obstacles must stay on one side of the divider"
            half = UPPER if r1 < divider else LOWER
            obstacle = {"color": color, "col": c, "r0": r0, "r1": r1, "half": half}
            obstacles.append(obstacle)

            for rr in range(r0, r1 + 1):
                key = (rr, c)
                assert key not in obstacle_at
                obstacle_at[key] = obstacle
            r += 1

    assert obstacles, "expected at least one obstacle bar"
    return obstacles, obstacle_at


def _correct_detour_rule(color: int, half: str) -> int:
    if half == UPPER:
        return -1 if color == BLUE else 1
    return 1 if color == BLUE else -1


def _upper_color_rule(color: int, _half: str) -> int:
    return -1 if color == BLUE else 1


def _lower_color_rule(color: int, _half: str) -> int:
    return 1 if color == BLUE else -1


def _half_only_rule(_color: int, half: str) -> int:
    return -1 if half == UPPER else 1


def _half_only_flipped_rule(_color: int, half: str) -> int:
    return 1 if half == UPPER else -1


def _paint_cell(out: np.ndarray, src: np.ndarray, r: int, c: int) -> None:
    if int(src[r, c]) == GREY:
        return
    assert int(src[r, c]) in (BLACK, RED)
    out[r, c] = RED


def _paint_horizontal(out: np.ndarray, src: np.ndarray, r: int, c0: int, c1: int) -> None:
    if c0 > c1:
        return
    for c in range(c0, c1 + 1):
        _paint_cell(out, src, r, c)


def _paint_vertical(out: np.ndarray, src: np.ndarray, c: int, r0: int, r1: int) -> None:
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        _paint_cell(out, src, r, c)


def _solve(grid: np.ndarray, detour_rule: DetourRule) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN, GREY}

    h, w = g.shape
    divider = _find_divider_row(g)
    start_r, start_c = _find_marker(g, divider)
    _obstacles, obstacle_at = _extract_obstacles(g, divider)

    out = g.copy()
    _paint_cell(out, g, start_r, start_c)
    current_r, current_c = start_r, start_c
    encounters: list[dict] = []

    while current_c < w - 1:
        hit = None
        for next_c in range(current_c + 1, w):
            if int(g[current_r, next_c]) in (BLUE, GREEN):
                hit = (current_r, next_c)
                break

        if hit is None:
            _paint_horizontal(out, g, current_r, current_c + 1, w - 1)
            break

        hit_r, hit_c = hit
        turn_c = hit_c - 1
        _paint_horizontal(out, g, current_r, current_c + 1, turn_c)

        obstacle = obstacle_at[(hit_r, hit_c)]
        sign = int(detour_rule(int(obstacle["color"]), str(obstacle["half"])))
        assert sign in (-1, 1)

        raw_target_r = int(obstacle["r0"]) - 1 if sign < 0 else int(obstacle["r1"]) + 1
        target_r = raw_target_r + sign if raw_target_r == divider else raw_target_r
        assert 0 <= target_r < h
        assert int(g[target_r, hit_c]) == BLACK, "detour landing cell must be background"

        _paint_vertical(out, g, turn_c, current_r, target_r)
        _paint_cell(out, g, target_r, hit_c)
        encounters.append(
            {
                "color": int(obstacle["color"]),
                "half": str(obstacle["half"]),
                "col": int(hit_c),
                "lane_before": int(current_r),
                "lane_after": int(target_r),
                "crossed_divider": bool(raw_target_r == divider),
            }
        )
        current_r, current_c = target_r, hit_c

    return out, encounters


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the red path from the left marker, detouring around each vertical bar with a half-plane-dependent direction rule.

    Input structure:
    - The grid has black background, one red marker on the left border, and a full-width horizontal gray divider row.
    - Blue and green one-cell-thick vertical obstacle bars lie entirely in either the upper or lower half.
    - The divider row separates the two half-planes and stays visible in the output.

    Rule:
    - Start at the red marker and extend a red path to the right.
    - When the path is about to enter an obstacle column, turn one column early and detour around that bar.
    - In the upper half, blue detours upward and green detours downward; in the lower half, the same color-to-direction mapping is reversed.
    - If a detour crosses the gray divider, skip painting the gray cell itself and continue on the other side; obstacle pixels and divider pixels remain unchanged.

    Key insight:
    - Obstacle color alone is insufficient: the same blue or green bar forces opposite detours depending on whether the bar belongs to the upper or lower half.
    """
    return _solve(grid, _correct_detour_rule)[0]


def _gen_upper_blue(lane: int, divider: int) -> tuple[int, int, int]:
    target = random.randint(max(1, divider - 4), min(lane - 1, divider - 3))
    r0 = target + 1
    r1 = random.randint(max(lane, r0 + 1), min(divider - 1, r0 + 3))
    return r0, r1, target


def _gen_upper_green_cross(lane: int, divider: int) -> tuple[int, int, int]:
    r1 = divider - 1
    r0 = random.randint(max(0, r1 - 3), min(lane, r1 - 1))
    return r0, r1, divider + 1


def _gen_lower_blue(lane: int, divider: int, height: int) -> tuple[int, int, int]:
    target = random.randint(max(lane + 1, divider + 3), min(divider + 4, height - 2))
    r1 = target - 1
    r0 = random.randint(max(divider + 1, r1 - 3), min(lane, r1 - 1))
    return r0, r1, target


def _gen_lower_green_cross(lane: int, divider: int, height: int) -> tuple[int, int, int]:
    r0 = divider + 1
    r1 = random.randint(max(lane, r0 + 1), min(height - 1, r0 + 3))
    return r0, r1, divider - 1


def _build_plan(height: int, divider: int, start_half: str) -> tuple[int, list[dict]]:
    if start_half == UPPER:
        lane = divider - random.choice((2, 3))
        builders = (
            (UPPER, BLUE, lambda x: _gen_upper_blue(x, divider)),
            (UPPER, GREEN, lambda x: _gen_upper_green_cross(x, divider)),
            (LOWER, BLUE, lambda x: _gen_lower_blue(x, divider, height)),
            (LOWER, GREEN, lambda x: _gen_lower_green_cross(x, divider, height)),
        )
    else:
        lane = divider + random.choice((2, 3))
        builders = (
            (LOWER, BLUE, lambda x: _gen_lower_blue(x, divider, height)),
            (LOWER, GREEN, lambda x: _gen_lower_green_cross(x, divider, height)),
            (UPPER, BLUE, lambda x: _gen_upper_blue(x, divider)),
            (UPPER, GREEN, lambda x: _gen_upper_green_cross(x, divider)),
        )

    plan: list[dict] = []
    current_lane = lane
    for half, color, builder in builders:
        r0, r1, next_lane = builder(current_lane)
        assert r0 <= current_lane <= r1
        assert 2 <= r1 - r0 + 1 <= 4
        if half == UPPER:
            assert r1 < divider
        else:
            assert r0 > divider
        plan.append({"half": half, "color": color, "r0": r0, "r1": r1})
        current_lane = next_lane

    return lane, plan


def _wrong_rule_changes_output(grid: np.ndarray, correct_out: np.ndarray, detour_rule: DetourRule) -> bool:
    try:
        wrong_out, _ = _solve(grid, detour_rule)
    except AssertionError:
        return True
    return not np.array_equal(wrong_out, correct_out)


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        out, encounters = _solve(grid, _correct_detour_rule)
    except AssertionError:
        return False

    divider = _find_divider_row(grid)
    combos = {(entry["half"], entry["color"]) for entry in encounters}
    if combos != {(UPPER, BLUE), (UPPER, GREEN), (LOWER, BLUE), (LOWER, GREEN)}:
        return False
    if sum(entry["crossed_divider"] for entry in encounters) < 2:
        return False
    if not np.array_equal(out[divider, :], grid[divider, :]):
        return False
    if not _wrong_rule_changes_output(grid, out, _upper_color_rule):
        return False
    if not _wrong_rule_changes_output(grid, out, _lower_color_rule):
        return False
    if not _wrong_rule_changes_output(grid, out, _half_only_rule):
        return False
    if not _wrong_rule_changes_output(grid, out, _half_only_flipped_rule):
        return False
    return True


def _generate_candidate() -> np.ndarray:
    height = random.choice((15, 17, 19, 21))
    divider = height // 2
    start_half = random.choice((UPPER, LOWER))
    start_row, plan = _build_plan(height, divider, start_half)

    columns = []
    col = random.randint(3, 4)
    for _ in range(len(plan)):
        columns.append(col)
        col += random.randint(3, 5)
    width = min(col + random.randint(2, 4), 29)
    assert columns[-1] < width - 1

    grid = np.zeros((height, width), dtype=int)
    grid[divider, :] = GREY
    grid[start_row, 0] = RED

    for obstacle, c in zip(plan, columns):
        grid[int(obstacle["r0"]) : int(obstacle["r1"]) + 1, c] = int(obstacle["color"])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_candidate()
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid half-plane detour input")
