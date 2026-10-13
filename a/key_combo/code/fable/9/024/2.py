from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE_CAP = 1
RED_CAP = 2
SPRITE = 3
WALL = 5
DYE_COLORS = (4, 6, 7, 8, 9)

GEN_RETRIES = 400


def _all_wall_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == WALL)]


def _lane_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    walls = _all_wall_columns(grid)
    assert len(walls) >= 2, "expected at least two full-height wall columns"
    assert walls[0] == 0 and walls[-1] == grid.shape[1] - 1, "outermost columns must be walls"

    lanes: list[tuple[int, int]] = []
    for left, right in zip(walls, walls[1:]):
        width = right - left - 1
        assert width >= 1, "adjacent walls must leave room for a lane"
        lanes.append((left, right))
    return lanes


def _parse_lane(grid: np.ndarray, left_wall: int, right_wall: int) -> dict:
    H = grid.shape[0]
    cols = slice(left_wall + 1, right_wall)
    lane = grid[:, cols]

    cap_cells: list[tuple[int, int, int]] = []
    for c in range(lane.shape[1]):
        if int(lane[0, c]) == BLUE_CAP:
            cap_cells.append((0, c, 1))
        if int(lane[H - 1, c]) == RED_CAP:
            cap_cells.append((H - 1, c, -1))
    assert len(cap_cells) == 1, "each lane needs exactly one direction cap"
    cap_r, cap_c_local, direction = cap_cells[0]

    sprite_coords = np.argwhere(lane == SPRITE)
    assert sprite_coords.shape == (1, 2), "each lane needs exactly one sprite"
    sprite_r = int(sprite_coords[0, 0])
    sprite_c_local = int(sprite_coords[0, 1])
    assert 0 < sprite_r < H - 1, "sprite must be away from the boundary rows"
    assert sprite_c_local == cap_c_local, "sprite and cap must share the lane axis"

    gates: dict[int, int] = {}
    for r in range(1, H - 1):
        row = lane[r]
        vals = set(map(int, row))
        if len(vals) == 1:
            color = next(iter(vals))
            if color not in {BG, SPRITE}:
                gates[int(r)] = int(color)
        else:
            assert vals <= {BG, SPRITE}, "non-gate rows may only contain background and the sprite"

    dyed_rows = [r for r, color in gates.items() if color != WALL]
    bounce_rows = [r for r, color in gates.items() if color == WALL]
    assert bounce_rows, "each lane must contain at least one grey bounce gate"
    assert dyed_rows, "each lane must contain at least one colored dye gate"

    return {
        "left_wall": int(left_wall),
        "right_wall": int(right_wall),
        "sprite_r": sprite_r,
        "sprite_c": int(left_wall + 1 + sprite_c_local),
        "direction": int(direction),
        "gates": gates,
        "cap_row": int(cap_r),
    }


def _simulate_lane(height: int, sprite_r: int, direction: int, gates: dict[int, int]) -> dict:
    current_r = int(sprite_r)
    current_dir = int(direction)
    current_color = SPRITE
    remaining = {int(r): int(color) for r, color in gates.items()}
    history: list[tuple[int, int]] = []

    while True:
        next_r = current_r + current_dir
        if next_r < 1 or next_r > height - 2:
            break
        if next_r in remaining:
            gate_color = remaining.pop(next_r)
            history.append((int(next_r), int(gate_color)))
            if gate_color == WALL:
                current_dir *= -1
                continue
            current_r = int(next_r)
            current_color = int(gate_color)
            continue
        current_r = int(next_r)

    visited_rows = [r for r, _color in history]
    visited_dyes = [(r, color) for r, color in history if color != WALL]
    visited_bounces = [r for r, color in history if color == WALL]
    untouched = sorted(set(map(int, gates)) - set(visited_rows))
    return {
        "final_r": int(current_r),
        "final_color": int(current_color),
        "history": history,
        "visited_rows": visited_rows,
        "visited_dyes": visited_dyes,
        "visited_bounces": visited_bounces,
        "untouched_rows": untouched,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each lane sprite through dye gates and bounce gates, then leave only the final parked sprite and the untouched gates.

    Input structure:
    - Full-height grey wall columns partition the grid into several vertical lanes.
    - Each lane has one cap marker on its lane axis: a blue top cap means the sprite starts by moving downward, and a red bottom cap means it starts by moving upward.
    - Each lane contains one single-cell sprite plus several one-cell-thick horizontal gate lines spanning the lane width; colored gates dye the sprite while grey gates make it bounce.

    Rule:
    - Simulate each sprite independently, moving one row at a time along its lane axis.
    - Crossing a colored gate recolors the sprite to that gate's color and removes that whole gate line.
    - Hitting a grey gate removes that whole gate line and reverses the sprite without entering the gate row; removed gates stay gone if the sprite later returns past them.
    - The sprite keeps moving until it reaches the last in-lane cell at the current end of the lane; untouched gates, caps, and walls remain unchanged.

    Key insight:
    - Grey gates are one-use bounce mirrors, so the decisive dye may sit behind the start point or beyond a gate that later disappears; the final sprite color is the last colored gate crossed on the full path, not the deepest visible gate in one direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    out = g.copy()
    H, _W = g.shape
    for left_wall, right_wall in _lane_bounds(g):
        lane = _parse_lane(g, left_wall, right_wall)
        result = _simulate_lane(H, lane["sprite_r"], lane["direction"], lane["gates"])

        out[lane["sprite_r"], lane["sprite_c"]] = BG
        for row in result["visited_rows"]:
            out[row, left_wall + 1 : right_wall] = BG
        out[result["final_r"], lane["sprite_c"]] = result["final_color"]

    return out


def _reflect_row(height: int, row: int) -> int:
    return int(height - 1 - row)


def _spaced_choices(lo: int, hi: int, counts: tuple[int, ...], min_gap: int = 2) -> tuple[int, ...] | None:
    if lo > hi:
        return None
    values = list(range(lo, hi + 1))
    all_choices: list[tuple[int, ...]] = []
    for count in counts:
        if count == 0:
            all_choices.append(())
            continue
        for combo in itertools.combinations(values, count):
            if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1)):
                all_choices.append(tuple(map(int, combo)))
    if not all_choices:
        return None
    return random.choice(all_choices)


def _random_untouched_colors(count: int) -> list[int]:
    colors: list[int] = []
    for _ in range(count):
        if random.random() < 0.35:
            colors.append(WALL)
        else:
            colors.append(random.choice(DYE_COLORS))
    return colors


def _build_down_pattern_one(height: int) -> dict | None:
    for _ in range(120):
        sprite_r = random.randint(5, height - 7)
        grey_r = random.randint(sprite_r + 3, height - 4)

        above = _spaced_choices(2, sprite_r - 1, (1, 2))
        between = _spaced_choices(sprite_r + 1, grey_r - 1, (1, 2))
        below = _spaced_choices(grey_r + 1, height - 3, (1, 2))
        if above is None or between is None or below is None:
            continue

        colored_rows = list(above) + list(between)
        if len(colored_rows) > len(DYE_COLORS):
            continue

        colors = random.sample(DYE_COLORS, len(colored_rows))
        gates: dict[int, int] = {int(grey_r): WALL}
        for row, color in zip(colored_rows, colors):
            gates[int(row)] = int(color)
        for row, color in zip(below, _random_untouched_colors(len(below))):
            gates[int(row)] = int(color)

        return {"direction": 1, "pattern": 1, "sprite_r": int(sprite_r), "gates": gates}
    return None


def _build_down_pattern_two(height: int) -> dict | None:
    for _ in range(160):
        grey_top = random.randint(4, height - 10)
        sprite_r = random.randint(grey_top + 3, height - 7)
        grey_bottom = random.randint(sprite_r + 3, height - 4)

        untouched_above = _spaced_choices(2, grey_top - 1, (1, 2))
        mid_up = _spaced_choices(grey_top + 1, sprite_r - 1, (1, 2))
        mid_down = _spaced_choices(sprite_r + 1, grey_bottom - 1, (1, 2))
        final_below = _spaced_choices(grey_bottom + 1, height - 3, (1, 2))
        if None in {untouched_above, mid_up, mid_down, final_below}:
            continue

        colored_rows = list(mid_up) + list(mid_down) + list(final_below)
        if len(colored_rows) > len(DYE_COLORS):
            continue

        colors = random.sample(DYE_COLORS, len(colored_rows))
        gates: dict[int, int] = {int(grey_top): WALL, int(grey_bottom): WALL}

        cursor = 0
        for group in (mid_up, mid_down, final_below):
            for row in group:
                gates[int(row)] = int(colors[cursor])
                cursor += 1
        for row, color in zip(untouched_above, _random_untouched_colors(len(untouched_above))):
            gates[int(row)] = int(color)

        return {"direction": 1, "pattern": 2, "sprite_r": int(sprite_r), "gates": gates}
    return None


def _mirror_lane_spec(height: int, spec: dict) -> dict:
    return {
        "direction": -1,
        "pattern": int(spec["pattern"]),
        "sprite_r": _reflect_row(height, int(spec["sprite_r"])),
        "gates": {_reflect_row(height, int(row)): int(color) for row, color in spec["gates"].items()},
    }


def _make_lane_spec(height: int, direction: int, pattern: int) -> dict | None:
    if pattern == 1:
        spec = _build_down_pattern_one(height)
    else:
        spec = _build_down_pattern_two(height)
    if spec is None:
        return None
    if direction == 1:
        return spec
    return _mirror_lane_spec(height, spec)


def _lane_output_signature(height: int, lane_spec: dict) -> tuple[int, int, int]:
    sim = _simulate_lane(height, lane_spec["sprite_r"], lane_spec["direction"], lane_spec["gates"])
    return int(sim["final_r"]), int(sim["final_color"]), len(sim["visited_rows"])


def _paint_input(height: int, widths: list[int], lane_specs: list[dict]) -> np.ndarray:
    width = 1 + sum(widths) + len(widths)
    grid = np.full((height, width), BG, dtype=int)

    wall = 0
    grid[:, wall] = WALL
    for lane_w, spec in zip(widths, lane_specs):
        next_wall = wall + lane_w + 1
        grid[:, next_wall] = WALL
        axis_c = wall + 1 + lane_w // 2

        if int(spec["direction"]) == 1:
            grid[0, axis_c] = BLUE_CAP
        else:
            grid[height - 1, axis_c] = RED_CAP

        grid[int(spec["sprite_r"]), axis_c] = SPRITE
        for row, color in spec["gates"].items():
            grid[int(row), wall + 1 : next_wall] = int(color)

        wall = next_wall

    return grid


def _validate_lane_semantics(height: int, lane: dict) -> bool:
    sim = _simulate_lane(height, lane["sprite_r"], lane["direction"], lane["gates"])
    if not sim["visited_dyes"] or not sim["visited_bounces"] or not sim["untouched_rows"]:
        return False

    first_grey = sim["visited_bounces"][0]
    last_dye_row, last_dye_color = sim["visited_dyes"][-1]
    if last_dye_color == SPRITE:
        return False

    if lane["pattern"] == 1:
        if len(sim["visited_bounces"]) != 1:
            return False
        if lane["direction"] == 1 and not (last_dye_row < lane["sprite_r"]):
            return False
        if lane["direction"] == -1 and not (last_dye_row > lane["sprite_r"]):
            return False
    else:
        if len(sim["visited_bounces"]) != 2:
            return False
        if lane["direction"] == 1 and not (last_dye_row > first_grey):
            return False
        if lane["direction"] == -1 and not (last_dye_row < first_grey):
            return False

    first_dye_color = sim["visited_dyes"][0][1]
    if last_dye_color == first_dye_color:
        return False
    return True


def _validate_generated_grid(grid: np.ndarray, lane_specs: list[dict]) -> bool:
    height = grid.shape[0]
    lanes = [_parse_lane(grid, left, right) for left, right in _lane_bounds(grid)]
    if len(lanes) != len(lane_specs):
        return False

    solved = main(grid)
    if np.array_equal(solved, grid):
        return False

    seen_sig = set()
    for parsed, spec in zip(lanes, lane_specs):
        if parsed["direction"] != spec["direction"]:
            return False
        if parsed["sprite_r"] != spec["sprite_r"]:
            return False
        if parsed["gates"] != spec["gates"]:
            return False
        if not _validate_lane_semantics(height, spec):
            return False
        sig = _lane_output_signature(height, spec)
        if sig in seen_sig:
            return False
        seen_sig.add(sig)

    patterns = {int(spec["pattern"]) for spec in lane_specs}
    directions = {int(spec["direction"]) for spec in lane_specs}
    if len(lane_specs) >= 2 and patterns != {1, 2}:
        return False
    if len(lane_specs) >= 2 and directions != {-1, 1}:
        return False
    return True


def _choose_layout() -> tuple[int, list[int], list[dict]] | None:
    height = random.randint(15, 21)
    lane_count = random.randint(2, 4)
    widths = [random.choice((3, 5)) for _ in range(lane_count)]
    total_width = 1 + sum(widths) + lane_count
    if total_width > 30:
        return None

    patterns = [1, 2]
    while len(patterns) < lane_count:
        patterns.append(random.choice((1, 2)))
    random.shuffle(patterns)

    directions = [1, -1]
    while len(directions) < lane_count:
        directions.append(random.choice((1, -1)))
    random.shuffle(directions)

    lane_specs: list[dict] = []
    for direction, pattern in zip(directions, patterns):
        spec = _make_lane_spec(height, direction, pattern)
        if spec is None:
            return None
        lane_specs.append(spec)

    return height, widths, lane_specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _choose_layout()
        if layout is None:
            continue
        height, widths, lane_specs = layout
        grid = _paint_input(height, widths, lane_specs)
        if _validate_generated_grid(grid, lane_specs):
            return grid
    raise RuntimeError("failed to generate a valid lane-gate puzzle")
