from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


GREY = 5
GREEN = 3
RED = 2
ARC_COLORS = tuple(range(10))

GRID_CONTAINER_COUNT = (3, 5)
GRID_CAPACITY_RANGE = (2, 6)
GRID_GAP_RANGE = (1, 5)
GRID_MARGIN_RANGE = (1, 4)
GRID_EXTRA_WIDTH_RANGE = (0, 4)
GEN_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _scan_containers(grid: np.ndarray, wall_color: int) -> list[dict[str, int]]:
    h, w = grid.shape
    found: dict[tuple[int, int], dict[str, int]] = {}
    for left in range(w - 4):
        for bottom in range(1, h):
            if not np.all(grid[bottom, left : left + 5] == int(wall_color)):
                continue
            top = int(bottom)
            while top > 0 and int(grid[top - 1, left]) == int(wall_color) and int(grid[top - 1, left + 4]) == int(wall_color):
                top -= 1
            if top == bottom:
                continue
            if np.any(grid[top:bottom, left + 1 : left + 4] == int(wall_color)):
                continue
            key = (int(left), int(bottom))
            found[key] = {
                "left": int(left),
                "right": int(left + 4),
                "top": int(top),
                "bottom": int(bottom),
                "capacity": int(bottom - top),
            }
    return sorted(found.values(), key=lambda item: (item["left"], item["top"]))


def _choose_wall_color(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    order = [GREY] if GREY in colors else []
    order.extend(color for color in colors if color != GREY)
    best_color = None
    best_containers: list[dict[str, int]] = []
    best_score = None
    for color in order:
        containers = _scan_containers(grid, int(color))
        score = (len(containers), sum(container["capacity"] for container in containers), int(color == GREY))
        if best_score is None or score > best_score:
            best_score = score
            best_color = int(color)
            best_containers = containers
    assert best_color is not None and best_containers, "expected at least one cup-like container"
    return int(best_color), best_containers


def _read_liquid(grid: np.ndarray, bg: int, wall_color: int, container: dict[str, int]) -> dict[str, object]:
    top = int(container["top"])
    bottom = int(container["bottom"])
    left = int(container["left"])
    right = int(container["right"])
    patch = grid[top:bottom, left + 1 : right]
    mask = (patch != int(bg)) & (patch != int(wall_color))
    pixels = [int(v) for v in patch[mask].tolist()]
    counts = Counter(pixels)
    primary = None if not counts else int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])
    return {
        "pixel_count": int(mask.sum()),
        "counts": counts,
        "primary": primary,
    }


def _pipe_state(grid: np.ndarray, wall_color: int, left_container: dict[str, int], right_container: dict[str, int]) -> str | None:
    if int(left_container["bottom"]) != int(right_container["bottom"]):
        return None
    row = int(left_container["bottom"])
    c0 = int(left_container["right"])
    c1 = int(right_container["left"])
    if c1 - c0 < 2:
        return None
    segment = grid[row, c0 : c1 + 1]
    if int(segment[0]) != int(wall_color) or int(segment[-1]) != int(wall_color):
        return None
    inner = segment[1:-1]
    non_wall = [int(v) for v in inner.tolist() if int(v) != int(wall_color)]
    if len(non_wall) != 1:
        return None
    valve = int(non_wall[0])
    if valve not in (GREEN, RED):
        return None
    if not np.all((segment == int(wall_color)) | (segment == int(valve))):
        return None
    return "open" if valve == GREEN else "closed"


def _open_groups(pipe_states: list[str | None]) -> list[list[int]]:
    groups: list[list[int]] = []
    current = [0]
    for i, state in enumerate(pipe_states):
        if state == "open":
            current.append(i + 1)
        else:
            groups.append(current)
            current = [i + 1]
    groups.append(current)
    return groups


def _group_output_color(group_data: list[dict[str, object]]) -> int | None:
    counts: Counter[int] = Counter()
    leftmost_color = None
    for data in group_data:
        pixel_count = int(data["pixel_count"])
        primary = data["primary"]
        if pixel_count > 0 and leftmost_color is None:
            leftmost_color = None if primary is None else int(primary)
        for color, amount in data["counts"].items():
            counts[int(color)] += int(amount)
    if not counts:
        return None
    best = max(counts.values())
    winners = {int(color) for color, amount in counts.items() if int(amount) == int(best)}
    if len(winners) == 1:
        return next(iter(winners))
    if leftmost_color in winners:
        return int(leftmost_color)
    for data in group_data:
        primary = data["primary"]
        if primary in winners:
            return int(primary)
    raise AssertionError("tie should resolve to a leftmost liquid color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Equalize liquid levels inside every open-valve connected group of cups.

    Input structure:
    - The most common color is the background.
    - Grey cup-like containers have outer width 5 and inner width 3.
    - Some adjacent cups are linked on their bottom row by a horizontal pipe that
      contains one valve pixel: green means open, red means closed.
    - Each cup holds zero or more full rows of colored liquid.

    Rule:
    - Only cups connected through open valves belong to the same group.
    - Inside each group, conserve total liquid pixels and redistribute them
      evenly across cups from the bottom upward.
    - If the total is not divisible by the number of cups, the leftmost cups get
      one extra liquid row first.
    - If multiple liquid colors occur in one open group, the equalized liquid
      uses the most frequent color; ties go to the leftmost liquid-bearing cup.
    - Walls, pipes, and valve pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    wall_color, containers = _choose_wall_color(g, bg)
    assert containers, "expected at least one container"

    liquids = [_read_liquid(g, bg, wall_color, container) for container in containers]
    pipe_states = [
        _pipe_state(g, wall_color, containers[i], containers[i + 1])
        for i in range(len(containers) - 1)
    ]
    groups = _open_groups(pipe_states)

    out = g.copy()
    for container in containers:
        top = int(container["top"])
        bottom = int(container["bottom"])
        left = int(container["left"])
        right = int(container["right"])
        out[top:bottom, left + 1 : right] = int(bg)

    for group in groups:
        group_liquids = [liquids[index] for index in group]
        total_pixels = sum(int(data["pixel_count"]) for data in group_liquids)
        assert total_pixels % 3 == 0, "liquid rows should have fixed width 3"
        total_rows = total_pixels // 3
        base_rows, extra = divmod(total_rows, len(group))
        fill_color = _group_output_color(group_liquids)
        for local_idx, container_index in enumerate(group):
            rows = int(base_rows + (local_idx < extra))
            if rows == 0 or fill_color is None:
                continue
            container = containers[container_index]
            bottom = int(container["bottom"])
            left = int(container["left"])
            right = int(container["right"])
            out[bottom - rows : bottom, left + 1 : right] = int(fill_color)

    return out


def _sample_pipe_states(count: int) -> list[str]:
    edge_count = count - 1
    assert edge_count >= 2, "generator uses at least three containers"
    case = random.choice(("all_open", "open_gaps", "open_closed", "mixed"))
    if case == "all_open":
        return ["open"] * edge_count
    if case == "open_gaps":
        while True:
            states = [random.choice(("open", "none")) for _ in range(edge_count)]
            if "open" in states and "none" in states:
                return states
    if case == "open_closed":
        while True:
            states = [random.choice(("open", "closed")) for _ in range(edge_count)]
            if "open" in states and "closed" in states:
                return states
    while True:
        states = [random.choice(("open", "closed", "none")) for _ in range(edge_count)]
        if "open" in states and any(state != "open" for state in states):
            return states


def _build_layout() -> tuple[int, int, int, list[int], list[int], list[str], list[int]]:
    container_count = random.randint(*GRID_CONTAINER_COUNT)
    capacities = [random.randint(*GRID_CAPACITY_RANGE) for _ in range(container_count)]
    gaps = [random.randint(*GRID_GAP_RANGE) for _ in range(container_count - 1)]
    states = _sample_pipe_states(container_count)

    top_margin = random.randint(*GRID_MARGIN_RANGE)
    bottom_margin = random.randint(1, 2)
    baseline = top_margin + max(capacities)
    height = baseline + 1 + bottom_margin

    structure_width = 5 * container_count + sum(gaps)
    width = structure_width + random.randint(*GRID_MARGIN_RANGE) + random.randint(*GRID_MARGIN_RANGE) + random.randint(*GRID_EXTRA_WIDTH_RANGE)
    left = random.randint(1, width - structure_width - 1)

    x_positions = [left]
    for gap in gaps:
        x_positions.append(x_positions[-1] + 5 + int(gap))
    return height, width, baseline, x_positions, capacities, states, gaps


def _feasible_targets(capacities: list[int]) -> list[list[int]]:
    feasible: list[list[int]] = []
    group_size = len(capacities)
    for base_rows in range(min(capacities) + 1):
        for extra in range(group_size):
            rows = [int(base_rows + (i < extra)) for i in range(group_size)]
            if all(rows[i] <= capacities[i] for i in range(group_size)):
                feasible.append(rows)
    return feasible


def _choose_target_rows(capacities: list[int], prefer_remainder: bool) -> list[int]:
    feasible = _feasible_targets(capacities)
    nonzero = [rows for rows in feasible if sum(rows) > 0]
    pool = nonzero or feasible
    if prefer_remainder:
        with_remainder = [rows for rows in pool if sum(rows) % len(rows) != 0]
        if with_remainder:
            pool = with_remainder
    return list(random.choice(pool))


def _search_composition(capacities: list[int], total: int, avoid: list[int] | None) -> list[int] | None:
    suffix_caps = [0] * (len(capacities) + 1)
    for i in range(len(capacities) - 1, -1, -1):
        suffix_caps[i] = suffix_caps[i + 1] + int(capacities[i])

    result = None

    def dfs(index: int, remaining: int, current: list[int]) -> bool:
        nonlocal result
        if index == len(capacities):
            if remaining == 0 and (avoid is None or current != avoid):
                result = list(current)
                return True
            return False
        low = max(0, remaining - suffix_caps[index + 1])
        high = min(int(capacities[index]), remaining)
        values = list(range(low, high + 1))
        random.shuffle(values)
        for value in values:
            current.append(int(value))
            if dfs(index + 1, remaining - int(value), current):
                return True
            current.pop()
        return False

    dfs(0, int(total), [])
    return result


def _sample_input_rows(capacities: list[int], target_rows: list[int]) -> list[int]:
    total = int(sum(target_rows))
    alternative = _search_composition(capacities, total, target_rows)
    if alternative is not None:
        return alternative
    exact = _search_composition(capacities, total, None)
    assert exact is not None
    return exact


def _winning_color_from_rows(rows: list[int], colors: list[int | None]) -> int | None:
    counts: Counter[int] = Counter()
    leftmost_color = None
    for row_count, color in zip(rows, colors, strict=True):
        if row_count <= 0 or color is None:
            continue
        if leftmost_color is None:
            leftmost_color = int(color)
        counts[int(color)] += int(row_count)
    if not counts:
        return None
    best = max(counts.values())
    winners = {int(color) for color, amount in counts.items() if int(amount) == int(best)}
    if len(winners) == 1:
        return next(iter(winners))
    if leftmost_color in winners:
        return int(leftmost_color)
    raise AssertionError("generator tie-break should resolve by leftmost liquid color")


def _pick_color_assignment(rows: list[int], color_pool: list[int], forced_mode: str | None) -> tuple[list[int | None], int | None]:
    colors: list[int | None] = [None] * len(rows)
    active = [i for i, row_count in enumerate(rows) if row_count > 0]
    if not active:
        return colors, None

    modes = ["uniform", "majority", "tie"]
    if forced_mode is not None and forced_mode in modes:
        modes.remove(forced_mode)
        modes.insert(0, forced_mode)
    else:
        random.shuffle(modes)

    usable_pool = color_pool[: min(3, len(color_pool))]
    for mode in modes:
        if mode == "uniform":
            color = random.choice(usable_pool)
            for idx in active:
                colors[idx] = int(color)
            return colors, int(color)
        if len(active) < 2 or len(usable_pool) < 2:
            continue
        assignments = []
        for assignment in itertools.product(usable_pool, repeat=len(active)):
            counts: Counter[int] = Counter()
            for idx, color in zip(active, assignment, strict=True):
                counts[int(color)] += int(rows[idx])
            best = max(counts.values())
            winners = {int(color) for color, amount in counts.items() if int(amount) == int(best)}
            if mode == "majority":
                if len(counts) < 2 or len(winners) != 1:
                    continue
            elif mode == "tie":
                if len(winners) < 2 or int(assignment[0]) not in winners:
                    continue
            assignments.append(tuple(int(color) for color in assignment))
        if assignments:
            chosen = list(random.choice(assignments))
            for idx, color in zip(active, chosen, strict=True):
                colors[idx] = int(color)
            return colors, _winning_color_from_rows(rows, colors)

    color = random.choice(usable_pool)
    for idx in active:
        colors[idx] = int(color)
    return colors, int(color)


def _draw_case(
    grid: np.ndarray,
    bg: int,
    x_positions: list[int],
    capacities: list[int],
    baseline: int,
    pipe_states: list[str],
    input_rows: list[int],
    input_colors: list[int | None],
) -> None:
    del bg
    for left, capacity, row_count, color in zip(x_positions, capacities, input_rows, input_colors, strict=True):
        top = int(baseline - capacity)
        bottom = int(baseline)
        grid[top : bottom + 1, left] = GREY
        grid[top : bottom + 1, left + 4] = GREY
        grid[bottom, left : left + 5] = GREY
        if row_count > 0:
            grid[bottom - row_count : bottom, left + 1 : left + 4] = int(color)

    for i, state in enumerate(pipe_states):
        if state == "none":
            continue
        row = int(baseline)
        start = int(x_positions[i] + 4)
        end = int(x_positions[i + 1])
        grid[row, start : end + 1] = GREY
        valve_col = random.randint(start + 1, end - 1)
        grid[row, valve_col] = GREEN if state == "open" else RED


def _expected_output(
    grid: np.ndarray,
    bg: int,
    x_positions: list[int],
    capacities: list[int],
    baseline: int,
    pipe_states: list[str],
    target_rows: list[int],
    output_colors: list[int | None],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for left, capacity in zip(x_positions, capacities, strict=True):
        top = int(baseline - capacity)
        bottom = int(baseline)
        out[top:bottom, left + 1 : left + 4] = int(bg)

    for left, capacity, row_count, color in zip(x_positions, capacities, target_rows, output_colors, strict=True):
        if row_count <= 0 or color is None:
            continue
        bottom = int(baseline)
        out[bottom - row_count : bottom, left + 1 : left + 4] = int(color)
    return out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts)) == int(bg)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height, width, baseline, x_positions, capacities, pipe_states, gaps = _build_layout()
    bg_choices = [color for color in ARC_COLORS if color not in {GREY, GREEN, RED}]
    bg = int(random.choice(bg_choices))
    liquid_pool = [color for color in ARC_COLORS if color not in {bg, GREY, GREEN, RED}]
    if len(liquid_pool) < 2:
        return None

    groups = _open_groups(pipe_states)
    target_rows = [0] * len(capacities)
    input_rows = [0] * len(capacities)
    input_colors: list[int | None] = [None] * len(capacities)
    output_colors: list[int | None] = [None] * len(capacities)
    any_change = False

    prefer_mixed_group = random.choice(groups)
    prefer_remainder_group = random.choice(groups)
    for group in groups:
        group_caps = [capacities[idx] for idx in group]
        target = _choose_target_rows(group_caps, prefer_remainder=(group == prefer_remainder_group))
        current = _sample_input_rows(group_caps, target)
        forced_mode = None
        if group == prefer_mixed_group and sum(current) > 0 and len(group) > 1:
            forced_mode = random.choice(("majority", "tie"))
        colors, winner = _pick_color_assignment(current, liquid_pool, forced_mode)
        for local_idx, container_idx in enumerate(group):
            target_rows[container_idx] = int(target[local_idx])
            input_rows[container_idx] = int(current[local_idx])
            input_colors[container_idx] = colors[local_idx]
            output_colors[container_idx] = winner
        if current != target:
            any_change = True
        if len({color for color in colors if color is not None}) > 1:
            any_change = True

    if not any_change:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    _draw_case(grid, bg, x_positions, capacities, baseline, pipe_states, input_rows, input_colors)
    expected = _expected_output(grid, bg, x_positions, capacities, baseline, pipe_states, target_rows, output_colors)

    if not _bg_is_dominant(grid, bg):
        return None
    if _mode_color(grid) != bg:
        return None
    if any(gap < 1 and state != "none" for gap, state in zip(gaps, pipe_states, strict=True)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid container-equalization input")
