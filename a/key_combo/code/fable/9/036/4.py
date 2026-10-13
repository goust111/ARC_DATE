import random

import numpy as np


BG = 0
PIVOT = 5
SEP = 7
ARM_PALETTE = (1, 2, 3, 4, 6, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_RETRIES = 400


def _rotate_dir(direction: int, quarter_turns: int) -> int:
    return (int(direction) + int(quarter_turns)) % 4


def _render_panel(size: int, pivot: tuple[int, int], arms: dict[int, tuple[int, int]]) -> np.ndarray:
    panel = np.full((size, size), BG, dtype=int)
    pr, pc = pivot
    panel[pr, pc] = PIVOT
    for color, (length, direction) in arms.items():
        dr, dc = DIRS[direction]
        for step in range(1, length + 1):
            rr = pr + dr * step
            cc = pc + dc * step
            assert 0 <= rr < size and 0 <= cc < size
            assert panel[rr, cc] == BG
            panel[rr, cc] = int(color)
    return panel


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    size = int(g.shape[0])
    H, W = map(int, g.shape)
    assert H == size
    assert (W + 1) % (size + 1) == 0, "input must be a row of equal square panels"
    panel_count = (W + 1) // (size + 1)
    assert panel_count >= 3

    sep_cols = [k * (size + 1) - 1 for k in range(1, panel_count)]
    sep_color = int(g[0, sep_cols[0]])
    assert sep_color not in (BG, PIVOT)
    for col in sep_cols:
        assert np.all(g[:, col] == sep_color), "separator bars must be solid"

    panels = []
    for idx in range(panel_count):
        start = idx * (size + 1)
        panels.append(g[:, start : start + size].copy())
    return panels, sep_color


def _parse_arm(panel: np.ndarray, color: int, pivot: tuple[int, int]) -> tuple[int, int]:
    pr, pc = pivot
    coords = np.argwhere(panel == int(color))
    assert coords.shape[0] >= 1
    rows = {int(r) for r, _ in coords}
    cols = {int(c) for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "each arm must be a straight segment"

    points = {tuple(map(int, rc)) for rc in coords}
    adjacent = []
    for direction, (dr, dc) in enumerate(DIRS):
        first = (pr + dr, pc + dc)
        if first in points:
            adjacent.append(direction)
    assert len(adjacent) == 1, "arm must touch the pivot at exactly one end"

    direction = adjacent[0]
    dr, dc = DIRS[direction]
    length = len(points)
    expected = {(pr + dr * step, pc + dc * step) for step in range(1, length + 1)}
    assert points == expected, "arm cells must be contiguous from the pivot"
    return length, direction


def _parse_panel(panel: np.ndarray) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    pivots = np.argwhere(panel == PIVOT)
    assert pivots.shape == (1, 2), "each panel must contain exactly one grey pivot"
    pivot = tuple(map(int, pivots[0]))

    colors = [int(c) for c in np.unique(panel) if int(c) not in (BG, PIVOT)]
    assert 1 <= len(colors) <= 2, "each panel must contain one or two arms"
    arms = {color: _parse_arm(panel, color, pivot) for color in colors}

    lengths = [length for length, _direction in arms.values()]
    assert len(lengths) == len(set(lengths)), "arm lengths must be distinct"
    directions = [direction for _length, direction in arms.values()]
    assert len(directions) == len(set(directions)), "arms may not overlap"
    return pivot, arms


def _validate_sequence(panels: list[np.ndarray]) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    parsed = [_parse_panel(panel) for panel in panels]
    pivot = parsed[0][0]
    colors = set(parsed[0][1])
    lengths = {color: parsed[0][1][color][0] for color in colors}

    for next_pivot, next_arms in parsed[1:]:
        assert next_pivot == pivot, "pivot must stay at the same coordinates"
        assert set(next_arms) == colors, "all arms must persist through the sequence"
        for color in colors:
            assert next_arms[color][0] == lengths[color], "arm lengths must stay constant"

    for arms_now, arms_next in zip((entry[1] for entry in parsed), (entry[1] for entry in parsed[1:])):
        for color in colors:
            length, direction = arms_now[color]
            next_length, next_direction = arms_next[color]
            assert next_length == length
            assert next_direction == _rotate_dir(direction, length), "rotation step must equal arm length"

    return pivot, parsed[-1][1]


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    size = panels[0].shape[0]
    full = np.full((size, len(panels) * size + len(panels) - 1), SEP, dtype=int)
    for idx, panel in enumerate(panels):
        start = idx * (size + 1)
        full[:, start : start + size] = panel
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next panel in a separator-split sequence of rotating arms.

    Input structure:
    - The grid is one row of equal square panels separated by solid vertical bars.
    - Every panel contains the same grey pivot pixel and one or two colored straight
      arms extending horizontally or vertically from that pivot.
    - Arm colors persist across panels, and each arm keeps its own length.

    Rule:
    - Read every arm's color, length, and direction in each panel.
    - Verify that from one panel to the next, each arm rotates clockwise by
      `length × 90°` around the shared pivot.
    - From the last panel, rotate every arm one more step by its own length and
      output that single next panel.

    Key insight:
    - The angular step is encoded by the arm's visible length, so different arms
      can seem to spin in opposite directions even though both are rotating clockwise.
    """
    panels, _sep_color = _split_panels(np.asarray(grid, dtype=int))
    pivot, last_arms = _validate_sequence(panels)

    next_arms = {
        color: (length, _rotate_dir(direction, length))
        for color, (length, direction) in last_arms.items()
    }
    return _render_panel(panels[0].shape[0], pivot, next_arms)


def _choose_case(rng: random.Random) -> list[int]:
    roll = rng.random()
    if roll < 0.58:
        return [1, 3]
    if roll < 0.80:
        return [2]
    if roll < 0.90:
        return [1]
    return [3]


def _choose_panel_count(rng: random.Random, lengths: list[int]) -> int:
    max_len = max(lengths)
    if max_len == 3:
        return 3
    if max_len == 2:
        return rng.choice([4, 5])
    return rng.choice([3, 4, 5])


def _choose_panel_size(rng: random.Random, panel_count: int, max_len: int) -> int:
    min_size = max(5, 2 * max_len + 1)
    max_size = (30 - (panel_count - 1)) // panel_count
    assert min_size <= max_size
    return rng.randint(min_size, max_size)


def _initial_directions(rng: random.Random, lengths: list[int]) -> dict[int, int]:
    if lengths == [1, 3]:
        base = rng.randrange(4)
        other = (base + rng.choice([1, 3])) % 4
        return {1: base, 3: other}
    return {lengths[0]: rng.randrange(4)}


def _build_states(lengths: list[int], colors: list[int], panel_count: int, base_dirs: dict[int, int]) -> list[dict[int, tuple[int, int]]]:
    states = []
    for t in range(panel_count + 1):
        arms = {}
        used_dirs = set()
        for color, length in zip(colors, lengths):
            direction = _rotate_dir(base_dirs[length], t * length)
            assert direction not in used_dirs
            used_dirs.add(direction)
            arms[color] = (length, direction)
        states.append(arms)
    return states


def _no_global_step(states: list[dict[int, tuple[int, int]]]) -> bool:
    first = states[0]
    if len(first) < 2:
        return True
    step_set = {
        (states[1][color][1] - first[color][1]) % 4
        for color in first
    }
    return len(step_set) > 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lengths = _choose_case(rng)
    panel_count = _choose_panel_count(rng, lengths)
    size = _choose_panel_size(rng, panel_count, max(lengths))

    margin = max(lengths)
    pivot = (
        rng.randint(margin, size - margin - 1),
        rng.randint(margin, size - margin - 1),
    )
    colors = rng.sample(ARM_PALETTE, len(lengths))
    rng.shuffle(lengths)
    base_dirs = _initial_directions(rng, sorted(lengths))
    states = _build_states(lengths, colors, panel_count, base_dirs)

    if len(lengths) == 2:
        deltas = [(states[t][colors[0]][1] - states[t][colors[1]][1]) % 4 for t in range(panel_count + 1)]
        if any(delta == 0 for delta in deltas):
            return None

    panels = [_render_panel(size, pivot, arms) for arms in states[:-1]]
    output = _render_panel(size, pivot, states[-1])
    grid = _assemble_input(panels)

    try:
        parsed_panels, _ = _split_panels(grid)
        assert len(parsed_panels) == panel_count
        assert np.array_equal(main(grid), output)
        sigs = {hash(panel.tobytes()) for panel in panels}
        assert len(sigs) >= 2
        if len(lengths) == 2:
            step_set = {
                (states[1][color][1] - states[0][color][1]) % 4
                for color in colors
            }
            assert step_set == {1, 3}
        else:
            assert lengths[0] in (1, 2, 3)
        assert _no_global_step(states)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
