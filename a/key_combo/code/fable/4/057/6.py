from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBSTACLE = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (10, 20)
GRID_WIDTH_RANGE = (12, 24)
PATH_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 120


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each top-edge source downward to its same-colored endpoint, sidestepping gray blockers toward the target side.

    Input structure:
    - The background is black.
    - Several distinct source pixels sit on the top edge; each source color appears exactly once more as an interior endpoint.
    - Gray obstacle cells may block the cell directly below a path and force a horizontal detour.

    Rule:
    - For each top-edge source, repeatedly try to move one cell downward toward its same-colored endpoint.
    - If the cell below is gray, move one cell horizontally on the current row toward the endpoint's column, then try downward again.
    - Paint every visited background cell with the source color and stop when the path reaches its matching endpoint pixel.

    Key insight:
    - The sidestep direction is not globally fixed: every blocked path chooses left or right from its own endpoint position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    source_colors = [int(color) for color in g[0] if int(color) not in (BACKGROUND, OBSTACLE)]
    assert source_colors, "expected at least one top-edge source"
    assert len(source_colors) == len(set(source_colors)), "top-edge source colors must be distinct"

    endpoints: dict[int, tuple[int, int]] = {}
    sources: dict[int, tuple[int, int]] = {}
    for color in source_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each path color must appear exactly twice in the input"
        top = [(int(r), int(c)) for r, c in coords if int(r) == 0]
        other = [(int(r), int(c)) for r, c in coords if int(r) != 0]
        assert len(top) == 1 and len(other) == 1, "each color needs one top source and one interior endpoint"
        sources[color] = top[0]
        endpoints[color] = other[0]

    order = sorted(source_colors, key=lambda color: sources[color][1])
    out = g.copy()

    for color in order:
        r, c = sources[color]
        target_r, target_c = endpoints[color]
        seen = {(r, c)}

        while (r, c) != (target_r, target_c):
            assert r + 1 < height or (r, c) == (target_r, target_c), "path ran out of rows before reaching its endpoint"

            if r + 1 < height and (r + 1, c) == (target_r, target_c):
                r += 1
                out[r, c] = color
                break

            below = int(g[r + 1, c])
            if below == OBSTACLE:
                step = (target_c > c) - (target_c < c)
                assert step != 0, "blocked path must have its target on one side"
                nc = c + step
                assert 0 <= nc < width, "horizontal sidestep left the grid"
                assert (r, nc) == (target_r, target_c) or int(g[r, nc]) == BACKGROUND, "sideways destination must be clear"
                c = nc
            else:
                assert below == BACKGROUND, "downward motion may only pass through background or the endpoint"
                r += 1

            assert (r, c) not in seen, "path should not loop"
            seen.add((r, c))
            out[r, c] = color

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_path(
    source_col: int,
    target_col: int,
    target_row: int,
    turn_rows: list[int],
    run_lengths: list[int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    current_r = 0
    current_c = int(source_col)
    direction = (target_col > source_col) - (target_col < source_col)

    row_to_run = {int(row): int(run) for row, run in zip(turn_rows, run_lengths)}
    path = [(0, int(source_col))]
    blockers: set[tuple[int, int]] = set()

    for row in range(target_row):
        run = row_to_run.get(int(row), 0)
        for _ in range(run):
            blockers.add((current_r + 1, current_c))
            current_c += direction
            path.append((current_r, current_c))
        current_r += 1
        path.append((current_r, current_c))

    assert current_r == target_row and current_c == target_col
    return path, blockers


def _sample_corridors(width: int, widths: list[int]) -> list[tuple[int, int]] | None:
    required = sum(widths) + max(0, len(widths) - 1)
    if required > width:
        return None

    extra = width - required
    gap_sizes = [0] * (len(widths) + 1)
    for _ in range(extra):
        gap_sizes[random.randrange(len(gap_sizes))] += 1

    corridors = []
    start = gap_sizes[0]
    for index, corridor_width in enumerate(widths):
        left = start
        right = left + corridor_width - 1
        corridors.append((left, right))
        if index < len(widths) - 1:
            start = right + 2 + gap_sizes[index + 1]
    return corridors


def _sample_specs(height: int, width: int) -> list[dict[str, object]] | None:
    path_count = random.randint(*PATH_COUNT_RANGE)
    blocked_flags = [True] * path_count
    if path_count > 1 and random.random() < 0.75:
        blocked_flags[random.randrange(path_count)] = False
    random.shuffle(blocked_flags)

    colors = random.sample(PATH_COLORS, path_count)
    corridor_widths = [
        random.randint(2, 5) if is_blocked else random.randint(1, 3)
        for is_blocked in blocked_flags
    ]
    corridors = _sample_corridors(width, corridor_widths)
    if corridors is None:
        return None

    specs: list[dict[str, object]] = []
    for color, is_blocked, (left, right) in zip(colors, blocked_flags, corridors):
        source_col = random.randint(left, right)
        if is_blocked:
            options = [col for col in range(left, right + 1) if col != source_col]
            if not options:
                return None
            target_col = random.choice(options)
            delta = abs(target_col - source_col)
            max_turn_rows = min(delta, height - 2, 3)
            if max_turn_rows < 1:
                return None
            turn_count = random.randint(1, max_turn_rows)
            min_target_row = max(4, turn_count + 1)
            if min_target_row > height - 2:
                return None
            target_row = random.randint(min_target_row, height - 2)
            turn_count = random.randint(1, min(delta, target_row, 3))
            turn_rows = sorted(random.sample(range(target_row), turn_count))
            run_lengths = _positive_composition(delta, turn_count)
        else:
            target_col = source_col
            target_row = random.randint(4, height - 2)
            turn_rows = []
            run_lengths = []

        path, blockers = _build_path(source_col, target_col, target_row, turn_rows, run_lengths)
        interval = (min(source_col, target_col), max(source_col, target_col))
        specs.append(
            {
                "color": int(color),
                "source": (0, int(source_col)),
                "target": (int(target_row), int(target_col)),
                "path": path,
                "blockers": blockers,
                "interval": interval,
            }
        )
    return specs


def _valid_layout(height: int, width: int, specs: list[dict[str, object]]) -> bool:
    occupancy = np.zeros((height, width), dtype=bool)
    for spec in specs:
        for cell in spec["path"]:
            if occupancy[cell]:
                return False
            occupancy[cell] = True
        for cell in spec["blockers"]:
            if occupancy[cell]:
                return False
            occupancy[cell] = True
    return True


def _render_input(height: int, width: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        sr, sc = spec["source"]
        tr, tc = spec["target"]
        grid[sr, sc] = color
        grid[tr, tc] = color
        for r, c in spec["blockers"]:
            grid[r, c] = OBSTACLE
    return grid


def _render_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["path"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        specs = _sample_specs(height, width)
        if specs is None or not _valid_layout(height, width, specs):
            continue

        if not any(spec["blockers"] for spec in specs):
            continue

        inp = _render_input(height, width, specs)
        expected = _render_output(inp, specs)

        try:
            actual = main(inp)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
