from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
UP = -1
DOWN = 1
PALETTE = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400


def _find_separator_row(grid: np.ndarray) -> int:
    sep_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert sep_rows.size == 1, "expected a unique full grey separator row"
    sep = int(sep_rows[0])
    assert sep > 0, "separator row must have a legend row above it"
    return sep


def _legend_lengths(legend_row: np.ndarray) -> dict[int, int]:
    lengths: dict[int, int] = {}
    width = int(legend_row.shape[0])
    c = 0
    while c < width:
        color = int(legend_row[c])
        if color == BG:
            c += 1
            continue
        assert color not in (GREY, MAROON), "legend row must only contain legend colors"
        start = c
        while c < width and int(legend_row[c]) == color:
            c += 1
        assert color not in lengths, "each legend color must occupy one contiguous segment"
        lengths[color] = c - start
    assert lengths, "expected at least one legend segment"
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow vertical rays from seed pixels using segment lengths encoded by the top legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = map(int, g.shape)

    sep = _find_separator_row(g)
    legend = _legend_lengths(g[sep - 1])

    seeds: list[tuple[int, int, int, int, int]] = []
    for rel_r, c in np.argwhere(g[sep + 1 :] != BG):
        r = int(rel_r + sep + 1)
        c = int(c)
        color = int(g[r, c])
        if color == MAROON:
            continue
        assert color != GREY, "grey is reserved for the separator row"
        assert color in legend, "every seed color must appear in the legend"

        has_above = r > 0 and int(g[r - 1, c]) == MAROON
        has_below = r + 1 < height and int(g[r + 1, c]) == MAROON
        assert has_above ^ has_below, "each seed needs exactly one vertical maroon marker"

        direction = DOWN if has_above else UP
        seeds.append((r, c, color, direction, int(legend[color])))

    assert seeds, "expected at least one seed below the separator"

    out = g.copy()
    out[out == MAROON] = BG
    for r, c, color, direction, length in seeds:
        for step in range(length):
            rr = r + direction * step
            if 0 <= rr < height:
                out[rr, c] = color

    return out


def _sample_legend() -> tuple[list[int], list[int]]:
    for _ in range(100):
        n_colors = random.randint(3, 6)
        colors = random.sample(PALETTE, n_colors)
        lengths = [random.randint(1, 5) for _ in range(n_colors)]
        total_width = sum(lengths)
        if 10 <= total_width <= 24 and max(lengths) >= 2:
            return colors, lengths
    raise RuntimeError("failed to sample a valid legend")


def _candidate_rows(height: int, length: int, direction: int, truncated: bool | None) -> list[int]:
    if direction == UP:
        rows = list(range(2, height - 1))
        if truncated is True:
            return [r for r in rows if r - (length - 1) < 0]
        if truncated is False:
            return [r for r in rows if r - (length - 1) >= 0]
        return rows

    rows = list(range(3, height))
    if truncated is True:
        return [r for r in rows if r + (length - 1) >= height]
    if truncated is False:
        return [r for r in rows if r + (length - 1) < height]
    return rows


def _render_case(
    height: int, legend_colors: list[int], legend_lengths: list[int], seeds: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    width = int(sum(legend_lengths))
    grid = np.full((height, width), BG, dtype=int)

    color_to_length: dict[int, int] = {}
    c0 = 0
    for color, length in zip(legend_colors, legend_lengths):
        grid[0, c0 : c0 + length] = int(color)
        color_to_length[int(color)] = int(length)
        c0 += length

    grid[1, :] = GREY

    for r, c, color, direction in seeds:
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)
        marker_r = r + 1 if direction == UP else r - 1
        assert int(grid[marker_r, c]) == BG
        grid[marker_r, c] = MAROON

    out = grid.copy()
    out[out == MAROON] = BG
    for r, c, color, direction in seeds:
        length = int(color_to_length[int(color)])
        for step in range(length):
            rr = r + direction * step
            if 0 <= rr < height:
                out[rr, c] = int(color)

    return grid, out


def _choose_special_mode(color_to_length: dict[int, int]) -> str:
    options = ["none", "none", "up_cross"]
    if any(length >= 4 for length in color_to_length.values()):
        options.append("up_trunc")
    if any(length >= 3 for length in color_to_length.values()):
        options.append("down_trunc")
    return random.choice(options)


def _pick_color(
    legend_colors: list[int], color_to_length: dict[int, int], used_colors: list[int], need_long: bool
) -> int:
    choices = legend_colors[:]
    if need_long:
        long_choices = [color for color in choices if color_to_length[color] >= 2]
        if long_choices:
            choices = long_choices
    if used_colors and len(set(used_colors)) == 1 and len(legend_colors) > 1:
        diff_choices = [color for color in choices if color != used_colors[0]]
        if diff_choices:
            choices = diff_choices
    return int(random.choice(choices))


def _choose_row(height: int, length: int, direction: int) -> int:
    want_truncation = random.random() < 0.4
    rows = _candidate_rows(height, length, direction, want_truncation)
    if not rows:
        rows = _candidate_rows(height, length, direction, not want_truncation)
    if not rows:
        rows = _candidate_rows(height, length, direction, None)
    assert rows, "every seed must have at least one valid row"
    return int(random.choice(rows))


def _build_attempt() -> tuple[np.ndarray, np.ndarray]:
    legend_colors, legend_lengths = _sample_legend()
    color_to_length = {int(color): int(length) for color, length in zip(legend_colors, legend_lengths)}
    width = int(sum(legend_lengths))
    height = random.randint(8, 18)
    n_seeds = random.randint(2, min(6, width))
    columns = random.sample(range(width), n_seeds)

    seeds: list[tuple[int, int, int, int]] = []
    used_colors: list[int] = []

    mode = _choose_special_mode(color_to_length)
    if mode == "up_cross":
        choices = [color for color, length in color_to_length.items() if length >= 2]
        color = int(random.choice(choices))
        seeds.append((2, int(columns[len(seeds)]), color, UP))
        used_colors.append(color)
    elif mode == "up_trunc":
        choices = [color for color, length in color_to_length.items() if length >= 4]
        color = int(random.choice(choices))
        seeds.append((2, int(columns[len(seeds)]), color, UP))
        used_colors.append(color)
    elif mode == "down_trunc":
        choices = [color for color, length in color_to_length.items() if length >= 3]
        color = int(random.choice(choices))
        seeds.append((height - 2, int(columns[len(seeds)]), color, DOWN))
        used_colors.append(color)

    remaining = n_seeds - len(seeds)
    directions: list[int] = []
    present = {direction for _r, _c, _color, direction in seeds}
    if n_seeds >= 2 and not present:
        directions.extend([UP, DOWN])
    elif n_seeds >= 2 and present == {UP}:
        directions.append(DOWN)
    elif n_seeds >= 2 and present == {DOWN}:
        directions.append(UP)
    while len(directions) < remaining:
        directions.append(random.choice([UP, DOWN]))
    directions = directions[:remaining]
    random.shuffle(directions)

    while len(seeds) < n_seeds:
        direction = int(directions[len(seeds) - (1 if mode != "none" else 0)] if mode != "none" else directions[len(seeds)])
        need_long = not any(color_to_length[color] >= 2 for color in used_colors)
        color = _pick_color(legend_colors, color_to_length, used_colors, need_long=need_long)
        row = _choose_row(height, color_to_length[color], direction)
        seeds.append((row, int(columns[len(seeds)]), color, direction))
        used_colors.append(color)

    grid, out = _render_case(height, legend_colors, legend_lengths, seeds)

    grown_cells = 0
    for r, _c, color, direction in seeds:
        length = color_to_length[color]
        for step in range(1, length):
            rr = r + direction * step
            if 0 <= rr < height:
                grown_cells += 1
    assert grown_cells > 0, "expected at least one ray to extend beyond its seed"

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _build_attempt()
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
