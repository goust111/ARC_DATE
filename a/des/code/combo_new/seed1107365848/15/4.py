from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _separator_color(grid: np.ndarray) -> int:
    counts: Counter[int] = Counter()
    height, width = grid.shape
    for row in range(height):
        value = int(grid[row, 0])
        if value != BLACK and np.all(grid[row] == value):
            counts[value] += 1
    for col in range(width):
        value = int(grid[0, col])
        if value != BLACK and np.all(grid[:, col] == value):
            counts[value] += 1
    assert counts, "expected at least one full separator row or column"
    return int(counts.most_common(1)[0][0])


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    previous = -1
    for stop in [*sorted(separators), size]:
        start = previous + 1
        end = stop - 1
        if start <= end:
            bounds.append((start, end))
        previous = stop
    assert bounds, "expected at least one non-separator band"
    return bounds


def _step_region(region: tuple[int, int], direction: str) -> tuple[int, int]:
    row, col = region
    if direction == "up":
        return row - 1, col
    if direction == "down":
        return row + 1, col
    if direction == "left":
        return row, col - 1
    assert direction == "right"
    return row, col + 1


def _extract_pointer(region: np.ndarray, separator: int) -> dict[str, int | str]:
    mask = (region != BLACK) & (region != separator)
    coords = np.argwhere(mask)
    assert len(coords) >= 2, "each region must contain one line segment"

    colors = {int(region[int(r), int(c)]) for r, c in coords}
    assert len(colors) == 1, "pointer must be monochrome"
    color = colors.pop()

    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        col_values = sorted(int(c) for _, c in coords)
        assert col_values == list(range(col_values[0], col_values[-1] + 1)), "horizontal pointer must be contiguous"
        length = len(col_values)
        touches_left = col_values[0] == 0
        touches_right = col_values[-1] == region.shape[1] - 1
        assert touches_left ^ touches_right, "horizontal pointer must touch exactly one side wall"
        direction = "right" if touches_left else "left"
        assert 0 <= row < region.shape[0]
        return {"color": color, "length": length, "direction": direction}

    if len(cols) == 1:
        col = cols[0]
        row_values = sorted(int(r) for r, _ in coords)
        assert row_values == list(range(row_values[0], row_values[-1] + 1)), "vertical pointer must be contiguous"
        length = len(row_values)
        touches_top = row_values[0] == 0
        touches_bottom = row_values[-1] == region.shape[0] - 1
        assert touches_top ^ touches_bottom, "vertical pointer must touch exactly one top/bottom wall"
        direction = "down" if touches_top else "up"
        assert 0 <= col < region.shape[1]
        return {"color": color, "length": length, "direction": direction}

    raise AssertionError("pointer must be a straight horizontal or vertical segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each pointed-at neighboring region with the winning pointer color.

    Input structure:
    - The background is black.
    - Full grey rows and columns split the grid into rectangular table-like regions.
    - Each region contains exactly one short non-black non-grey straight line segment.
    - One endpoint of that segment sits on a region boundary; the other endpoint is inside the region.

    Rule:
    - Detect each segment's direction from its boundary-touching endpoint toward its free endpoint.
    - Move to the immediately adjacent region in that direction.
    - Fill all black cells in that target region with the segment's color.
    - If several segments point to the same target region, keep only the color of the longest segment.
    - Preserve the original line segments and the grey separator lines.

    Key insight:
    - The colored segment is a pointer: it does not paint its own region, it paints the neighboring region lying in the segment's pointing direction, with segment length breaking conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator = _separator_color(g)
    height, width = g.shape
    separator_rows = [row for row in range(height) if np.all(g[row] == separator)]
    separator_cols = [col for col in range(width) if np.all(g[:, col] == separator)]
    row_bands = _bands(height, separator_rows)
    col_bands = _bands(width, separator_cols)

    winners: dict[tuple[int, int], tuple[int, int]] = {}
    for region_row, (r0, r1) in enumerate(row_bands):
        for region_col, (c0, c1) in enumerate(col_bands):
            pointer = _extract_pointer(g[r0 : r1 + 1, c0 : c1 + 1], separator)
            target = _step_region((region_row, region_col), str(pointer["direction"]))
            assert 0 <= target[0] < len(row_bands) and 0 <= target[1] < len(col_bands), "pointer must target an adjacent region"

            candidate = (int(pointer["length"]), int(pointer["color"]))
            current = winners.get(target)
            if current is None or candidate[0] > current[0]:
                winners[target] = candidate
            elif candidate[0] == current[0]:
                raise AssertionError("target region has an equal-length conflict")

    out = g.copy()
    for (region_row, region_col), (_length, color) in winners.items():
        r0, r1 = row_bands[region_row]
        c0, c1 = col_bands[region_col]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = color
    return out


def _sample_band_sizes(count: int, rng: random.Random) -> list[int]:
    max_size = min(7, (30 - (count - 1)) // count)
    assert max_size >= 5, "band sizes must fit ARC bounds"
    return [rng.randint(5, max_size) for _ in range(count)]


def _build_layout(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_bands: list[tuple[int, int]] = []
    cursor = 0
    for index, band_height in enumerate(row_heights):
        row_bands.append((cursor, cursor + band_height - 1))
        cursor += band_height
        if index != len(row_heights) - 1:
            grid[cursor, :] = GREY
            cursor += 1

    col_bands: list[tuple[int, int]] = []
    cursor = 0
    for index, band_width in enumerate(col_widths):
        col_bands.append((cursor, cursor + band_width - 1))
        cursor += band_width
        if index != len(col_widths) - 1:
            grid[:, cursor] = GREY
            cursor += 1

    return grid, row_bands, col_bands


def _feasible_directions(region: tuple[int, int], n_rows: int, n_cols: int) -> list[str]:
    row, col = region
    directions: list[str] = []
    if row > 0:
        directions.append("up")
    if row < n_rows - 1:
        directions.append("down")
    if col > 0:
        directions.append("left")
    if col < n_cols - 1:
        directions.append("right")
    return directions


def _incoming_options(target: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], str]]:
    row, col = target
    options: list[tuple[tuple[int, int], str]] = []
    if row > 0:
        options.append(((row - 1, col), "down"))
    if row < n_rows - 1:
        options.append(((row + 1, col), "up"))
    if col > 0:
        options.append(((row, col - 1), "right"))
    if col < n_cols - 1:
        options.append(((row, col + 1), "left"))
    return options


def _assign_directions(n_rows: int, n_cols: int, rng: random.Random) -> dict[tuple[int, int], str]:
    regions = [(row, col) for row in range(n_rows) for col in range(n_cols)]
    directions = {region: rng.choice(_feasible_directions(region, n_rows, n_cols)) for region in regions}

    if len(regions) >= 4 and rng.random() < 0.65:
        target_candidates = [region for region in regions if len(_incoming_options(region, n_rows, n_cols)) >= 2]
        rng.shuffle(target_candidates)
        used_sources: set[tuple[int, int]] = set()
        max_forced_targets = 1 + int(len(regions) >= 9 and rng.random() < 0.5)
        forced_targets = 0
        for target in target_candidates:
            options = [item for item in _incoming_options(target, n_rows, n_cols) if item[0] not in used_sources]
            if len(options) < 2:
                continue
            choice_count = 2 if len(options) == 2 or rng.random() < 0.8 else 3
            chosen = rng.sample(options, choice_count)
            for source, direction in chosen:
                directions[source] = direction
                used_sources.add(source)
            forced_targets += 1
            if forced_targets >= max_forced_targets:
                break

    return directions


def _incoming_map(directions: dict[tuple[int, int], str]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    incoming: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for source, direction in directions.items():
        incoming[_step_region(source, direction)].append(source)
    return incoming


def _assign_lengths(directions: dict[tuple[int, int], str], rng: random.Random) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], list[tuple[int, int]]]]:
    incoming = _incoming_map(directions)
    lengths: dict[tuple[int, int], int] = {}
    for sources in incoming.values():
        if len(sources) == 1:
            lengths[sources[0]] = rng.randint(2, 4)
            continue

        winner = rng.choice(sources)
        winner_length = rng.choice((3, 4))
        lengths[winner] = winner_length
        loser_choices = (2,) if winner_length == 3 else (2, 3)
        for source in sources:
            if source != winner:
                lengths[source] = rng.choice(loser_choices)
    return lengths, incoming


def _assign_colors(incoming: dict[tuple[int, int], list[tuple[int, int]]], rng: random.Random) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    for sources in incoming.values():
        if len(sources) > 1:
            palette = list(PALETTE)
            rng.shuffle(palette)
            for source, color in zip(sources, palette):
                colors[source] = color
    for sources in incoming.values():
        for source in sources:
            colors.setdefault(source, rng.choice(PALETTE))
    return colors


def _draw_pointer(
    grid: np.ndarray,
    row_bounds: tuple[int, int],
    col_bounds: tuple[int, int],
    direction: str,
    length: int,
    color: int,
    rng: random.Random,
) -> None:
    r0, r1 = row_bounds
    c0, c1 = col_bounds
    if direction in {"left", "right"}:
        row = rng.randint(r0 + 1, r1 - 1)
        if direction == "right":
            grid[row, c0 : c0 + length] = color
        else:
            grid[row, c1 - length + 1 : c1 + 1] = color
        return

    col = rng.randint(c0 + 1, c1 - 1)
    if direction == "down":
        grid[r0 : r0 + length, col] = color
    else:
        grid[r1 - length + 1 : r1 + 1, col] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)
    row_heights = _sample_band_sizes(n_rows, rng)
    col_widths = _sample_band_sizes(n_cols, rng)
    grid, row_bands, col_bands = _build_layout(row_heights, col_widths)

    directions = _assign_directions(n_rows, n_cols, rng)
    if n_rows > 1 and n_cols > 1:
        orientations = {("horizontal" if direction in {"left", "right"} else "vertical") for direction in directions.values()}
        if len(orientations) < 2:
            return None

    lengths, incoming = _assign_lengths(directions, rng)
    colors = _assign_colors(incoming, rng)

    for region, direction in directions.items():
        row, col = region
        _draw_pointer(
            grid,
            row_bands[row],
            col_bands[col],
            direction,
            lengths[region],
            colors[region],
            rng,
        )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
