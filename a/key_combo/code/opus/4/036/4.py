from __future__ import annotations

import random

import numpy as np


BG = 0
GREEN = 3
CYAN = 8
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 9)

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
OPPOSITE = [2, 3, 0, 1]

GRID_MIN = 14
GRID_MAX = 22
N_CROSSES_RANGE = (2, 4)
MAX_LONG_ARM = 5
MAX_SHORT_ARM = 4
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
PLACE_ATTEMPTS = 160
NOISE_ATTEMPTS = 80

NOISE_PATTERNS = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _foreground_color(grid: np.ndarray) -> int | None:
    colors = sorted({int(v) for v in np.asarray(grid).ravel() if int(v) != BG})
    if not colors:
        return None
    if len(colors) != 1:
        raise AssertionError("expected exactly one non-background input color")
    return colors[0]


def _arm_length(grid: np.ndarray, fg: int, center: tuple[int, int], direction: int) -> int:
    r, c = center
    dr, dc = DIRS4[direction]
    length = 0
    rr, cc = r + dr, c + dc
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == fg:
        length += 1
        rr += dr
        cc += dc
    return length


def _cross_analysis(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    fg = _foreground_color(g)
    if fg is None:
        return []

    h, w = g.shape
    analyses: list[dict[str, object]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != fg:
                continue
            if any(int(g[r + dr, c + dc]) != fg for dr, dc in DIRS4):
                continue
            lengths = [_arm_length(g, fg, (r, c), d) for d in range(4)]
            longest = int(np.argmax(lengths))
            if lengths.count(lengths[longest]) != 1:
                raise AssertionError("cross must have a unique longest arm")
            analyses.append(
                {
                    "center": (r, c),
                    "lengths": tuple(int(x) for x in lengths),
                    "longest": longest,
                }
            )

    analyses.sort(key=lambda item: item["center"])
    return analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each plus cross, mark it green, and shoot a cyan ray in that cross's uniquely longest-arm direction.

    Input structure:
    - The grid has black background and exactly one foreground color.
    - The foreground contains one or more recognizable plus centers: a foreground cell whose four orthogonal neighbors are also foreground.
    - Each plus may sit inside a larger same-color shape, but every valid plus has one uniquely longest arm.

    Rule:
    - Find every plus center in the original grid and measure its four arm lengths by counting consecutive foreground cells from the center outward.
    - For each plus, choose the unique longest-arm direction, then recolor the center and its four orthogonal neighbors green.
    - From the center, extend a cyan ray to the grid edge along that chosen direction, recoloring only background cells; existing foreground and green cells are preserved while the ray passes through them.

    Key insight:
    - The ray direction is not fixed globally: each cross independently chooses its ray from a local comparison of its four arm lengths.
    """
    g = np.asarray(grid, dtype=int)
    analyses = _cross_analysis(g)
    if not analyses:
        return g.copy()

    fg = _foreground_color(g)
    assert fg is not None
    out = g.copy()

    for item in analyses:
        r, c = item["center"]
        out[r, c] = GREEN
        for dr, dc in DIRS4:
            out[r + dr, c + dc] = GREEN

    for item in analyses:
        r, c = item["center"]
        dr, dc = DIRS4[item["longest"]]
        rr, cc = r + dr, c + dc
        while _in_bounds(out.shape, rr, cc):
            if int(out[rr, cc]) == BG:
                out[rr, cc] = CYAN
            rr += dr
            cc += dc

    return out


def _cross_cells(center: tuple[int, int], lengths: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for direction, arm_len in enumerate(lengths):
        dr, dc = DIRS4[direction]
        for step in range(1, arm_len + 1):
            cells.add((r + dr * step, c + dc * step))
    return cells


def _shape_can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in occupied and nbr not in cells:
                return False
    return True


def _sample_lengths_from_capacities(capacities: tuple[int, int, int, int], longest: int) -> tuple[int, int, int, int] | None:
    max_long = min(MAX_LONG_ARM, capacities[longest] - 1)
    if max_long < 3:
        return None

    long_len = random.randint(3, max_long)
    lengths = [0, 0, 0, 0]
    lengths[longest] = long_len

    for direction in range(4):
        if direction == longest:
            continue
        max_short = min(MAX_SHORT_ARM, long_len - 1, capacities[direction] - 1)
        if max_short < 1:
            return None
        lengths[direction] = random.randint(1, max_short)

    return tuple(lengths)


def _try_place_random_cross(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    preferred_longest: int,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        r = random.randint(2, h - 3)
        c = random.randint(2, w - 3)
        capacities = (r, w - 1 - c, h - 1 - r, c)
        lengths = _sample_lengths_from_capacities(capacities, preferred_longest)
        if lengths is None:
            continue
        center = (r, c)
        cells = _cross_cells(center, lengths)
        if not _shape_can_place(cells, occupied, shape):
            continue
        return {
            "center": center,
            "lengths": lengths,
            "longest": preferred_longest,
            "cells": cells,
        }
    return None


def _sample_fixed_lengths(longest: int) -> tuple[int, int, int, int]:
    long_len = random.randint(3, 5)
    lengths = [0, 0, 0, 0]
    lengths[longest] = long_len
    for direction in range(4):
        if direction == longest:
            continue
        lengths[direction] = random.randint(1, min(3, long_len - 1))
    return tuple(lengths)


def _try_place_crossing_pair(shape: tuple[int, int], second_longest: int) -> list[dict[str, object]] | None:
    h, w = shape
    first_longest = 1  # canonical: first cross shoots to the right

    for _ in range(PAIR_ATTEMPTS):
        first_lengths = _sample_fixed_lengths(first_longest)
        second_lengths = _sample_fixed_lengths(second_longest)

        min_row = max(first_lengths[0] + 1, second_lengths[0] + 1)
        max_row = h - 2 - max(first_lengths[2], second_lengths[2])
        if min_row > max_row:
            continue
        row = random.randint(min_row, max_row)

        gap = random.randint(1, 3)
        min_c1 = first_lengths[3] + 1
        max_c1 = w - 2 - second_lengths[1] - (first_lengths[1] + gap + 1 + second_lengths[3])
        if min_c1 > max_c1:
            continue
        c1 = random.randint(min_c1, max_c1)
        c2 = c1 + first_lengths[1] + gap + 1 + second_lengths[3]

        first = {
            "center": (row, c1),
            "lengths": first_lengths,
            "longest": first_longest,
        }
        second = {
            "center": (row, c2),
            "lengths": second_lengths,
            "longest": second_longest,
        }
        first["cells"] = _cross_cells(first["center"], first["lengths"])
        second["cells"] = _cross_cells(second["center"], second["lengths"])

        if first["cells"] & second["cells"]:
            continue
        if not _shape_can_place(first["cells"], set(), shape):
            continue
        if not _shape_can_place(second["cells"], first["cells"], shape):
            continue
        return [first, second]

    return None


def _rotate_offsets(pattern: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    coords = [(r, c) for r, c in pattern]
    for _ in range(turns % 4):
        coords = [(-c, r) for r, c in coords]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _try_place_noise(shape: tuple[int, int], occupied: set[tuple[int, int]], count: int) -> list[set[tuple[int, int]]] | None:
    h, w = shape
    placed: list[set[tuple[int, int]]] = []
    occupied_now = set(occupied)
    for _ in range(count):
        success = False
        for _ in range(NOISE_ATTEMPTS):
            pattern = random.choice(NOISE_PATTERNS)
            pattern = _rotate_offsets(pattern, random.randint(0, 3))
            ph = max(r for r, _ in pattern) + 1
            pw = max(c for _, c in pattern) + 1
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            cells = {(top + r, left + c) for r, c in pattern}
            if not _shape_can_place(cells, occupied_now, shape):
                continue
            placed.append(cells)
            occupied_now |= cells
            success = True
            break
        if not success:
            return None
    return placed


def _paint_grid(shape: tuple[int, int], fg: int, crosses: list[dict[str, object]], noise: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for cross in crosses:
        for r, c in cross["cells"]:
            grid[r, c] = fg
    for cells in noise:
        for r, c in cells:
            grid[r, c] = fg
    return grid


def _green_mask(shape: tuple[int, int], analyses: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in analyses:
        r, c = item["center"]
        mask[r, c] = True
        for dr, dc in DIRS4:
            mask[r + dr, c + dc] = True
    return mask


def _has_crossing_green_intersection(analyses: list[dict[str, object]], shape: tuple[int, int]) -> bool:
    green = _green_mask(shape, analyses)
    for item in analyses:
        r, c = item["center"]
        own = np.zeros(shape, dtype=bool)
        own[r, c] = True
        for dr, dc in DIRS4:
            own[r + dr, c + dc] = True
        dr, dc = DIRS4[item["longest"]]
        rr, cc = r + dr, c + dc
        while _in_bounds(shape, rr, cc):
            if green[rr, cc] and not own[rr, cc]:
                return True
            rr += dr
            cc += dc
    return False


def _validate_generated_instance(
    grid: np.ndarray,
    *,
    expected_crosses: int,
    require_crossing: bool,
) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    fg = _foreground_color(g)
    assert fg in INPUT_COLORS
    analyses = _cross_analysis(g)
    assert len(analyses) == expected_crosses
    assert len({item["longest"] for item in analyses}) >= 2

    for item in analyses:
        lengths = item["lengths"]
        assert max(lengths) >= 3
        ordered = sorted(lengths)
        assert ordered[-1] > ordered[-2]

    out = main(g)
    green = _green_mask(g.shape, analyses)
    assert np.array_equal(out == GREEN, green)
    assert np.any(out == CYAN)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == fg and not green[r, c]:
                assert int(out[r, c]) == fg

    for item in analyses:
        r, c = item["center"]
        dr, dc = DIRS4[item["longest"]]
        rr, cc = r + dr, c + dc
        saw_background = False
        while _in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) == BG:
                saw_background = True
                assert int(out[rr, cc]) == CYAN
            rr += dr
            cc += dc
        assert saw_background

    if require_crossing:
        assert _has_crossing_green_intersection(analyses, g.shape)


def _sample_direction_list(count: int) -> list[int]:
    while True:
        dirs = [random.randint(0, 3) for _ in range(count)]
        if len(set(dirs)) >= 2:
            return dirs


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    fg = random.choice(INPUT_COLORS)
    n_crosses = random.randint(*N_CROSSES_RANGE)
    require_crossing = random.random() < 0.6

    crosses: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()

    if require_crossing:
        second_longest = random.choice((0, 2, 3))
        pair = _try_place_crossing_pair((h, w), second_longest)
        if pair is None:
            return None
        crosses.extend(pair)
        for cross in pair:
            occupied |= cross["cells"]
        planned_dirs = [second_longest]
        while len(planned_dirs) < n_crosses - 1:
            planned_dirs.append(random.randint(0, 3))
    else:
        planned_dirs = _sample_direction_list(n_crosses)

    if not require_crossing:
        for direction in planned_dirs:
            cross = _try_place_random_cross((h, w), occupied, direction)
            if cross is None:
                return None
            crosses.append(cross)
            occupied |= cross["cells"]
    else:
        for direction in planned_dirs[1:]:
            cross = _try_place_random_cross((h, w), occupied, direction)
            if cross is None:
                return None
            crosses.append(cross)
            occupied |= cross["cells"]

    noise_count = random.choices([0, 1, 2], weights=[3, 3, 1], k=1)[0]
    noise = _try_place_noise((h, w), occupied, noise_count)
    if noise is None:
        return None

    grid = _paint_grid((h, w), fg, crosses, noise)
    grid = np.rot90(grid, k=random.randint(0, 3))
    _validate_generated_instance(grid, expected_crosses=n_crosses, require_crossing=require_crossing)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GEN_ATTEMPTS} attempts")
