from __future__ import annotations

import numpy as np


BLACK = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
GENERATION_RETRIES = 400


def _compress_runs(values: list[int]) -> tuple[list[int], list[int]]:
    colors: list[int] = []
    lengths: list[int] = []
    for value in values:
        value = int(value)
        if not colors or colors[-1] != value:
            colors.append(value)
            lengths.append(1)
        else:
            lengths[-1] += 1
    return colors, lengths


def _line_majorities(grid: np.ndarray, axis: int) -> tuple[list[int], list[float]]:
    colors = [int(color) for color in np.unique(grid)]
    axis_means: dict[int, float] = {}
    for color in colors:
        coords = np.argwhere(grid == color)[:, axis]
        axis_means[color] = float(coords.mean())

    majorities: list[int] = []
    ratios: list[float] = []
    for index in range(grid.shape[axis]):
        line = grid[index, :] if axis == 0 else grid[:, index]
        values, counts = np.unique(line, return_counts=True)
        max_count = int(counts.max())
        winners = [int(value) for value, count in zip(values, counts) if int(count) == max_count]
        if len(winners) == 1:
            winner = winners[0]
        else:
            distances = [(abs(axis_means[color] - index), color) for color in winners]
            min_distance = min(distance for distance, _color in distances)
            closest = [color for distance, color in distances if distance == min_distance]
            assert len(closest) == 1, "tied line majority must be resolved geometrically"
            winner = closest[0]
        majorities.append(winner)
        ratios.append(max_count / line.size)
    return majorities, ratios


def _axis_profile(grid: np.ndarray, axis: int) -> dict:
    majorities, ratios = _line_majorities(grid, axis)
    colors, thicknesses = _compress_runs(majorities)
    unique_colors = {int(color) for color in np.unique(grid)}
    return {
        "axis": int(axis),
        "majorities": majorities,
        "ratios": ratios,
        "colors": colors,
        "thicknesses": thicknesses,
        "score": float(np.mean(ratios)),
        "covers_all_colors_once": len(colors) == len(unique_colors) and set(colors) == unique_colors,
    }


def _choose_band_axis(grid: np.ndarray) -> dict:
    profiles = [_axis_profile(grid, axis) for axis in (0, 1)]
    valid = [profile for profile in profiles if profile["covers_all_colors_once"]]
    if len(valid) == 1:
        return valid[0]
    if len(valid) == 2:
        ordered = sorted(valid, key=lambda profile: profile["score"], reverse=True)
        assert ordered[0]["score"] > ordered[1]["score"], "band axis must be uniquely favored"
        return ordered[0]

    ordered = sorted(profiles, key=lambda profile: profile["score"], reverse=True)
    assert ordered[0]["score"] > ordered[1]["score"], "band axis must be uniquely favored"
    return ordered[0]


def _spiral_strokes(colors: list[int], lengths: list[int]) -> list[tuple[tuple[int, int], int]]:
    assert len(colors) == len(lengths) and colors
    assert all(int(length) >= 1 for length in lengths)

    strokes: list[tuple[tuple[int, int], int]] = []
    r = 0
    c = 0

    strokes.append(((r, c), int(colors[0])))
    for _ in range(int(lengths[0]) - 1):
        c += 1
        strokes.append(((r, c), int(colors[0])))

    for index in range(1, len(lengths)):
        dr, dc = DIRS[index % 4]
        for _ in range(int(lengths[index])):
            r += dr
            c += dc
            strokes.append(((r, c), int(colors[index])))

    return strokes


def _draw_spiral(colors: list[int], lengths: list[int]) -> np.ndarray:
    strokes = _spiral_strokes(colors, lengths)
    rows = [coord[0] for coord, _color in strokes]
    cols = [coord[1] for coord, _color in strokes]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for (r, c), color in strokes:
        out[r - min_r, c - min_c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the jagged color bands as a (color, thickness) program and execute it as a clockwise spiral drawing.

    Input structure:
    - The grid is fully tiled by a small set of non-black colors, with no separate background.
    - Along exactly one axis, each line has a clear majority color and those line majorities compress into ordered broad bands.
    - Band boundaries are jagged noise, but the majority-run order still recovers the intended band sequence and each band's thickness.

    Rule:
    - Determine the unique band axis from the line-majority compression.
    - Compress the majority colors along that axis into an ordered band list, and use each run length as that band's thickness.
    - On a black canvas, start at the top-left cell and draw the first segment to the right with the first band's color and thickness.
    - For every later band, turn ninety degrees clockwise, start from the adjacent next cell, and draw a segment whose color and length come from that band.
    - Return the exact bounding box of the whole spiral trace.

    Key insight:
    - The noisy color bands are not the final picture; they encode a turtle-style instruction stream where band thickness becomes spiral step length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert BLACK not in set(map(int, np.unique(g))), "input must not use black as a background"

    profile = _choose_band_axis(g)
    return _draw_spiral(profile["colors"], profile["thicknesses"])


def _sample_spiral_lengths(rng: np.random.Generator) -> list[int] | None:
    band_count = int(rng.integers(5, 8))
    first = int(rng.integers(2, 5))
    second = int(rng.integers(2, 5))
    lengths = [first, second]

    width = first
    height = second + 1
    for seg_index in range(2, band_count):
        extension = int(rng.integers(1, 4))
        if seg_index % 2 == 0:
            lengths.append(width + extension - 1)
            width += extension
        else:
            lengths.append(height + extension - 1)
            height += extension

    if sum(lengths) > 27:
        return None
    if max(width, height) > 18:
        return None
    if len(set(lengths)) < 2:
        return None

    strokes = _spiral_strokes([1] * len(lengths), lengths)
    coords = [coord for coord, _color in strokes]
    if len(set(coords)) != len(coords):
        return None

    out = _draw_spiral(list(range(1, len(lengths) + 1)), lengths)
    if min(out.shape) < 3:
        return None
    return lengths


def _random_walk_offsets(
    rng: np.random.Generator,
    width: int,
    amplitude: int,
) -> np.ndarray:
    offsets = np.zeros(width, dtype=int)
    for idx in range(1, width):
        step_choices = (-1, 0, 1) if amplitude == 1 else (-2, -1, 0, 1, 2)
        step = int(step_choices[int(rng.integers(len(step_choices)))])
        if abs(step) == 2 and rng.random() < 0.5:
            step = int(np.sign(step))
        offsets[idx] = int(np.clip(offsets[idx - 1] + step, -amplitude, amplitude))
    return offsets


def _build_row_bands(
    colors: list[int],
    thicknesses: list[int],
    cross_dim: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    band_dim = sum(thicknesses)
    boundary_count = len(thicknesses) - 1
    offsets = []
    for idx in range(boundary_count):
        min_adjacent = min(thicknesses[idx], thicknesses[idx + 1])
        amplitude = 1 if min_adjacent <= 4 else 2
        boundary_offsets = _random_walk_offsets(rng, cross_dim, amplitude)

        active = np.zeros(cross_dim, dtype=bool)
        target = int(rng.integers(max(3, cross_dim // 4), max(4, cross_dim // 2)))
        for _ in range(cross_dim):
            if int(active.sum()) >= target:
                break
            span = int(rng.integers(1, min(4, cross_dim) + 1))
            start = int(rng.integers(0, cross_dim - span + 1))
            active[start : start + span] = True
        boundary_offsets = np.where(active, boundary_offsets, 0)
        if not np.any(boundary_offsets):
            return None
        offsets.append(boundary_offsets)

    offsets_arr = np.asarray(offsets, dtype=int) if offsets else np.zeros((0, cross_dim), dtype=int)
    nominal = np.cumsum(thicknesses[:-1], dtype=int)
    boundaries = nominal[:, None] + offsets_arr
    if boundary_count:
        if np.any(boundaries < 1) or np.any(boundaries > band_dim - 1):
            return None
        if np.any(boundaries[1:] <= boundaries[:-1]):
            return None

    grid = np.empty((band_dim, cross_dim), dtype=int)
    for col in range(cross_dim):
        starts = [0]
        if boundary_count:
            starts.extend(int(x) for x in boundaries[:, col])
        starts.append(band_dim)
        for band_idx, color in enumerate(colors):
            grid[starts[band_idx] : starts[band_idx + 1], col] = int(color)

    if boundary_count and not np.any(offsets_arr != 0):
        return None
    return grid


def _expected_output(colors: list[int], thicknesses: list[int]) -> np.ndarray:
    return _draw_spiral(colors, thicknesses)


def _valid_instance(
    grid: np.ndarray,
    axis: int,
    colors: list[int],
    thicknesses: list[int],
    expected: np.ndarray,
) -> bool:
    try:
        profiles = [_axis_profile(grid, ax) for ax in (0, 1)]
        chosen = _choose_band_axis(grid)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if chosen["axis"] != axis:
        return False
    if chosen["colors"] != colors or chosen["thicknesses"] != thicknesses:
        return False

    other = profiles[1 - axis]
    correct = profiles[axis]
    if not correct["covers_all_colors_once"]:
        return False
    if other["covers_all_colors_once"]:
        return False
    if correct["score"] - other["score"] < 0.12:
        return False
    if min(correct["ratios"]) < 0.55:
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    axis = int(rng.integers(2))
    thicknesses = _sample_spiral_lengths(rng)
    if thicknesses is None:
        return None

    band_count = len(thicknesses)
    colors = [int(color) for color in rng.choice(COLOR_POOL, size=band_count, replace=False)]
    cross_dim = int(rng.integers(max(10, band_count + 4), 19))
    row_grid = _build_row_bands(colors, thicknesses, cross_dim, rng)
    if row_grid is None:
        return None

    grid = row_grid if axis == 0 else row_grid.T
    expected = _expected_output(colors, thicknesses)

    if not _valid_instance(grid, axis, colors, thicknesses, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-band input")
