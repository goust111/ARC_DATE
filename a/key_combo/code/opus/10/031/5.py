from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
HEIGHT_CHOICES = tuple(range(3, 11))  # 3 * H <= 30
WIDTH_CHOICES = tuple(range(3, 9))  # 2 * W <= 30
GENERATION_RETRIES = 400


def _tile_alternating_reflections(grid: np.ndarray) -> np.ndarray:
    middle = np.concatenate((np.fliplr(grid), grid), axis=1)
    flipped = np.flipud(middle)
    return np.concatenate((flipped, middle, flipped), axis=0)


def _big_orbit_positions(height: int, width: int, row: int, col: int) -> list[tuple[int, int]]:
    return [(band * height + row, block * width + col) for band in range(3) for block in range(2)]


def _source_orbit(height: int, width: int, row: int, col: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (row, width - 1 - col),
                (height - 1 - row, col),
                (height - 1 - row, width - 1 - col),
            }
        )
    )


def _all_source_orbits(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for row in range(height):
        for col in range(width):
            orbit = _source_orbit(height, width, row, col)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _colors_present(grid: np.ndarray, *, exclude: Iterable[int] = ()) -> list[int]:
    excluded = set(int(color) for color in exclude)
    return sorted(int(color) for color in np.unique(grid) if int(color) not in excluded)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the damaged source pattern into a 3x2 alternating-reflection canvas and repair every green placeholder from its cross-tile reflection orbit.

    Input structure:
    - The input is a small black-background source tile containing a multicolor foreground pattern.
    - Some foreground cells are replaced by green placeholder pixels that mark damaged locations.
    - The source tile may be asymmetric, but once it is mirrored into a 3-by-2 layout, each local position has a six-cell cross-tile reflection orbit.

    Rule:
    - Build the middle band by concatenating a left-right reflection of the source beside the original source.
    - Flip that band vertically to obtain the top and bottom bands, then stack flipped, original, flipped to form the 3x2 tiled canvas.
    - For each local source coordinate, examine the six tiled cells at that same local coordinate across the 3-by-2 canvas.
    - If the orbit's non-green cells show exactly one visible color, replace every green cell in that orbit with that color; otherwise leave the orbit unchanged.

    Key insight:
    - The repair signal does not come from nearby neighbors in the small tile; it comes from the redundant copies created by alternating reflections, where one local coordinate gathers all reflected evidence for that orbit.
    """
    source = np.asarray(grid, dtype=int)
    assert source.ndim == 2, "expected a 2D grid"

    height, width = map(int, source.shape)
    tiled = _tile_alternating_reflections(source)
    out = tiled.copy()

    for row in range(height):
        for col in range(width):
            orbit = _big_orbit_positions(height, width, row, col)
            visible = {int(tiled[r, c]) for r, c in orbit if int(tiled[r, c]) != GREEN}
            if len(visible) != 1:
                continue
            fill_color = visible.pop()
            for r, c in orbit:
                if int(tiled[r, c]) == GREEN:
                    out[r, c] = fill_color

    return out


def _pick_with_parity(rng: np.random.Generator, values: tuple[int, ...], parity: int) -> int:
    choices = [value for value in values if value % 2 == parity]
    return int(rng.choice(choices))


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    parity_case = int(rng.integers(0, 4))
    height_parity = parity_case // 2
    width_parity = parity_case % 2
    height = _pick_with_parity(rng, HEIGHT_CHOICES, height_parity)
    width = _pick_with_parity(rng, WIDTH_CHOICES, width_parity)
    return height, width


def _fill_uniform_orbits(
    rng: np.random.Generator,
    clean: np.ndarray,
    colors: list[int],
    orbits: list[tuple[tuple[int, int], ...]],
) -> list[tuple[tuple[int, int], ...]]:
    nontrivial = [orbit for orbit in orbits if len(orbit) > 1]
    rng.shuffle(nontrivial)
    max_fill = min(len(nontrivial) - 1, 7)
    if max_fill < 2:
        return []
    target_count = int(rng.integers(2, max_fill + 1))
    selected = nontrivial[:target_count]

    forced_colors = colors.copy()
    rng.shuffle(forced_colors)
    for idx, orbit in enumerate(selected):
        if idx < len(forced_colors):
            color = int(forced_colors[idx])
        else:
            color = int(rng.choice(colors))
        for row, col in orbit:
            clean[row, col] = color
    return selected


def _add_asymmetric_accents(
    rng: np.random.Generator,
    clean: np.ndarray,
    colors: list[int],
    filled_orbits: list[tuple[tuple[int, int], ...]],
    all_orbits: list[tuple[tuple[int, int], ...]],
) -> None:
    blocked = {coord for orbit in filled_orbits for coord in orbit}
    candidate_orbits = [orbit for orbit in all_orbits if not all(coord in blocked for coord in orbit)]
    if not candidate_orbits:
        return
    rng.shuffle(candidate_orbits)
    accent_count = int(rng.integers(1, min(4, len(candidate_orbits)) + 1))

    added = 0
    for orbit in candidate_orbits:
        open_cells = [coord for coord in orbit if coord not in blocked and int(clean[coord]) == BLACK]
        if not open_cells:
            continue
        cell = open_cells[int(rng.integers(len(open_cells)))]
        clean[cell] = int(rng.choice(colors))
        blocked.add(cell)
        added += 1
        if added >= accent_count:
            break


def _allocate_damage_counts(
    rng: np.random.Generator,
    capacities: list[int],
    target: int,
) -> list[int] | None:
    counts = [0] * len(capacities)
    remaining = int(target)
    while remaining > 0:
        choices = [idx for idx, cap in enumerate(capacities) if counts[idx] < cap]
        if not choices:
            return None
        idx = int(rng.choice(choices))
        counts[idx] += 1
        remaining -= 1
    return counts


def _damage_input(
    rng: np.random.Generator,
    clean: np.ndarray,
    repairable_orbits: list[tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    foreground = int(np.count_nonzero(clean != BLACK))
    lower = max(1, math.ceil(0.20 * foreground))
    upper = math.floor(0.40 * foreground)
    capacities = [len(orbit) - 1 for orbit in repairable_orbits]
    max_damage = sum(capacities)
    feasible_upper = min(upper, max_damage)
    if feasible_upper < lower:
        return None

    target_damage = int(rng.integers(lower, feasible_upper + 1))
    counts = _allocate_damage_counts(rng, capacities, target_damage)
    if counts is None:
        return None

    damaged = clean.copy()
    for orbit, count in zip(repairable_orbits, counts):
        if count == 0:
            continue
        indices = rng.choice(len(orbit), size=count, replace=False)
        for index in np.atleast_1d(indices):
            row, col = orbit[int(index)]
            damaged[row, col] = GREEN
    return damaged


def _semantic_checks(
    inp: np.ndarray,
    clean: np.ndarray,
    repairable_orbits: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> bool:
    foreground = int(np.count_nonzero(inp != BLACK))
    green_count = int(np.count_nonzero(inp == GREEN))
    if foreground == 0 or green_count == 0:
        return False
    ratio = green_count / foreground
    if not (0.20 <= ratio <= 0.40):
        return False
    if set(_colors_present(inp, exclude={BLACK, GREEN})) != set(colors):
        return False
    if not (len(colors) in (2, 3)):
        return False
    if np.all(inp[inp != BLACK] == GREEN):
        return False

    clean_colors = _colors_present(clean, exclude={BLACK})
    if set(clean_colors) != set(colors):
        return False

    if np.array_equal(clean, np.fliplr(clean)) and np.array_equal(clean, np.flipud(clean)):
        return False

    expected = _tile_alternating_reflections(clean)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == GREEN):
        return False

    height, width = map(int, inp.shape)
    tiled_inp = _tile_alternating_reflections(inp)
    for row in range(height):
        for col in range(width):
            orbit = _big_orbit_positions(height, width, row, col)
            values = [int(tiled_inp[r, c]) for r, c in orbit]
            if GREEN not in values:
                continue
            visible = {value for value in values if value != GREEN}
            if len(visible) != 1:
                return False

    for orbit in repairable_orbits:
        damaged_here = sum(int(inp[row, col] == GREEN) for row, col in orbit)
        if damaged_here >= len(orbit):
            return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height, width = _sample_shape(rng)
    color_count = int(rng.integers(2, 4))
    colors = [int(color) for color in rng.choice(PALETTE, size=color_count, replace=False)]

    clean = np.zeros((height, width), dtype=int)
    orbits = _all_source_orbits(height, width)
    repairable_orbits = _fill_uniform_orbits(rng, clean, colors, orbits)
    _add_asymmetric_accents(rng, clean, colors, repairable_orbits, orbits)

    if int(np.count_nonzero(clean != BLACK)) < max(6, color_count * 3):
        return None

    inp = _damage_input(rng, clean, repairable_orbits)
    if inp is None:
        return None
    if not _semantic_checks(inp, clean, repairable_orbits, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-reflection repair input")
