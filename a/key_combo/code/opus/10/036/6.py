from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 4
PATTERN_COLORS = (1, 2, 3)
SIZE_CHOICES = (7, 9, 11)
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
Orbit = tuple[Coord, ...]


def _turn_clockwise(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _spiral_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    mask = np.zeros((height, width), dtype=bool)
    r = c = 0
    dr, dc = 0, 1
    mask[r, c] = True

    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < height and 0 <= nc < width):
            dr, dc = _turn_clockwise(dr, dc)
            continue
        if mask[nr, nc]:
            break

        ar, ac = r + 2 * dr, c + 2 * dc
        if 0 <= ar < height and 0 <= ac < width and mask[ar, ac]:
            dr, dc = _turn_clockwise(dr, dc)
            continue

        r, c = nr, nc
        mask[r, c] = True

    return mask


def _orbit_members(shape: tuple[int, int], r: int, c: int) -> Orbit:
    height, width = map(int, shape)
    candidates = [
        (r, c),
        (r, width - 1 - c),
        (height - 1 - r, c),
        (height - 1 - r, width - 1 - c),
    ]
    orbit: list[Coord] = []
    seen: set[Coord] = set()
    for coord in candidates:
        if coord not in seen:
            seen.add(coord)
            orbit.append(coord)
    return tuple(orbit)


def _reflection_orbits(shape: tuple[int, int]) -> list[Orbit]:
    height, width = map(int, shape)
    seen: set[frozenset[Coord]] = set()
    orbits: list[Orbit] = []
    for r in range(height):
        for c in range(width):
            orbit = _orbit_members(shape, r, c)
            key = frozenset(orbit)
            if key in seen:
                continue
            seen.add(key)
            orbits.append(orbit)
    return orbits


def _candidate_markers(grid: np.ndarray, orbits: list[Orbit]) -> list[int]:
    colors = sorted({int(value) for value in grid.flatten()})
    candidates: list[int] = []
    for color in colors:
        valid = True
        for orbit in orbits:
            orbit_values = [int(grid[r, c]) for r, c in orbit]
            known = {value for value in orbit_values if value != color}
            if len(known) != 1:
                valid = False
                break
        if valid:
            candidates.append(color)
    return candidates


def _infer_marker(grid: np.ndarray, orbits: list[Orbit]) -> int:
    candidates = _candidate_markers(grid, orbits)
    assert len(candidates) == 1, "expected exactly one reflection-hole marker color"
    return int(candidates[0])


def _apply_transform(grid: np.ndarray, marker: int, spiral: np.ndarray, orbits: list[Orbit]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for orbit in orbits:
        orbit_values = [int(grid[r, c]) for r, c in orbit]
        if marker not in orbit_values:
            continue
        known = {value for value in orbit_values if value != marker}
        assert len(known) == 1, "every marked orbit must reveal exactly one non-marker color"
        fill_color = known.pop()
        for r, c in orbit:
            if int(grid[r, c]) != marker:
                continue
            out[r, c] = fill_color if spiral[r, c] else BACKGROUND
    return out


def _pure_symmetry_fill(grid: np.ndarray, marker: int, orbits: list[Orbit]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for orbit in orbits:
        orbit_values = [int(grid[r, c]) for r, c in orbit]
        if marker not in orbit_values:
            continue
        known = {value for value in orbit_values if value != marker}
        assert len(known) == 1
        fill_color = known.pop()
        for r, c in orbit:
            if int(grid[r, c]) == marker:
                out[r, c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a hidden clockwise spiral as a mask that decides which symmetric holes are restored and which are erased.

    Input structure:
    - The grid is a square two-axis reflection pattern built from 2-3 non-black colors.
    - Some cells are replaced by one structurally unique marker color that acts as a hole.
    - Every marked cell belongs to a horizontal/vertical reflection orbit whose non-marker cells all share one color.

    Rule:
    - Trace the same clockwise gap spiral as in the standard top-left spiral task: start at (0,0), move right, turn clockwise at walls or when the cell two steps ahead is already on the path, and stop when the next step would revisit the path.
    - Infer the unique marker color from the reflection-orbit structure.
    - For each marked cell, copy the orbit color back in if that coordinate lies on the spiral path; otherwise replace it with black.
    - Leave every non-marker cell unchanged.

    Key insight:
    - The spiral is invisible in the final picture and acts only as a gate: symmetry tells you the missing color, while the spiral tells you whether that hole is repaired or deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"

    orbits = _reflection_orbits(g.shape)
    marker = _infer_marker(g, orbits)
    spiral = _spiral_mask(g.shape)
    return _apply_transform(g, marker, spiral, orbits)


def _build_pattern(size: int, palette: list[int]) -> np.ndarray:
    half = size // 2 + 1
    seed = np.full((half, half), -1, dtype=int)
    grid = np.full((size, size), BACKGROUND, dtype=int)

    for r in range(half):
        for c in range(half):
            neighbor_choices = []
            if r > 0:
                neighbor_choices.append(int(seed[r - 1, c]))
            if c > 0:
                neighbor_choices.append(int(seed[r, c - 1]))
            if r > 0 and c > 0:
                neighbor_choices.append(int(seed[r - 1, c - 1]))
            neighbor_choices = [color for color in neighbor_choices if color >= 0]

            if neighbor_choices and random.random() < 0.65:
                color = random.choice(neighbor_choices)
            else:
                color = random.choice(palette)

            seed[r, c] = color
            for rr, cc in _orbit_members((size, size), r, c):
                grid[rr, cc] = color

    flat_coords = [(r, c) for r in range(half) for c in range(half)]
    for color in palette:
        if np.any(grid == color):
            continue
        r, c = random.choice(flat_coords)
        for rr, cc in _orbit_members((size, size), r, c):
            grid[rr, cc] = color

    return grid


def _mixed_spiral_orbits(size: int) -> tuple[np.ndarray, list[Orbit]]:
    spiral = _spiral_mask((size, size))
    mixed: list[Orbit] = []
    for orbit in _reflection_orbits((size, size)):
        if len(orbit) != 4:
            continue
        spiral_count = sum(int(spiral[r, c]) for r, c in orbit)
        if 0 < spiral_count < 4:
            mixed.append(orbit)
    return spiral, mixed


def _choose_known_index(orbit: Orbit, spiral: np.ndarray, require_mixed: bool) -> int:
    spiral_idxs = [idx for idx, (r, c) in enumerate(orbit) if spiral[r, c]]
    off_idxs = [idx for idx, (r, c) in enumerate(orbit) if not spiral[r, c]]

    if require_mixed:
        if len(spiral_idxs) == 1:
            return random.choice(off_idxs)
        return random.choice(spiral_idxs)

    choices: list[list[int]] = []
    if len(spiral_idxs) == 1:
        choices.extend([spiral_idxs, off_idxs])
    else:
        choices.extend([spiral_idxs, off_idxs])
    return random.choice(random.choice(choices))


def _recolor_orbit(grid: np.ndarray, orbit: Orbit, color: int) -> None:
    for r, c in orbit:
        grid[r, c] = color


def _attempt_generate() -> np.ndarray | None:
    size = random.choice(SIZE_CHOICES)
    spiral, mixed_orbits = _mixed_spiral_orbits(size)
    if len(mixed_orbits) < 2:
        return None

    palette = random.sample(list(PATTERN_COLORS), random.randint(2, 3))
    orbit_count = random.randint(2, min(4, len(mixed_orbits)))
    selected_orbits = random.sample(mixed_orbits, orbit_count)

    base = _build_pattern(size, palette)
    selected_colors = {int(base[orbit[0][0], orbit[0][1]]) for orbit in selected_orbits}
    if len(selected_colors) < 2:
        recolor_target = random.choice(selected_orbits)
        current = int(base[recolor_target[0][0], recolor_target[0][1]])
        alternatives = [color for color in palette if color != current]
        if alternatives:
            _recolor_orbit(base, recolor_target, random.choice(alternatives))

    inp = base.copy()
    known_positions: set[Coord] = set()
    mixed_outcome_orbit_found = False

    forced_mixed_orbit = random.choice(selected_orbits)
    for orbit in selected_orbits:
        known_index = _choose_known_index(orbit, spiral, require_mixed=(orbit == forced_mixed_orbit))
        hole_outcomes = []
        for idx, coord in enumerate(orbit):
            if idx == known_index:
                known_positions.add(coord)
                continue
            r, c = coord
            inp[r, c] = MARKER
            hole_outcomes.append(bool(spiral[r, c]))
        if any(hole_outcomes) and not all(hole_outcomes):
            mixed_outcome_orbit_found = True

    if not mixed_outcome_orbit_found:
        return None

    if int(np.sum(inp == MARKER)) not in (6, 9, 12):
        return None

    orbits = _reflection_orbits(inp.shape)
    candidates = _candidate_markers(inp, orbits)
    if candidates != [MARKER]:
        return None

    out = main(inp)
    expected = base.copy()
    repaired = 0
    erased = 0
    for r, c in np.argwhere(inp == MARKER):
        r = int(r)
        c = int(c)
        if spiral[r, c]:
            repaired += 1
            expected[r, c] = int(base[r, c])
        else:
            erased += 1
            expected[r, c] = BACKGROUND

    if repaired == 0 or erased == 0:
        return None
    if not np.array_equal(out, expected):
        return None

    symmetry_only = _pure_symmetry_fill(inp, MARKER, orbits)
    erase_only = np.where(inp == MARKER, BACKGROUND, inp)
    if np.array_equal(out, symmetry_only):
        return None
    if np.array_equal(out, erase_only):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
