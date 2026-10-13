from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 2
PATTERN_COLORS = (1, 3, 4, 7, 8)
GRID_SIZES = (13, 15, 17)
GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

Coord = tuple[int, int]


def _orbit_positions(size: int, r: int, c: int) -> list[Coord]:
    center = size // 2
    dr = int(r) - center
    dc = int(c) - center
    orbit = [
        (center + dr, center + dc),
        (center - dc, center + dr),
        (center - dr, center - dc),
        (center + dc, center - dr),
    ]

    unique: list[Coord] = []
    seen: set[Coord] = set()
    for pos in orbit:
        if pos not in seen:
            unique.append(pos)
            seen.add(pos)
    return unique


def _iter_orbits(size: int):
    seen: set[Coord] = set()
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = _orbit_positions(size, r, c)
            seen.update(orbit)
            yield orbit


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete fourfold rotationally symmetric color groups while red markers block missing-cell filling.

    Input structure:
    - The grid is a square odd-sized canvas with black background and a center of rotation.
    - A colored motif uses 2-3 non-black, non-red colors and is approximately 90-degree rotationally symmetric.
    - Some cells in a rotational orbit are missing as black holes, and some original colored cells are replaced by red markers.

    Rule:
    - Split the grid into 90-degree rotational orbits around the center.
    - For each orbit, read its non-black, non-red cells as the orbit's target color; if no such cell exists, the whole orbit becomes black.
    - Keep existing colored cells, fill black holes only when the orbit has no red marker, and erase every red marker to black.

    Key insight:
    - The red marker acts on its entire rotational orbit: it disappears itself and simultaneously forbids filling every black hole in that same orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"

    out = np.full_like(g, BACKGROUND)
    for orbit in _iter_orbits(h):
        values = [int(g[r, c]) for r, c in orbit]
        colors = {value for value in values if value not in (BACKGROUND, MARKER)}
        if not colors:
            continue

        assert len(colors) == 1, "each rotational orbit must have a single target color"
        target = next(iter(colors))
        blocked = MARKER in values

        for (r, c), value in zip(orbit, values):
            if value not in (BACKGROUND, MARKER):
                out[r, c] = target
            elif value == BACKGROUND and not blocked:
                out[r, c] = target

    return out


def _representatives(size: int) -> list[Coord]:
    center = size // 2
    return [(r, c) for r in range(center) for c in range(center, size)]


def _domain_neighbors(size: int, cell: Coord) -> list[Coord]:
    center = size // 2
    r, c = cell
    neighbors: list[Coord] = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < center and center <= nc < size:
            neighbors.append((nr, nc))
    return neighbors


def _grow_connected_reps(size: int, target_size: int) -> set[Coord] | None:
    reps = set(_representatives(size))
    seed = random.choice(tuple(reps))
    region = {seed}

    while len(region) < target_size:
        frontier = {
            nbr
            for cell in region
            for nbr in _domain_neighbors(size, cell)
            if nbr not in region
        }
        if not frontier:
            return None

        if random.random() < 0.7:
            choices = []
            for nbr in frontier:
                support = sum((adj in region) for adj in _domain_neighbors(size, nbr))
                choices.extend([nbr] * max(1, support))
            region.add(random.choice(choices))
        else:
            region.add(random.choice(tuple(frontier)))

    rows = {r for r, _c in region}
    cols = {c for _r, c in region}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return region


def _assign_colors(active_reps: set[Coord], palette: tuple[int, ...]) -> dict[Coord, int]:
    seeds = random.sample(list(active_reps), len(palette))
    assignment: dict[Coord, int] = {}

    for rep in active_reps:
        distances = [
            abs(rep[0] - seed[0]) + abs(rep[1] - seed[1])
            for seed in seeds
        ]
        best = min(distances)
        choices = [idx for idx, dist in enumerate(distances) if dist == best]
        assignment[rep] = int(palette[random.choice(choices)])

    return assignment


def _paint_input(
    size: int,
    color_by_rep: dict[Coord, int],
) -> tuple[np.ndarray, set[Coord], set[Coord]]:
    grid = np.full((size, size), BACKGROUND, dtype=int)
    reps = list(color_by_rep)
    active_count = len(reps)

    marker_count = random.randint(2, min(4, active_count - 3))
    marker_reps = set(random.sample(reps, marker_count))

    fillable_pool = [rep for rep in reps if rep not in marker_reps]
    fillable_max = min(6, len(fillable_pool) - 1)
    fillable_count = random.randint(2, fillable_max)
    fillable_reps = set(random.sample(fillable_pool, fillable_count))

    for rep, color in color_by_rep.items():
        orbit = _orbit_positions(size, *rep)
        slots = list(range(len(orbit)))
        random.shuffle(slots)

        red_slot: int | None = None
        black_slots: set[int] = set()
        if rep in marker_reps:
            red_slot = slots[0]
            black_count = random.randint(1, 2)
            black_slots = set(slots[1 : 1 + black_count])
        elif rep in fillable_reps:
            black_count = random.randint(1, len(orbit) - 1)
            black_slots = set(slots[:black_count])

        for idx, (r, c) in enumerate(orbit):
            if red_slot is not None and idx == red_slot:
                grid[r, c] = MARKER
            elif idx in black_slots:
                continue
            else:
                grid[r, c] = int(color)

    return grid, marker_reps, fillable_reps


def _valid_generated_input(
    grid: np.ndarray,
    color_by_rep: dict[Coord, int],
    marker_reps: set[Coord],
    fillable_reps: set[Coord],
) -> bool:
    if grid.shape[0] not in GRID_SIZES or grid.shape[0] != grid.shape[1]:
        return False

    if not (2 <= int(np.count_nonzero(grid == MARKER)) <= 4):
        return False

    visible_colors = sorted({int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, MARKER)})
    if len(visible_colors) < 2 or len(visible_colors) > 3:
        return False

    saw_blocked = False
    saw_fillable = False
    for rep, target in color_by_rep.items():
        orbit = _orbit_positions(grid.shape[0], *rep)
        values = [int(grid[r, c]) for r, c in orbit]
        orbit_colors = {value for value in values if value not in (BACKGROUND, MARKER)}
        if orbit_colors != {int(target)}:
            return False

        red_count = values.count(MARKER)
        black_count = values.count(BACKGROUND)
        color_count = len(values) - red_count - black_count
        if color_count <= 0:
            return False

        if rep in marker_reps:
            if red_count != 1 or black_count < 1:
                return False
            saw_blocked = True
        elif rep in fillable_reps:
            if red_count != 0 or black_count < 1:
                return False
            saw_fillable = True
        else:
            if red_count != 0 or black_count != 0:
                return False

    if not saw_blocked or not saw_fillable:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    changed_to_color = np.argwhere((grid == BACKGROUND) & (out != BACKGROUND))
    changed_to_black = np.argwhere((grid == MARKER) & (out == BACKGROUND))
    if changed_to_color.size == 0 or changed_to_black.size == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    max_orbits = min(22, (size // 2) * 3)
    target_orbits = random.randint(8, max_orbits)
    active_reps = _grow_connected_reps(size, target_orbits)
    if active_reps is None or len(active_reps) < 8:
        return None

    color_count = 2 if random.random() < 0.55 else 3
    palette = tuple(random.sample(PATTERN_COLORS, color_count))
    color_by_rep = _assign_colors(active_reps, palette)
    grid, marker_reps, fillable_reps = _paint_input(size, color_by_rep)

    if not _valid_generated_input(grid, color_by_rep, marker_reps, fillable_reps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-blocking ARC input")
