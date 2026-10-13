from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
BLACK = 0


def _unique_mode_and_noise_count(region: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, region.ravel()))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty quadrant"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    dominant = int(ranked[0][0])
    noise_count = int(region.size - ranked[0][1])
    return dominant, noise_count


def _analyze_quadrants(grid: np.ndarray) -> tuple[int, int, list[tuple[str, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    grey_rows = [r for r in range(height) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(width) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 1, "expected exactly one full grey separator row"
    assert len(grey_cols) == 1, "expected exactly one full grey separator column"

    cross_r = int(grey_rows[0])
    cross_c = int(grey_cols[0])
    assert 0 < cross_r < height - 1, "separator row must leave nonempty top and bottom quadrants"
    assert 0 < cross_c < width - 1, "separator column must leave nonempty left and right quadrants"

    specs = [
        ("lu", slice(0, cross_r), slice(0, cross_c)),
        ("ru", slice(0, cross_r), slice(cross_c + 1, width)),
        ("ld", slice(cross_r + 1, height), slice(0, cross_c)),
        ("rd", slice(cross_r + 1, height), slice(cross_c + 1, width)),
    ]

    info: list[tuple[str, int, int]] = []
    for name, row_slice, col_slice in specs:
        region = g[row_slice, col_slice]
        dominant, noise_count = _unique_mode_and_noise_count(region)
        info.append((name, dominant, noise_count))
    return cross_r, cross_c, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four diagonal rays from the separator center, with each ray length given by a quadrant's minority-pixel count.

    Input structure:
    - A unique full grey row and a unique full grey column split the grid into four nonempty rectangular quadrants.
    - Each quadrant has one dominant color and a small number of off-color noise pixels.
    - The four quadrants occupy the fixed spatial roles left-up, right-up, left-down, and right-down around the grey cross.

    Rule:
    - Find the separator cross and treat its intersection as the reference center.
    - In each quadrant, identify the unique dominant color by frequency; every other pixel in that quadrant counts as noise.
    - Let each quadrant's total noise-pixel count be the ray length for that quadrant's diagonal direction.
    - Return a black square of size (2*max_noise+1) with the center cell grey, then draw colored diagonal rays from the center toward the matching quadrant directions.

    Key insight:
    - The visible output is not encoded by any explicit object shape: the solver must first denoise each quadrant to recover its majority color, then use the leftover minority-pixel count as the diagonal length tied to that quadrant's position.
    """
    _, _, info = _analyze_quadrants(grid)
    max_noise = max(noise_count for _name, _dominant, noise_count in info)
    size = 2 * max_noise + 1
    center = max_noise

    out = np.full((size, size), BLACK, dtype=int)
    out[center, center] = GREY

    directions = {
        "lu": (-1, -1),
        "ru": (-1, 1),
        "ld": (1, -1),
        "rd": (1, 1),
    }
    for name, dominant, noise_count in info:
        dr, dc = directions[name]
        for step in range(1, noise_count + 1):
            out[center + dr * step, center + dc * step] = int(dominant)

    return out


# --------------- generator (att2) ---------------

DOMINANT_POOL = (1, 2, 3, 4)
NOISE_POOL = tuple(range(10))
GRID_HEIGHT_RANGE = (11, 17)
GRID_WIDTH_RANGE = (11, 17)
MIN_QUADRANT_SIZE = 4
GEN_MAX_ATTEMPTS = 400


def _quadrant_bounds(height: int, width: int, cross_r: int, cross_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "lu": (0, cross_r, 0, cross_c),
        "ru": (0, cross_r, cross_c + 1, width),
        "ld": (cross_r + 1, height, 0, cross_c),
        "rd": (cross_r + 1, height, cross_c + 1, width),
    }


def _quadrant_anchor(name: str, cross_r: int, cross_c: int) -> tuple[int, int]:
    if name == "lu":
        return cross_r - 1, cross_c - 1
    if name == "ru":
        return cross_r - 1, cross_c + 1
    if name == "ld":
        return cross_r + 1, cross_c - 1
    return cross_r + 1, cross_c + 1


def _all_cells(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]


def _neighbors8(cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r, c = cell
    r0, r1, c0, c1 = bounds
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if r0 <= rr < r1 and c0 <= cc < c1:
                out.append((rr, cc))
    return out


def _pick_cluster_cells(
    candidates: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    count: int,
    seed: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    available = set(candidates)
    if seed is None or seed not in available:
        seed = random.choice(candidates)
    chosen = [seed]
    available.remove(seed)
    frontier = set(_neighbors8(seed, bounds)) & available

    while len(chosen) < count:
        if frontier:
            weights = []
            options = list(frontier)
            for cell in options:
                neighbor_hits = sum(nei in chosen for nei in _neighbors8(cell, bounds))
                weights.append((neighbor_hits + 1) ** 2)
            cell = random.choices(options, weights=weights, k=1)[0]
            frontier.remove(cell)
        else:
            cell = random.choice(list(available))
        chosen.append(cell)
        available.remove(cell)
        frontier |= set(_neighbors8(cell, bounds)) & available
    return chosen


def _pick_scattered_cells(
    candidates: list[tuple[int, int]],
    count: int,
    seed: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    available = list(candidates)
    if seed is None or seed not in available:
        seed = random.choice(available)
    chosen = [seed]
    available.remove(seed)

    while len(chosen) < count:
        best_cell = None
        best_score = None
        random.shuffle(available)
        for cell in available:
            dist = min(abs(cell[0] - rr) + abs(cell[1] - cc) for rr, cc in chosen)
            if best_score is None or dist > best_score:
                best_score = dist
                best_cell = cell
        assert best_cell is not None
        chosen.append(best_cell)
        available.remove(best_cell)
    return chosen


def _pick_noise_positions(
    bounds: tuple[int, int, int, int],
    count: int,
    preferred: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    candidates = _all_cells(r0, r1, c0, c1)
    assert len(candidates) >= count

    style = random.choice(("cluster", "scatter", "mixed"))
    if style == "cluster":
        return _pick_cluster_cells(candidates, bounds, count, preferred)
    if style == "scatter":
        return _pick_scattered_cells(candidates, count, preferred)

    half = max(1, count // 2)
    cluster = _pick_cluster_cells(candidates, bounds, half, preferred)
    remaining = [cell for cell in candidates if cell not in cluster]
    scattered = _pick_scattered_cells(remaining, count - half, None) if count > half else []
    return cluster + scattered


def _noise_values(count: int, dominant: int, mode: str) -> list[int]:
    palette = [color for color in NOISE_POOL if color != dominant]
    if mode == "mono":
        value = random.choice(palette)
        return [value] * count
    if mode == "duo":
        colors = random.sample(palette, 2)
        values = [colors[0], colors[1]]
        while len(values) < count:
            values.append(random.choice(colors))
        random.shuffle(values)
        return values

    distinct = random.randint(1, min(3, count, len(palette)))
    colors = random.sample(palette, distinct)
    values = colors[:]
    while len(values) < count:
        values.append(random.choice(colors))
    random.shuffle(values)
    return values


def _render_expected(dominants: dict[str, int], counts: dict[str, int]) -> np.ndarray:
    max_noise = max(counts.values())
    center = max_noise
    out = np.full((2 * max_noise + 1, 2 * max_noise + 1), BLACK, dtype=int)
    out[center, center] = GREY
    directions = {
        "lu": (-1, -1),
        "ru": (-1, 1),
        "ld": (1, -1),
        "rd": (1, 1),
    }
    for name, dominant in dominants.items():
        dr, dc = directions[name]
        for step in range(1, counts[name] + 1):
            out[center + dr * step, center + dc * step] = int(dominant)
    return out


def _rank_order(values: list[int]) -> tuple[int, ...]:
    return tuple(index for index, _value in sorted(enumerate(values), key=lambda item: (item[1], item[0])))


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    planned_dominants: dict[str, int],
    planned_counts: dict[str, int],
    near_center_quadrants: set[str],
    repeated_noise_quadrants: set[str],
) -> bool:
    try:
        cross_r, cross_c, info = _analyze_quadrants(grid)
    except AssertionError:
        return False

    found_dominants = {name: dominant for name, dominant, _noise_count in info}
    found_counts = {name: noise_count for name, _dominant, noise_count in info}
    if found_dominants != planned_dominants:
        return False
    if found_counts != planned_counts:
        return False
    if len(set(found_dominants.values())) != 4:
        return False
    if sorted(found_counts.values()) != [1, 2, 3, 4]:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    bounds = _quadrant_bounds(grid.shape[0], grid.shape[1], cross_r, cross_c)
    areas = []
    counts = []
    for name in ("lu", "ru", "ld", "rd"):
        r0, r1, c0, c1 = bounds[name]
        areas.append((r1 - r0) * (c1 - c0))
        counts.append(found_counts[name])
        anchor = _quadrant_anchor(name, cross_r, cross_c)
        if name in near_center_quadrants and int(grid[anchor]) == planned_dominants[name]:
            return False

        region = grid[r0:r1, c0:c1]
        noise_colors = [int(v) for v in region.ravel() if int(v) != planned_dominants[name]]
        if name in repeated_noise_quadrants and len(set(noise_colors)) == len(noise_colors):
            return False

    area_order = _rank_order(areas)
    count_order = _rank_order(counts)
    if area_order == count_order or area_order == count_order[::-1]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if height < 2 * MIN_QUADRANT_SIZE + 1 or width < 2 * MIN_QUADRANT_SIZE + 1:
        return None

    cross_r = random.randint(MIN_QUADRANT_SIZE, height - MIN_QUADRANT_SIZE - 1)
    cross_c = random.randint(MIN_QUADRANT_SIZE, width - MIN_QUADRANT_SIZE - 1)
    bounds = _quadrant_bounds(height, width, cross_r, cross_c)

    names = ["lu", "ru", "ld", "rd"]
    dominant_values = random.sample(list(DOMINANT_POOL), 4)
    noise_counts = random.sample([1, 2, 3, 4], 4)
    dominants = {name: dominant_values[i] for i, name in enumerate(names)}
    counts = {name: noise_counts[i] for i, name in enumerate(names)}

    grid = np.full((height, width), GREY, dtype=int)
    for name in names:
        r0, r1, c0, c1 = bounds[name]
        grid[r0:r1, c0:c1] = dominants[name]

    near_center_quadrants = set(random.sample(names, random.randint(2, 3)))
    repeated_noise_quadrants: set[str] = set()
    repeat_candidates = [name for name in names if counts[name] >= 2]
    repeated_noise_quadrants.add(random.choice(repeat_candidates))
    duo_candidates = [name for name in names if counts[name] >= 3 and name not in repeated_noise_quadrants]
    if duo_candidates:
        repeated_noise_quadrants.add(random.choice(duo_candidates))

    for name in names:
        preferred = _quadrant_anchor(name, cross_r, cross_c) if name in near_center_quadrants else None
        positions = _pick_noise_positions(bounds[name], counts[name], preferred)
        if name in repeated_noise_quadrants:
            mode = "mono" if counts[name] == 2 or random.random() < 0.5 else "duo"
        else:
            mode = "free"
        values = _noise_values(counts[name], dominants[name], mode)
        for (r, c), value in zip(positions, values):
            grid[r, c] = int(value)

    expected = _render_expected(dominants, counts)
    if not _semantic_checks(grid, expected, dominants, counts, near_center_quadrants, repeated_noise_quadrants):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid noisy-cross instance")
