from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400

DIRECTION_ORDER = ("up", "right", "down", "left")
DIR_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
SIDE_CELLS = {
    "up": ((-1, -1), (-1, 0), (-1, 1)),
    "right": ((-1, 1), (0, 1), (1, 1)),
    "down": ((1, -1), (1, 0), (1, 1)),
    "left": ((-1, -1), (0, -1), (1, -1)),
}


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates: list[tuple[int, int]] = []

    for r_raw, c_raw in np.argwhere((g != BLACK) & (g != GREY)):
        r = int(r_raw)
        c = int(c_raw)
        if not (0 < r < height - 1 and 0 < c < width - 1):
            continue

        patch = g[r - 1 : r + 2, c - 1 : c + 2]
        outer = patch.copy()
        outer[1, 1] = BLACK
        if np.any((outer != BLACK) & (outer != GREY)):
            continue
        if int(np.count_nonzero(outer == GREY)) < 4:
            continue
        candidates.append((r, c))

    assert len(candidates) == 1, "expected exactly one grey-framed seed"
    seed = candidates[0]
    sr, sc = seed
    for r_raw, c_raw in np.argwhere(g == GREY):
        r = int(r_raw)
        c = int(c_raw)
        assert abs(r - sr) <= 1 and abs(c - sc) <= 1, "all grey cells must belong to the seed frame"
    return seed


def _branch_directions(grid: np.ndarray, seed: tuple[int, int]) -> list[str]:
    g = np.asarray(grid, dtype=int)
    r, c = seed
    directions: list[str] = []
    for direction in DIRECTION_ORDER:
        if any(int(g[r + dr, c + dc]) != GREY for dr, dc in SIDE_CELLS[direction]):
            directions.append(direction)
    assert directions, "expected at least one open side"
    for direction in directions:
        dr, dc = DIR_STEP[direction]
        assert int(g[r + dr, c + dc]) != GREY, "branch opening must include the side midpoint"
    return directions


def _structure_cells(
    seed: tuple[int, int], directions: tuple[str, ...] | list[str], length: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r, c = seed
    grey_cells = {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    }
    for direction in directions:
        dr, dc = DIR_STEP[direction]
        grey_cells.remove((r + dr, c + dc))

    branch_cells: set[tuple[int, int]] = set()
    for direction in directions:
        dr, dc = DIR_STEP[direction]
        for step in range(1, length + 1):
            branch_cells.add((r + dr * step, c + dc * step))
    return grey_cells, branch_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique colored seed inside the grey 3x3 frame, read which frame sides
    have gaps, and grow one branch per open side. Branch length equals the global
    count of the seed color, and each branch endpoint becomes maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    seed = _find_seed(g)
    seed_r, seed_c = seed
    seed_color = int(g[seed_r, seed_c])
    length = int(np.count_nonzero(g == seed_color))
    assert 2 <= length <= 6, "seed color count must be in [2, 6]"

    directions = _branch_directions(g, seed)

    out = g.copy()
    for direction in directions:
        dr, dc = DIR_STEP[direction]
        for step in range(1, length + 1):
            rr = seed_r + dr * step
            cc = seed_c + dc * step
            assert 0 <= rr < height and 0 <= cc < width, "branch must stay in bounds"
            assert int(g[rr, cc]) == BLACK, "branch path must be empty in the input"
            out[rr, cc] = seed_color
        out[seed_r + dr * length, seed_c + dc * length] = MAROON

    return out


def _random_directions(rng: random.Random) -> tuple[str, ...]:
    directions = tuple(direction for direction in DIRECTION_ORDER if rng.random() < 0.5)
    if directions:
        return directions
    return (rng.choice(DIRECTION_ORDER),)


def _expanded_cells(
    cells: set[tuple[int, int]], height: int, width: int, radius: int = 1
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(height - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(width - 1, c + radius) + 1):
                expanded.add((rr, cc))
    return expanded


def _pick_free_cell(
    rng: random.Random, height: int, width: int, blocked: set[tuple[int, int]]
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in blocked
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _generate_counts(rng: random.Random) -> tuple[int, dict[int, int]]:
    seed_count = rng.randint(2, 6)
    other_color_count = rng.randint(1, 3)
    colors = rng.sample(list(INPUT_COLORS), 1 + other_color_count)
    counts = {int(colors[0]): seed_count}
    for color in colors[1:]:
        counts[int(color)] = rng.randint(2, 6)
    return int(colors[0]), counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    directions = _random_directions(rng)
    seed_color, counts = _generate_counts(rng)
    seed_count = counts[seed_color]
    total_singletons = sum(counts.values())

    min_height = 3 + seed_count * int("up" in directions) + seed_count * int("down" in directions)
    min_width = 3 + seed_count * int("left" in directions) + seed_count * int("right" in directions)
    base_side = max(11, int(np.ceil(np.sqrt(total_singletons * 7))))
    height_low = max(min_height + 2, base_side)
    width_low = max(min_width + 2, base_side)
    height = rng.randint(height_low, height_low + 6)
    width = rng.randint(width_low, width_low + 6)

    row_min = seed_count if "up" in directions else 1
    row_max = height - 1 - seed_count if "down" in directions else height - 2
    col_min = seed_count if "left" in directions else 1
    col_max = width - 1 - seed_count if "right" in directions else width - 2
    if row_min > row_max or col_min > col_max:
        return None

    seed = (rng.randint(row_min, row_max), rng.randint(col_min, col_max))
    grey_cells, branch_cells = _structure_cells(seed, directions, seed_count)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[seed] = seed_color
    for r, c in grey_cells:
        grid[r, c] = GREY

    protected = set(grey_cells)
    protected.add(seed)
    protected.update(branch_cells)
    blocked = _expanded_cells(protected, height, width, radius=1)

    placement_items = list(counts.items())
    rng.shuffle(placement_items)
    placement_items.sort(key=lambda item: item[1], reverse=True)

    for color, total_count in placement_items:
        remaining = total_count - (1 if color == seed_color else 0)
        for _ in range(remaining):
            cell = _pick_free_cell(rng, height, width, blocked)
            if cell is None:
                return None
            grid[cell] = color
            blocked.update(_expanded_cells({cell}, height, width, radius=1))

    if not _validate_generated(grid, seed, directions, seed_color, seed_count):
        return None
    return grid


def _validate_generated(
    grid: np.ndarray,
    seed: tuple[int, int],
    directions: tuple[str, ...],
    seed_color: int,
    seed_count: int,
) -> bool:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if _find_seed(g) != seed:
        return False

    colored = np.argwhere((g != BLACK) & (g != GREY))
    for r_raw, c_raw in colored:
        r = int(r_raw)
        c = int(c_raw)
        neighbors = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) not in (BLACK, GREY):
                neighbors += 1
        if neighbors != 0:
            return False

    for color in np.unique(g):
        color_int = int(color)
        if color_int in (BLACK, GREY):
            continue
        count = int(np.count_nonzero(g == color_int))
        if not 2 <= count <= 6:
            return False
    if int(np.count_nonzero(g == seed_color)) != seed_count:
        return False

    _, branch_cells = _structure_cells(seed, directions, seed_count)
    out = main(g)
    if out.shape != g.shape:
        return False

    for r in range(height):
        for c in range(width):
            if (r, c) in branch_cells:
                continue
            if int(out[r, c]) != int(g[r, c]):
                return False

    seed_r, seed_c = seed
    for direction in directions:
        dr, dc = DIR_STEP[direction]
        for step in range(1, seed_count):
            rr = seed_r + dr * step
            cc = seed_c + dc * step
            if int(out[rr, cc]) != seed_color:
                return False
        end_r = seed_r + dr * seed_count
        end_c = seed_c + dc * seed_count
        if int(out[end_r, end_c]) != MAROON:
            return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
