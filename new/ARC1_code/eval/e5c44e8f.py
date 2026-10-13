from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


def _marker_color(g: np.ndarray, bg: int) -> int:
    singletons = [
        color
        for color in colors_present(g, exclude={bg})
        if int(np.sum(g == color)) == 1
    ]
    assert len(singletons) == 1, "expected exactly one single-pixel spiral seed color"
    return singletons[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the single seed pixel into a clipped square spiral, stopping before the first blocker on that spiral.

    Input structure
    - The dominant background contains exactly one single-pixel foreground seed.
    - There may also be foreground blocker pixels of one other color.
    - The output keeps the same shape and preserves all original non-background pixels.

    Rule
    - Starting at the seed, follow directions up, right, down, left, with leg lengths 2, 2, 4, 4, 6, 6, and so on.
    - Paint in-bounds background cells on the spiral with the seed color; out-of-bounds steps are skipped but still count.
    - If the next in-bounds spiral cell is a blocker, stop before painting it.

    Key insight
    - The spiral is traced in the unbounded plane and clipped to the grid, which is why separate visible segments can appear when later legs re-enter the canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    seed_color = _marker_color(g, bg)

    foreground = set(colors_present(g, exclude={bg}))
    blocker_colors = foreground - {seed_color}
    assert len(blocker_colors) <= 1, "expected at most one blocker color"

    seed_coords = np.argwhere(g == seed_color)
    assert seed_coords.shape == (1, 2), "expected exactly one spiral seed pixel"
    r, c = map(int, seed_coords[0])
    H, W = g.shape
    max_delta = max(r, H - 1 - r, c, W - 1 - c)
    max_leg_length = 2 * (max_delta + 1)

    out = g.copy()
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_index = 0
    leg_length = 2

    while leg_length <= max_leg_length:
        dr, dc = directions[direction_index]

        for _ in range(leg_length):
            r += dr
            c += dc
            if not in_bounds(g.shape, r, c):
                continue

            color = int(g[r, c])
            if color in blocker_colors:
                return out

            assert color == bg, "spiral should cross only background before a blocker"
            out[r, c] = seed_color

        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SHAPE = (11, 11)
SEED_COLOR, BLOCKER_COLOR = 3, 2

# Spiral geometry and valid painted-path range.
SPIRAL_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
INITIAL_LEG_LENGTH, LEG_LENGTH_STEP = 2, 2
MIN_PAINTED_CELLS, MAX_PAINTED_CELLS = 5, 59

# Seed, blocker, scatter, and retry constraints.
SEED_COORD_RANGE = range(4, 7)
BLOCKER_COUNT_OPTIONS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MIN_SEED_BLOCKER_MANHATTAN, MIN_SCATTER_CHEBYSHEV = 2, 2
GENERATION_ATTEMPTS = 300

def _gen_spiral_path(seed_r: int, seed_c: int, shape: tuple[int, int] = GRID_SHAPE) -> list[tuple[int, int]]:
    h, w = shape
    r, c = seed_r, seed_c
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg_length = LEG_LENGTH_STEP * (max_delta + 1)
    direction_index, leg_length = 0, INITIAL_LEG_LENGTH
    path: list[tuple[int, int]] = []

    while leg_length <= max_leg_length:
        dr, dc = SPIRAL_DIRECTIONS[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            if 0 <= r < h and 0 <= c < w:
                path.append((r, c))

        direction_index = (direction_index + 1) % len(SPIRAL_DIRECTIONS)
        if direction_index in (0, 2):
            leg_length += LEG_LENGTH_STEP

    return path

def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    seed_pos = np.argwhere(inp == SEED_COLOR)
    if seed_pos.shape != (1, 2):
        raise ValueError("generated input must contain exactly one seed")
    seed_r, seed_c = map(int, seed_pos[0])
    path = _gen_spiral_path(seed_r, seed_c, inp.shape)
    out = inp.copy()
    for r, c in path:
        if int(inp[r, c]) == BLOCKER_COLOR:
            break
        out[r, c] = SEED_COLOR
    return out

def _gen_pick_scattered(candidates: list[tuple[int, int]], count: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    picked, remaining = [], candidates[:]
    random.shuffle(remaining)
    while len(picked) < count:
        valid = [
            (r, c)
            for r, c in remaining
            if all(max(abs(r - rr), abs(c - cc)) >= MIN_SCATTER_CHEBYSHEV for rr, cc in occupied)
        ]
        if not valid:
            return None
        cell = random.choice(valid)
        picked.append(cell)
        occupied.add(cell)
        remaining = [
            (r, c)
            for r, c in remaining
            if (r, c) != cell and max(abs(r - cell[0]), abs(c - cell[1])) >= MIN_SCATTER_CHEBYSHEV
        ]
    return picked

def _place_unblocked_seed(inp: np.ndarray) -> None:
    seed_options = []
    for seed_r in SEED_COORD_RANGE:
        for seed_c in SEED_COORD_RANGE:
            if MIN_PAINTED_CELLS <= len(_gen_spiral_path(seed_r, seed_c, inp.shape)) <= MAX_PAINTED_CELLS:
                seed_options.append((seed_r, seed_c))
    seed_r, seed_c = random.choice(seed_options)
    inp[seed_r, seed_c] = SEED_COLOR

def _terminator_indices(path: list[tuple[int, int]], seed_r: int, seed_c: int) -> list[int]:
    return [
        i
        for i, (r, c) in enumerate(path)
        if MIN_PAINTED_CELLS <= i <= MAX_PAINTED_CELLS
        and abs(r - seed_r) + abs(c - seed_c) >= MIN_SEED_BLOCKER_MANHATTAN
    ]

def _extra_blocker_candidates(shape, seed, stop, before_stop):
    h, w = shape
    seed_r, seed_c = seed
    blocked = {seed, stop}
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in blocked
        and (r, c) not in before_stop
        and abs(r - seed_r) + abs(c - seed_c) >= MIN_SEED_BLOCKER_MANHATTAN
    ]

def _place_blockers_case(inp: np.ndarray, total_blockers: int) -> bool:
    seed_r = random.randint(SEED_COORD_RANGE.start, SEED_COORD_RANGE.stop - 1)
    seed_c = random.randint(SEED_COORD_RANGE.start, SEED_COORD_RANGE.stop - 1)
    inp[seed_r, seed_c] = SEED_COLOR
    path = _gen_spiral_path(seed_r, seed_c, inp.shape)

    terminator_indices = _terminator_indices(path, seed_r, seed_c)
    if not terminator_indices:
        return False
    stop_i = random.choice(terminator_indices)
    stop_r, stop_c = path[stop_i]
    inp[stop_r, stop_c] = BLOCKER_COLOR

    allowed_extra = _extra_blocker_candidates(inp.shape, (seed_r, seed_c), (stop_r, stop_c), set(path[:stop_i]))
    extra_blockers = _gen_pick_scattered(allowed_extra, total_blockers - 1, {(stop_r, stop_c)})
    if extra_blockers is None:
        return False
    for r, c in extra_blockers:
        inp[r, c] = BLOCKER_COLOR
    return True

def _passes_oracle(inp: np.ndarray) -> bool:
    expected = _gen_expected_output(inp)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)

def _gen_attempt() -> np.ndarray | None:
    inp = np.zeros(GRID_SHAPE, dtype=int)
    total_blockers = random.choice(BLOCKER_COUNT_OPTIONS)

    if total_blockers == 0:
        _place_unblocked_seed(inp)
    elif not _place_blockers_case(inp, total_blockers):
        return None

    return inp if _passes_oracle(inp) else None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate valid e5c44e8f input")
