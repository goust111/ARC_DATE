from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _roll(grid: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return np.roll(grid, shift=shift, axis=(0, 1))


def _quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "expected odd grid dimensions"

    mid_r = h // 2
    mid_c = w // 2
    qh, qw = mid_r, mid_c
    assert qh == h - mid_r - 1 and qw == w - mid_c - 1, "separator cross must be centered"
    assert qh == qw and qh > 0, "expected four equal square quadrants"

    return (
        g[:mid_r, :mid_c].copy(),
        g[:mid_r, mid_c + 1 :].copy(),
        g[mid_r + 1 :, :mid_c].copy(),
        g[mid_r + 1 :, mid_c + 1 :].copy(),
    )


def _find_increment(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> tuple[int, int]:
    side = int(first.shape[0])
    assert first.shape == second.shape == third.shape == (side, side), "quadrants must match"

    candidates: list[tuple[int, int]] = []
    for dr in range(side):
        for dc in range(side):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(_roll(first, (dr, dc)), second) and np.array_equal(
                _roll(second, (dr, dc)), third
            ):
                candidates.append((dr, dc))

    assert len(candidates) == 1, "expected exactly one arithmetic cyclic increment"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the fourth quadrant pattern from a three-term cyclic-shift arithmetic sequence.

    Input structure:
    - The grid has odd height and width, with a separator cross on the middle row and middle column.
    - The separator splits the grid into four equal square quadrants.
    - The top-left, top-right, and bottom-left quadrants are the same pattern under cyclic rolls.
    - The bottom-right quadrant is unrelated distractor content.

    Rule:
    - Search for the unique cyclic shift (dr, dc) that rolls the top-left quadrant into the top-right quadrant.
    - Verify that applying the same shift again rolls the top-right quadrant into the bottom-left quadrant.
    - Apply that same shift one more time to the bottom-left quadrant.
    - Output the resulting square quadrant only.

    Key insight:
    - Three quadrants are not "matching pictures" but consecutive terms of a shift sequence in toroidal coordinates; the missing answer is the next term, not the odd one out.
    """
    tl, tr, bl, _br = _quadrants(np.asarray(grid, dtype=int))
    shift = _find_increment(tl, tr, bl)
    return _roll(bl, shift).copy()


GEN_ATTEMPTS = 600
SIDE_RANGE = (4, 12)
COLOR_POOL = tuple(range(10))
BRANCHES = ("row_only", "col_only", "both_axes", "ambiguous_half")


def _all_rolls_unique(pattern: np.ndarray) -> bool:
    side = int(pattern.shape[0])
    seen: set[bytes] = set()
    for dr in range(side):
        for dc in range(side):
            key = _roll(pattern, (dr, dc)).tobytes()
            if key in seen:
                return False
            seen.add(key)
    return True


def _pairwise_distinct(arrays: Iterable[np.ndarray]) -> bool:
    seen: set[bytes] = set()
    for arr in arrays:
        key = np.asarray(arr, dtype=int).tobytes()
        if key in seen:
            return False
        seen.add(key)
    return True


def _seeded_pattern(side: int, palette: list[int]) -> np.ndarray:
    seed_count = random.randint(3, min(6, len(palette) + 1, side * side))
    seeds: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for i in range(seed_count):
        while True:
            cell = (random.randrange(side), random.randrange(side))
            if cell not in occupied:
                occupied.add(cell)
                break
        if i < len(palette):
            color = palette[i]
        else:
            color = random.choice(palette)
        seeds.append((cell[0], cell[1], color))

    grid = np.zeros((side, side), dtype=int)
    for r in range(side):
        for c in range(side):
            best_idx = 0
            best_key = None
            for idx, (sr, sc, _color) in enumerate(seeds):
                key = (abs(r - sr) + abs(c - sc), idx)
                if best_key is None or key < best_key:
                    best_idx = idx
                    best_key = key
            grid[r, c] = seeds[best_idx][2]

    mutate_count = random.randint(side, max(side, (side * side) // 3))
    cells = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(cells)
    for r, c in cells[:mutate_count]:
        grid[r, c] = random.choice(palette)
    return grid


def _make_base_pattern(side: int, separator_color: int) -> np.ndarray | None:
    usable = [c for c in COLOR_POOL if c != separator_color]
    palette_size = random.randint(3, min(5, len(usable)))
    palette = random.sample(usable, palette_size)

    for _ in range(60):
        pattern = _seeded_pattern(side, palette)
        if len(set(map(int, pattern.ravel()))) < 3:
            continue
        if not _all_rolls_unique(pattern):
            continue
        return pattern
    return None


def _choose_side(branch: str) -> int:
    if branch == "ambiguous_half":
        return random.choice([n for n in range(SIDE_RANGE[0], SIDE_RANGE[1] + 1) if n % 2 == 0])
    return random.randint(*SIDE_RANGE)


def _choose_shift(side: int, branch: str) -> tuple[int, int]:
    if branch == "row_only":
        return random.randrange(1, side), 0
    if branch == "col_only":
        return 0, random.randrange(1, side)
    if branch == "both_axes":
        return random.randrange(1, side), random.randrange(1, side)

    while True:
        dr = random.randrange(side)
        dc = random.randrange(side)
        if dr == 0 and dc == 0:
            continue
        if (dr % 2 == 1) or (dc % 2 == 1):
            return dr, dc


def _half_ambiguity_count(side: int, doubled_shift: tuple[int, int]) -> int:
    hit_count = 0
    target_r, target_c = doubled_shift
    for dr in range(side):
        for dc in range(side):
            if (2 * dr) % side == target_r and (2 * dc) % side == target_c:
                hit_count += 1
    return hit_count


def _random_distractor(side: int, separator_color: int, forbidden: Iterable[np.ndarray]) -> np.ndarray | None:
    usable = [c for c in COLOR_POOL if c != separator_color]
    palette_size = random.randint(3, min(5, len(usable)))
    palette = random.sample(usable, palette_size)
    forbidden_keys = {np.asarray(arr, dtype=int).tobytes() for arr in forbidden}

    for _ in range(80):
        candidate = _seeded_pattern(side, palette)
        key = candidate.tobytes()
        if key not in forbidden_keys:
            return candidate
    return None


def _assemble(
    tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray, separator_color: int
) -> np.ndarray:
    side = int(tl.shape[0])
    grid = np.full((2 * side + 1, 2 * side + 1), separator_color, dtype=int)
    grid[:side, :side] = tl
    grid[:side, side + 1 :] = tr
    grid[side + 1 :, :side] = bl
    grid[side + 1 :, side + 1 :] = br
    return grid


def _valid_example(
    grid: np.ndarray,
    expected_output: np.ndarray,
    chosen_shift: tuple[int, int],
    branch: str,
    distractor: np.ndarray,
) -> bool:
    try:
        tl, tr, bl, br = _quadrants(grid)
        inferred = _find_increment(tl, tr, bl)
        actual = main(grid)
    except Exception:
        return False

    if inferred != chosen_shift:
        return False
    if not np.array_equal(actual, expected_output):
        return False
    if np.array_equal(br, distractor) is False:
        return False
    if not _pairwise_distinct([tl, tr, bl, expected_output, br]):
        return False
    if branch == "ambiguous_half":
        doubled = ((2 * chosen_shift[0]) % tl.shape[0], (2 * chosen_shift[1]) % tl.shape[0])
        if _half_ambiguity_count(tl.shape[0], doubled) < 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(BRANCHES)
    side = _choose_side(branch)
    separator_color = random.choice(COLOR_POOL)
    base = _make_base_pattern(side, separator_color)
    if base is None:
        return None

    shift = _choose_shift(side, branch)
    tl = base
    tr = _roll(base, shift)
    bl = _roll(base, ((2 * shift[0]) % side, (2 * shift[1]) % side))
    out = _roll(base, ((3 * shift[0]) % side, (3 * shift[1]) % side))

    if not _pairwise_distinct([tl, tr, bl, out]):
        return None
    if branch == "ambiguous_half":
        doubled = ((2 * shift[0]) % side, (2 * shift[1]) % side)
        if _half_ambiguity_count(side, doubled) < 2:
            return None

    br = _random_distractor(side, separator_color, [tl, tr, bl, out])
    if br is None:
        return None

    grid = _assemble(tl, tr, bl, br, separator_color)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not _valid_example(grid, out, shift, branch, br):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid extrapolated-quadrant input")
