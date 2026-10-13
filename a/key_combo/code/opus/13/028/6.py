import itertools
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
YELLOW = 4
PINK = 6

MAIN_WIDTH = 7
HEIGHT_RANGE = (8, 12)
MAX_GENERATION_ATTEMPTS = 800
GROUP_COLUMNS = ((3,), (2, 4), (1, 5), (0, 6))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BACKGROUND)
    return grid.copy() if box is None else _crop_bbox(grid, box)


def _denoise_vertical_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w % 2 == 1, "main pattern width must be odd"

    out = g.copy()
    mirror_sum = w - 1
    for color in (BLUE, YELLOW):
        for r, c in np.argwhere(g == color):
            rr = int(r)
            cc = int(c)
            mc = mirror_sum - cc
            if int(g[rr, mc]) != color:
                out[rr, cc] = BACKGROUND
    return out


def _transform_main_region(main_region: np.ndarray, shift: int) -> np.ndarray:
    rolled = np.roll(np.asarray(main_region, dtype=int), shift=int(shift), axis=0)
    cleaned = _denoise_vertical_symmetry(rolled)
    return _crop_nonzero(cleaned)


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the control column, row-roll the pattern, symmetrically denoise each color, then crop.

    Input structure:
    - The input is a rectangular grid with a one-cell-wide left control column.
    - The control column is pink except for exactly one red marker pixel whose row encodes a cyclic downward shift.
    - The remaining columns form a black-background pattern using blue and yellow foreground pixels.

    Rule:
    - Read the red marker row index r from the control column and remove that column.
    - Cyclically shift the remaining pattern downward by r rows.
    - For blue and yellow independently, keep only pixels whose mirror across the main region's central vertical axis has the same color.
    - Crop the cleaned result to the bounding box of all non-background pixels.

    Key insight:
    - The control strip does not belong to the final picture; it only tells you how to align the rows before testing per-color mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    control = g[:, 0]
    marker_rows = np.flatnonzero(control == RED)
    assert marker_rows.size == 1
    assert np.all((control == PINK) | (control == RED))

    shift = int(marker_rows[0])
    return _transform_main_region(g[:, 1:], shift)


def _row_from_groups(owners: tuple[int, ...]) -> np.ndarray:
    row = np.zeros(MAIN_WIDTH, dtype=int)
    for owner, cols in zip(owners, GROUP_COLUMNS):
        for c in cols:
            row[c] = int(owner)
    return row


ROW_PATTERNS = [
    _row_from_groups(owners)
    for owners in itertools.product((BACKGROUND, BLUE, YELLOW), repeat=len(GROUP_COLUMNS))
    if 1 <= np.count_nonzero(_row_from_groups(owners)) <= 4
]


def _color_count(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == color))


def _build_clean_pattern(height: int) -> np.ndarray | None:
    clean = np.zeros((height, MAIN_WIDTH), dtype=int)
    occupied_count = random.randint(4, min(6, height))
    occupied_rows = {0, height - 1}
    while len(occupied_rows) < occupied_count:
        occupied_rows.add(random.randrange(1, height - 1))
    ordered_rows = sorted(occupied_rows)

    used_patterns = []
    for row in ordered_rows:
        choices = ROW_PATTERNS[:]
        random.shuffle(choices)
        picked = None
        for candidate in choices:
            if row in (0, height - 1):
                if np.count_nonzero(candidate) < 2:
                    continue
            if row == height - 1 and used_patterns and np.array_equal(candidate, used_patterns[0]):
                continue
            picked = candidate
            break
        if picked is None:
            return None
        clean[row] = picked
        used_patterns.append(picked)

    box = _bbox(clean != BACKGROUND)
    if box is None:
        return None
    if box[0] != 0 or box[2] != height - 1:
        return None
    if box[3] - box[1] + 1 < 3:
        return None

    total = int(np.count_nonzero(clean))
    if not (8 <= total <= 14):
        return None
    if _color_count(clean, BLUE) < 3 or _color_count(clean, YELLOW) < 3:
        return None
    return clean


def _candidate_noise_positions(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    mirror_sum = w - 1
    primary = []
    secondary = []
    for r in range(h):
        has_color = np.any(grid[r] == color)
        for c in range(w):
            if grid[r, c] != BACKGROUND:
                continue
            mc = mirror_sum - c
            if mc == c or grid[r, mc] == color:
                continue
            dest = primary if has_color else secondary
            dest.append((r, c))
    random.shuffle(primary)
    random.shuffle(secondary)
    return primary + secondary


def _add_noise(clean: np.ndarray) -> np.ndarray | None:
    noisy = clean.copy()
    noise_total = 0
    for color in (BLUE, YELLOW):
        count = random.randint(1, 2)
        for _ in range(count):
            candidates = _candidate_noise_positions(noisy, color)
            if not candidates:
                return None
            r, c = candidates[0]
            noisy[r, c] = color
            noise_total += 1

    total = int(np.count_nonzero(noisy))
    if not (12 <= total <= 18):
        return None
    if noise_total < 2:
        return None
    cleaned = _denoise_vertical_symmetry(noisy)
    if not np.array_equal(cleaned, clean):
        return None
    return noisy


def _all_wrong_shifts_differ(input_main: np.ndarray, expected: np.ndarray, shift: int) -> bool:
    height = input_main.shape[0]
    for other in range(height):
        if other == shift:
            continue
        alt = _transform_main_region(input_main, other)
        if np.array_equal(alt, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    shift = random.randrange(height)

    clean = _build_clean_pattern(height)
    if clean is None:
        return None

    noisy = _add_noise(clean)
    if noisy is None:
        return None

    input_main = np.roll(noisy, shift=-shift, axis=0)
    grid = np.full((height, MAIN_WIDTH + 1), PINK, dtype=int)
    grid[shift, 0] = RED
    grid[:, 1:] = input_main

    expected = _crop_nonzero(clean)
    if not np.array_equal(main(grid), expected):
        return None
    if not _all_wrong_shifts_differ(input_main, expected, shift):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
