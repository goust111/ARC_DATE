from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

MAX_GENERATION_ATTEMPTS = 400
INPUT_HEIGHT_RANGE = (8, 15)
INPUT_WIDTH_RANGE = (5, 10)
TOP_BAND_CHOICES = (2, 3)
BLUE_COUNT_RANGE = (3, 6)
RESIDUE_PROBABILITY = 0.7


def _top_band_height(grid: np.ndarray) -> int:
    for r in range(grid.shape[0]):
        if np.all(grid[r] == BLACK):
            assert r > 0, "the top edge strip must have positive height"
            return int(r)
    raise AssertionError("expected a full-black separator row below the top edge strip")


def _tile_from_sprite(sprite: np.ndarray) -> np.ndarray:
    middle = np.concatenate((np.fliplr(sprite), sprite), axis=1)
    mirrored = np.flipud(middle)
    return np.concatenate((mirrored, middle, mirrored), axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the marked top edge strip to the bottom, crop that migrated strip as a sprite, and tile it with alternating reflections.

    Input structure:
    - The grid has a black background, is taller than it is wide, and its top 2-3 rows form the active edge strip.
    - Blue pixels in that strip define the payload pattern, while exactly one red marker sits on the leftmost or rightmost column inside the strip.
    - Below the strip there is a full-black separator row; lower rows may contain stray blue residue pixels that are not part of the payload.

    Rule:
    - Read the top edge strip from row 0 down to, but not including, the first full-black row.
    - Copy only the blue pattern from that strip to the bottom strip of the same height, keeping the same columns.
    - Erase the original top strip, erase the red marker, and erase every other blue residue pixel.
    - Use the resulting bottom strip rectangle as sprite S, build the middle band as [left-right mirror of S | S], then place its top-bottom mirror above and below it.

    Key insight:
    - The output tile is not taken from an arbitrary crop of the input; it is the cleaned migrated edge-strip pattern after the marker-selected transfer step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    band_h = _top_band_height(g)
    marker_coords = np.argwhere(g == RED)
    assert len(marker_coords) == 1, "expected exactly one red marker"
    marker_r, marker_c = map(int, marker_coords[0])
    assert marker_r < band_h, "the marker must lie inside the top edge strip"
    assert marker_c in (0, w - 1), "the marker must lie on the left or right edge"

    moved = np.zeros_like(g)
    moved[-band_h:, :] = np.where(g[:band_h, :] == BLUE, BLUE, BLACK)
    sprite = moved[-band_h:, :]
    assert sprite.shape == (band_h, w)
    return _tile_from_sprite(sprite)


def _sample_dimensions() -> tuple[int, int, int]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        band_h = random.choice(TOP_BAND_CHOICES)
        width = random.randint(*INPUT_WIDTH_RANGE)
        min_height = max(INPUT_HEIGHT_RANGE[0], width + 1, 2 * band_h + 3)
        max_height = min(INPUT_HEIGHT_RANGE[1], width + 5)
        if min_height <= max_height:
            height = random.randint(min_height, max_height)
            return height, width, band_h
    raise AssertionError("failed to sample valid input dimensions")


def _valid_sprite(mask: np.ndarray) -> bool:
    blue_count = int(mask.sum())
    if not (BLUE_COUNT_RANGE[0] <= blue_count <= BLUE_COUNT_RANGE[1]):
        return False
    if np.any(mask.sum(axis=1) == 0):
        return False

    active_cols = np.flatnonzero(mask.any(axis=0))
    if len(active_cols) < 3:
        return False

    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    return True


def _sample_sprite(height: int, width: int) -> np.ndarray:
    playable_cols = list(range(1, width - 1))
    assert len(playable_cols) >= 3, "need interior columns for the sprite and edge marker"

    for _ in range(MAX_GENERATION_ATTEMPTS):
        count = random.randint(BLUE_COUNT_RANGE[0], min(BLUE_COUNT_RANGE[1], height * (width - 2)))
        chosen: set[tuple[int, int]] = set()

        base_cols = random.sample(playable_cols, k=min(len(playable_cols), height))
        for r in range(height):
            chosen.add((r, base_cols[r % len(base_cols)]))

        while len(chosen) < count:
            chosen.add((random.randrange(height), random.choice(playable_cols)))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        if _valid_sprite(mask):
            return np.where(mask, BLUE, BLACK)

    raise AssertionError("failed to sample a non-degenerate sprite")


def _pick_marker_row(sprite: np.ndarray, marker_col: int) -> int:
    candidates = [r for r in range(sprite.shape[0]) if int(sprite[r, marker_col]) == BLACK]
    assert candidates, "expected at least one free edge cell for the red marker"
    return random.choice(candidates)


def _place_bottom_residue(
    grid: np.ndarray,
    sprite: np.ndarray,
    band_h: int,
    occupied: set[tuple[int, int]],
) -> None:
    bottom_r0 = grid.shape[0] - band_h
    candidates = [
        (bottom_r0 + r, c)
        for r in range(band_h)
        for c in range(grid.shape[1])
        if int(sprite[r, c]) == BLACK
    ]
    assert candidates, "need at least one empty bottom-strip cell for a residue counterexample"
    cell = random.choice(candidates)
    grid[cell] = BLUE
    occupied.add(cell)


def _add_extra_residue(
    grid: np.ndarray,
    band_h: int,
    occupied: set[tuple[int, int]],
    extra_count: int,
) -> None:
    rows = range(band_h + 1, grid.shape[0])
    candidates = [(r, c) for r in rows for c in range(grid.shape[1]) if (r, c) not in occupied]
    random.shuffle(candidates)
    for r, c in candidates[:extra_count]:
        grid[r, c] = BLUE
        occupied.add((r, c))


def _build_input(
    height: int,
    width: int,
    band_h: int,
    sprite: np.ndarray,
    marker_on_left: bool,
    add_residue: bool,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:band_h, :] = sprite

    marker_col = 0 if marker_on_left else width - 1
    marker_row = _pick_marker_row(sprite, marker_col)
    grid[marker_row, marker_col] = RED

    occupied = {(marker_row, marker_col)}
    if add_residue:
        _place_bottom_residue(grid, sprite, band_h, occupied)
        extra_count = random.randint(0, 2)
        _add_extra_residue(grid, band_h, occupied, extra_count)

    assert np.all(grid[band_h, :] == BLACK), "the separator row must remain fully black"
    return grid


def _wrong_no_erase_output(grid: np.ndarray) -> np.ndarray:
    band_h = _top_band_height(grid)
    bottom_band = np.where(grid[-band_h:, :] == BLUE, BLUE, BLACK)
    return _tile_from_sprite(bottom_band)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width, band_h = _sample_dimensions()
        sprite = _sample_sprite(band_h, width)
        marker_on_left = random.choice((False, True))
        add_residue = random.random() < RESIDUE_PROBABILITY
        grid = _build_input(height, width, band_h, sprite, marker_on_left, add_residue)

        expected = _tile_from_sprite(sprite)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if add_residue and np.array_equal(_wrong_no_erase_output(grid), expected):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
