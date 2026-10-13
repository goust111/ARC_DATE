from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PATTERN_COLORS = (1, 2)
PERIOD_CHOICES = (4, 6, 8)
HEIGHT_RANGE = (4, 8)
WIDTH_MULTIPLIERS = {
    4: (3, 4, 5, 6),
    6: (2, 3, 4),
    8: (2, 3),
}
MAX_GENERATION_ATTEMPTS = 500


def _period_matches_visible_pairs(grid: np.ndarray, bg: int, period: int) -> bool:
    h, w = grid.shape
    compared = False
    for r in range(h):
        for c in range(w - period):
            if int(grid[r, c]) == int(bg) or int(grid[r, c + period]) == int(bg):
                continue
            compared = True
            if int(grid[r, c]) != int(grid[r, c + period]):
                return False
    return compared


def _minimal_visible_period(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    assert h > 0 and w > 1, "expected a non-empty grid wider than one column"
    assert np.any(grid != bg), "expected a visible non-background motif"

    for period in range(1, w):
        if _period_matches_visible_pairs(grid, bg, period):
            return period
    raise AssertionError("failed to infer a visible horizontal period")


def _reconstruct_tile(grid: np.ndarray, bg: int, period: int) -> np.ndarray:
    h, _w = grid.shape
    tile = np.full((h, period), int(bg), dtype=int)
    seen = np.zeros((h, period), dtype=bool)

    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        slot = cc % period
        color = int(grid[rr, cc])
        if seen[rr, slot]:
            assert int(tile[rr, slot]) == color, "visible motif is inconsistent within a period class"
        else:
            tile[rr, slot] = color
            seen[rr, slot] = True

    assert seen.all(), "expected the visible prefix to expose every tile column"
    return tile


def _tile_horizontally(tile: np.ndarray, width: int) -> np.ndarray:
    h, period = tile.shape
    out = np.empty((h, width), dtype=int)
    for c in range(width):
        out[:, c] = tile[:, c % period]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the visible periodic motif, then append a half-period-shifted vertical mirror.

    Input structure:
    - The input is a 2D ARC grid whose dominant background is black.
    - Every row contains the left prefix of the same blue/red horizontal motif, and the right side is background.
    - The visible motif repeats with some smallest horizontal column period P.

    Rule:
    - Find the smallest positive column period P such that every visible non-background cell matches the visible cell P columns to its right whenever that comparison exists.
    - Reconstruct the P-column tile from the visible prefix and repeat it horizontally until the full input width is filled.
    - Mirror that completed top half vertically by reversing the row order.
    - Cyclically shift every mirrored row right by floor(P/2) columns, then stack it below the completed top half.

    Key insight:
    - The same inferred period controls two different steps: it tells you how to finish the top pattern and also how far to offset the mirrored bottom half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = BACKGROUND
    period = _minimal_visible_period(g, bg)
    tile = _reconstruct_tile(g, bg, period)
    top = _tile_horizontally(tile, g.shape[1])
    bottom = np.roll(top[::-1], shift=period // 2, axis=1)
    return np.vstack((top, bottom))


def _cutoff_choices(width: int, period: int) -> list[int]:
    start = period + period // 2
    stop = min(width - 1, 2 * period + period // 2)
    return [cutoff for cutoff in range(start, stop + 1) if cutoff % period != 0]


def _random_tile(height: int, period: int) -> np.ndarray:
    return np.random.choice(PATTERN_COLORS, size=(height, period)).astype(int)


def _full_period_top(tile: np.ndarray, repeats: int) -> np.ndarray:
    return np.tile(tile, (1, repeats))


def _semantic_check(inp: np.ndarray, expected_period: int, cutoff: int) -> bool:
    if inp.ndim != 2:
        return False

    h, w = inp.shape
    if not (1 <= h <= 15 and 1 <= w <= 30):
        return False
    if w % expected_period != 0 or expected_period % 2 != 0:
        return False
    if not (expected_period + expected_period // 2 <= cutoff < w):
        return False
    if cutoff % expected_period == 0:
        return False
    if not np.all(inp[:, cutoff:] == BACKGROUND):
        return False
    if np.any(inp[:, :cutoff] == BACKGROUND):
        return False
    if not np.any(inp == PATTERN_COLORS[0]) or not np.any(inp == PATTERN_COLORS[1]):
        return False

    inferred = _minimal_visible_period(inp, BACKGROUND)
    if inferred != expected_period:
        return False

    out = main(inp)
    if out.shape != (2 * h, w):
        return False
    top = out[:h]
    bottom = out[h:]
    if not np.array_equal(bottom, np.roll(top[::-1], shift=expected_period // 2, axis=1)):
        return False
    if not np.array_equal(top[:, :cutoff], inp[:, :cutoff]):
        return False
    if not np.array_equal(top, _tile_horizontally(_reconstruct_tile(inp, BACKGROUND, inferred), w)):
        return False
    if np.array_equal(top, top[::-1]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    repeats = random.choice(WIDTH_MULTIPLIERS[period])
    width = period * repeats
    height = random.randint(*HEIGHT_RANGE)
    if width <= height:
        return None

    cutoff_options = _cutoff_choices(width, period)
    if not cutoff_options:
        return None
    cutoff = random.choice(cutoff_options)

    tile = _random_tile(height, period)
    top = _full_period_top(tile, repeats)
    inp = top.copy()
    inp[:, cutoff:] = BACKGROUND

    if np.array_equal(tile, tile[::-1]):
        return None
    if _minimal_visible_period(top, BACKGROUND) != period:
        return None
    if not _semantic_check(inp, period, cutoff):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-mirror-shift ARC input")
