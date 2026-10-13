from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

TOP_H = 6
SEP_H = 1
LOWER_H = 7
GRID_H = TOP_H + SEP_H + LOWER_H

PERIOD_CHOICES = (3, 4, 5)
WIDTH_RANGE = (16, 26)
VISIBLE_REPEAT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _visible_prefix_width(top: np.ndarray, bg: int) -> int:
    cols = np.flatnonzero(np.any(top != bg, axis=0))
    assert len(cols) > 0, "top panel must contain a visible motif"
    width = int(cols[-1]) + 1
    assert not np.any(top[:, width:] != bg), "top panel must end with a clean black suffix"
    return width


def _minimal_period(prefix: np.ndarray) -> int:
    _, width = prefix.shape
    for period in range(1, width + 1):
        residues = np.arange(width) % period
        if np.array_equal(prefix, prefix[:, residues]):
            return period
    raise AssertionError("prefix must have a finite period")


def _render_output_from_parts(grid: np.ndarray, period: int, marker_cols: list[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(sep_rows) == 1
    sep = int(sep_rows[0])

    top = g[:sep]
    assert top.shape[0] > 0
    assert g.shape[0] >= sep + 2 + top.shape[0]

    template = top[:, :period]
    out = g.copy()
    out[sep + 1 :, :] = BG

    stamp_r0 = sep + 2
    for c in marker_cols:
        c0 = int(c)
        c1 = min(g.shape[1], c0 + period)
        if c0 >= c1:
            continue
        out[stamp_r0 : stamp_r0 + top.shape[0], c0:c1] = template[:, : c1 - c0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the upper motif's minimal column period and use it as both stamp width and lower-panel spacing.

    Input structure:
    - The grid has a solid grey separator row splitting an upper motif panel from a lower stamping panel.
    - Above the separator, a red/blue periodic strip starts at the left edge and stops after a visible prefix; the rest is black.
    - Below the separator, the first row is a black marker row containing some green pixels, and the remaining rows form a blank black canvas.

    Rule:
    - Find the smallest positive column period P that explains the visible upper prefix.
    - Take columns 0..P-1 of the upper panel as the stamp template.
    - Erase the green marker row, then stamp that template into the lower canvas starting one row below the markers at every marked column, clipping on the right edge if needed.
    - Keep the upper panel and the grey separator row unchanged.

    Key insight:
    The same detected period controls two different things at once: which columns define the template and how the lower marker columns are meant to align.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(sep_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(sep_rows[0])

    top = g[:sep]
    lower = g[sep + 1 :]
    assert top.shape[0] > 0 and lower.shape[0] >= 1 + top.shape[0], "expected marker row plus a canvas matching the top height"

    visible_w = _visible_prefix_width(top, bg)
    period = _minimal_period(top[:, :visible_w])

    marker_row = lower[0]
    marker_cols = [int(c) for c in np.flatnonzero(marker_row == GREEN)]
    return _render_output_from_parts(g, period, marker_cols)


def _repeat_tile(tile: np.ndarray, width: int) -> np.ndarray:
    reps = (width + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (1, reps))[:, :width]


def _gen_tile(period: int) -> np.ndarray:
    for _ in range(200):
        tile = np.random.choice([BLUE, RED], size=(TOP_H, period))
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(period)}) < 2:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a non-degenerate tile")


def _choose_visible_width(width: int, period: int) -> int:
    max_full_repeats = min(VISIBLE_REPEAT_RANGE[1], (width - 1) // period)
    assert max_full_repeats >= VISIBLE_REPEAT_RANGE[0]
    full_repeats = random.randint(VISIBLE_REPEAT_RANGE[0], max_full_repeats)
    max_partial = min(period - 1, width - 1 - full_repeats * period)
    partial = random.randint(0, max_partial)
    return full_repeats * period + partial


def _choose_marker_cols(width: int, period: int) -> list[int]:
    positions = list(range(0, width, period))
    assert len(positions) >= 2

    want_truncation = width % period != 0 and random.random() < 0.55
    forced = {positions[-1]} if want_truncation else set()

    max_pick = len(positions) - 1
    count = random.randint(1, max_pick)
    while len(forced) > count:
        count += 1

    remaining = [c for c in positions if c not in forced]
    chosen = set(forced)
    if count > len(chosen):
        chosen.update(random.sample(remaining, count - len(chosen)))

    if len(chosen) == len(positions):
        chosen.remove(random.choice(sorted(chosen - forced or chosen)))

    return sorted(chosen)


def _build_input(period: int, width: int, visible_w: int, tile: np.ndarray, marker_cols: list[int]) -> np.ndarray:
    grid = np.full((GRID_H, width), BG, dtype=int)
    top = _repeat_tile(tile, visible_w)
    grid[:TOP_H, :visible_w] = top
    grid[TOP_H, :] = GREY
    for c in marker_cols:
        grid[TOP_H + 1, c] = GREEN
    return grid


def _validate_generated_case(inp: np.ndarray, period: int, visible_w: int, marker_cols: list[int], tile: np.ndarray) -> bool:
    top = inp[:TOP_H]
    prefix = top[:, :visible_w]
    all_positions = list(range(0, inp.shape[1], period))

    if _minimal_period(prefix) != period:
        return False
    for smaller in range(1, period):
        if np.array_equal(prefix, prefix[:, np.arange(visible_w) % smaller]):
            return False

    if visible_w < 2 * period or visible_w >= inp.shape[1]:
        return False
    if len(marker_cols) == 0 or len(marker_cols) >= len(all_positions):
        return False
    if any(c % period != 0 for c in marker_cols):
        return False

    if len(set(map(int, tile.ravel()))) < 2:
        return False

    expected = _render_output_from_parts(inp, period, marker_cols)
    predicted = main(inp)
    if not np.array_equal(predicted, expected):
        return False

    stamp_everywhere = _render_output_from_parts(inp, period, all_positions)
    if np.array_equal(stamp_everywhere, expected):
        return False

    if np.array_equal(expected, inp):
        return False

    lower_canvas = expected[TOP_H + 2 :, :]
    if not np.any(lower_canvas != BG):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    width = random.randint(max(WIDTH_RANGE[0], 2 * period + 2), WIDTH_RANGE[1])
    visible_w = _choose_visible_width(width, period)

    for _ in range(200):
        tile = _gen_tile(period)
        prefix = _repeat_tile(tile, visible_w)
        if _minimal_period(prefix) != period:
            continue

        marker_cols = _choose_marker_cols(width, period)
        inp = _build_input(period, width, visible_w, tile, marker_cols)
        if _validate_generated_case(inp, period, visible_w, marker_cols, tile):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
