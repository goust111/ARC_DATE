from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _canonicalize(g: np.ndarray, bg: int) -> tuple[np.ndarray, bool]:
    coords = np.argwhere(g != bg)
    assert len(coords) > 0, "expected a non-background diagonal pattern"

    diag_offsets = coords[:, 1] - coords[:, 0]
    anti_offsets = coords[:, 1] + coords[:, 0]
    diag_span = int(diag_offsets.max() - diag_offsets.min())
    anti_span = int(anti_offsets.max() - anti_offsets.min())
    assert diag_span != anti_span, "expected one structurally dominant diagonal direction"

    if diag_span < anti_span:
        return g, False
    return np.fliplr(g), True


def _learn_period(
    g: np.ndarray,
    bg: int,
    min_offset: int,
    max_offset: int,
) -> tuple[int, dict[tuple[int, int], int]]:
    height, width = map(int, g.shape)
    observations: list[tuple[int, int, int]] = []
    for row in range(height):
        for offset in range(min_offset, max_offset + 1):
            col = row + offset
            if 0 <= col < width:
                observations.append((row, offset, int(g[row, col])))

    assert observations, "expected observable diagonal-band cells"
    for period in range(1, height + 1):
        by_phase: dict[tuple[int, int], int] = {}
        ok = True
        for row, offset, color in observations:
            key = (offset, row % period)
            if key in by_phase and by_phase[key] != color:
                ok = False
                break
            by_phase[key] = color
        if ok:
            return period, by_phase

    raise AssertionError("expected a consistent row-periodic diagonal pattern")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the input's diagonal color band onto a canvas with doubled height and width.

    Input structure:
    - The background is the dominant color.
    - Foreground cells form a narrow row-periodic band along one diagonal direction.
    - If the band slopes up-right/down-left, a horizontal flip makes it a down-right band in the top-left quadrant.

    Rule:
    - Canonicalize the band so the input is the top-left quadrant of the larger pattern.
    - Learn the occupied diagonal offsets and the smallest row period of colors, including background gaps within the band.
    - Paint the same offset/period rule onto a 2H x 2W background canvas, then undo the horizontal flip if needed.

    Key insight:
    The input is a corner crop of a larger periodic diagonal pattern, not a pixel-scaled image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = int(get_bg(g))
    canonical, flipped = _canonicalize(g, bg)
    coords = np.argwhere(canonical != bg)
    offsets = coords[:, 1] - coords[:, 0]
    min_offset = int(offsets.min())
    max_offset = int(offsets.max())

    period, pattern = _learn_period(canonical, bg, min_offset, max_offset)

    out = np.full((2 * height, 2 * width), bg, dtype=int)
    for row in range(2 * height):
        for offset in range(min_offset, max_offset + 1):
            col = row + offset
            if 0 <= col < 2 * width:
                key = (offset, row % period)
                assert key in pattern, "period must define every output band phase"
                color = pattern[key]
                if color != bg:
                    out[row, col] = color

    assert np.array_equal(out[:height, :width], canonical), "learned pattern must reproduce the input crop"
    if flipped:
        out = np.fliplr(out)
    return out

# --------------- generator (o5x) ---------------

# Color palette.
_GEN_BG = 0
_GEN_FG_COLORS = tuple(range(1, 10))

# Canvas and structural bounds.
_GEN_SIZE_RANGE = (5, 12)
_GEN_PERIOD_RANGE = (2, 4)
_GEN_MIN_SIZE_PER_PERIOD = 2
_GEN_BAND_WIDTHS = (1, 3)
_GEN_BAND_OFFSETS = {1: (0,), 3: (-1, 0, 1)}

# Tile color and gap rules.
_GEN_MIN_COLOR_COUNT = 2
_GEN_MAX_COLOR_COUNT = 4
_GEN_MAX_BG_GAPS = 2
_GEN_PERIOD4_TWO_COLOR_ROW_INDICES = (0, 0, 1, 0)

# Rejection sampling.
_GEN_RETRY_LIMIT = 300


def _gen_exact_period(tile: np.ndarray) -> bool:
    period = int(tile.shape[0])
    for smaller in range(1, period):
        if np.array_equal(tile, np.roll(tile, -smaller, axis=0)):
            return False
    return True


def _gen_row_color_indices(period: int, color_count: int) -> list[int]:
    if period == 4 and color_count == 2:
        return list(_GEN_PERIOD4_TWO_COLOR_ROW_INDICES)
    indices = list(range(min(period, color_count)))
    while len(indices) < period:
        indices.append(random.randrange(color_count))
    return indices


def _gen_unprotected_cells(tile: np.ndarray, protected: set[tuple[int, int]]) -> list[tuple[int, int]]:
    period, band_width = map(int, tile.shape)
    return [
        (row, col)
        for row in range(period)
        for col in range(band_width)
        if (row, col) not in protected
    ]


def _gen_make_tile(period: int, band_width: int, colors: list[int]) -> np.ndarray:
    tile = np.empty((period, band_width), dtype=int)
    for row in range(period):
        for col in range(band_width):
            tile[row, col] = random.choice(colors)

    protected: set[tuple[int, int]] = set()
    for row, color_index in enumerate(_gen_row_color_indices(period, len(colors))):
        tile[row, 0] = colors[color_index]
        protected.add((row, 0))

    for col in range(1, band_width):
        row = random.randrange(period)
        tile[row, col] = random.choice(colors)
        protected.add((row, col))

    present = set(int(x) for x in tile.ravel())
    for color in colors:
        if color in present:
            continue
        choices = _gen_unprotected_cells(tile, protected)
        if not choices:
            break
        row, col = random.choice(choices)
        tile[row, col] = color
        protected.add((row, col))
        present.add(color)

    gap_choices = _gen_unprotected_cells(tile, protected)
    random.shuffle(gap_choices)
    gap_count = random.randint(0, min(_GEN_MAX_BG_GAPS, len(gap_choices)))
    for row, col in gap_choices[:gap_count]:
        tile[row, col] = _GEN_BG

    if set(colors) - set(int(x) for x in tile.ravel()):
        raise ValueError("tile lost a required color")
    if any(np.all(tile[:, col] == _GEN_BG) for col in range(band_width)):
        raise ValueError("tile lost a band offset")
    if not _gen_exact_period(tile):
        raise ValueError("tile period collapsed")
    return tile


def _gen_paint_output(size: int, tile: np.ndarray) -> np.ndarray:
    out = np.full((2 * size, 2 * size), _GEN_BG, dtype=int)
    band_width = int(tile.shape[1])
    offsets = _GEN_BAND_OFFSETS[band_width]
    period = int(tile.shape[0])
    for row in range(2 * size):
        for tile_col, offset in enumerate(offsets):
            col = row + offset
            if 0 <= col < 2 * size:
                color = int(tile[row % period, tile_col])
                if color != _GEN_BG:
                    out[row, col] = color
    return out


def _gen_sample_design() -> tuple[int, int, int, list[int]]:
    size = random.randint(*_GEN_SIZE_RANGE)
    min_period, max_period = _GEN_PERIOD_RANGE
    valid_periods = [
        period
        for period in range(min_period, max_period + 1)
        if size >= _GEN_MIN_SIZE_PER_PERIOD * period
    ]
    period = random.choice(valid_periods)
    band_width = random.choice(_GEN_BAND_WIDTHS)
    max_colors = min(_GEN_MAX_COLOR_COUNT, period * band_width)
    color_count = random.randint(_GEN_MIN_COLOR_COUNT, max_colors)
    colors = random.sample(_GEN_FG_COLORS, color_count)
    return size, period, band_width, colors


def _gen_crop_for_direction(canonical_out: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    flipped = random.choice([False, True])
    if flipped:
        return np.fliplr(canonical_out[:size, :size]).copy(), np.fliplr(canonical_out)
    return canonical_out[:size, :size].copy(), canonical_out


def _gen_attempt() -> np.ndarray | None:
    size, period, band_width, colors = _gen_sample_design()

    try:
        tile = _gen_make_tile(period, band_width, colors)
    except ValueError:
        return None

    canonical_out = _gen_paint_output(size, tile)
    inp, expected = _gen_crop_for_direction(canonical_out, size)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRY_LIMIT} attempts")
