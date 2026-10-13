from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_040_8",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste

BG = Color.BLACK
PALETTE = tuple(color for color in range(10) if color != BG)
GRID_ROW_RANGE = (10, 18)
WIDTH_BY_MARKER_COUNT = {
    1: (10, 18),
    2: (12, 20),
    3: (16, 24),
}
BAND_HEIGHTS = (2, 3)
MARKER_COUNTS = (1, 2, 3)
MIN_COLUMN_GAP = {
    1: 1,
    2: 3,
    3: 4,
}
MIN_SURVIVING_COLUMNS = {
    1: 6,
    2: 4,
    3: 2,
}
GEN_ATTEMPTS = 400


def _top_band_height(grid: np.ndarray, bg: int) -> int:
    band_h = 0
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == int(bg)):
            break
        assert np.all(grid[r, :] != int(bg)), "top band rows must span the full width"
        band_h += 1
    assert band_h >= 2, "expected a top band of height 2 or 3"
    assert np.all(grid[band_h:, :] == int(bg)), "expected only the top band to be non-background"
    return band_h


def _band_main_color(band: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(band[band != int(bg)], return_counts=True)
    assert len(colors) >= 1
    return int(colors[np.argmax(counts)])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    band_h = _top_band_height(g, bg)
    band = g[:band_h, :].copy()
    base_color = _band_main_color(band, bg)
    marker_mask = (band != int(bg)) & (band != int(base_color))
    markers = np.argwhere(marker_mask)
    assert markers.shape[0] >= 1, "expected at least one embedded marker"

    out = g.copy()
    mirrored_band = np.flipud(band)
    paste(out, mirrored_band, g.shape[0] - band_h, 0)

    radius = int(markers.shape[0] - 1)
    mirror_row_start = g.shape[0] - band_h
    for r, c in markers:
        mr = int(g.shape[0] - 1 - r)
        mc = int(c)
        r0 = max(mirror_row_start, mr - radius)
        r1 = min(g.shape[0] - 1, mr + radius)
        c0 = max(0, mc - radius)
        c1 = min(g.shape[1] - 1, mc + radius)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the top color band to the bottom, then let the mirrored marker pixels erode square holes whose radius is set by the total marker count.

    Input structure:
    - The grid has a black background with a full-width colored band occupying the top 2 or 3 rows.
    - Most band pixels share one main color, while 1 to 3 single-pixel markers of other colors are embedded inside the band.
    - The marker pixels are not on the topmost band row, and every row below the band is background.

    Rule:
    - Keep the original top band unchanged.
    - Copy that band to the bottom of the grid after flipping its row order vertically.
    - Count the total number of marker pixels T in the original band.
    - For each original marker, find its corresponding position inside the mirrored bottom copy and erase every mirrored pixel within Chebyshev distance at most T-1 from that position.

    Key insight:
    - The non-main-color band pixels play two roles at once: they are copied into the mirrored band, and their total count globally sets the local square erosion radius for every mirrored marker.
    """
    return _solve(grid)


def _sample_marker_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(width))
    random.shuffle(cols)
    picked: list[int] = []
    min_gap = MIN_COLUMN_GAP[count]
    for col in cols:
        if all(abs(col - prev) >= min_gap for prev in picked):
            picked.append(int(col))
            if len(picked) == count:
                picked.sort()
                return picked
    return None


def _sample_marker_colors(base_color: int, count: int) -> list[int]:
    choices = [color for color in PALETTE if color != int(base_color)]
    if count > 1 and random.random() < 0.5:
        shared = random.choice(choices)
        return [int(shared)] * count
    return [int(random.choice(choices)) for _ in range(count)]


def _build_grid(
    height: int,
    width: int,
    band_h: int,
    base_color: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), int(BG), dtype=int)
    grid[:band_h, :] = int(base_color)
    for r, c, color in markers:
        grid[int(r), int(c)] = int(color)
    return grid


def _visual_checks(inp: np.ndarray, out: np.ndarray, marker_count: int) -> bool:
    band_h = _top_band_height(inp, BG)
    mirror_before = np.flipud(inp[:band_h, :])
    mirror_after = out[-band_h:, :]

    if not np.array_equal(out[:band_h, :], inp[:band_h, :]):
        return False
    if not np.all(out[band_h : out.shape[0] - band_h, :] == int(BG)):
        return False
    if not np.any(mirror_after == int(BG)):
        return False
    if not np.any(mirror_after != int(BG)):
        return False

    erased = (mirror_before != int(BG)) & (mirror_after == int(BG))
    if int(erased.sum()) < marker_count:
        return False

    surviving_columns = int(np.sum(np.any(mirror_after != int(BG), axis=0)))
    if surviving_columns < MIN_SURVIVING_COLUMNS[marker_count]:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.choice(MARKER_COUNTS)
    band_h = random.choice(BAND_HEIGHTS)
    height_lo = max(GRID_ROW_RANGE[0], 2 * band_h + 4)
    height_hi = GRID_ROW_RANGE[1]
    width_lo, width_hi = WIDTH_BY_MARKER_COUNT[marker_count]
    height = random.randint(height_lo, height_hi)
    width = random.randint(width_lo, width_hi)

    columns = _sample_marker_columns(width, marker_count)
    if columns is None:
        return None

    base_color = random.choice(PALETTE)
    marker_colors = _sample_marker_colors(base_color, marker_count)
    row_choices = list(range(1, band_h))
    markers = [
        (random.choice(row_choices), col, color)
        for col, color in zip(columns, marker_colors)
    ]

    grid = _build_grid(height, width, band_h, int(base_color), markers)
    out = main(grid)

    if not _visual_checks(grid, out, marker_count):
        return None
    if np.array_equal(out[-band_h:, :], np.flipud(grid[:band_h, :])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 040/8")
