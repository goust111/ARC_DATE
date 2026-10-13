from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.append(arc_utils_path)

try:
    from arc_utils import colors_present, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = {int(v) for v in np.asarray(grid).ravel()}
        if exclude is not None:
            cols -= {int(v) for v in exclude}
        return sorted(cols)


def _extract_bands(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    bands: list[dict[str, int]] = []
    for color in colors_present(grid, exclude={bg}):
        cols = np.where(grid == color)[1]
        assert cols.size > 0, "each foreground color must appear"
        c0, c1 = int(cols.min()), int(cols.max())
        region = grid[:, c0 : c1 + 1]
        region_colors = {int(v) for v in np.unique(region)}
        assert region_colors <= {int(bg), int(color)}, "each band may contain only its own color and background"
        assert np.any(region == color), "band must contain foreground"
        assert np.any(region == bg), "band must contain gaps to fill"
        assert np.any(grid[:, c0] == color) and np.any(grid[:, c1] == color), "band boundaries must be visible"
        bands.append(
            {
                "c0": c0,
                "c1": c1,
                "color": int(color),
                "gaps": int(np.count_nonzero(region == bg)),
            }
        )

    assert len(bands) >= 2, "expected at least two bands"
    bands.sort(key=lambda band: band["c0"])
    for left, right in zip(bands, bands[1:]):
        assert left["c1"] < right["c0"], "bands must not overlap"
    return bands


def _apply_rule(grid: np.ndarray, bands: list[dict[str, int]], bg: int) -> np.ndarray:
    out = grid.copy()
    n = len(bands)
    for index, band in enumerate(bands):
        fill_color = bands[(index + band["gaps"]) % n]["color"]
        c0, c1 = band["c0"], band["c1"]
        src_region = grid[:, c0 : c1 + 1]
        dst_region = out[:, c0 : c1 + 1]
        dst_region[src_region == bg] = fill_color
    return out


def _fill_with_fixed_neighbor(grid: np.ndarray, bands: list[dict[str, int]], bg: int) -> np.ndarray:
    fake_bands = [dict(band) for band in bands]
    for band in fake_bands:
        band["gaps"] = 1
    return _apply_rule(grid, fake_bands, bg)


def _fill_with_own_color(grid: np.ndarray, bands: list[dict[str, int]], bg: int) -> np.ndarray:
    out = grid.copy()
    for band in bands:
        c0, c1 = band["c0"], band["c1"]
        src_region = grid[:, c0 : c1 + 1]
        dst_region = out[:, c0 : c1 + 1]
        dst_region[src_region == bg] = band["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical band's background gaps with the color found by jumping right by that band's own gap count in cyclic order.

    Input structure:
    - The dominant color is the background.
    - Every other color occupies one non-overlapping vertical band, with only that band's color and background inside its column span.
    - Each band contains one or more background gap cells inside the span of its own color.

    Rule:
    - Order the bands from left to right as a cyclic sequence.
    - For a band with g background gap cells, move g bands to the right in that cyclic order, wrapping around as needed.
    - Recolor all g gaps inside the source band with the destination band's color, while preserving existing foreground pixels and all background outside the bands.

    Key insight:
    - The holes are both the missing pixels and the step counter: you must count each band's own gaps first, then use that count as a cyclic jump length to choose the borrowing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    bands = _extract_bands(g, bg)
    return _apply_rule(g, bands, bg)


GRID_ROWS_RANGE = (8, 12)
BAND_COUNT_RANGE = (3, 5)
BAND_WIDTH_RANGE = (2, 4)
SEPARATOR_WIDTH_RANGE = (1, 2)
SIDE_MARGIN_RANGE = (0, 2)
MAX_GAP_COUNT = 8
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))
_MODE_NAMES = ("plain_jump", "wrapped_jump", "mixed_jump")
_mode_bag: list[str] = []


def _next_mode() -> str:
    global _mode_bag
    if not _mode_bag:
        _mode_bag = random.sample(list(_MODE_NAMES), k=len(_MODE_NAMES))
    return _mode_bag.pop()


def _sample_layout(n_bands: int) -> tuple[int, list[int], list[int], int, int] | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    widths = [random.randint(*BAND_WIDTH_RANGE) for _ in range(n_bands)]
    separators = [random.randint(*SEPARATOR_WIDTH_RANGE) for _ in range(n_bands - 1)]
    left_margin = random.randint(*SIDE_MARGIN_RANGE)
    right_margin = random.randint(*SIDE_MARGIN_RANGE)
    total_cols = left_margin + right_margin + sum(widths) + sum(separators)
    if total_cols > 30:
        return None
    return rows, widths, separators, left_margin, right_margin


def _sample_gap_counts(n_bands: int, max_count: int, mode: str) -> list[int] | None:
    if max_count < n_bands + 2:
        return None

    counts = {1, n_bands}
    if mode == "wrapped_jump":
        counts.add(n_bands + 2)
    else:
        counts.add(2)

    if any(count > max_count for count in counts):
        return None

    pool = [count for count in range(1, max_count + 1) if count not in counts]
    if mode == "mixed_jump":
        large_pool = [count for count in pool if count > n_bands]
        if large_pool:
            forced = random.choice(large_pool)
            counts.add(forced)
            pool.remove(forced)

    needed = n_bands - len(counts)
    if needed < 0 or len(pool) < needed:
        return None

    counts_list = list(counts) + random.sample(pool, needed)
    random.shuffle(counts_list)
    return counts_list


def _gap_cells(rows: int, c0: int, width: int, gap_count: int) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(rows) for c in range(c0, c0 + width)]
    required_fg = {(random.randrange(rows), c) for c in range(c0, c0 + width)}
    candidates = [cell for cell in cells if cell not in required_fg]
    if gap_count <= 0 or gap_count > len(candidates):
        return None
    return set(random.sample(candidates, gap_count))


def _paint_band(grid: np.ndarray, bg: int, c0: int, width: int, color: int, gap_count: int) -> bool:
    rows = grid.shape[0]
    holes = _gap_cells(rows, c0, width, gap_count)
    if holes is None:
        return False

    grid[:, c0 : c0 + width] = color
    for r, c in holes:
        grid[r, c] = bg
    return True


def _oracle_output(grid: np.ndarray, specs: list[dict[str, int]], bg: int) -> np.ndarray:
    out = grid.copy()
    n = len(specs)
    for index, band in enumerate(specs):
        fill_color = specs[(index + band["gaps"]) % n]["color"]
        c0, c1 = band["c0"], band["c1"]
        src_region = grid[:, c0 : c1 + 1]
        dst_region = out[:, c0 : c1 + 1]
        dst_region[src_region == bg] = fill_color
    return out


def _validate_case(grid: np.ndarray, specs: list[dict[str, int]], bg: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    counts = Counter(map(int, grid.ravel()))
    fg_counts = [count for color, count in counts.items() if color != bg]
    if not fg_counts or counts[bg] <= max(fg_counts):
        return False

    try:
        actual_specs = _extract_bands(grid, bg)
    except AssertionError:
        return False

    if len(actual_specs) != len(specs):
        return False
    for actual, intended in zip(actual_specs, specs):
        if actual != intended:
            return False

    n = len(actual_specs)
    gap_counts = [band["gaps"] for band in actual_specs]
    if len(set(gap_counts)) != n:
        return False
    if 1 not in gap_counts:
        return False
    if not any(gap % n == 0 for gap in gap_counts):
        return False
    if not any(gap % n not in (0, 1) for gap in gap_counts):
        return False

    expected = _oracle_output(grid, specs, bg)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.array_equal(actual, _fill_with_fixed_neighbor(grid, actual_specs, bg)):
        return False
    if np.array_equal(actual, _fill_with_own_color(grid, actual_specs, bg)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_bands = random.randint(*BAND_COUNT_RANGE)
    layout = _sample_layout(n_bands)
    if layout is None:
        return None
    rows, widths, separators, left_margin, right_margin = layout

    max_safe_gap = min(rows * width - width for width in widths)
    max_gap = min(MAX_GAP_COUNT, max_safe_gap)
    mode = _next_mode()
    gap_counts = _sample_gap_counts(n_bands, max_gap, mode)
    if gap_counts is None:
        return None

    colors = random.sample(PALETTE, n_bands + 1)
    bg, *band_colors = colors
    total_cols = left_margin + right_margin + sum(widths) + sum(separators)
    grid = np.full((rows, total_cols), bg, dtype=int)

    specs: list[dict[str, int]] = []
    col = left_margin
    for index, (width, color, gap_count) in enumerate(zip(widths, band_colors, gap_counts)):
        if not _paint_band(grid, bg, col, width, color, gap_count):
            return None
        specs.append({"c0": col, "c1": col + width - 1, "color": int(color), "gaps": int(gap_count)})
        col += width
        if index < n_bands - 1:
            col += separators[index]

    if not _validate_case(grid, specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-count cyclic-band input")
