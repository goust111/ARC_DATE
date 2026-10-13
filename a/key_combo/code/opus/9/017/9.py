from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, colors_present, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = sorted(int(v) for v in np.unique(grid))
        if exclude is None:
            return cols
        banned = {int(v) for v in exclude}
        return [c for c in cols if c not in banned]


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
TOP_HEIGHT_RANGE = (5, 9)
BOTTOM_HEIGHT_RANGE = (6, 11)
BAND_COUNT_OPTIONS = (3, 4)
BAND_WIDTH_RANGE = (2, 5)
SIDE_MARGIN_RANGE = (1, 4)
INNER_GAP_RANGE = (1, 4)
GEN_RETRY_LIMIT = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == Color.GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _ordered_bands(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bands = []
    for color in colors_present(grid, exclude={bg, Color.GREY}):
        cols = np.where(np.any(grid == color, axis=0))[0]
        assert cols.size > 0, f"missing visible cells for color {color}"
        c0 = int(cols.min())
        c1 = int(cols.max())
        band_colors = set(map(int, np.unique(grid[:, c0 : c1 + 1]))) - {bg, Color.GREY}
        assert band_colors == {int(color)}, f"band span for color {color} contains extra colors {band_colors}"
        bands.append((c0, c1, int(color)))
    bands.sort()
    assert 3 <= len(bands) <= 4, "expected 3 or 4 distinct vertical bands"
    for (c0, c1, _), (n0, n1, _) in zip(bands, bands[1:]):
        assert c1 < n0, f"bands overlap: {(c0, c1)} and {(n0, n1)}"
    return bands


def _signal_color(grid: np.ndarray, sep: int, bands: list[tuple[int, int, int]]) -> int:
    upper = grid[:sep, :]
    lower = grid[sep + 1 :, :]
    growth = []
    for _c0, _c1, color in bands:
        upper_count = int(np.count_nonzero(upper == color))
        lower_count = int(np.count_nonzero(lower == color))
        growth.append((lower_count - upper_count, color))
    best = max(delta for delta, _ in growth)
    winners = [color for delta, color in growth if delta == best]
    assert len(winners) == 1, f"expected a unique max growth color, got {growth}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the upper band gaps with the strongest growth color and the lower band gaps with cyclic neighbor colors.

    Input structure:
    - The background is black, and exactly one full grey row splits the grid into an upper half and a lower half.
    - Both halves contain 3 or 4 non-overlapping vertical color bands; each band keeps the same color and column span across the separator.
    - Inside each band's columns, some cells already have the band color while the remaining cells are black gaps.

    Rule:
    - Find the grey separator row and the left-to-right order of the color bands.
    - For each band color C, count its pixels above and below the separator and compute the growth difference D_C - U_C.
    - Let S be the unique color with the largest growth difference.
    - In the upper half, recolor every black gap inside any band span to S.
    - In the lower half, recolor each band's black gaps to the color of the next band to its right, wrapping from the last band back to the first.
    - Keep the separator row, all existing colored pixels, and all background outside the band spans unchanged.

    Key insight:
    - The two halves use different completion rules: the upper half depends on a global cross-region comparison, while the lower half depends only on the cyclic left-to-right band order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep = _find_separator_row(g)
    bands = _ordered_bands(g, bg)
    signal = _signal_color(g, sep, bands)

    out = g.copy()
    for c0, c1, _color in bands:
        upper_band = out[:sep, c0 : c1 + 1]
        upper_band[upper_band == bg] = signal

    for idx, (c0, c1, _color) in enumerate(bands):
        fill_color = bands[(idx + 1) % len(bands)][2]
        lower_band = out[sep + 1 :, c0 : c1 + 1]
        lower_band[lower_band == bg] = fill_color

    return out


def _sample_mask(height: int, width: int, count: int, *, ensure_every_col: bool) -> np.ndarray | None:
    area = height * width
    if not (1 <= count < area):
        return None

    mask = np.zeros((height, width), dtype=bool)
    chosen: set[tuple[int, int]] = set()
    if ensure_every_col:
        if count < width:
            return None
        for c in range(width):
            chosen.add((random.randrange(height), c))

    remaining = [(r, c) for r in range(height) for c in range(width) if (r, c) not in chosen]
    need = count - len(chosen)
    if need < 0 or need > len(remaining):
        return None
    for cell in random.sample(remaining, need):
        chosen.add(cell)

    for r, c in chosen:
        mask[r, c] = True

    if ensure_every_col and not np.all(mask.any(axis=0)):
        return None
    if np.all(mask) or not np.any(mask):
        return None
    return mask


def _choose_counts(top_areas: list[int], bottom_areas: list[int], signal_idx: int) -> tuple[list[int], list[int]] | None:
    upper_counts = []
    lower_caps = []
    for top_area, bottom_area in zip(top_areas, bottom_areas):
        upper_cap = min(top_area - 1, max(2, int(round(top_area * random.uniform(0.25, 0.55)))))
        upper_cap = max(1, upper_cap)
        upper = random.randint(1, upper_cap)
        lower_min = upper + 1
        lower_max = max(
            lower_min,
            min(bottom_area - 1, max(lower_min, int(round(bottom_area * random.uniform(0.55, 0.85))))),
        )
        if lower_min > bottom_area - 1:
            return None
        upper_counts.append(upper)
        lower_caps.append((lower_min, lower_max))

    non_signal_growths = []
    for idx, upper in enumerate(upper_counts):
        if idx == signal_idx:
            continue
        lower_min, lower_max = lower_caps[idx]
        max_growth = lower_max - upper
        if max_growth < 1:
            return None
        growth = random.randint(1, min(max_growth, 5))
        non_signal_growths.append(growth)

    base_max = max(non_signal_growths)
    sig_lower_min, sig_lower_max = lower_caps[signal_idx]
    sig_upper = upper_counts[signal_idx]
    sig_growth_min = base_max + 1
    sig_growth_max = sig_lower_max - sig_upper
    if sig_growth_min > sig_growth_max:
        return None
    sig_growth = random.randint(sig_growth_min, max(sig_growth_min, min(sig_growth_max, base_max + 7)))

    lower_counts = []
    non_signal_iter = iter(non_signal_growths)
    for idx, upper in enumerate(upper_counts):
        if idx == signal_idx:
            lower_counts.append(upper + sig_growth)
        else:
            lower_counts.append(upper + next(non_signal_iter))

    return upper_counts, lower_counts


def _band_columns(widths: list[int], gaps: list[int], left_margin: int) -> list[tuple[int, int]]:
    cols = []
    cursor = left_margin
    for idx, width in enumerate(widths):
        cols.append((cursor, cursor + width - 1))
        cursor += width
        if idx < len(gaps):
            cursor += gaps[idx]
    return cols


def _build_case() -> np.ndarray | None:
    band_count = random.choice(BAND_COUNT_OPTIONS)
    top_h = random.randint(*TOP_HEIGHT_RANGE)
    bottom_h = random.randint(*BOTTOM_HEIGHT_RANGE)
    widths = [random.randint(*BAND_WIDTH_RANGE) for _ in range(band_count)]
    gaps = [random.randint(*INNER_GAP_RANGE) for _ in range(band_count - 1)]
    left_margin = random.randint(*SIDE_MARGIN_RANGE)
    right_margin = random.randint(*SIDE_MARGIN_RANGE)
    total_w = left_margin + sum(widths) + sum(gaps) + right_margin
    total_h = top_h + 1 + bottom_h
    if total_w > 30 or total_h > 30:
        return None

    colors = random.sample(PALETTE, band_count)
    signal_idx = random.randrange(band_count)
    cols = _band_columns(widths, gaps, left_margin)
    top_areas = [top_h * width for width in widths]
    bottom_areas = [bottom_h * width for width in widths]
    counts = _choose_counts(top_areas, bottom_areas, signal_idx)
    if counts is None:
        return None
    upper_counts, lower_counts = counts

    grid = np.zeros((total_h, total_w), dtype=int)
    sep = top_h
    grid[sep, :] = Color.GREY

    for idx, ((c0, c1), color) in enumerate(zip(cols, colors)):
        width = c1 - c0 + 1
        top_mask = _sample_mask(top_h, width, upper_counts[idx], ensure_every_col=upper_counts[idx] >= width and random.random() < 0.65)
        bottom_mask = _sample_mask(bottom_h, width, lower_counts[idx], ensure_every_col=True)
        if top_mask is None or bottom_mask is None:
            return None
        if not np.any(~top_mask):
            return None
        if not np.any(~bottom_mask):
            return None
        top_view = grid[:top_h, c0 : c1 + 1]
        bottom_view = grid[sep + 1 :, c0 : c1 + 1]
        top_view[top_mask] = color
        bottom_view[bottom_mask] = color

    return grid


def _validate_input(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], int]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    bg = get_bg(grid)
    assert bg == Color.BLACK, "generator must keep black as dominant background"
    sep = _find_separator_row(grid)
    bands = _ordered_bands(grid, bg)
    signal = _signal_color(grid, sep, bands)

    upper = grid[:sep, :]
    lower = grid[sep + 1 :, :]
    deltas = {}
    for c0, c1, color in bands:
        top_span = upper[:, c0 : c1 + 1]
        bot_span = lower[:, c0 : c1 + 1]
        assert np.any(top_span == color), f"top half missing color {color}"
        assert np.any(bot_span == color), f"bottom half missing color {color}"
        assert np.any(top_span == Color.BLACK), f"top band {color} needs a visible gap"
        assert np.any(bot_span == Color.BLACK), f"bottom band {color} needs a visible gap"
        assert np.all((top_span == color) | (top_span == Color.BLACK)), "upper band contains invalid colors"
        assert np.all((bot_span == color) | (bot_span == Color.BLACK)), "lower band contains invalid colors"
        deltas[color] = int(np.count_nonzero(bot_span == color) - np.count_nonzero(top_span == color))

    assert list(deltas.values()).count(max(deltas.values())) == 1
    assert signal in deltas
    return sep, bands, signal


def _validate_output(inp: np.ndarray, out: np.ndarray, sep: int, bands: list[tuple[int, int, int]], signal: int) -> None:
    assert out.shape == inp.shape
    assert np.array_equal(out[sep, :], inp[sep, :]), "separator row must stay unchanged"

    covered = np.zeros(inp.shape, dtype=bool)
    covered[sep, :] = True
    top_changes = 0
    bottom_changes = 0
    for idx, (c0, c1, color) in enumerate(bands):
        covered[:sep, c0 : c1 + 1] = True
        covered[sep + 1 :, c0 : c1 + 1] = True

        top_in = inp[:sep, c0 : c1 + 1]
        top_out = out[:sep, c0 : c1 + 1]
        assert np.all(top_out[top_in == color] == color), "existing upper pixels must stay unchanged"
        assert np.all(top_out[top_in == Color.BLACK] == signal), "upper gaps must become the signal color"
        top_changes += int(np.count_nonzero((top_in == Color.BLACK) & (top_out == signal)))

        fill_color = bands[(idx + 1) % len(bands)][2]
        bottom_in = inp[sep + 1 :, c0 : c1 + 1]
        bottom_out = out[sep + 1 :, c0 : c1 + 1]
        assert np.all(bottom_out[bottom_in == color] == color), "existing lower pixels must stay unchanged"
        assert np.all(bottom_out[bottom_in == Color.BLACK] == fill_color), "lower gaps must use the right-neighbor color"
        bottom_changes += int(np.count_nonzero((bottom_in == Color.BLACK) & (bottom_out == fill_color)))

    assert top_changes > 0 and bottom_changes > 0, "both half-rules must visibly affect the output"
    assert np.array_equal(out[~covered], inp[~covered]), "background outside bands must stay unchanged"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_case()
        if grid is None:
            continue
        try:
            sep, bands, signal = _validate_input(grid)
            out = main(grid)
            _validate_output(grid, out, sep, bands, signal)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
