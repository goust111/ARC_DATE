from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _segment_pairs(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    horizontals: list[tuple[int, int, int, int]] = []
    verticals: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = (tuple(map(int, rc)) for rc in coords)
        same_row = r0 == r1
        same_col = c0 == c1
        assert same_row ^ same_col, "same-color markers must share exactly one axis"
        if same_row:
            c_lo, c_hi = sorted((c0, c1))
            assert c_hi - c_lo >= 2, "horizontal segment must have a strict interior"
            horizontals.append((r0, c_lo, c_hi, color))
        else:
            r_lo, r_hi = sorted((r0, r1))
            assert r_hi - r_lo >= 2, "vertical segment must have a strict interior"
            verticals.append((c0, r_lo, r_hi, color))
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Materialize only the hidden horizontal-vertical segment intersections as 3x3 two-color stamps.

    Input structure:
    - The background is black, and every non-background color appears exactly twice.
    - Each same-color pair lies either on one row or on one column, defining an
      implicit horizontal or vertical segment between the two markers.
    - Some implicit segments may never intersect and serve only as distractors.

    Rule:
    - Treat every same-color pair as an invisible axis-aligned segment.
    - For every strict intersection between a horizontal segment and a vertical
      segment, place a 3x3 stamp centered on the crossing cell.
    - Paint the four diagonal neighbors with the horizontal segment color, the
      four orthogonal neighbors with the vertical segment color, keep the center
      as background, and erase all original marker pixels.

    Key insight:
    - The input contains only segment endpoints; the answer is recovered from the
      unseen geometry of where horizontal and vertical pairs cross inside both
      segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    horizontals, verticals = _segment_pairs(g, bg)

    out = np.full_like(g, bg)
    written: dict[tuple[int, int], int] = {}
    for hr, hc0, hc1, h_color in horizontals:
        for vc, vr0, vr1, v_color in verticals:
            if hc0 < vc < hc1 and vr0 < hr < vr1:
                assert 1 <= hr < g.shape[0] - 1 and 1 <= vc < g.shape[1] - 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = hr + dr, vc + dc
                        if dr == 0 and dc == 0:
                            value = bg
                        elif abs(dr) == 1 and abs(dc) == 1:
                            value = h_color
                        else:
                            value = v_color
                        key = (rr, cc)
                        if key in written:
                            assert written[key] == value, "overlapping stamps must agree"
                        written[key] = value

    for (r, c), value in written.items():
        out[r, c] = value
    return out


_BG = 0
_PALETTE = tuple(range(1, 10))
_MIN_SIZE = 14
_MAX_SIZE = 30
_COUNT_OPTIONS = (1, 2, 3)
_CENTER_GAP = 4
_MAX_GENERATION_ATTEMPTS = 400


def _sample_spaced(count: int, low: int, high: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    for _ in range(120):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _free_band_positions(lo: int, hi: int, used: set[int]) -> list[int]:
    return [pos for pos in range(lo, hi + 1) if pos not in used]


def _build_cross_segments(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    colors: list[int],
) -> int:
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    color_idx = 0

    for row in rows:
        left = random.randint(0, min_col - 1)
        right = random.randint(max_col + 1, grid.shape[1] - 1)
        color = colors[color_idx]
        color_idx += 1
        grid[row, left] = color
        grid[row, right] = color

    for col in cols:
        top = random.randint(0, min_row - 1)
        bottom = random.randint(max_row + 1, grid.shape[0] - 1)
        color = colors[color_idx]
        color_idx += 1
        grid[top, col] = color
        grid[bottom, col] = color

    return color_idx


def _add_decoy_horizontals(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    colors: list[int],
    color_idx: int,
) -> int:
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    candidates = _free_band_positions(min_row, max_row, set(rows))
    random.shuffle(candidates)
    if not candidates:
        return color_idx

    left_span = list(range(0, min_col))
    right_span = list(range(max_col + 1, grid.shape[1]))
    decoy_cap = min(2, len(colors) - color_idx, len(candidates))
    decoy_count = random.randint(0, decoy_cap)
    for row in candidates[:decoy_count]:
        options: list[tuple[int, int]] = []
        if len(left_span) >= 2:
            options.append(tuple(sorted(random.sample(left_span, 2))))
        if len(right_span) >= 2:
            options.append(tuple(sorted(random.sample(right_span, 2))))
        if not options:
            break
        c0, c1 = random.choice(options)
        color = colors[color_idx]
        color_idx += 1
        grid[row, c0] = color
        grid[row, c1] = color
    return color_idx


def _add_decoy_verticals(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    colors: list[int],
    color_idx: int,
) -> int:
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    candidates = _free_band_positions(min_col, max_col, set(cols))
    random.shuffle(candidates)
    if not candidates:
        return color_idx

    top_span = list(range(0, min_row))
    bottom_span = list(range(max_row + 1, grid.shape[0]))
    decoy_cap = min(2, len(colors) - color_idx, len(candidates))
    decoy_count = random.randint(0, decoy_cap)
    for col in candidates[:decoy_count]:
        options: list[tuple[int, int]] = []
        if len(top_span) >= 2:
            options.append(tuple(sorted(random.sample(top_span, 2))))
        if len(bottom_span) >= 2:
            options.append(tuple(sorted(random.sample(bottom_span, 2))))
        if not options:
            break
        r0, r1 = random.choice(options)
        color = colors[color_idx]
        color_idx += 1
        grid[r0, col] = color
        grid[r1, col] = color
    return color_idx


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_MIN_SIZE, _MAX_SIZE)
    w = random.randint(_MIN_SIZE, _MAX_SIZE)
    grid = np.full((h, w), _BG, dtype=int)

    h_count = random.choice(_COUNT_OPTIONS)
    v_count = random.choice(_COUNT_OPTIONS)
    rows = _sample_spaced(h_count, 2, h - 3, _CENTER_GAP)
    cols = _sample_spaced(v_count, 2, w - 3, _CENTER_GAP)
    if rows is None or cols is None:
        return None

    total_colors = min(len(_PALETTE), h_count + v_count + 4)
    colors = random.sample(_PALETTE, total_colors)
    color_idx = _build_cross_segments(grid, rows, cols, colors)
    color_idx = _add_decoy_horizontals(grid, rows, cols, colors, color_idx)
    color_idx = _add_decoy_verticals(grid, rows, cols, colors, color_idx)

    expected_crossings = len(rows) * len(cols)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.count_nonzero(out != _BG) // 8 != expected_crossings:
        return None
    if np.count_nonzero(out != _BG) != expected_crossings * 8:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
