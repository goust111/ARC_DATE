from __future__ import annotations

from collections import Counter
import math

import numpy as np


LEGEND_COLOR_POOL = np.array([6, 7, 9], dtype=int)
MARKER_COLOR_POOL = np.array([1, 2, 3, 4, 5, 8], dtype=int)
COUNT_PATTERNS = {
    2: [(1, 2), (1, 3), (2, 4), (3, 5)],
    3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
}
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """Return legend entries as (col, color, height), scanning row-0 anchored bars."""
    g = np.asarray(grid, dtype=int)
    entries: list[tuple[int, int, int]] = []
    for c in range(g.shape[1]):
        color = int(g[0, c])
        if color == bg:
            continue
        k = 1
        while k < g.shape[0] and int(g[k, c]) == color:
            k += 1
        assert np.all(g[:k, c] == color), "legend entry must be a single-color vertical bar"
        entries.append((c, color, k))

    assert entries, "expected at least one legend entry"
    return entries


def _marker_region_top(legend: list[tuple[int, int, int]]) -> int:
    return max(k for _c, _color, k in legend) + 2


def _marker_counts(grid: np.ndarray, start_row: int, bg: int) -> Counter[int]:
    region = grid[start_row:, :]
    colors = region[region != bg]
    return Counter(map(int, colors.tolist()))


def _ranked_marker_colors(counts: Counter[int]) -> list[int]:
    assert counts, "expected at least one marker color"
    freqs = list(counts.values())
    assert len(set(freqs)) == len(freqs), "marker-color counts must be unique"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [int(color) for color, _count in ranked]


def _draw_hollow_square(out: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    r, c = center
    h, w = out.shape
    assert 0 <= r - radius and r + radius < h and 0 <= c - radius and c + radius < w
    for rr in range(r - radius, r + radius + 1):
        for cc in range(c - radius, c + radius + 1):
            if max(abs(rr - r), abs(cc - c)) != radius:
                continue
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each legend height as both a count-rank index and an outline radius, then draw the corresponding hollow squares.

    Input structure:
    - A black background contains 2-3 single-column legend bars touching row 0, separated by black gaps.
    - Each legend bar has a unique color K and a unique height k; the heights cover 1..N.
    - Below two blank separator rows, the main region contains isolated single-pixel markers in N non-legend colors.

    Rule:
    - Read every legend entry as (K, k), where K is the outline color and k is the lookup index.
    - Count the main-region markers by color and rank those colors from most frequent to least frequent.
    - For each legend entry, select the marker color at rank k and draw a centered hollow square of side (2k+1) around every marker of that color, using outline color K.
    - Keep the marker pixels unchanged and clear the entire legend area to black in the output.

    Key insight:
    - Legend height has a double role: it first indexes into the frequency ranking, and the same number also sets the outline size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == 0, "expected black background"

    legend = _parse_legend(g, bg)
    legend_heights = sorted(k for _c, _color, k in legend)
    assert legend_heights == list(range(1, len(legend) + 1)), "legend heights must cover 1..N"
    assert len({color for _c, color, _k in legend}) == len(legend), "legend colors must be distinct"

    start_row = _marker_region_top(legend)
    assert start_row <= g.shape[0], "grid too short for legend separation"
    if start_row - 2 < g.shape[0]:
        assert np.all(g[max(legend_heights) : start_row, :] == bg), "separator rows must be empty"

    counts = _marker_counts(g, start_row, bg)
    assert len(counts) == len(legend), "marker color count must match legend entry count"
    ranked_colors = _ranked_marker_colors(counts)
    marker_positions = {
        color: [(int(r), int(c)) for r, c in np.argwhere(g[start_row:, :] == color) + np.array([start_row, 0])]
        for color in ranked_colors
    }

    out = np.zeros_like(g)
    for color, coords in marker_positions.items():
        for r, c in coords:
            out[r, c] = int(color)

    for _col, outline_color, k in legend:
        target_color = ranked_colors[k - 1]
        for center in marker_positions[target_color]:
            _draw_hollow_square(out, center, k, outline_color)
            out[center] = int(target_color)

    return out


def _expected_from_semantics(
    shape: tuple[int, int],
    legend: list[tuple[int, int, int]],
    marker_positions: dict[int, list[tuple[int, int]]],
    ranked_colors: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, coords in marker_positions.items():
        for r, c in coords:
            out[r, c] = int(color)
    for _col, outline_color, k in legend:
        for center in marker_positions[ranked_colors[k - 1]]:
            _draw_hollow_square(out, center, k, outline_color)
            out[center] = ranked_colors[k - 1]
    return out


def _legend_columns(rng: np.random.Generator, n_entries: int, width: int) -> list[int]:
    gaps = [int(rng.integers(1, 4)) for _ in range(n_entries - 1)]
    span = 1 + sum(g + 1 for g in gaps)
    start = int(rng.integers(0, width - span + 1))
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)
    return cols


def _grid_profile(n_entries: int, total_markers: int, rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int]]]:
    radius = n_entries
    step = 2 * radius + 1
    top_margin = 2 * radius
    options: list[tuple[int, int, int, int, int, int]] = []
    for n_rows in range(2, 6):
        min_h = top_margin + (n_rows - 1) * step + radius + 1
        if min_h > 30:
            continue
        for n_cols in range(2, 6):
            min_w = radius + (n_cols - 1) * step + radius + 1
            if min_w > 30:
                continue
            if n_rows * n_cols >= total_markers:
                aspect = max(min_h, min_w) / min(min_h, min_w)
                if aspect <= 2.2:
                    options.append((n_rows, n_cols, min_h, min_w, abs(n_rows - n_cols), min_h * min_w))
    assert options, "expected at least one feasible lattice profile"

    best_area = min(area for *_rest, area in options)
    compact = [opt for opt in options if opt[-1] <= best_area + step * step]
    best_balance = min(delta for *_rest, delta, _area in compact)
    compact = [opt for opt in compact if opt[-2] <= best_balance + 1]
    n_rows, n_cols, min_h, min_w, _delta, _area = compact[int(rng.integers(len(compact)))]

    base_h = max(min_h, 12)
    base_w = max(min_w, 12)
    extra_h = int(rng.integers(0, min(4, 30 - base_h) + 1))
    extra_w = int(rng.integers(0, min(5, 30 - base_w) + 1))
    row_shift = int(rng.integers(0, extra_h + 1))
    col_shift = int(rng.integers(0, extra_w + 1))
    h = base_h + extra_h
    w = base_w + extra_w

    centers = [
        (top_margin + row_shift + r * step, radius + col_shift + c * step)
        for r in range(n_rows)
        for c in range(n_cols)
    ]
    return h, w, centers


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _parse_legend(g, bg)
    n_entries = len(legend)
    assert n_entries in (2, 3), "generator must emit 2 or 3 legend entries"
    assert sorted(k for _c, _color, k in legend) == list(range(1, n_entries + 1))

    start_row = _marker_region_top(legend)
    counts = _marker_counts(g, start_row, bg)
    assert len(counts) == n_entries
    assert len(set(counts.values())) == n_entries

    max_k = max(k for _c, _color, k in legend)
    marker_coords = [(int(r), int(c)) for r, c in np.argwhere(g[start_row:, :] != bg) + np.array([start_row, 0])]
    for r, c in marker_coords:
        assert max_k <= r < g.shape[0] - max_k
        assert max_k <= c < g.shape[1] - max_k
        neighborhood = g[r - max_k : r + max_k + 1, c - max_k : c + max_k + 1]
        assert int(np.count_nonzero(neighborhood != bg)) == 1, "marker neighborhoods must be isolated"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_entries = int(rng.choice([2, 3]))
    legend_colors = [int(v) for v in rng.choice(LEGEND_COLOR_POOL, size=n_entries, replace=False)]
    marker_colors = [int(v) for v in rng.choice(MARKER_COLOR_POOL, size=n_entries, replace=False)]

    base_counts = list(COUNT_PATTERNS[n_entries][int(rng.integers(len(COUNT_PATTERNS[n_entries])))])
    counts_for_colors = list(rng.permutation(base_counts))
    count_map = {color: int(count) for color, count in zip(marker_colors, counts_for_colors)}
    ranked_colors = [color for color, _count in sorted(count_map.items(), key=lambda item: (-item[1], item[0]))]

    total_markers = sum(counts_for_colors)
    h, w, centers = _grid_profile(n_entries, total_markers, rng)
    if len(centers) < total_markers:
        return None

    chosen_idx = rng.choice(len(centers), size=total_markers, replace=False)
    chosen_centers = [centers[int(i)] for i in chosen_idx]
    rng.shuffle(chosen_centers)

    marker_positions: dict[int, list[tuple[int, int]]] = {}
    cursor = 0
    for color in marker_colors:
        count = count_map[color]
        marker_positions[color] = chosen_centers[cursor : cursor + count]
        cursor += count

    grid = np.zeros((h, w), dtype=int)
    heights = list(rng.permutation(np.arange(1, n_entries + 1)))
    legend_entries: list[tuple[int, int, int]] = []
    for col, color, k in zip(_legend_columns(rng, n_entries, w), legend_colors, heights):
        grid[:k, col] = int(color)
        legend_entries.append((col, int(color), int(k)))

    for color, coords in marker_positions.items():
        for r, c in coords:
            grid[r, c] = int(color)

    try:
        _validate_input(grid)
    except AssertionError:
        return None

    expected = _expected_from_semantics(grid.shape, legend_entries, marker_positions, ranked_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
