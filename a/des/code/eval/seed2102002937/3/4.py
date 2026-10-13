from __future__ import annotations

import numpy as np


BG = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_TEMPLATE_SCAN = 6
GEN_RETRIES = 400
RNG = np.random.default_rng()


def _minimal_unit(seq: list[int]) -> list[int]:
    seq = [int(x) for x in seq]
    assert seq, "template sequence must be non-empty"
    for period in range(1, len(seq) + 1):
        unit = seq[:period]
        if all(color == unit[i % period] for i, color in enumerate(seq)):
            return unit
    raise AssertionError("failed to extract a repeating unit")


def _find_template_width(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert w >= 4, "grid must leave room for template and fill regions"

    scan_w = min(w - 1, MAX_TEMPLATE_SCAN)
    prefix = grid[:, :scan_w]
    used_cols = np.flatnonzero(np.any(prefix != BG, axis=0))
    assert used_cols.size > 0, "template zone not found"
    lower = int(used_cols[-1]) + 1
    if lower < w and bool(np.all(np.any(grid[:, :lower] != BG, axis=1))):
        return lower

    col_counts = np.count_nonzero(grid != BG, axis=0)
    dense_threshold = max(2, (h + 1) // 2)
    candidates: list[tuple[int, int, int]] = []
    for width in range(max(2, lower), scan_w + 1):
        if not bool(np.all(np.any(grid[:, :width] != BG, axis=1))):
            continue
        if not bool(np.any(grid[:, width - 1] != BG)):
            continue
        dense_suffix = int(np.count_nonzero(col_counts[width:] >= dense_threshold))
        candidates.append((dense_suffix, width, -int(np.count_nonzero(grid[:, width - 1] != BG))))

    if candidates:
        return min(candidates)[1]
    return lower


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_w = _find_template_width(g)
    assert 0 < template_w < g.shape[1], "bad template width"

    out = g.copy()
    active_cols = np.any(g[:, template_w:] != BG, axis=0)
    for r in range(g.shape[0]):
        seq = [int(color) for color in g[r, :template_w] if int(color) != BG]
        unit = _minimal_unit(seq)
        for offset, is_active in enumerate(active_cols):
            if not bool(is_active):
                continue
            c = template_w + offset
            if out[r, c] == BG:
                out[r, c] = unit[offset % len(unit)]

    return out


def _sample_periods(height: int, template_w: int) -> list[int]:
    max_period = min(3, template_w)
    periods = RNG.integers(1, max_period + 1, size=height).tolist()
    if max_period > 1 and len(set(periods)) == 1:
        idx = int(RNG.integers(height))
        choices = [p for p in range(1, max_period + 1) if p != periods[idx]]
        periods[idx] = int(RNG.choice(choices))
    return [int(p) for p in periods]


def _sample_seq_len(period: int, template_w: int) -> int:
    min_len = max(period, 2)
    choices = list(range(min_len, template_w + 1))
    non_multiples = [length for length in choices if length % period != 0]
    if non_multiples and RNG.random() < 0.5:
        return int(RNG.choice(non_multiples))
    return int(RNG.choice(choices))


def _make_template_row(template_w: int, period: int) -> tuple[np.ndarray, list[int]]:
    unit = RNG.choice(COLOR_POOL, size=period, replace=False).astype(int).tolist()
    seq_len = _sample_seq_len(period, template_w)
    seq = [unit[i % period] for i in range(seq_len)]

    row = np.zeros(template_w, dtype=int)
    if seq_len == template_w:
        positions = list(range(template_w))
    else:
        middle = sorted(
            RNG.choice(np.arange(1, template_w - 1), size=seq_len - 2, replace=False).astype(int).tolist()
        )
        positions = [0, *middle, template_w - 1]
    for pos, color in zip(positions, seq):
        row[pos] = int(color)

    return row, unit


def _expected_output(grid: np.ndarray, template_w: int, row_units: list[list[int]]) -> np.ndarray:
    out = grid.copy()
    active_cols = np.any(grid[:, template_w:] != BG, axis=0)
    for r, unit in enumerate(row_units):
        for offset, is_active in enumerate(active_cols):
            if not bool(is_active):
                continue
            c = template_w + offset
            if out[r, c] == BG:
                out[r, c] = int(unit[offset % len(unit)])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(4, 10))
    template_w = int(RNG.choice([3, 4, 5]))
    fill_w = int(RNG.integers(6, 13))
    width = template_w + fill_w

    grid = np.zeros((height, width), dtype=int)
    periods = _sample_periods(height, template_w)
    row_units: list[list[int]] = []
    template_colors_used: set[int] = set()

    for r, period in enumerate(periods):
        row, unit = _make_template_row(template_w, period)
        grid[r, :template_w] = row
        row_units.append(unit)
        template_colors_used.update(map(int, unit))

    active_count = int(RNG.integers(2, fill_w))
    active_offsets = sorted(RNG.choice(np.arange(fill_w), size=active_count, replace=False).astype(int).tolist())
    marker_palette = [color for color in COLOR_POOL.tolist() if color not in template_colors_used]
    if not marker_palette:
        marker_palette = COLOR_POOL.tolist()

    max_markers_per_col = min(height, 3)
    for offset in active_offsets:
        marker_count = int(RNG.integers(1, max_markers_per_col + 1))
        rows = RNG.choice(np.arange(height), size=marker_count, replace=False).astype(int).tolist()
        for r in rows:
            grid[r, template_w + offset] = int(RNG.choice(marker_palette))

    expected = _expected_output(grid, template_w, row_units)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
