from __future__ import annotations

from collections import Counter

import numpy as np


ARC_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 500
DIMENSION_CYCLE = (
    (3, 4),
    (4, 3),
    (5, 5),
    (4, 4),
    (5, 3),
    (3, 5),
    (5, 4),
    (4, 5),
)

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _runs_between_stops(length: int, stops: list[int]) -> list[tuple[int, int]]:
    stop_set = set(map(int, stops))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(int(length)):
        if i in stop_set:
            if start is not None:
                runs.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _compress_regions(grid: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert max(height, width) <= 30, "ARC grids must be at most 30x30"

    candidates: list[tuple[np.ndarray, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(g)):
        sep_rows = [r for r in range(height) if bool(np.all(g[r, :] == color))]
        sep_cols = [c for c in range(width) if bool(np.all(g[:, c] == color))]
        if not sep_rows or not sep_cols:
            continue

        sep_mask = np.zeros(g.shape, dtype=bool)
        sep_mask[sep_rows, :] = True
        sep_mask[:, sep_cols] = True
        if not np.array_equal(g == color, sep_mask):
            continue

        row_runs = _runs_between_stops(height, sep_rows)
        col_runs = _runs_between_stops(width, sep_cols)
        if len(row_runs) < 2 or len(col_runs) < 2:
            continue

        compact = np.empty((len(row_runs), len(col_runs)), dtype=int)
        ok = True
        for rr, (r0, r1) in enumerate(row_runs):
            for cc, (c0, c1) in enumerate(col_runs):
                block = g[r0 : r1 + 1, c0 : c1 + 1]
                values = np.unique(block)
                if len(values) != 1 or int(values[0]) == color:
                    ok = False
                    break
                compact[rr, cc] = int(values[0])
            if not ok:
                break
        if ok:
            candidates.append((compact, int(color), row_runs, col_runs))

    assert len(candidates) == 1, "expected one full-row/full-column separator partition"
    return candidates[0]


def _repair_compact(compact: np.ndarray) -> np.ndarray:
    colors = np.asarray(compact, dtype=int)
    rows, cols = map(int, colors.shape)
    assert rows >= 2 and cols >= 2, "expected a rectangular region map"

    mismatches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            mate = (rows - 1 - r, cols - 1 - c)
            if (r, c) >= mate:
                continue
            if int(colors[r, c]) != int(colors[mate]):
                mismatches.append(((r, c), mate))

    assert len(mismatches) == 1, "expected exactly one broken 180-degree pair"
    (a_r, a_c), (b_r, b_c) = mismatches[0]
    a_color = int(colors[a_r, a_c])
    b_color = int(colors[b_r, b_c])
    counts = Counter(map(int, colors.ravel()))

    a_is_bad = counts[a_color] == 1 and counts[b_color] > 1
    b_is_bad = counts[b_color] == 1 and counts[a_color] > 1
    assert a_is_bad != b_is_bad, "exactly one mismatched color must be globally unique"

    out = colors.copy()
    if a_is_bad:
        out[a_r, a_c] = b_color
    else:
        out[b_r, b_c] = a_color
    return out.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress an uneven separator partition, then repair the one broken 180-degree symmetric region.

    Input structure:
    - Full horizontal and vertical separator bars divide the grid into monochrome rectangular regions.
    - The region row heights and column widths may differ, so raw pixel geometry is not the symmetry frame.
    - After compression to one cell per region, the region-color map has one off-color cell breaking
      180-degree point symmetry; that off color occurs exactly once in the compact map.

    Rule:
    - Infer the separator color from the full separator rows and columns, and collapse each rectangle
      between them to its single color.
    - Find the unique 180-degree symmetric pair whose colors differ.
    - In that pair, replace the globally unique color with its partner's repeated color and output the
      repaired compact region map.

    Key insight:
    - The relevant symmetry exists only after rectangular partitions are compressed; the unique color in
      the lone mismatched pair identifies which side is damaged.
    """
    compact, _sep_color, _row_runs, _col_runs = _compress_regions(grid)
    return _repair_compact(compact)


def _symmetry_pairs(rows: int, cols: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int] | None]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    center: tuple[int, int] | None = None
    for r in range(rows):
        for c in range(cols):
            mate = (rows - 1 - r, cols - 1 - c)
            if (r, c) > mate:
                continue
            if (r, c) == mate:
                center = (r, c)
            else:
                pairs.append(((r, c), mate))
    return pairs, center


def _random_sizes(count: int, rng: np.random.Generator) -> list[int]:
    for _ in range(200):
        sizes = list(map(int, rng.integers(2, 6, size=int(count))))
        if len(set(sizes)) > 1 and sizes != list(reversed(sizes)):
            return sizes
    raise RuntimeError("failed to sample varied non-palindromic region sizes")


def _make_compact_pair(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    call_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    palette = list(map(int, rng.permutation(ARC_COLORS)))
    sep_color = palette[0]
    anomaly_color = palette[1]
    region_colors = palette[2:]
    assert len(region_colors) >= 2

    pairs, center = _symmetry_pairs(rows, cols)
    assert len(pairs) >= 2
    anomaly_pair_idx = int(rng.integers(len(pairs)))
    extra_choices = [idx for idx in range(len(pairs)) if idx != anomaly_pair_idx]
    extra_pair_idx = int(rng.choice(extra_choices))
    partner_color = int(rng.choice(region_colors))

    repaired = np.empty((rows, cols), dtype=int)
    for idx, (a, b) in enumerate(pairs):
        color = partner_color if idx == extra_pair_idx else int(rng.choice(region_colors))
        repaired[a] = color
        repaired[b] = color

    if center is not None:
        repaired[center] = partner_color

    corrupt = repaired.copy()
    anomaly_pair = pairs[anomaly_pair_idx]
    anomaly_pos = anomaly_pair[call_index % 2]
    partner_pos = anomaly_pair[1 - (call_index % 2)]
    repaired[anomaly_pos] = partner_color
    repaired[partner_pos] = partner_color
    corrupt[partner_pos] = partner_color
    corrupt[anomaly_pos] = anomaly_color

    counts = Counter(map(int, corrupt.ravel()))
    assert counts[anomaly_color] == 1
    assert counts[partner_color] > 1
    unique_colors = [color for color, count in counts.items() if count == 1]
    assert unique_colors == [anomaly_color]
    assert np.array_equal(_repair_compact(corrupt), repaired)
    return corrupt, repaired, sep_color


def _render_partition(
    compact: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    sep_color: int,
) -> np.ndarray:
    rows, cols = map(int, compact.shape)
    height = sum(row_heights) + rows - 1
    width = sum(col_widths) + cols - 1
    grid = np.full((height, width), int(sep_color), dtype=int)

    r0 = 0
    for rr, block_h in enumerate(row_heights):
        c0 = 0
        for cc, block_w in enumerate(col_widths):
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = int(compact[rr, cc])
            c0 += block_w + 1
        r0 += block_h + 1
    return grid


def _gen_attempt(call_index: int) -> np.ndarray | None:
    rng = _RNG
    rows, cols = DIMENSION_CYCLE[call_index % len(DIMENSION_CYCLE)]
    if rng.random() < 0.25:
        rows = int(rng.integers(3, 6))
        cols = int(rng.integers(3, 6))

    row_heights = _random_sizes(rows, rng)
    col_widths = _random_sizes(cols, rng)
    corrupt, repaired, sep_color = _make_compact_pair(rows, cols, rng, call_index)
    grid = _render_partition(corrupt, row_heights, col_widths, sep_color)
    if max(grid.shape) > 30:
        return None

    compact, found_sep, found_rows, found_cols = _compress_regions(grid)
    if found_sep != sep_color:
        return None
    if [r1 - r0 + 1 for r0, r1 in found_rows] != row_heights:
        return None
    if [c1 - c0 + 1 for c0, c1 in found_cols] != col_widths:
        return None
    if not np.array_equal(compact, corrupt):
        return None
    if not np.array_equal(main(grid), repaired):
        return None
    return grid.astype(int)


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    call_index = _GEN_COUNTER
    _GEN_COUNTER += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid separator-partition symmetry input")
