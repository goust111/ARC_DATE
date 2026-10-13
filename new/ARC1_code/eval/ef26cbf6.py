from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    dividers = set(int(i) for i in divider_indices)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(length):
        if i in dividers:
            if start is not None:
                runs.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _full_divider_indices(g: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    row_dividers = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
    col_dividers = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
    return row_dividers, col_dividers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each mask region in a divider table using the singleton marker color in its aligned row or column.

    Input structure
    - The dominant color is the background.
    - One foreground color forms complete horizontal and/or vertical divider lines, partitioning the grid into rectangular cells.
    - One repeated foreground color is the mask color to be recolored.
    - Each marker cell contains exactly one non-background, non-divider, non-mask pixel.

    Rule
    - Split the grid into rectangular cells between the divider lines.
    - For every cell containing mask pixels, find the unique marker cell that shares its table row or table column.
    - Replace the mask pixels in that cell with the aligned marker color.
    - Preserve the background, dividers, and marker pixels.

    Key insight
    - The divider table makes the marker-to-mask assignment positional: matching is by shared row or column, not by the mask shape itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    divider_candidates = []
    for color in colors_present(g, exclude={bg}):
        rows, cols = _full_divider_indices(g, int(color))
        if rows or cols:
            divider_candidates.append(int(color))
    assert len(divider_candidates) == 1, "expected one color forming full divider lines"
    divider = divider_candidates[0]

    row_dividers, col_dividers = _full_divider_indices(g, divider)
    row_runs = _non_divider_runs(g.shape[0], row_dividers)
    col_runs = _non_divider_runs(g.shape[1], col_dividers)
    assert row_runs and col_runs, "divider lines must leave at least one cell"

    regions = [
        (ri, ci, r0, r1, c0, c1, g[r0 : r1 + 1, c0 : c1 + 1])
        for ri, (r0, r1) in enumerate(row_runs)
        for ci, (c0, c1) in enumerate(col_runs)
    ]

    role_colors = colors_present(g, exclude={bg, divider})
    mask_candidates = [
        int(color)
        for color in role_colors
        if any(int(np.sum(region == int(color))) > 1 for *_, region in regions)
    ]
    assert len(mask_candidates) == 1, "expected one repeated mask color inside table cells"
    mask_color = mask_candidates[0]

    markers: dict[tuple[int, int], int] = {}
    mask_regions: list[tuple[int, int, int, int, int, int]] = []
    for ri, ci, r0, r1, c0, c1, region in regions:
        has_mask = bool(np.any(region == mask_color))
        marker_colors = colors_present(region, exclude={bg, divider, mask_color})
        if has_mask:
            assert not marker_colors, "mask cells must not also contain marker pixels"
            mask_regions.append((ri, ci, r0, r1, c0, c1))
        elif marker_colors:
            assert len(marker_colors) == 1, "each marker cell must have one marker color"
            marker_color = int(marker_colors[0])
            assert int(np.sum(region == marker_color)) == 1, "marker color must be a singleton in its cell"
            markers[(ri, ci)] = marker_color

    assert mask_regions, "expected at least one mask cell"
    assert markers, "expected at least one marker cell"

    out = g.copy()
    for ri, ci, r0, r1, c0, c1 in mask_regions:
        aligned_markers = [
            marker_color
            for (marker_ri, marker_ci), marker_color in markers.items()
            if marker_ri == ri or marker_ci == ci
        ]
        assert len(aligned_markers) == 1, "each mask cell must have one aligned marker"
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        region_out[region_out == mask_color] = aligned_markers[0]

    return out

# --------------- generator (o5x) ---------------

# Colors.
_GEN_BG = 0
_GEN_MASK_COLOR = 1
_GEN_DIVIDER = 4
_GEN_MARKER_COLORS = [2, 3, 5, 6, 7, 8, 9]

# Layout choices: (marker-mask pairs, cell size), limited by the 7-11 grid bound.
_GEN_LAYOUT_OPTIONS = [(2, 3), (3, 3), (2, 5)]
_GEN_LAYOUT_WEIGHTS = [1, 2, 1]
_GEN_PAIR_AXIS_BANDS = 2

# Mask pattern density.
_GEN_MIN_MASK_PIXELS = 3
_GEN_MAX_MASK_PIXELS = 7

# Retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_cell_origin(index: int, cell_size: int) -> int:
    return int(index) * (int(cell_size) + 1)


def _gen_draw_dividers(grid: np.ndarray, n_rows: int, n_cols: int, cell_size: int) -> None:
    for ri in range(1, n_rows):
        grid[ri * (cell_size + 1) - 1, :] = _GEN_DIVIDER
    for ci in range(1, n_cols):
        grid[:, ci * (cell_size + 1) - 1] = _GEN_DIVIDER


def _gen_choose_layout() -> tuple[int, int, int, int]:
    n_pairs, cell_size = random.choices(_GEN_LAYOUT_OPTIONS, weights=_GEN_LAYOUT_WEIGHTS, k=1)[0]
    if random.choice([True, False]):
        return n_pairs, n_pairs, _GEN_PAIR_AXIS_BANDS, cell_size
    return n_pairs, _GEN_PAIR_AXIS_BANDS, n_pairs, cell_size


def _gen_pair_cells(pair_i: int, n_rows: int, n_pairs: int, marker_first_band: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_rows == n_pairs:
        marker_col = 0 if marker_first_band else 1
        return (pair_i, marker_col), (pair_i, 1 - marker_col)

    marker_row = 0 if marker_first_band else 1
    return (marker_row, pair_i), (1 - marker_row, pair_i)


def _gen_pattern_positions(cell_size: int, pattern_index: int, signatures: list[int]) -> list[int]:
    total = cell_size * cell_size
    count = random.randint(_GEN_MIN_MASK_PIXELS, min(_GEN_MAX_MASK_PIXELS, total - 2))
    signature = signatures[pattern_index]
    non_signature_positions = [pos for pos in range(total) if pos not in signatures]
    return [signature] + random.sample(non_signature_positions, count - 1)


def _gen_render_tables(n_pairs: int, n_rows: int, n_cols: int, cell_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Render matching input/output tables for one sampled layout."""
    height = n_rows * cell_size + n_rows - 1
    width = n_cols * cell_size + n_cols - 1
    output = np.full((height, width), _GEN_BG, dtype=int)
    _gen_draw_dividers(output, n_rows, n_cols, cell_size)
    inp = output.copy()

    marker_colors = random.sample(_GEN_MARKER_COLORS, n_pairs)
    marker_first_band = random.choice([True, False])
    signatures = random.sample(range(cell_size * cell_size), n_pairs)

    for pair_i, marker_color in enumerate(marker_colors):
        marker_cell, mask_cell = _gen_pair_cells(pair_i, n_rows, n_pairs, marker_first_band)
        marker_r = _gen_cell_origin(marker_cell[0], cell_size) + cell_size // 2
        marker_c = _gen_cell_origin(marker_cell[1], cell_size) + cell_size // 2
        output[marker_r, marker_c] = marker_color
        inp[marker_r, marker_c] = marker_color

        mask_r0 = _gen_cell_origin(mask_cell[0], cell_size)
        mask_c0 = _gen_cell_origin(mask_cell[1], cell_size)
        for pos in _gen_pattern_positions(cell_size, pair_i, signatures):
            dr, dc = divmod(pos, cell_size)
            output[mask_r0 + dr, mask_c0 + dc] = marker_color
            inp[mask_r0 + dr, mask_c0 + dc] = _GEN_MASK_COLOR

    return inp, output


def _gen_attempt() -> np.ndarray | None:
    n_pairs, n_rows, n_cols, cell_size = _gen_choose_layout()
    inp, output = _gen_render_tables(n_pairs, n_rows, n_cols, cell_size)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
