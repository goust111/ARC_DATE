from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_mask, bbox_w, colors_present, find_components, get_bg


def _true_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = int(len(flags))
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]):
            j += 1
        intervals.append((int(i), int(j)))
        i = j + 1
    return intervals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the empty gutters of a rectangular lattice of grey 2x2 blocks.

    Input structure:
    - The background is the most common color.
    - The only foreground color forms a complete rectangular array of separated 2x2 blocks.
    - Blank row bands and column bands between adjacent block bands are the lattice gutters.

    Rule:
    - Keep the grey blocks unchanged.
    - Paint gutter background pixels red inside the foreground bounding rectangle.
    - Paint gutter background pixels blue where the same gutter rows or columns extend outside that bounding rectangle.

    Key insight:
    - The grey blocks define row bands and column bands; their intervening blank bands are the lines to color, and the block-array bbox separates internal gutters from exterior extensions.
    """
    g = np.asarray(grid, dtype=int)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1
    block_color = int(colors[0])

    block_mask = g == block_color
    block_bbox = bbox(block_mask)
    assert block_bbox is not None
    inside_block_bbox = bbox_mask(block_mask)

    components = find_components(g, bg=bg, connectivity=4, colors=[block_color])
    assert components
    for component in components:
        assert int(component["area"]) == 4
        assert bbox_h(component["bbox"]) == 2
        assert bbox_w(component["bbox"]) == 2

    row_bands = _true_intervals(block_mask.any(axis=1))
    col_bands = _true_intervals(block_mask.any(axis=0))
    assert len(row_bands) >= 2
    assert len(col_bands) >= 2
    assert all(e - s + 1 == 2 for s, e in row_bands)
    assert all(e - s + 1 == 2 for s, e in col_bands)

    row_band_mask = np.zeros(g.shape[0], dtype=bool)
    col_band_mask = np.zeros(g.shape[1], dtype=bool)
    for s, e in row_bands:
        row_band_mask[s : e + 1] = True
    for s, e in col_bands:
        col_band_mask[s : e + 1] = True
    assert np.array_equal(block_mask, row_band_mask[:, None] & col_band_mask[None, :])

    row_gutters = np.zeros(g.shape[0], dtype=bool)
    col_gutters = np.zeros(g.shape[1], dtype=bool)
    for (_prev_s, prev_e), (next_s, _next_e) in zip(row_bands, row_bands[1:]):
        assert next_s > prev_e + 1
        row_gutters[prev_e + 1 : next_s] = True
    for (_prev_s, prev_e), (next_s, _next_e) in zip(col_bands, col_bands[1:]):
        assert next_s > prev_e + 1
        col_gutters[prev_e + 1 : next_s] = True
    assert bool(row_gutters.any())
    assert bool(col_gutters.any())

    gutter_mask = (row_gutters[:, None] | col_gutters[None, :]) & (g == bg)

    out = g.copy()
    out[gutter_mask & inside_block_bbox] = Color.RED
    out[gutter_mask & ~inside_block_bbox] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas and color model.
GRID_SIZE = 10
BACKGROUND_COLOR = 0
BLOCK_COLOR = 5
RED_GUTTER_COLOR = 2
BLUE_GUTTER_COLOR = 1

# Lattice geometry.
BLOCK_SIZE = 2
MIN_BANDS = 2
MAX_BANDS = 3
MIN_GUTTER_WIDTH = 1
MAX_GUTTER_WIDTH = 2
MAX_SIDE_MARGIN = 4
MIN_TOTAL_MARGIN = 1

# Rejection sampling.
GEN_RETRIES = 300


def _gen_axis_layout(rng: np.random.Generator, n_bands: int) -> tuple[list[int], int]:
    gutter_widths = [
        int(rng.integers(MIN_GUTTER_WIDTH, MAX_GUTTER_WIDTH + 1))
        for _ in range(n_bands - 1)
    ]
    margin = GRID_SIZE - n_bands * BLOCK_SIZE - sum(gutter_widths)
    if margin < 0:
        return [], -1

    min_before = max(0, margin - MAX_SIDE_MARGIN)
    max_before = min(MAX_SIDE_MARGIN, margin)
    before = int(rng.integers(min_before, max_before + 1))

    starts: list[int] = []
    pos = before
    for i in range(n_bands):
        starts.append(pos)
        pos += BLOCK_SIZE
        if i < len(gutter_widths):
            pos += gutter_widths[i]
    return starts, margin


def _gen_gutter_masks(block_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row and column masks for blank lattice gutters between block bands."""
    row_bands = _true_intervals(block_mask.any(axis=1))
    col_bands = _true_intervals(block_mask.any(axis=0))
    row_gutters = np.zeros(block_mask.shape[0], dtype=bool)
    col_gutters = np.zeros(block_mask.shape[1], dtype=bool)

    for bands, gutters in ((row_bands, row_gutters), (col_bands, col_gutters)):
        for (_prev_s, prev_e), (next_s, _next_e) in zip(bands, bands[1:]):
            gutters[prev_e + 1 : next_s] = True
    return row_gutters, col_gutters


def _render_lattice(row_starts: list[int], col_starts: list[int]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    for r in row_starts:
        for c in col_starts:
            grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = BLOCK_COLOR
    return grid


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    block_mask = inp == BLOCK_COLOR
    row_gutters, col_gutters = _gen_gutter_masks(block_mask)
    inside = bbox_mask(block_mask)

    gutter_mask = (row_gutters[:, None] | col_gutters[None, :]) & (inp == BACKGROUND_COLOR)
    out = inp.copy()
    out[gutter_mask & inside] = RED_GUTTER_COLOR
    out[gutter_mask & ~inside] = BLUE_GUTTER_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_row_bands = int(rng.integers(MIN_BANDS, MAX_BANDS + 1))
    n_col_bands = int(rng.integers(MIN_BANDS, MAX_BANDS + 1))

    row_starts, row_margin = _gen_axis_layout(rng, n_row_bands)
    col_starts, col_margin = _gen_axis_layout(rng, n_col_bands)
    if row_margin < 0 or col_margin < 0 or row_margin + col_margin < MIN_TOTAL_MARGIN:
        return None

    return _render_lattice(row_starts, col_starts)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, _gen_expected_output(inp)):
            return inp
    raise RuntimeError("failed to generate valid 137f0df0 input")
