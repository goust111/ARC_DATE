from __future__ import annotations

import numpy as np


BLACK = 0
PINK = 6
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4, 5, 8)
TILE_REPEATS = (3, 3)
MAX_GENERATION_ATTEMPTS = 400


def _extract_bands(is_divider: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal non-divider intervals as [start, end) bands."""
    bands: list[tuple[int, int]] = []
    i = 0
    n = int(is_divider.shape[0])
    while i < n:
        while i < n and bool(is_divider[i]):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not bool(is_divider[j]):
            j += 1
        bands.append((i, j))
        i = j
    return bands


def _compress_regions(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_bands = _extract_bands(np.all(g == BLACK, axis=1))
    col_bands = _extract_bands(np.all(g == BLACK, axis=0))
    assert row_bands and col_bands, "expected at least one region band on each axis"

    out = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            colors = np.unique(region)
            colors = colors[colors != BLACK]
            assert colors.size <= 1, "each region must contain at most one non-black color"
            if colors.size == 1:
                out[rr, cc] = int(colors[0])
    return out


def _annotate_shadow(compressed: np.ndarray) -> np.ndarray:
    tile = np.asarray(compressed, dtype=int).copy()
    rows, cols = tile.shape
    for r in range(1, rows):
        for c in range(cols):
            if int(compressed[r, c]) == BLACK:
                continue
            tr, tc = r - 1, (c - 1) % cols
            tile[tr, tc] = ORANGE if int(compressed[tr, tc]) != BLACK else PINK
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress divider-separated regions, mark an up-left wrapping shadow on the compressed grid, then tile it 3x3.

    Input structure:
    - The grid is partitioned by one-cell-thick full-black divider rows and columns into rectangular regions.
    - Each region is either empty black or contains scattered pixels of exactly one non-black color.
    - Colors 6 and 7 are absent from the input; the divider geometry defines the compressed grid layout.

    Rule:
    - Locate every full-black divider row and column, and compress each rectangular region to one output cell.
    - A region compresses to its unique non-black color, or to black if the region is empty.
    - For each non-black compressed cell at (r, c) with r > 0, project a shadow to (r-1, (c-1) mod W):
      place pink (6) on black targets, or orange (7) when the shadow lands on an existing compressed foreground cell.
    - Tile the shadow-annotated compressed grid three times vertically and three times horizontally.

    Key insight:
    - The real tile is the region-level compressed layout, and the only wrap is horizontal, so overlap sites in the
      compressed grid become orange markers before the whole annotated tile is repeated.
    """
    compressed = _compress_regions(np.asarray(grid, dtype=int))
    tile = _annotate_shadow(compressed)
    return np.tile(tile, TILE_REPEATS)


def _mask_has_required_structure(mask: np.ndarray) -> bool:
    rows, cols = mask.shape
    if rows < 2 or cols < 2:
        return False
    if not np.all(mask.any(axis=1)):
        return False
    if not np.all(mask.any(axis=0)):
        return False
    if int(mask.sum()) < 3:
        return False

    merge_exists = False
    pink_exists = False
    wrap_used = False
    for r in range(1, rows):
        for c in range(cols):
            if not bool(mask[r, c]):
                continue
            tr, tc = r - 1, (c - 1) % cols
            if c == 0:
                wrap_used = True
            if bool(mask[tr, tc]):
                merge_exists = True
            else:
                pink_exists = True
    return merge_exists and pink_exists and wrap_used


def _random_compressed_grid(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rows = int(rng.integers(2, 6))
        cols = int(rng.integers(2, 6))
        p = float(rng.uniform(0.35, 0.7))
        mask = rng.random((rows, cols)) < p
        if not _mask_has_required_structure(mask):
            continue

        compressed = np.zeros((rows, cols), dtype=int)
        coords = np.argwhere(mask)
        colors = rng.choice(np.array(INPUT_COLORS, dtype=int), size=len(coords), replace=True)
        if len(coords) >= 2 and len(set(map(int, colors))) == 1:
            idx = int(rng.integers(len(coords)))
            alternatives = [c for c in INPUT_COLORS if c != int(colors[idx])]
            colors[idx] = int(rng.choice(np.array(alternatives, dtype=int)))
        for (r, c), color in zip(coords, colors):
            compressed[int(r), int(c)] = int(color)

        tile = _annotate_shadow(compressed)
        if not np.any(tile == PINK):
            continue
        if not np.any(tile == ORANGE):
            continue
        return compressed
    raise RuntimeError("failed to generate a valid compressed grid")


def _region_sprite(height: int, width: int, color: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse region pattern that still covers every local row and column."""
    sprite = np.zeros((height, width), dtype=int)
    used_cols: set[int] = set()
    for r in range(height):
        c = int(rng.integers(width))
        sprite[r, c] = color
        used_cols.add(c)

    for c in range(width):
        if c in used_cols:
            continue
        r = int(rng.integers(height))
        sprite[r, c] = color

    candidates = [(r, c) for r in range(height) for c in range(width) if sprite[r, c] == BLACK]
    rng.shuffle(candidates)
    extra = int(rng.integers(0, max(1, (height * width) // 3) + 1))
    for r, c in candidates[:extra]:
        sprite[r, c] = color
    return sprite


def _render_input(compressed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows, cols = compressed.shape
    max_region_h = min(5, (30 - (rows - 1)) // rows)
    max_region_w = min(5, (30 - (cols - 1)) // cols)
    assert max_region_h >= 2 and max_region_w >= 2

    heights = [int(rng.integers(2, max_region_h + 1)) for _ in range(rows)]
    widths = [int(rng.integers(2, max_region_w + 1)) for _ in range(cols)]

    grid_h = sum(heights) + (rows - 1)
    grid_w = sum(widths) + (cols - 1)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    r0 = 0
    for rr, h in enumerate(heights):
        c0 = 0
        for cc, w in enumerate(widths):
            color = int(compressed[rr, cc])
            if color != BLACK:
                grid[r0 : r0 + h, c0 : c0 + w] = _region_sprite(h, w, color, rng)
            c0 += w + 1
        r0 += h + 1

    recon = _compress_regions(grid)
    assert recon.shape == compressed.shape
    assert np.array_equal(recon, compressed), "rendered grid must compress back exactly"
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        compressed = _random_compressed_grid(rng)
        grid = _render_input(compressed, rng)
        try:
            out = main(grid)
        except Exception:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if not np.any(out == PINK) or not np.any(out == ORANGE):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input grid")
