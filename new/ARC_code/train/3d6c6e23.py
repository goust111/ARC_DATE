import math

import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move each vertical stream of pixels into a bottom-aligned centered pile.

    Input structure
    - A mostly background grid contains one or more vertical source columns of
      foreground pixels.
    - Each source column has n^2 foreground pixels, and the top-to-bottom color
      sequence splits into monochrome groups of lengths 1, 3, 5, ...

    Rule
    - Clear the canvas to background.
    - For each source column, read the foreground colors from top to bottom and
      rebuild them at the bottom below the same column as odd-width centered
      rows of lengths 1, 3, 5, ...

    Key insight
    - The scattered vertical pixels are falling grains; a square number of
      grains decomposes into the odd row widths of a centered pyramid.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    source_cols: list[tuple[int, list[int]]] = []
    for c in range(W):
        sequence = [int(g[r, c]) for r in range(H) if int(g[r, c]) != bg]
        if sequence:
            source_cols.append((c, sequence))

    assert source_cols

    out = np.full_like(g, bg)
    for center_c, sequence in source_cols:
        height = math.isqrt(len(sequence))
        assert height * height == len(sequence)
        assert height <= H

        idx = 0
        for layer_idx in range(height):
            width = 2 * layer_idx + 1
            layer = sequence[idx : idx + width]
            assert len(layer) == width
            assert len(set(layer)) == 1

            r = H - height + layer_idx
            c0 = center_c - layer_idx
            c1 = center_c + layer_idx
            assert 0 <= c0 <= c1 < W
            assert np.all(out[r, c0 : c1 + 1] == bg)

            out[r, c0 : c1 + 1] = layer[0]
            idx += width

        assert idx == len(sequence)

    return out

# --------------- generator (att1, Claude) ---------------

# Canvas and palette bounds.
GRID_HEIGHT_RANGE = (16, 30)
GRID_WIDTH_RANGE = (9, 13)
BACKGROUND_COLOR = 0
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Source-column, pyramid, placement, and retry choices.
SOURCE_COLUMN_COUNTS = [1, 2]
PYRAMID_HEIGHTS = [2, 3]
MIN_FOOTPRINT_GAP = 1
MAX_GENERATION_ATTEMPTS = 300


def _second_center_candidates(lo, hi, first_center, first_height, height):
    first_radius = first_height - 1
    radius = height - 1
    right_edge = first_center + first_radius
    left_edge = first_center - first_radius

    candidates = []
    right_lo = max(lo, right_edge + radius + MIN_FOOTPRINT_GAP)
    if right_lo <= hi:
        candidates.extend(range(right_lo, hi + 1))
    left_hi = min(hi, left_edge - radius - MIN_FOOTPRINT_GAP)
    if left_hi >= lo:
        candidates.extend(range(lo, left_hi + 1))
    return candidates


def _choose_centers(width, heights):
    """Pick source columns whose bottom pyramid footprints do not overlap."""
    centers = []
    for i, height in enumerate(heights):
        margin = height - 1
        lo, hi = margin, width - 1 - margin
        if lo > hi:
            return None
        if i == 0:
            centers.append(random.randint(lo, hi))
            continue

        candidates = _second_center_candidates(lo, hi, centers[0], heights[0], height)
        if not candidates:
            return None
        centers.append(random.choice(candidates))
    return centers


def _flatten_layers(colors):
    sequence = []
    for layer_idx, color in enumerate(colors):
        sequence.extend([color] * (2 * layer_idx + 1))
    return sequence


def _build_pyramid_data(centers, heights):
    """Return (center_col, height, flattened_layer_sequence) tuples."""
    pyramid_data = []
    for center, height in zip(centers, heights):
        colors = [random.choice(FOREGROUND_COLORS) for _ in range(height)]
        pyramid_data.append((center, height, _flatten_layers(colors)))
    return pyramid_data


def _draw_pyramids(shape, pyramid_data):
    """Render bottom-aligned output pyramids from flattened layer sequences."""
    H, W = shape
    out = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    for center, height, sequence in pyramid_data:
        idx = 0
        for layer_idx in range(height):
            width = 2 * layer_idx + 1
            r = H - height + layer_idx
            c0 = center - layer_idx
            out[r, c0:c0 + width] = sequence[idx]
            idx += width
    return out


def _scatter_source_columns(shape, pyramid_data):
    """Scatter each flattened layer sequence above its pyramid top row."""
    H, W = shape
    inp = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    for center, height, sequence in pyramid_data:
        top_pyramid_row = H - height
        num_pixels = height * height
        available_rows = list(range(top_pyramid_row))
        if len(available_rows) < num_pixels:
            return None
        chosen_rows = sorted(random.sample(available_rows, num_pixels))
        for row, color in zip(chosen_rows, sequence):
            inp[row, center] = color
    return inp


def _gen_attempt():
    """Generate a single input grid for the pyramid-scatter task."""
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    K = random.choice(SOURCE_COLUMN_COUNTS)
    heights = [random.choice(PYRAMID_HEIGHTS) for _ in range(K)]

    centers = _choose_centers(W, heights)
    if centers is None:
        return None

    pyramid_data = _build_pyramid_data(centers, heights)
    out = _draw_pyramids((H, W), pyramid_data)
    inp = _scatter_source_columns((H, W), pyramid_data)
    if inp is not None:
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            pass
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input in {MAX_GENERATION_ATTEMPTS} attempts")
