from __future__ import annotations

from collections import Counter

import numpy as np
import random


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j < len(mask) and bool(mask[j]):
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _background_color(g: np.ndarray) -> int:
    edge = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    vals = np.unique(edge)
    assert vals.size == 1, "expected a single blank border background color"
    return int(vals[0])


def _dominant_foreground_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g[g != bg]))
    assert counts, "expected foreground pixels"
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "expected a unique repeated motif color"
    return int(best[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored slot map inside the marked motif and rebuild the motif lattice with those colors.

    Input structure:
    - A blank background surrounds a rectangular lattice of identical motifs.
    - The motif rows and columns are separated by one background line.
    - One motif has colored marker pixels in its internal odd-row/odd-column slots.

    Rule:
    - The motif size is 2R+1 by 2C+1, where R by C is the motif lattice size.
    - Read the marked motif's odd internal slots as the color map for the R by C output motifs.
    - Slots that still contain the repeated motif color map to that repeated motif color.
    - Rebuild the compact lattice on a canvas filled with the repeated motif color,
      painting each motif mask with its decoded color.

    Key insight:
    - The holes and filled odd slots inside one motif are a miniature legend for recoloring the entire repeated motif lattice.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    base = _dominant_foreground_color(g, bg)

    occupied = g != bg
    row_runs = _runs(occupied.any(axis=1))
    col_runs = _runs(occupied.any(axis=0))
    assert row_runs and col_runs, "expected a nonempty motif lattice"

    tile_heights = [r1 - r0 + 1 for r0, r1 in row_runs]
    tile_widths = [c1 - c0 + 1 for c0, c1 in col_runs]
    assert len(set(tile_heights)) == 1, "motif rows must have equal height"
    assert len(set(tile_widths)) == 1, "motif columns must have equal width"
    assert all(
        row_runs[i + 1][0] - row_runs[i][1] == 2 for i in range(len(row_runs) - 1)
    ), "expected one blank row between motif rows"
    assert all(
        col_runs[i + 1][0] - col_runs[i][1] == 2 for i in range(len(col_runs) - 1)
    ), "expected one blank column between motif columns"

    lattice_h = len(row_runs)
    lattice_w = len(col_runs)
    tile_h = tile_heights[0]
    tile_w = tile_widths[0]
    assert tile_h == 2 * lattice_h + 1, "motif height must encode the number of motif rows"
    assert tile_w == 2 * lattice_w + 1, "motif width must encode the number of motif columns"

    tiles = []
    marked_tiles = []
    for r0, r1 in row_runs:
        tile_row = []
        for c0, c1 in col_runs:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)
            tile_row.append(tile)
            if np.any((tile != bg) & (tile != base)):
                marked_tiles.append(tile)
        tiles.append(tile_row)
    assert len(marked_tiles) == 1, "expected exactly one marked motif"

    legend_tile = marked_tiles[0]
    slot_mask = np.zeros((tile_h, tile_w), dtype=bool)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            slot_mask[1 + 2 * lr, 1 + 2 * lc] = True

    marker_mask = (legend_tile != bg) & (legend_tile != base)
    assert np.all(slot_mask[marker_mask]), "marker pixels must occupy legend slots"
    assert np.all(
        legend_tile[~slot_mask] == base
    ), "non-slot motif cells must be repeated-color foreground"

    motif_mask = legend_tile == base
    for tile_row in tiles:
        for tile in tile_row:
            assert np.array_equal(tile == base, motif_mask), "all motifs must share the same repeated-color mask"

    color_map = np.empty((lattice_h, lattice_w), dtype=int)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            value = int(legend_tile[1 + 2 * lr, 1 + 2 * lc])
            assert value != bg, "each legend slot must name a motif color"
            color_map[lr, lc] = value

    out_h = lattice_h * tile_h + lattice_h - 1
    out_w = lattice_w * tile_w + lattice_w - 1
    out = np.full((out_h, out_w), base, dtype=int)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            r0 = lr * (tile_h + 1)
            c0 = lc * (tile_w + 1)
            block = out[r0 : r0 + tile_h, c0 : c0 + tile_w]
            block[motif_mask] = int(color_map[lr, lc])
    return out

# --------------- generator (o5x) ---------------
# CFG: background and foreground color sampling.
BG_COLOR, COLOR_BOUNDS = 0, (1, 9)
# CFG: lattice geometry and slot placement.
LATTICE_SIZE_BOUNDS, SLOT_OFFSET, SLOT_STRIDE, TILE_GAP = (2, 3), 1, 2, 1
# CFG: hole and marker-color sampling.
HOLE_COUNT_BOUNDS, MIN_HOLE_FRACTION, MAX_HOLE_FRACTION = (3, 5), (1, 2), (5, 6)
MARKER_COLOR_BOUNDS = (3, 4)
# CFG: canvas padding and retry budget.
BASE_PADDING, EXTRA_PAD_SIDE_BOUNDS, EXTRA_PAD_AMOUNT_BOUNDS, GEN_RETRIES = 1, (1, 2), (1, 2), 300

def _slot_positions(lattice_h: int, lattice_w: int) -> list[tuple[int, int]]:
    return [(SLOT_OFFSET + SLOT_STRIDE * r, SLOT_OFFSET + SLOT_STRIDE * c) for r in range(lattice_h) for c in range(lattice_w)]

def _slot_index(r: int, c: int) -> tuple[int, int]:
    return (r - SLOT_OFFSET) // SLOT_STRIDE, (c - SLOT_OFFSET) // SLOT_STRIDE

def _allowed_hole_counts(n_slots: int) -> list[int]:
    min_num, min_den = MIN_HOLE_FRACTION
    max_num, max_den = MAX_HOLE_FRACTION
    low, high = HOLE_COUNT_BOUNDS
    return [k for k in range(low, high + 1) if k <= n_slots and min_den * k >= min_num * n_slots and max_den * k <= max_num * n_slots]

def _sample_motif(lattice_h, lattice_w, tile_h, tile_w, base):
    """Return slot positions, repeated motif mask, and legend color map."""
    slot_positions = _slot_positions(lattice_h, lattice_w)
    n_holes = random.choice(_allowed_hole_counts(len(slot_positions)))
    hole_positions = set(random.sample(slot_positions, n_holes))
    marker_palette = [color for color in range(COLOR_BOUNDS[0], COLOR_BOUNDS[1] + 1) if color != base]
    n_marker_colors = random.randint(MARKER_COLOR_BOUNDS[0], min(MARKER_COLOR_BOUNDS[1], n_holes))
    marker_colors = random.sample(marker_palette, n_marker_colors)
    marker_values = marker_colors + [random.choice(marker_colors) for _ in range(n_holes - n_marker_colors)]
    random.shuffle(marker_values)
    motif_mask = np.ones((tile_h, tile_w), dtype=bool)
    color_map = np.full((lattice_h, lattice_w), base, dtype=int)
    for (r, c), color in zip(sorted(hole_positions), marker_values):
        motif_mask[r, c] = False
        color_map[_slot_index(r, c)] = color
    return slot_positions, motif_mask, color_map

def _paint_output(color_map: np.ndarray, motif_mask: np.ndarray, base: int) -> np.ndarray:
    lattice_h, lattice_w = color_map.shape
    tile_h, tile_w = motif_mask.shape
    out = np.full((lattice_h * tile_h + lattice_h - 1, lattice_w * tile_w + lattice_w - 1), base, dtype=int)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            r0 = lr * (tile_h + TILE_GAP)
            c0 = lc * (tile_w + TILE_GAP)
            out[r0 : r0 + tile_h, c0 : c0 + tile_w][motif_mask] = int(color_map[lr, lc])
    return out

def _paint_input_lattice(color_map, motif_mask, slot_positions, base, shape):
    """Render the motif lattice with exactly one tile replaced by the legend."""
    lattice_h, lattice_w = color_map.shape
    tile_h, tile_w = motif_mask.shape
    unmarked_tile = np.where(motif_mask, base, BG_COLOR)
    lattice = np.full(shape, BG_COLOR, dtype=int)
    legend_r = random.randrange(lattice_h)
    legend_c = random.randrange(lattice_w)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            tile = unmarked_tile.copy()
            if (lr, lc) == (legend_r, legend_c):
                for sr, sc in slot_positions:
                    tile[sr, sc] = int(color_map[_slot_index(sr, sc)])
            r0 = lr * (tile_h + TILE_GAP)
            c0 = lc * (tile_w + TILE_GAP)
            lattice[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return lattice

def _pad_lattice(lattice: np.ndarray) -> np.ndarray:
    pads = [BASE_PADDING, BASE_PADDING, BASE_PADDING, BASE_PADDING]
    for side in random.sample(range(4), random.randint(*EXTRA_PAD_SIDE_BOUNDS)):
        pads[side] += random.randint(*EXTRA_PAD_AMOUNT_BOUNDS)
    top, bottom, left, right = pads
    return np.pad(lattice, ((top, bottom), (left, right)), constant_values=BG_COLOR)

def _gen_attempt() -> np.ndarray | None:
    lattice_h = random.randint(*LATTICE_SIZE_BOUNDS)
    lattice_w = random.randint(*LATTICE_SIZE_BOUNDS)
    tile_h = SLOT_STRIDE * lattice_h + SLOT_OFFSET
    tile_w = SLOT_STRIDE * lattice_w + SLOT_OFFSET
    base = random.randint(*COLOR_BOUNDS)
    slot_positions, motif_mask, color_map = _sample_motif(lattice_h, lattice_w, tile_h, tile_w, base)
    out = _paint_output(color_map, motif_mask, base)
    inp = _pad_lattice(_paint_input_lattice(color_map, motif_mask, slot_positions, base, out.shape))
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid 1be83260 input")
