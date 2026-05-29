# Combo variant of dfadab01 (index 21)
# combo_spec: D5/3+D10/2
# Name: 3x3小字形+密集排列重叠

from __future__ import annotations

import random

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

MARKER_COLORS = tuple(GLYPHS.keys())
OUTPUT_COLORS = tuple(color for color, _ in GLYPHS.values())
GLYPH_SIZE = 3
SPACING = 2
GRID_CHOICES = (12, 14, 16, 18, 20)
MAX_GEN_RETRIES = 400
LEGEND_PROB = 0.65


# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - Original dfadab01 treated every real marker as the top-left anchor of a
#      4x4 fixed glyph, and generation kept marker spacing at least 4 so those
#      glyphs never touched each other.
#    - This combo shrinks every dictionary entry to 3x3 and simultaneously
#      places markers on a denser lattice with step 2.  A row like markers at
#      columns [1, 3, 5] used to create three disjoint 4x4 blocks; now it
#      creates three 3x3 glyphs whose footprints overlap on shared columns.
#    - Example with marker color 2 (yellow frame) at top-lefts (1,1) and (1,3):
#        old output footprint width with spacing >= 4: two separate 4x4 frames
#        new output footprint width with spacing 2: two 3x3 frames sharing the
#        middle column, producing one fused 3x5 strip.
#    - The legend idea stays the same, but its sample glyph is also 3x3 and the
#      label marker moves to one cell below-right of the 3x3 box, i.e. (+3,+3).
#
# b) Complexity calibration:
#    - The solver still uses one fixed glyph dictionary; only the glyph size and
#      spatial density change.  That keeps the rule learnable from a few pairs.
#    - The generator does not randomize arbitrary per-sample overlap rules.
#      Overlap comes only from geometry: same 3x3 glyphs plus step-2 placement.
#    - To avoid adding a second hidden rule about which color should win in a
#      mixed overlap, the generator groups touching markers into monochrome dense
#      blocks.  Different marker colors can still appear, but only in regions
#      separated by at least one empty lattice row or column.
#
# c) Original assumptions that break:
#    - `_has_source_glyph(..., r - 4, c - 4, ...)` is no longer correct because
#      the legend offset now depends on a 3x3 glyph, not a 4x4 glyph.
#    - `_paint_clipped` cannot assume a 4x4 loop bound.
#    - The old spec guaranteed "glyphs never overlap"; that assumption is gone.
#      The core paint logic still works, but only if the generator avoids
#      ambiguous cross-color overlaps that humans could not infer from examples.
#    - The legend label skip check must still happen before painting; otherwise a
#      dense layout would incorrectly expand the below-right label marker.
#
# d) Placement legality:
#    - Real markers are placed on a regular lattice with spacing exactly 2.
#    - Each occupied component is a full rectangle in lattice coordinates, so the
#      output always contains many overlap cells yet the input remains easy to
#      parse as a regular marker arrangement.
#    - Different-colored rectangles are separated by a blank lattice row and/or
#      column, which guarantees their 3x3 footprints do not overlap at all.
#    - Legends are inserted only into an empty 4x4 region (3x3 sample + label),
#      so they never collide with real markers.
#
# e) Visual observability:
#    - The input still clearly shows the controlling single-pixel markers, while
#      the output visibly turns them into a woven carpet because the lattice step
#      is smaller than the glyph size.
#    - Keeping the layout regular makes the "dense overlap" change readable by
#      eye; the solver is not depending on hidden numeric properties of colors.
#    - Optional legends provide a direct local example of the new 3x3 alphabet,
#      which is important because the glyphs are smaller than in the original.
#
# f) Visual cleanliness:
#    - Every overlap-connected cluster is monochrome, so the output looks dense
#      rather than noisy.  Humans see interleaving repeated motifs, not arbitrary
#      color clashes caused by an unannounced z-order rule.
#    - Empty separator rows/columns keep multi-color scenes legible.
#
# g) Inferability of the change:
#    - The only color mapping remains the original fixed marker->glyph-color
#      mapping.  No extra color transformation is introduced.
#    - The new information a solver must infer is geometric: glyphs are 3x3, and
#      neighboring markers are packed closely enough that their expansions merge.
#
# Post-change solver assumptions that the generator explicitly guarantees:
#    - Every real marker is a single pixel in {2, 3, 5, 8}.
#    - Any legend label marker has a matching 3x3 sample glyph exactly at
#      (r - 3, c - 3); real markers never satisfy that condition by accident.
#    - Every overlap-connected set of real markers is monochrome, so the solver's
#      deterministic paint order never needs to resolve a human-invisible
#      cross-color conflict.
#    - Every generated grid contains at least one pair of real markers at spacing
#      2, so overlap is not optional or degenerate.


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    mh, mw = mask.shape
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    patch = g[r0 : r0 + mh, c0 : c0 + mw]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    for dr in range(mh):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(mw):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - GLYPH_SIZE, cc - GLYPH_SIZE, glyph_color, mask):
                continue
            markers.append((rr, cc, marker_color))
    return markers


def _coverage_from_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    cov = np.zeros(shape, dtype=int)
    h, w = shape
    for r0, c0, marker_color in markers:
        mask = GLYPHS[marker_color][1]
        for dr in range(GLYPH_SIZE):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(GLYPH_SIZE):
                cc = c0 + dc
                if 0 <= cc < w and bool(mask[dr, dc]):
                    cov[rr, cc] += 1
    return cov


def overlap_stats(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    markers = _collect_real_markers(g)
    cov = _coverage_from_markers(g.shape, markers)
    return {
        "num_markers": len(markers),
        "overlap_cells": int(np.count_nonzero(cov >= 2)),
        "max_cover": int(cov.max(initial=0)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control marker into a fixed 3x3 glyph, allowing dense overlaps.

    Input structure:
    - The background is black.  Real control markers are single pixels in colors
      2, 3, 5, and/or 8, typically arranged on a regular step-2 lattice.
    - Some inputs also include one legend entry: a 3x3 sample glyph already
      drawn in its output color, with a matching label marker at the sample's
      below-right corner offset (+3, +3).

    Rule:
    - Replace each real marker by its learned 3x3 glyph: 2->yellow frame,
      3->blue diamond, 5->pink diagonal blocks, 8->orange X.
    - Paint each glyph with its top-left corner at the marker position and clip
      any part that falls outside the grid.
    - Ignore legend samples entirely and skip expanding the below-right label
      marker that points to a legend sample.
    - Because markers are packed more tightly than the 3x3 glyph size, nearby
      glyphs are allowed to overlap and merge into denser output regions.

    Key insight:
    - The colored pixels are still only instructions, but the learned alphabet is
      smaller and the placement lattice is tighter.  Solving now requires seeing
      that the same fixed glyph dictionary is being stamped densely enough that
      neighboring stamps interlock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *OUTPUT_COLORS})

    real_markers = _collect_real_markers(g)
    assert real_markers, "expected at least one real marker"

    out = np.zeros_like(g)
    for marker_color in MARKER_COLORS:
        glyph_color, mask = GLYPHS[marker_color]
        for r, c, mc in real_markers:
            if mc == marker_color:
                _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _sample_separator_index(length: int, rng: random.Random) -> int | None:
    if length < 5 or rng.random() >= 0.55:
        return None
    options = list(range(2, length - 2))
    if not options:
        return None
    return rng.choice(options)


def _segment_bounds(length: int, sep: int | None) -> list[tuple[int, int]]:
    if sep is None:
        return [(0, length)]
    return [(0, sep), (sep + 1, length)]


def _build_marker_lattice(rng: random.Random, rows: int, cols: int) -> np.ndarray | None:
    row_sep = _sample_separator_index(rows, rng)
    col_sep = _sample_separator_index(cols, rng)
    occupied_ratio = ((rows - int(row_sep is not None)) * (cols - int(col_sep is not None))) / (rows * cols)
    if occupied_ratio < 0.6:
        return None

    lattice = np.zeros((rows, cols), dtype=int)
    blocks: list[tuple[slice, slice]] = []
    for r0, r1 in _segment_bounds(rows, row_sep):
        for c0, c1 in _segment_bounds(cols, col_sep):
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                blocks.append((slice(r0, r1), slice(c0, c1)))
    if not blocks:
        return None

    used_types = rng.sample(list(MARKER_COLORS), k=rng.randint(1, min(4, len(blocks))))
    for block in blocks:
        lattice[block] = rng.choice(used_types)

    return lattice


def _choose_geometry(rng: random.Random) -> tuple[int, int, int, int, int, int]:
    h = rng.choice(GRID_CHOICES)
    w = rng.choice(GRID_CHOICES)

    max_rows = min(7, 1 + (h - 1) // SPACING)
    max_cols = min(7, 1 + (w - 1) // SPACING)
    rows = rng.randint(4, max_rows)
    cols = rng.randint(4, max_cols)

    max_r0 = h - 1 - SPACING * (rows - 1)
    max_c0 = w - 1 - SPACING * (cols - 1)
    start_r = rng.randint(0, max_r0)
    start_c = rng.randint(0, max_c0)
    return h, w, rows, cols, start_r, start_c


def _place_markers(grid: np.ndarray, lattice: np.ndarray, start_r: int, start_c: int) -> None:
    rows, cols = lattice.shape
    for rr in range(rows):
        for cc in range(cols):
            marker = int(lattice[rr, cc])
            if marker == 0:
                continue
            grid[start_r + rr * SPACING, start_c + cc * SPACING] = marker


def _find_legend_spot(grid: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - GLYPH_SIZE):
        for c0 in range(w - GLYPH_SIZE):
            if r0 + GLYPH_SIZE >= h or c0 + GLYPH_SIZE >= w:
                continue
            block = grid[r0 : r0 + GLYPH_SIZE + 1, c0 : c0 + GLYPH_SIZE + 1]
            if np.all(block == 0):
                candidates.append((r0, c0))
    if not candidates:
        return None
    corner_pref = []
    for r0, c0 in candidates:
        score = min(r0, c0, h - (r0 + GLYPH_SIZE + 1), w - (c0 + GLYPH_SIZE + 1))
        corner_pref.append((score, r0, c0))
    corner_pref.sort()
    best_band = corner_pref[: max(1, len(corner_pref) // 3)]
    _, r0, c0 = rng.choice(best_band)
    return r0, c0


def _add_legend(grid: np.ndarray, rng: random.Random) -> bool:
    spot = _find_legend_spot(grid, rng)
    if spot is None:
        return False

    used_markers = [color for color in MARKER_COLORS if np.any(grid == color)]
    if not used_markers:
        return False

    marker_color = rng.choice(used_markers)
    glyph_color, mask = GLYPHS[marker_color]
    r0, c0 = spot
    for dr in range(GLYPH_SIZE):
        for dc in range(GLYPH_SIZE):
            if bool(mask[dr, dc]):
                grid[r0 + dr, c0 + dc] = glyph_color
    grid[r0 + GLYPH_SIZE, c0 + GLYPH_SIZE] = marker_color
    return True


def _old_style_footprint_difference(grid: np.ndarray) -> int:
    old_masks: dict[int, tuple[int, np.ndarray]] = {
        2: (
            4,
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        3: (
            1,
            np.array(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                ],
                dtype=bool,
            ),
        ),
        5: (
            6,
            np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        8: (
            7,
            np.array(
                [
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ],
                dtype=bool,
            ),
        ),
    }

    g = np.asarray(grid, dtype=int)
    new_out = main(g)
    old_out = np.zeros_like(g)
    for r0, c0, marker_color in _collect_real_markers(g):
        glyph_color, mask = old_masks[marker_color]
        mh, mw = mask.shape
        for dr in range(mh):
            rr = r0 + dr
            if rr < 0 or rr >= g.shape[0]:
                continue
            for dc in range(mw):
                cc = c0 + dc
                if 0 <= cc < g.shape[1] and bool(mask[dr, dc]):
                    old_out[rr, cc] = glyph_color
    return int(np.count_nonzero(new_out != old_out))


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GEN_RETRIES):
        h, w, rows, cols, start_r, start_c = _choose_geometry(rng)
        lattice = _build_marker_lattice(rng, rows, cols)
        if lattice is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        _place_markers(grid, lattice, start_r, start_c)

        if rng.random() < LEGEND_PROB:
            _add_legend(grid, rng)

        stats = overlap_stats(grid)
        if stats["num_markers"] < 4:
            continue
        if stats["overlap_cells"] <= 0 or stats["max_cover"] < 2:
            continue
        if _old_style_footprint_difference(grid) <= 0:
            continue

        out = main(grid)
        assert out.shape == grid.shape
        assert np.count_nonzero(out) > 0
        return grid

    raise RuntimeError("failed to generate a dense-overlap dfadab01 combo sample")
