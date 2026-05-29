# Variant of 269e22fb
# Dimension: D14 — 符号编码：颜色与图案的映射方式 (symbol_instructions)
# Change: 二值图案直接映射为两种颜色，无额外编码层... → 颜色图例指示映射关系

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the D14 variant
# -----------------------------------
# spec:
#   preserved_core:
#     motif: same hard-coded 20x20 binary pattern
#     transform_family: same 8 dihedral variants
#     output: still repaint the full 20x20 oriented motif with the crop palette
#     crop_uniqueness: still enforced by rejection sampling
#   input_layout:
#     frame:
#       role: isolate the crop from the canvas edge so the legend is visibly separate
#       color: one extra ARC color distinct from both motif colors
#       thickness:
#         top: 1
#         bottom: 1
#         right: 1
#         left_band: 2
#     legend:
#       location: left edge band
#       shape: two stacked 2x1 swatches in column 0
#       semantics:
#         upper swatch (rows 1:3, col 0): motif foreground color
#         lower swatch (rows 3:5, col 0): motif background color
#       separator: column 1 remains frame-colored, so the legend does not merge into the crop
#     crop:
#       location: strict interior rectangle grid[1:-1, 2:-1]
#       size:
#         rows: 8-16
#         cols: 8-12
#       content: exact crop from one oriented motif, painted with the legend-indicated fg/bg colors
#   behavior_trace:
#     original:
#       - the whole input grid was the crop
#       - main() tried both input colors as possible foreground
#       - polarity was inferred by brute force during window matching
#     variant:
#       - main() first peels off the frame and reads the edge legend
#       - the legend fixes which crop color means motif-1 and which means motif-0
#       - only that directed foreground mask is matched against the 20x20 motif variants
#     concrete_example:
#       - suppose the crop uses red and yellow, and the left legend shows red above yellow
#       - then red marks True cells of the binary motif and yellow marks False cells
#       - the solver must ignore the frame color, extract the interior crop, and match red-as-True only
#   complexity_calibration:
#     - The motif and crop uniqueness machinery stay unchanged, so the task is not diluted into a new geometry puzzle.
#     - The legend uses only one extra color and one fixed position convention, so the added symbolic step is visible but not noisy.
#     - The legend is large enough to be noticed in 2-4 training pairs, unlike single-pixel metadata that could look accidental.
#   implicit_assumptions_review:
#     - Original `colors_present(g)` assumed the full input had exactly two colors. That breaks here because the frame adds a third color.
#     - Original `visible = g == fg` assumed the whole grid was the crop. That is now false; the crop must be sliced out first.
#     - Original polarity loop over both colors is no longer the intended mechanism. The legend fixes fg/bg explicitly.
#     - The matching loop still assumes the visible crop is an exact rectangular window of one dihedral motif variant. The generator preserves this.
#   placement_legality:
#     - The crop always sits at rows 1..H and cols 2..W+1, so it never overlaps the legend or crosses the border.
#     - The legend always fits because crop height is at least 8, leaving room for four legend rows inside the framed canvas.
#     - Frame color is sampled distinct from crop colors, preventing legend disappearance or crop/frame ambiguity.
#   visual_observability:
#     - Human solvers can see a stable left-edge legend pattern across examples: upper chip -> filled motif cells, lower chip -> empty motif cells.
#     - The interior crop remains the same style as the source task, so the new concept is isolated to the encoding layer.
#   visual_cleanliness:
#     - Only three regions exist: frame, legend swatches, crop.
#     - No overlays or touching multicolor composites occur; the separator column keeps legend and crop distinct.
#   color_inference:
#     - Only three colors appear in the input.
#     - Two of them repeat inside the crop and reappear in the legend, giving enough support for the mapping rule.


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

_CROP_H_RANGE = (8, 16)
_CROP_W_RANGE = (8, 12)
_ARC_COLORS = tuple(range(10))
_MAX_ATTEMPTS = 1000


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _extract_layout(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 6 and W >= 5

    crop = g[1:-1, 2:-1]
    ch, cw = map(int, crop.shape)
    assert 8 <= ch <= 16 and 8 <= cw <= 12

    border_mask = np.ones_like(g, dtype=bool)
    border_mask[1:-1, 2:-1] = False
    border_vals = g[border_mask]
    frame = int(np.bincount(border_vals, minlength=10).argmax())

    fg = int(g[1, 0])
    bg = int(g[3, 0])
    assert fg != bg
    assert fg != frame and bg != frame

    assert np.all(g[1:3, 0] == fg)
    assert np.all(g[3:5, 0] == bg)
    assert np.all(g[:, 1] == frame)
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)
    assert np.all(g[:, -1] == frame)
    assert np.all(g[5:, 0] == frame)

    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2
    assert set(map(int, crop_colors)) == {fg, bg}
    return crop, fg, bg, frame


def _find_matches(crop: np.ndarray, fg: int) -> list[np.ndarray]:
    crop = np.asarray(crop, dtype=int)
    H, W = map(int, crop.shape)
    visible = crop == int(fg)

    matches: list[np.ndarray] = []
    for pattern in _dihedral_variants(_pattern_mask()):
        for r in range(20 - H + 1):
            for c in range(20 - W + 1):
                if np.array_equal(pattern[r : r + H, c : c + W], visible):
                    matches.append(pattern)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from an interior crop whose color polarity is given by an edge legend.

    Input structure:
    - A framed canvas contains a rectangular crop of one dihedral variant of the shared 20x20 binary motif.
    - Along the left edge, two stacked color swatches act as a legend: the upper swatch names the motif foreground color and the lower swatch names the motif background color.

    Rule:
    - Ignore the frame color and slice out the interior crop.
    - Read the legend to decide which crop color should be interpreted as binary 1 versus binary 0.
    - Match that directed boolean crop against every window of every dihedral motif variant, then repaint the uniquely matched full motif with the same foreground/background colors.

    Key insight:
    - The crop still comes from the same canonical 20x20 pattern as the source task, but the color meaning is no longer inferred by trying both polarities; it is explicitly declared by the border legend.
    """
    crop, fg, bg, _frame = _extract_layout(np.asarray(grid, dtype=int))
    matches = _find_matches(crop, fg)
    assert len(matches) == 1
    pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_colors() -> tuple[int, int, int]:
    fg, bg, frame = random.sample(_ARC_COLORS, 3)
    return int(fg), int(bg), int(frame)


def _build_input(crop: np.ndarray, fg: int, bg: int, frame: int) -> np.ndarray:
    H, W = map(int, crop.shape)
    inp = np.full((H + 2, W + 3), int(frame), dtype=int)
    inp[1:-1, 2:-1] = crop
    inp[1:3, 0] = int(fg)
    inp[3:5, 0] = int(bg)
    return inp


def generate_input() -> np.ndarray:
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    for _ in range(_MAX_ATTEMPTS):
        fg, bg, frame = _sample_colors()
        pattern = random.choice(variants)
        H = random.randint(*_CROP_H_RANGE)
        W = random.randint(*_CROP_W_RANGE)
        r = random.randint(0, 20 - H)
        c = random.randint(0, 20 - W)

        full = np.where(pattern, int(fg), int(bg)).astype(int)
        crop = full[r : r + H, c : c + W]

        if len(_find_matches(crop, fg)) != 1:
            continue

        inp = _build_input(crop, fg, bg, frame)
        if np.array_equal(main(inp), full):
            return inp

    raise RuntimeError("failed to generate a valid D14 sample")


def variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    crop, fg, bg, frame = _extract_layout(np.asarray(grid, dtype=int))
    all_colors = colors_present(np.asarray(grid, dtype=int))
    border_mask = np.ones_like(grid, dtype=bool)
    border_mask[1:-1, 2:-1] = False
    return {
        "input_color_count": len(all_colors),
        "crop_height": int(crop.shape[0]),
        "crop_width": int(crop.shape[1]),
        "frame_distinct": bool(frame not in {fg, bg}),
        "legend_fg_equals_top": bool(np.all(np.asarray(grid)[1:3, 0] == fg)),
        "legend_bg_equals_lower": bool(np.all(np.asarray(grid)[3:5, 0] == bg)),
        "border_frame_cells": int(np.sum(np.asarray(grid)[border_mask] == frame)),
    }
