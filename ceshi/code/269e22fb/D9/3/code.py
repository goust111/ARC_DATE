# Variant of 269e22fb
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 固定20x20输出，与基础图案尺寸相同... → 输出扩展为包含裁剪位置标记的22x22网格

from __future__ import annotations

import random

import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
# ---------------------------------------------------------------------------
# variant_spec:
#   intent:
#     - Keep the original recognition problem intact: the input is still a
#       two-color rectangular crop cut from one dihedral orientation of the
#       shared 20x20 binary motif.
#     - Change only the output geometry: instead of returning the bare 20x20
#       completion, return a 22x22 canvas whose central 20x20 area is the full
#       motif and whose 1-cell border annotates the crop location.
#   concretization:
#     input:
#       - shape: HxW, where H in [8,16] and W in [8,12]
#       - colors: exactly 2 visible motif colors, no dedicated marker color
#       - content: one rectangular crop of a rotated/flipped 20x20 motif
#     output:
#       - shape: 22x22
#       - interior: output[1:21, 1:21] is the matched full 20x20 motif
#       - border encoding:
#         - top and bottom borders mark the crop's column span
#         - left and right borders mark the crop's row span
#         - unmarked border cells use the motif background color
#         - marked border cells use one reserved marker color
#     worked_example:
#       - If the input is a 10x8 crop from rows 3..12 and cols 5..12 of the
#         matched motif, the output paints:
#         - marker on left/right border rows 4..13
#         - marker on top/bottom border cols 6..13
#         - the full 20x20 motif in the center
#   behavior_trace:
#     original_flow:
#       - match crop -> recover (fg, bg, orientation) -> repaint 20x20 motif
#     new_flow:
#       - match crop -> recover (fg, bg, orientation, row_offset, col_offset)
#       - repaint the same 20x20 motif in the center
#       - add four border segments that encode the crop window's row/col range
#     key_delta:
#       - offset recovery existed implicitly before but was discarded; D9 makes
#         it observable, so the solver must retain the unique window location.
#   complexity_calibration:
#     - The crop size distribution stays unchanged, preserving the original
#       pattern-recognition difficulty.
#     - Only one extra visible concept is introduced: a single marker color on
#       the border. This avoids noisy multi-color legends.
#     - The location signal is repeated on opposite sides (top+bottom, left+right)
#       so humans can infer "span on rows / span on cols" from a few examples.
#   implicit_assumptions_review:
#     original_main:
#       - `H <= 20 and W <= 20`: still valid because the input remains a crop
#         of the 20x20 motif.
#       - `len(colors) == 2`: still valid because the input structure is
#         unchanged; the marker color appears only in the output.
#       - matching loops over all variants and offsets: still valid and now must
#         keep `(r, c)` rather than discarding them.
#       - `return np.where(pattern, fg, bg)`: invalid after D9 because output is
#         no longer the motif's natural size and must expose the crop position.
#     new_main:
#       - assumes the crop has exactly one `(polarity, orientation, offset)`
#         match; `generate_input()` enforces this by rejection sampling.
#       - assumes the reserved marker color is absent from the input motif
#         colors; `generate_input()` samples motif colors from 1..9 while using
#         marker color 0.
#   placement_legality:
#     - The 20x20 motif always sits strictly inside the 22x22 frame, so border
#       marks can never overwrite motif pixels.
#     - Border spans are indexed as `1 + offset` through `1 + offset + size`,
#       which stays in-bounds because `0 <= offset <= 20 - size`.
#     - Top/bottom and left/right annotations are disjoint except for corners,
#       which remain background-colored for visual cleanliness.
#   visual_observability:
#     - Every output visibly differs from the original task: it is larger and
#       carries four border cues tied to the input crop's size and position.
#     - Offsets vary across generated examples, so the border markings are not a
#       fixed decoration.
#   visual_cleanliness:
#     - Only three semantic layers exist: motif foreground, motif background,
#       and border marker.
#     - The marker is confined to the outer frame, so no overlapping symbols
#       compete inside the motif area.
#   color_inference:
#     - A fixed reserved marker color keeps the rule legible across examples.
#       The generator avoids using that color for the input crop, so the border
#       marker remains a genuinely special output color instead of a hidden
#       color-id trick.


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

_MARKER_COLOR = 0
_MOTIF_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted({int(x) for x in np.asarray(grid).flat})


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = _colors_present(g)
    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))
    return matches


def _render_output(
    pattern: np.ndarray,
    fg: int,
    bg: int,
    crop_r: int,
    crop_c: int,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    out = np.full((22, 22), int(bg), dtype=int)
    out[1:21, 1:21] = np.where(pattern, int(fg), int(bg))
    out[0, 1 + crop_c : 1 + crop_c + crop_w] = int(_MARKER_COLOR)
    out[21, 1 + crop_c : 1 + crop_c + crop_w] = int(_MARKER_COLOR)
    out[1 + crop_r : 1 + crop_r + crop_h, 0] = int(_MARKER_COLOR)
    out[1 + crop_r : 1 + crop_r + crop_h, 21] = int(_MARKER_COLOR)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full motif and add a border that records where the crop came from.

    Input structure:
    - The grid is still a two-color rectangular crop taken from one rotation/reflection of the shared 20x20 binary motif.
    - The two input colors repaint motif foreground/background; the border marker color is not present in the input.

    Rule:
    - Try each input color as the motif foreground and compare that boolean crop against every offset in every dihedral orientation of the canonical motif.
    - Keep the unique match together with its crop row/column offset, then repaint the full 20x20 motif inside a 22x22 canvas.
    - Use the outer border to mark the crop span: top/bottom mark the matched columns, left/right mark the matched rows.

    Key insight:
    - The local crop still identifies the motif orientation, polarity, and exact window position; D9 simply makes that recovered window position visible in the output frame.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert 1 <= H <= 20 and 1 <= W <= 20

    colors = _colors_present(g)
    assert len(colors) == 2
    assert int(_MARKER_COLOR) not in colors

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern, crop_r, crop_c = matches[0]
    return _render_output(pattern, fg, bg, crop_r, crop_c, H, W)


def generate_input() -> np.ndarray:
    rng = random.Random()
    base = _pattern_mask()
    variants = _dihedral_variants(base)

    for _ in range(400):
        fg, bg = rng.sample(_MOTIF_PALETTE, 2)
        pattern = variants[rng.randrange(len(variants))]
        crop_h = rng.randint(8, 16)
        crop_w = rng.randint(8, 12)
        crop_r = rng.randint(0, 20 - crop_h)
        crop_c = rng.randint(0, 20 - crop_w)

        full = np.where(pattern, int(fg), int(bg)).astype(int)
        crop = full[crop_r : crop_r + crop_h, crop_c : crop_c + crop_w]

        matches = _find_matches(crop)
        if len(matches) != 1:
            continue

        expected = _render_output(pattern, fg, bg, crop_r, crop_c, crop_h, crop_w)
        try:
            actual = main(crop)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return crop

    raise RuntimeError("generate_input: failed after 400 attempts")
