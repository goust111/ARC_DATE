# Variant of 269e22fb
# Dimension: D13 — 图案唯一性保证机制 (meta)
# Change: 通过裁剪窗口尺寸约束（min(H,W)>=11或max(H,W)>=12保证100%唯一性）和运行时唯一性验证来保证... → 允许多解并用额外提示消歧

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# spec_variant:
#   unchanged_core:
#     motif: same hard-coded 20x20 binary mask
#     output: same full 20x20 motif recolored by the crop's two colors
#     transforms: same 8 dihedral variants
#   d13_change:
#     old_uniqueness: the crop itself had to identify exactly one (polarity, variant, offset)
#     new_uniqueness: the crop is intentionally allowed to be ambiguous, and a visible extra clue row
#       supplies one bit that selects the correct solution class
#   concrete_input_shape:
#     layout: 1 hint row + HxW crop beneath it
#     hint_row: mostly a third color; one endpoint cell is recolored
#     endpoint_meaning:
#       left endpoint marked  -> even variant index
#       right endpoint marked -> odd variant index
#     crop: still a two-color rectangle from one dihedral variant of the motif
#   behavior_trace:
#     original:
#       a 12x11 crop might match exactly one orientation/offset, so the solver can assert len(matches)==1
#     variant:
#       a 7x7 crop can match two placements, e.g. one even-indexed variant and one odd-indexed variant
#       the top clue row contributes a single visible bit, so the solver first finds both matches, then keeps
#       only the one whose variant-index parity matches the clue
#     difference:
#       uniqueness no longer comes from crop size alone; it comes from crop + clue jointly
#   complexity_calibration:
#     crop_sizes: restricted to 7..10 on each side
#     reason: these sizes still look like meaningful motif fragments, but many windows are ambiguous in
#       exactly two ways, making the clue visibly necessary without exploding the search space
#   implicit_assumption_audit_for_main:
#     - first row is always the clue row; generator enforces it
#     - clue row contains exactly one majority clue color plus one endpoint marker; generator enforces it
#     - lower crop uses exactly two colors and excludes the clue color; generator enforces three distinct colors
#     - ambiguous crop candidates split across opposite parities; precomputation filters for that
#   placement_legality:
#     crop is taken from a valid 20x20 window, so no out-of-bounds placement exists
#     clue row is external to the crop, so it never corrupts the observed motif fragment
#   visual_observability:
#     the extra top strip is easy to spot, and its left/right marker changes across examples
#     smaller crops make it plausible to a human that the strip is needed for disambiguation
#   visual_cleanliness:
#     only one extra row is added; there are no overlapping objects or dense decorations
#   color_reasoning:
#     color ids themselves carry no meaning
#     the rule uses position of the marker (left vs right), not which numeric color was chosen

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


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


@lru_cache(maxsize=1)
def _variants() -> tuple[np.ndarray, ...]:
    return tuple(_dihedral_variants(_pattern_mask()))


def _crop_matches(crop: np.ndarray) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    crop = np.asarray(crop, dtype=int)
    H, W = map(int, crop.shape)
    colors = colors_present(crop)
    assert len(colors) == 2

    matches: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = crop == int(fg)
        for variant_index, pattern in enumerate(_variants()):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), variant_index, r, c, pattern))
    return matches


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 8 and W >= 7

    clue_row = g[0]
    clue_colors = colors_present(clue_row)
    assert len(clue_colors) == 2
    counts = {color: int(np.sum(clue_row == color)) for color in clue_colors}
    clue_color = max(counts, key=counts.get)
    marker_positions = np.flatnonzero(clue_row != int(clue_color))
    assert len(marker_positions) == 1
    marker_col = int(marker_positions[0])
    assert marker_col in (0, W - 1)
    parity = 0 if marker_col == 0 else 1

    crop = g[1:, :]
    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2
    assert int(clue_color) not in crop_colors
    return crop, parity


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a small ambiguous crop into the full 20x20 motif using an extra clue row.

    Input structure:
    - The first row is a clue strip: one third color repeated across the row, plus one endpoint marker.
    - The remaining rows form a two-color rectangular crop from one dihedral variant of the shared 20x20 motif.
    - A left-end marker means an even-indexed variant; a right-end marker means an odd-indexed variant.

    Rule:
    - Remove the clue row and read the marker side to get the required variant-index parity.
    - Try each crop color as the motif foreground and match that boolean crop against every offset of every dihedral variant.
    - Keep the matches whose variant parity agrees with the clue, then repaint the full 20x20 motif with the matched foreground/background colors.

    Key insight:
    - The crop alone is intentionally too small to be unique, so solving requires combining the local crop evidence with the separate parity hint.
    """
    crop, parity = _parse_input(np.asarray(grid, dtype=int))

    matches = _crop_matches(crop)
    filtered = [match for match in matches if match[2] % 2 == parity]
    assert len(filtered) == 1

    fg, bg, _variant_index, _r, _c, pattern = filtered[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


@lru_cache(maxsize=1)
def _ambiguous_records() -> tuple[tuple[int, int, int, int, int], ...]:
    variants = [pattern.astype(np.uint8) for pattern in _variants()]
    records: list[tuple[int, int, int, int, int]] = []

    for H in range(7, 11):
        for W in range(7, 11):
            window_map: defaultdict[bytes, list[tuple[int, int, int, int]]] = defaultdict(list)
            for variant_index, pattern in enumerate(variants):
                for polarity in (1, 0):
                    visible = pattern if polarity == 1 else 1 - pattern
                    for r in range(20 - H + 1):
                        for c in range(20 - W + 1):
                            key = visible[r : r + H, c : c + W].tobytes()
                            window_map[key].append((polarity, variant_index, r, c))

            for variant_index, pattern in enumerate(variants):
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if window.min() == window.max():
                            continue
                        candidates = window_map[window.tobytes()]
                        if len(candidates) != 2:
                            continue
                        if {candidate[1] % 2 for candidate in candidates} != {0, 1}:
                            continue
                        records.append((H, W, variant_index, r, c))

    assert records
    return tuple(records)


def generate_input() -> np.ndarray:
    for _ in range(200):
        fg, bg, clue = random.sample(Color.ALL_COLORS, 3)
        H, W, variant_index, r, c = random.choice(_ambiguous_records())
        pattern = _variants()[variant_index]
        crop_mask = pattern[r : r + H, c : c + W]
        crop = np.where(crop_mask, int(fg), int(bg)).astype(int)

        matches = _crop_matches(crop)
        assert len(matches) == 2
        assert {match[2] % 2 for match in matches} == {0, 1}

        inp = np.full((H + 1, W), int(clue), dtype=int)
        inp[1:, :] = crop
        inp[0, 0 if variant_index % 2 == 0 else W - 1] = int(fg)

        expected = np.where(pattern, int(fg), int(bg)).astype(int)
        if np.array_equal(main(inp), expected):
            return inp

    raise AssertionError("failed to construct a valid D13 ambiguous sample")
