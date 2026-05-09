# Combo variant of 269e22fb (index 19)
# combo_spec: D7/0+D13/0
# Name: 小窗口多解加提示消歧

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# spec_combo:
#   unchanged_core:
#     motif: same fixed 20x20 binary pattern as the base task
#     output: still the full 20x20 motif recolored by the crop's two colors
#     transforms: still the 8 dihedral variants ordered as rot_k, then fliplr(rot_k)
#   fused_change:
#     d7_small_window:
#       the visible motif fragment is reduced to a deliberately tiny crop, here 5x5 / 5x6 / 6x5
#       these windows often admit several legal (polarity, variant, offset) matches
#     d13_hint_disambiguation:
#       the crop is wrapped by a 1-cell clue frame in a third color
#       exactly one frame corner is replaced by one of the crop colors
#       corner position encodes the rotation group k in {0,1,2,3}
#       marker color encodes the flip bit relative to the crop colors:
#         marker == crop foreground  -> unflipped member of that rotation group
#         marker == crop background  -> flipped member of that rotation group
#   concrete_pixel_layout:
#     input shape: (H+2) x (W+2)
#     center: tiny two-color crop from one dihedral variant
#     border: solid third-color frame except for one marked corner
#     corners clockwise:
#       top-left -> rotation group 0
#       top-right -> rotation group 1
#       bottom-right -> rotation group 2
#       bottom-left -> rotation group 3
#   behavior_trace:
#     original:
#       a medium crop was usually unique, so the solver only needed brute-force matching
#     fused:
#       example: a 5x5 crop may match several placements, including same-rotation alternatives and
#       different-rotation alternatives
#       the clue corner first selects one of the four rotation groups, then the marker color selects
#       which flip state inside that group is compatible with the observed crop colors
#       only after combining crop evidence with both hint dimensions does one candidate remain
#     difference:
#       uniqueness is no longer a property of crop size; ambiguity is intentional and the hint frame
#       carries the extra bits needed to recover the answer
#   complexity_calibration:
#     crop_sizes: restricted to a tiny band with enough structure to be visible but still ambiguous
#     generator_filter:
#       keep only records where the crop alone has multiple matches
#       keep only records where corner position alone is insufficient
#       keep only records where marker color alone is insufficient
#     reason:
#       this prevents degeneration into a puzzle that only needs corner position or only needs
#       marker color; both sub-signals genuinely matter
#   implicit_assumption_audit_for_main:
#     - the clue is always a 1-cell outer frame; generator constructs exactly that layout
#     - the frame uses one third color absent from the crop interior; generator samples 3 distinct colors
#     - there is exactly one marked border cell and it is a corner; generator enforces one-corner marking
#     - the interior crop uses exactly two colors and is non-monochrome; generator filters constant windows
#     - the crop must be ambiguous before reading the clue; generator keeps only windows with >1 match
#     - after applying both hint dimensions, exactly one candidate remains; generator precomputes only such windows
#   placement_legality:
#     the crop is always extracted from a legal 20x20 window
#     the clue frame is external to the crop, so no overlap corrupts motif pixels
#     the marked cell stays on the frame corner, so it never intrudes into the crop interior
#   visual_observability:
#     humans can immediately see a tiny central crop plus a conspicuous outer frame
#     corner location varies across samples, so the positional code is observable
#     the corner marker reuses one of the crop colors, so the color relation is also observable
#   visual_cleanliness:
#     only one extra layer is added around the crop, with a single marked corner
#     there are no stacked decorations inside the crop itself
#   inferability_of_change:
#     color ids themselves carry no hidden semantics
#     the only color-based rule is relational and visible: whether the corner marker matches the crop's
#     motif color or its background color
#   post_change_main_audit:
#     main parses the frame before matching, then filters brute-force crop matches by:
#       1. same rotation group as the marked corner
#       2. compatible flip bit as indicated by the marker color
#     generate_input guarantees both filters are necessary on every emitted sample

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


def _corner_positions(H: int, W: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, W - 1), (H - 1, W - 1), (H - 1, 0))


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 7 and W >= 7

    crop = g[1:-1, 1:-1]
    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2

    border_mask = np.ones_like(g, dtype=bool)
    border_mask[1:-1, 1:-1] = False
    border_colors = colors_present(g[border_mask])
    clue_candidates = [color for color in border_colors if int(color) not in crop_colors]
    assert len(clue_candidates) == 1
    clue_color = int(clue_candidates[0])
    assert clue_color not in crop_colors

    marker_positions = np.argwhere(border_mask & (g != clue_color))
    assert len(marker_positions) == 1
    marker_r, marker_c = map(int, marker_positions[0])
    corner_map = {pos: idx for idx, pos in enumerate(_corner_positions(H, W))}
    assert (marker_r, marker_c) in corner_map

    marker_color = int(g[marker_r, marker_c])
    assert marker_color in crop_colors
    return crop, int(corner_map[(marker_r, marker_c)]), marker_color


def _hint_compatible(
    match: tuple[int, int, int, int, int, np.ndarray],
    rotation_index: int,
    marker_color: int,
) -> bool:
    fg, bg, variant_index, _r, _c, _pattern = match
    if variant_index // 2 != rotation_index:
        return False
    return _marker_color_compatible(match, marker_color)


def _marker_color_compatible(
    match: tuple[int, int, int, int, int, np.ndarray],
    marker_color: int,
) -> bool:
    fg, bg, variant_index, _r, _c, _pattern = match
    if variant_index % 2 == 0:
        return marker_color == int(fg)
    return marker_color == int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a deliberately ambiguous tiny crop plus a coded corner hint.

    Input structure:
    - The center is a very small two-color crop cut from one dihedral variant of the shared 20x20 motif.
    - A 1-cell frame of a third color surrounds that crop.
    - Exactly one frame corner is recolored with one of the crop colors; its corner position encodes the
      rotation group, and its color relative to the crop encodes which flip state in that group is correct.

    Rule:
    - Remove the outer frame to recover the crop, and read the marked corner to obtain the rotation hint.
    - Try each crop color as the motif foreground and match that boolean crop against every offset of every
      dihedral variant.
    - Because the crop is intentionally too small, multiple matches can exist; keep only the matches whose
      rotation group and marker-color relation agree with the frame hint, then repaint the full 20x20 motif.

    Key insight:
    - The crop alone is not meant to be unique. The answer appears only after fusing three visible cues:
      the crop pattern, the marked corner's position, and whether that marker matches the motif color or the background color.
    """
    crop, rotation_index, marker_color = _parse_input(np.asarray(grid, dtype=int))

    matches = _crop_matches(crop)
    assert len(matches) > 1
    filtered = [match for match in matches if _hint_compatible(match, rotation_index, marker_color)]
    assert len(filtered) == 1

    fg, bg, _variant_index, _r, _c, pattern = filtered[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


@lru_cache(maxsize=1)
def _ambiguous_records() -> tuple[tuple[int, int, int, int, int], ...]:
    variants = [pattern.astype(np.uint8) for pattern in _variants()]
    records: list[tuple[int, int, int, int, int]] = []

    for H in (5, 6, 7):
        for W in (5, 6, 7):
            window_map: defaultdict[bytes, list[tuple[int, int, int, int]]] = defaultdict(list)
            for variant_index, pattern in enumerate(variants):
                for polarity in (1, 0):
                    visible = pattern if polarity == 1 else 1 - pattern
                    for r in range(20 - H + 1):
                        for c in range(20 - W + 1):
                            key = visible[r : r + H, c : c + W].tobytes()
                            window_map[key].append((polarity, variant_index, r, c))

            for variant_index, pattern in enumerate(variants):
                actual_parity = variant_index % 2
                actual_rotation = variant_index // 2
                for r in range(20 - H + 1):
                    for c in range(20 - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if window.min() == window.max():
                            continue

                        matches = window_map[window.tobytes()]
                        if len(matches) <= 1:
                            continue

                        actual_record = (1, variant_index, r, c)
                        survivors = [
                            match
                            for match in matches
                            if match[1] // 2 == actual_rotation
                            and match[0] == int((match[1] % 2) == actual_parity)
                        ]
                        if survivors != [actual_record]:
                            continue

                        has_same_rotation_distractor = any(
                            match[1] // 2 == actual_rotation and match != actual_record for match in matches
                        )
                        has_marker_compatible_other_rotation_distractor = any(
                            match[1] // 2 != actual_rotation
                            and match[0] == int((match[1] % 2) == actual_parity)
                            for match in matches
                        )
                        if not (
                            has_same_rotation_distractor
                            and has_marker_compatible_other_rotation_distractor
                        ):
                            continue

                        records.append((H, W, variant_index, r, c))

    assert records
    return tuple(records)


@lru_cache(maxsize=1)
def _records_by_variant() -> tuple[tuple[tuple[int, int, int, int, int], ...], ...]:
    grouped: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(8)]
    for record in _ambiguous_records():
        grouped[record[2]].append(record)
    assert all(grouped)
    return tuple(tuple(group) for group in grouped)


def generate_input() -> np.ndarray:
    for _ in range(200):
        fg, bg, clue = random.sample(Color.ALL_COLORS, 3)
        variant_index = random.randrange(8)
        H, W, variant_index, r, c = random.choice(_records_by_variant()[variant_index])
        pattern = _variants()[variant_index]
        crop_mask = pattern[r : r + H, c : c + W]
        crop = np.where(crop_mask, int(fg), int(bg)).astype(int)

        matches = _crop_matches(crop)
        assert len(matches) > 1
        actual_match = (int(fg), int(bg), variant_index, r, c)
        assert any(
            match[2] // 2 == variant_index // 2 and match[:5] != actual_match for match in matches
        )
        assert any(
            match[2] // 2 != variant_index // 2
            and _marker_color_compatible(match, int(fg if variant_index % 2 == 0 else bg))
            and match[:5] != actual_match
            for match in matches
        )

        inp = np.full((H + 2, W + 2), int(clue), dtype=int)
        inp[1:-1, 1:-1] = crop
        marker_r, marker_c = _corner_positions(H + 2, W + 2)[variant_index // 2]
        inp[marker_r, marker_c] = int(fg if variant_index % 2 == 0 else bg)

        expected = np.where(pattern, int(fg), int(bg)).astype(int)
        if np.array_equal(main(inp), expected):
            return inp

    raise AssertionError("failed to construct a valid combo sample")
