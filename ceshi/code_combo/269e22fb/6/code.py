# Combo variant of 269e22fb (index 6)
# combo_spec: D4/0+D10/2
# Name: 条件性双重变换输出
#
# Design analysis
# spec_like:
#   concrete_behavior:
#     - Input keeps the original task's physical layout: a 2-color rectangular crop cut from one
#       dihedral variant of the fixed 20x20 binary motif.
#     - The crop is still taken from the pre-transform full motif. This preserves the original
#       identification subproblem: infer orientation, offset, and polarity from a local window.
#     - After matching, the output no longer uses the legacy repainting rule. First invert the
#       recovered motif polarity, so cells that belonged to the motif foreground become background
#       color and vice versa.
#     - Then inspect the crop width. If the width is odd, rotate the full inverted motif 90 degrees
#       counterclockwise; if the width is even, keep the inverted motif orientation unchanged.
#     - Small trace:
#       legacy:  9x10 crop -> recover pattern V -> output paint(V, fg, bg)
#       combo:   9x10 crop -> recover pattern V -> invert to ~V -> width even -> output paint(~V, fg, bg)
#       legacy:  9x11 crop -> recover pattern V -> output paint(V, fg, bg)
#       combo:   9x11 crop -> recover pattern V -> invert to ~V -> width odd  -> output paint(rot90(~V), fg, bg)
#   complexity_calibration:
#     - The fusion introduces exactly two post-match layers with low branching factor: inversion is
#       always on, and rotation has only two states controlled by width parity.
#     - Width parity is directly visible from the rectangular crop, so the second rule is observable
#       without relying on hidden metadata such as sample index or color ids.
#     - The hard-coded motif is neither self-complementary nor 90-degree symmetric, so inversion and
#       odd-width rotation both create visible changes instead of collapsing back to the original task.
#   implicit_assumptions_review:
#     - `colors_present(grid)` must still return exactly two colors. The generator keeps that invariant.
#     - `_find_unique_match` assumes the input window came from an unmodified dihedral motif. The
#       generator therefore crops from the raw matched canvas, never from the post-transform output.
#     - The exhaustive search over both color polarities still works because the crop itself is not
#       rotated or inverted by the combo rule; all extra logic happens after full reconstruction.
#     - `assert len(matches) == 1` now depends on generator-side rejection exactly as before. Ambiguous
#       crops are retried until a single legacy match is found.
#     - The new width-parity branch assumes the full reconstruction is square. The motif is fixed at
#       20x20, so `np.rot90` preserves shape and remains valid for every accepted instance.
#     - The inversion layer assumes the two input colors are distinct. The palette sampler enforces this.
#   placement_validity:
#     - Crops remain axis-aligned windows fully inside the 20x20 source motif, with the original size
#       constraints H in [8,16] and W in [8,12].
#     - No extra markers or overlay objects are introduced, so there are no new overlap constraints.
#     - Because the conditional rotation is applied only after reconstruction, it cannot move the crop
#       out of bounds or invalidate the sampling geometry.
#   visual_observability:
#     - Humans can compare input colors to output colors and see that motif/background roles are swapped.
#     - Width parity is visible from the input rectangle itself, and the odd-width branch changes the
#       whole-image orientation, which is a strong global signal.
#     - With 2-4 examples, the learner can separate the always-on inversion from the parity-triggered
#       rotation because even-width examples show inversion alone while odd-width examples show both.
#   visual_cleanliness:
#     - The task still contains one motif and exactly two colors per grid, so the canvas stays as clean
#       as the original task. The fusion changes only interpretation, not scene clutter.
#   color_change_inference:
#     - The color mapping uses only the two already-present colors. No extra palette states are added.
#     - The inversion is structurally anchored to the recovered motif/background partition, making the
#       color swap inferable from shape rather than from arbitrary color-id arithmetic.
#   main_assumptions_after_change:
#     - The matching loops assume the crop dimensions satisfy H <= 20 and W <= 20; generation keeps the
#       original crop ranges, so every sampled window is legal.
#     - The final branch on `crop_width % 2` assumes both parity cases appear in training-style data.
#       The generator samples odd and even widths explicitly to avoid silent branch starvation.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present


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

_PALETTE = tuple(range(10))
_EVEN_WIDTHS = (8, 10, 12)
_ODD_WIDTHS = (9, 11)
_PARITY_CURSOR = 0
_WIDTH_CURSOR = {0: 0, 1: 0}


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern.copy()))

    assert len(matches) == 1
    return matches[0]


def _invert_pattern(pattern: np.ndarray) -> np.ndarray:
    return np.logical_not(pattern)


def _conditional_rotate(pattern: np.ndarray, crop_width: int) -> np.ndarray:
    if int(crop_width) % 2 == 1:
        return np.rot90(pattern, 1)
    return pattern


def _post_transform(pattern: np.ndarray, crop_width: int) -> np.ndarray:
    inverted = _invert_pattern(pattern)
    return _conditional_rotate(inverted, crop_width)


def _paint(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden 20x20 motif, invert its polarity, and then apply a width-triggered rotation.

    Input structure:
    - The grid is a two-color rectangular crop from one rotation/reflection of the shared 20x20 motif.
    - The crop shows the motif before any combo-specific output transform, so it only exposes local structure, its two colors, and the crop width.

    Rule:
    - Treat each input color in turn as the visible motif foreground and find the unique orientation/offset whose window matches the crop.
    - Reconstruct the full matched motif, then invert that binary motif so foreground and background roles swap in the final coloring.
    - If the crop width is odd, rotate the inverted full motif 90 degrees counterclockwise; if the width is even, leave the inverted motif as is, then paint it with the matched two colors.

    Key insight:
    - The task is now a two-layer deduction: first solve the original crop-to-motif identification problem, then notice that the completed answer always flips motif polarity and sometimes also rotates based on the observable width parity of the crop.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = map(int, g.shape)
    fg, bg, pattern = _find_unique_match(g)
    transformed = _post_transform(pattern, w)
    return _paint(transformed, fg, bg)


def generate_input() -> np.ndarray:
    global _PARITY_CURSOR

    target_parity = _PARITY_CURSOR % 2
    while True:
        fg, bg = map(int, np.random.choice(_PALETTE, size=2, replace=False))
        pattern = _VARIANTS[int(np.random.randint(len(_VARIANTS)))]

        H = int(np.random.randint(8, 17))
        width_pool = _ODD_WIDTHS if target_parity == 1 else _EVEN_WIDTHS
        width_index = _WIDTH_CURSOR[target_parity] % len(width_pool)
        W = int(width_pool[width_index])
        r = int(np.random.randint(0, 20 - H + 1))
        c = int(np.random.randint(0, 20 - W + 1))

        raw_full = _paint(pattern, fg, bg)
        inp = raw_full[r : r + H, c : c + W].copy()

        try:
            m_fg, m_bg, matched = _find_unique_match(inp)
        except AssertionError:
            continue

        legacy = _paint(pattern, fg, bg)
        inversion_only = _paint(_invert_pattern(pattern), fg, bg)
        expected = _paint(_post_transform(pattern, W), fg, bg)

        if (m_fg, m_bg) != (fg, bg):
            continue
        if not np.array_equal(matched, pattern):
            continue
        if np.array_equal(expected, legacy):
            continue
        if W % 2 == 0 and not np.array_equal(expected, inversion_only):
            continue
        if W % 2 == 1 and np.array_equal(expected, inversion_only):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        _WIDTH_CURSOR[target_parity] += 1
        _PARITY_CURSOR += 1
        return inp
