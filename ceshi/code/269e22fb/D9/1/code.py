# Variant of 269e22fb
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 固定20x20输出，与基础图案尺寸相同... → 输出与输入同尺寸但展示不同区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   inherited_core:
#     motif: fixed 20x20 binary pattern, still used in 8 dihedral orientations
#     input_colors: exactly 2 colors, one for motif foreground and one for motif background
#     identification: the visible crop must still uniquely determine orientation, offset, and polarity
#   changed_dimension:
#     name: D9 / grid_shape
#     old_output: the whole 20x20 matched motif
#     new_output: a window with the same HxW size as the input crop
#     new_content: the window whose position is point-reflected through the center of the 20x20 motif
#   concrete_behavior:
#     if the matched input crop is pattern[r:r+H, c:c+W], the output is
#     pattern[20-H-r:20-r, 20-W-c:20-c] with the same two colors.
#     The output therefore shows the "opposite side" of the same oriented motif.
#   behavior_trace:
#     original:
#       1. infer the unique (variant, r, c, fg, bg)
#       2. repaint the full 20x20 variant
#     variant:
#       1. infer the same unique (variant, r, c, fg, bg)
#       2. compute reflected offset (r2, c2) = (20-H-r, 20-W-c)
#       3. repaint only pattern[r2:r2+H, c2:c2+W]
#     difference:
#       the recognition step is unchanged; only the final projection target changes
#       from "entire motif" to "center-opposite window".
#   complexity_calibration:
#     no extra object types are introduced, so the task stays close to the source.
#     The main ambiguity risk remains crop non-uniqueness, so the generator preserves
#     the original uniqueness rejection test. To make the D9 change visually obvious,
#     it additionally rejects centered crops where the reflected window is the same
#     location and thus could collapse to an identity-looking example.
#   implicit_assumption_audit:
#     - colors_present(grid) == 2 is still required and is enforced by generation.
#     - H, W <= 20 still holds because the crop comes from the same 20x20 motif.
#     - A unique match is still required; otherwise the reflected partner window would
#       also be ambiguous, so the solver keeps assert len(matches) == 1.
#     - The old solver only stored the matched pattern because it always returned the
#       full motif. This variant also needs the matched offset (r, c), so matches now
#       store those coordinates explicitly.
#   placement_legality:
#     generation only slices valid windows from a 20x20 oriented motif, so the input
#     and reflected output are automatically in-bounds, non-overlapping concerns do not
#     arise, and the solver's reflection formula always lands on a legal HxW window.
#   visual_observability:
#     humans can see the rule from examples because output size now equals input size
#     and the shown region moves to the opposite side of the same global pattern.
#     The binary motif is asymmetric enough that opposite windows usually differ.
#   cleanliness:
#     still exactly two colors and one rectangular crop, so the images remain tidy.
#   color_inference:
#     colors are preserved verbatim; there is no hidden color remapping.


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

_SIZE = 20
_ROW_RANGE = (8, 16)
_COL_RANGE = (8, 12)
_ALL_COLORS = tuple(range(10))
_GEN_RETRIES = 400


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
    colors = colors_present(g)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(_pattern_mask()):
            for r in range(_SIZE - H + 1):
                for c in range(_SIZE - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a crop's matched motif placement and return the equally sized crop from the opposite side.

    Input structure:
    - The grid is a two-color rectangular window cut from one dihedral orientation of the shared 20x20 binary motif.
    - One input color marks motif cells and the other marks non-motif cells.

    Rule:
    - Try each input color as the motif foreground and locate the unique matching window across all rotations/reflections.
    - After finding the matching orientation and offset (r, c), compute the point-reflected offset inside the 20x20 frame.
    - Repaint that opposite HxW window with the same foreground/background colors.

    Key insight:
    - The visible crop still serves only as an identifier for the hidden full motif placement; once that placement is known, the task is to look up the window across the center rather than to reconstruct the whole motif.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= _SIZE and W <= _SIZE

    matches = _find_matches(g)
    assert len(matches) == 1

    fg, bg, pattern, r, c = matches[0]
    r2 = _SIZE - H - int(r)
    c2 = _SIZE - W - int(c)
    opposite = pattern[r2 : r2 + H, c2 : c2 + W]
    return np.where(opposite, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    """
    Generate a uniquely identifiable crop whose target is the center-opposite window.

    The generator keeps the original crop-size range and uniqueness filter, then rejects
    centered windows so the D9 behavior is visually active in the produced examples.
    """
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    rng = random.Random()

    for _ in range(_GEN_RETRIES):
        H = rng.randint(*_ROW_RANGE)
        W = rng.randint(*_COL_RANGE)
        pattern = variants[rng.randrange(len(variants))]
        r = rng.randint(0, _SIZE - H)
        c = rng.randint(0, _SIZE - W)
        r2 = _SIZE - H - int(r)
        c2 = _SIZE - W - int(c)
        if r2 == r and c2 == c:
            continue

        fg, bg = rng.sample(_ALL_COLORS, 2)
        crop = pattern[r : r + H, c : c + W]
        opposite = pattern[r2 : r2 + H, c2 : c2 + W]
        if np.array_equal(crop, opposite):
            continue

        inp = np.where(crop, int(fg), int(bg)).astype(int)
        matches = _find_matches(inp)
        if len(matches) != 1:
            continue

        expected = np.where(opposite, int(fg), int(bg)).astype(int)
        out = main(inp)
        if np.array_equal(out, expected):
            return inp

    raise RuntimeError("failed to generate a unique non-degenerate D9 sample")
