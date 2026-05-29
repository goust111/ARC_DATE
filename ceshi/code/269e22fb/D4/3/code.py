# Variant of 269e22fb
# Dimension: D4 — 变换：颜色重绘规则 (transform)
# Change: 用裁剪窗口中的前景色和背景色直接重绘完整图案（前景->color_a, 背景->color_b）... → 输出使用四色编码象限

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for this D4 variant
# ===================================
# a) Concrete behavior
#    - Keep the original 20x20 binary motif and the original dihedral search.
#    - Change only the repaint stage: the final 20x20 board is split into four fixed
#      10x10 quadrants, and each quadrant has its own ordered (fg, bg) color pair.
#    - The input is still a rectangular crop, but it must stay completely inside one
#      quadrant so it remains exactly two-colored. Those two colors tell us which
#      quadrant the crop came from, then the matched full motif is repainted with the
#      quadrant-specific palette for all four quadrants.
#    - Example trace:
#      original: a red/green crop that matches the motif is expanded into a full
#      red/green 20x20 output.
#      variant: the same local binary crop, if colored red/green and that pair is the
#      top-right quadrant code, is expanded into a four-color output where top-right is
#      red/green while the other three quadrants use their own fixed pairs.
#
# b) Complexity calibration
#    - If quadrant palettes were randomized per instance, the two unseen output colors
#      would be underdetermined from the test input alone. To keep the task human-
#      solvable, use one fixed four-color cycle across the whole task.
#    - The unordered color pairs are all distinct, so the crop colors identify one
#      quadrant without relying on color-id sorting.
#    - Crops are restricted to sizes 8..10 by 8..10. This is the smallest change that
#      still fits a whole crop inside a 10x10 quadrant while preserving enough local
#      structure for unique matching by rejection/pre-enumeration.
#
# c) Hidden assumptions in the original main()
#    - `assert len(colors) == 2` still holds, but now only because the generator keeps
#      every crop inside a single quadrant. Any cross-quadrant crop would violate the
#      two-color invariant.
#    - The old loop tried each input color as global foreground. That is no longer
#      valid: the crop colors now mean "which quadrant palette is active here", so the
#      foreground color must come from that quadrant's ordered pair.
#    - The old return `np.where(pattern, fg, bg)` assumed one global palette for the
#      whole board. That is the exact line that must change: repaint now happens per
#      quadrant with four different ordered pairs.
#
# d) Placement legality
#    - A valid crop must satisfy: H,W in 8..10; chosen start row/col keep the crop
#      fully inside one 10x10 quadrant; the crop must contain both colors; and the crop
#      must have exactly one legal match among all dihedral variants and all offsets
#      allowed in that same quadrant.
#    - The generator pre-enumerates only such legal windows, so `main()` sees only
#      inputs it is designed to solve.
#
# e) Visual observability
#    - The new rule is obvious in the outputs: instead of one uniform two-color motif,
#      the 20x20 answer is quartered into four color-coded regions.
#    - The inputs remain simple local crops, so the novel behavior is concentrated in
#      output recoloring rather than in extra clutter.
#
# f) Visual cleanliness
#    - No extra markers are added. The only new structure is the clean 10x10 quadrant
#      partition, which is easy to read and does not overlap the motif geometry.
#
# g) Color-rule inferability
#    - Use the fixed cycle:
#        TL = (blue, red)
#        TR = (red, green)
#        BL = (yellow, blue)
#        BR = (green, yellow)
#      Each unordered pair is unique, so the crop colors act as a visible quadrant code.
#      The same four colors recur across examples, letting a human infer the full map.
#
# New main() assumptions audited
#    - The crop colors must equal exactly one quadrant's unordered pair.
#    - The crop must lie fully inside that quadrant, otherwise the solver's restricted
#      offset search would miss the true placement.
#    - The quadrant pair also fixes local fg/bg polarity; the generator enforces this by
#      cutting the input from the already-rendered quadrant-coded output.


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

_QUADRANT_BOUNDS = (
    (0, 10, 0, 10),   # TL
    (0, 10, 10, 20),  # TR
    (10, 20, 0, 10),  # BL
    (10, 20, 10, 20), # BR
)

# Fixed four-color cycle shared by the whole task.
_QUADRANT_PAIRS = (
    (1, 2),  # TL: blue/red
    (2, 3),  # TR: red/green
    (4, 1),  # BL: yellow/blue
    (3, 4),  # BR: green/yellow
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


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)


def _quadrant_offsets(quadrant: int, H: int, W: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _QUADRANT_BOUNDS[quadrant]
    return [(r, c) for r in range(r0, r1 - H + 1) for c in range(c0, c1 - W + 1)]


def _render_output(pattern: np.ndarray) -> np.ndarray:
    out = np.zeros((20, 20), dtype=int)
    for quadrant, (r0, r1, c0, c1) in enumerate(_QUADRANT_BOUNDS):
        fg, bg = _QUADRANT_PAIRS[quadrant]
        out[r0:r1, c0:c1] = np.where(pattern[r0:r1, c0:c1], int(fg), int(bg))
    return out


def _quadrant_from_colors(colors: list[int]) -> int:
    color_set = {int(c) for c in colors}
    candidates = [
        quadrant
        for quadrant, (fg, bg) in enumerate(_QUADRANT_PAIRS)
        if {int(fg), int(bg)} == color_set
    ]
    assert len(candidates) == 1
    return int(candidates[0])


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 2

    quadrant = _quadrant_from_colors(colors)
    fg, _bg = _QUADRANT_PAIRS[quadrant]
    visible = g == int(fg)

    matches: list[tuple[int, int, np.ndarray]] = []
    for pattern in _VARIANTS:
        for r, c in _quadrant_offsets(quadrant, H, W):
            if np.array_equal(pattern[r : r + H, c : c + W], visible):
                matches.append((int(r), int(c), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full motif, but repaint each 10x10 quadrant with its own fixed two-color code.

    Input structure:
    - The grid is a two-color crop taken entirely from one 10x10 quadrant of a rotated/flipped 20x20 motif.
    - Each quadrant has a different ordered foreground/background color pair, so the crop colors identify the source quadrant.

    Rule:
    - Use the crop's two colors to determine which quadrant palette is present and therefore which local foreground color to read.
    - Match that foreground mask against every allowed offset inside that quadrant across all dihedral variants of the motif.
    - Rebuild the uniquely matched full 20x20 motif and repaint each quadrant with its own fixed color pair.

    Key insight:
    - The geometric search is the same as the original task; the new twist is that the observed colors are a quadrant code, not a global palette for the entire output.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 10 and W <= 10

    matches = _find_matches(g)
    assert len(matches) == 1
    _r, _c, pattern = matches[0]
    return _render_output(pattern)


def _enumerate_valid_cases() -> list[tuple[int, int, int, int, int, int]]:
    cases: list[tuple[int, int, int, int, int, int]] = []
    for variant_idx, pattern in enumerate(_VARIANTS):
        full = _render_output(pattern)
        for quadrant in range(4):
            for H in range(8, 11):
                for W in range(8, 11):
                    for r, c in _quadrant_offsets(quadrant, H, W):
                        crop = full[r : r + H, c : c + W]
                        if len(colors_present(crop)) != 2:
                            continue
                        matches = _find_matches(crop)
                        if len(matches) == 1:
                            cases.append((variant_idx, quadrant, H, W, r, c))
    assert cases
    return cases


_VALID_CASES = _enumerate_valid_cases()


def _build_case(spec: tuple[int, int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    variant_idx, _quadrant, H, W, r, c = spec
    pattern = _VARIANTS[variant_idx]
    output = _render_output(pattern)
    input_grid = output[r : r + H, c : c + W].copy()
    return input_grid, output


def generate_input() -> np.ndarray:
    spec = random.choice(_VALID_CASES)
    input_grid, output = _build_case(spec)
    assert np.array_equal(main(input_grid), output)
    return input_grid
