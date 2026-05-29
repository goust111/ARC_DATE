from __future__ import annotations

# Variant of 269e22fb
# Dimension: D8 — 颜色系统：输入颜色数量和语义 (meta)
# Change: 恰好2种不同的ARC颜色，任意颜色对均可出现，一种映射为前景一种映射为背景... → 三色输入含边框标记色

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the D8 change
# ---------------------------------
# a) Concrete behavior:
#    The original input was an HxW two-color crop taken directly from the 20x20 motif.
#    The variant wraps that same crop in a one-cell rectangular frame of a third color.
#    Example: an original 10x8 crop becomes a 12x10 input where row 0, row -1, col 0,
#    and col -1 are all the marker color. The inner 10x8 area is unchanged. The output
#    remains the full 20x20 motif in only the inner two colors.
#    Flow difference: original solver matched the whole input as the crop; the variant
#    must first detect that the perimeter is metadata, strip it, and only then run the
#    same crop-matching logic.
# b) Complexity calibration:
#    The extra color is visually obvious but does not add uncontrolled freedom.
#    Border width is fixed at exactly one cell and uses exactly one distinct color.
#    Interior crop sizes stay in the original ranges, so uniqueness comes from the same
#    binary motif rather than from arbitrary border patterns.
# c) Old implicit assumptions that no longer hold:
#    - `len(colors) == 2`: no longer true; we now require exactly 3 colors overall.
#    - `H, W` being the match window size: no longer true; the match window is `H-2, W-2`.
#    - Every visible cell participates in motif polarity: false in the variant because the
#      perimeter cells are border metadata and must be ignored during matching and output.
# d) Placement legality:
#    The generator first picks a legal original crop entirely inside the 20x20 motif, then
#    pads it by one cell on every side. This cannot overlap anything because the input only
#    contains the framed crop itself, and the output is still the standalone 20x20 motif.
# e) Visual observability:
#    Across examples, the new behavior is easy to see: every input has a uniform one-cell
#    frame in a third color, while outputs never contain that color. A human can infer that
#    the frame marks the crop boundary and should be discarded before completion.
# f) Visual cleanliness:
#    The border is separated from the motif by role and by color count. There is no stacking
#    of multiple annotations on the same cells, so the grid stays clean.
# g) Color-rule inferability:
#    The third color has only one structural role: the outer ring. The two interior colors
#    keep the original foreground/background semantics, and the border color never appears in
#    the output. This is directly readable from 2-3 examples.
# h) Quantitative anti-degeneration target:
#    Generated samples must always have exactly 3 colors, a uniform outer ring, and a border
#    color absent from the interior. Under these inputs the original two-color solver would
#    fail immediately, so the new border-stripping behavior is necessary rather than cosmetic.
# i) New solver assumptions to audit:
#    - The outermost ring is uniform and exactly one cell thick.
#    - The interior contains exactly 2 colors and excludes the border color.
#    - After stripping the ring, the interior crop still has a unique motif match.
#    `generate_input()` enforces all three conditions.


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
_CROP_H_RANGE = (8, 16)
_CROP_W_RANGE = (8, 12)
_MAX_GEN_ATTEMPTS = 400


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


def _match_candidates(inner: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    inner = np.asarray(inner, dtype=int)
    H, W = map(int, inner.shape)
    assert H <= 20 and W <= 20

    colors = colors_present(inner)
    assert len(colors) == 2

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = inner == int(fg)
        for pattern in _VARIANTS:
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(fg), int(bg), pattern))
    return matches


def _strip_border(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 3 and W >= 3

    colors = colors_present(g)
    assert len(colors) == 3

    perimeter = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    border_colors = colors_present(perimeter)
    assert len(border_colors) == 1
    border = int(border_colors[0])

    inner = g[1:-1, 1:-1]
    inner_colors = colors_present(inner)
    assert len(inner_colors) == 2
    assert border not in inner_colors
    return inner, border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a framed crop in the correct orientation.

    Input structure:
    - The grid uses exactly 3 colors.
    - A uniform one-cell border in the third color surrounds the actual crop.
    - The framed interior is the original two-color rectangular window from one dihedral variant of the shared motif.

    Rule:
    - Detect the perimeter marker color and remove that one-cell frame.
    - Treat the remaining two interior colors as candidate foreground/background assignments.
    - Match the stripped crop against every offset in every dihedral orientation, then repaint the full 20x20 motif using only the matched interior colors.

    Key insight:
    - The added third color is only metadata that marks the crop boundary; once the frame is stripped away, the task reduces to the same unique local-pattern identification as the base puzzle.
    """
    inner, _border = _strip_border(grid)
    matches = _match_candidates(inner)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _render_output(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _add_border(inner: np.ndarray, border: int) -> np.ndarray:
    H, W = map(int, inner.shape)
    framed = np.full((H + 2, W + 2), int(border), dtype=int)
    framed[1:-1, 1:-1] = inner
    return framed


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        fg, bg, border = random.sample(_PALETTE, 3)
        pattern = _VARIANTS[random.randrange(len(_VARIANTS))]
        output = _render_output(pattern, fg, bg)

        crop_h = random.randint(*_CROP_H_RANGE)
        crop_w = random.randint(*_CROP_W_RANGE)
        r0 = random.randint(0, 20 - crop_h)
        c0 = random.randint(0, 20 - crop_w)
        inner = output[r0 : r0 + crop_h, c0 : c0 + crop_w].copy()

        if len(_match_candidates(inner)) != 1:
            continue

        framed = _add_border(inner, border)
        if np.array_equal(main(framed), output):
            return framed

    raise RuntimeError("failed to generate a unique framed crop for variant D8")
