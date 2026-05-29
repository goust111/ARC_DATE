# Variant of 269e22fb
# Dimension: D17 — 图案的拓扑结构 (meta)
# Change: 平面拓扑，图案有明确边界，裁剪窗口不跨越边界... → 环面拓扑（上下/左右循环连接）

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the D17 torus variant
# -----------------------------------------
# variant_spec:
#   task: Recover the full 20x20 motif from a two-color crop taken on a torus.
#   unchanged_core:
#     - The hidden object is still the same fixed 20x20 binary motif.
#     - The motif may still appear in any of the 8 dihedral orientations.
#     - Input and output still use exactly 2 colors, with unknown fg/bg polarity.
#   changed_dimension:
#     - Original task: the crop is a planar rectangle fully contained inside the 20x20 frame.
#     - Variant: the 20x20 frame is treated as a torus, so the crop may wrap across the top/bottom
#       edge, the left/right edge, or both. The visible rectangle can therefore splice together
#       opposite sides of the motif.
#   concrete_behavior_trace:
#     - Original example logic:
#       A 10x10 crop starting at (14, 9) is only legal if rows 14..23 and cols 9..18 stay in-bounds,
#       so the original generator would reject it; only starts with r <= 10 and c <= 10 exist.
#     - Variant logic:
#       The same nominal start means rows [14,15,16,17,18,19,0,1,2,3] and cols [9..18].
#       The input rectangle shows a seam where the motif's bottom connects back to its top.
#     - Consequence:
#       Solver search must enumerate all 20x20 toroidal offsets instead of only the planar
#       offsets 0..20-H and 0..20-W.
#   complexity_calibration:
#     - The search space grows from at most 8 * 13 * 9 * 2 = 1872 planar hypotheses
#       to 8 * 20 * 20 * 2 = 6400 toroidal hypotheses.
#     - Exhaustive checks on this fixed motif show that every wrapped crop in the original size
#       family H in [8,16], W in [8,12] is uniquely identifying across all orientations and both
#       polarities. That lets the generator keep the original size range while forcing wraparound.
#   implicit_assumption_audit:
#     - `assert H <= 20 and W <= 20` still holds; toroidal crops cannot exceed one full loop.
#     - The original nested loops `for r in range(20-H+1)` and `for c in range(20-W+1)` relied on
#       the crop never crossing the border. This is false now and must become `range(20)` on both axes.
#     - The original slice `pattern[r:r+H, c:c+W]` assumed planar indexing. It must be replaced by
#       modular row/column lookup.
#     - Color polarity handling remains valid because the crop still contains exactly 2 colors.
#   placement_legality:
#     - There is still only one hidden object, so overlap is not an issue.
#     - Generator legality reduces to choosing H, W <= 20 and an offset where at least one axis wraps.
#     - The generator uses output-first construction and calls `main()` on the sampled input to verify
#       the solver's assumptions before returning.
#   visual_observability:
#     - The seam is directly visible: many inputs begin with a band that visually continues from the
#       opposite side of the 20x20 output instead of from a nearby local neighborhood.
#     - To avoid degeneration back to the planar task, the generator always wraps on row, column,
#       or both; it never returns a fully contained crop.
#   cleanliness:
#     - No extra markers are added. The grid stays as a single clean binary motif recolored with 2 colors.
#   solver_assumptions_after_change:
#     - The crop must be rectangular, two-colored, and no larger than 20x20.
#     - The crop must come from a dihedral variant of the fixed motif under toroidal extraction.
#     - The crop must remain uniquely identifying. `generate_input()` guarantees this by only sampling
#       wrapped windows from a size range already checked to be unique and by self-validating via `main()`.


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

_ARC_COLORS = tuple(range(10))


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _torus_crop(mask: np.ndarray, r: int, c: int, h: int, w: int) -> np.ndarray:
    rows = (np.arange(h) + int(r)) % 20
    cols = (np.arange(w) + int(c)) % 20
    return mask[np.ix_(rows, cols)]


def _paint(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _choose_offset(rng: random.Random, size: int, wrap_axis: bool) -> int:
    if wrap_axis:
        return rng.randrange(20 - size + 1, 20)
    return rng.randrange(0, 20 - size + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the shared 20x20 motif from a toroidal crop in the correct orientation.

    Input structure:
    - The grid is a two-color rectangular crop taken from one dihedral variant of the same 20x20 motif.
    - The hidden motif lives on a torus, so the crop may wrap past the top/bottom edge, the left/right edge, or both.

    Rule:
    - Treat each input color as a candidate foreground mask.
    - Compare that mask against every toroidal offset of every rotation/reflection of the canonical motif.
    - Repaint the uniquely matching full 20x20 motif using the recovered foreground/background colors.

    Key insight:
    - Even when the visible rectangle splices together opposite sides of the pattern, the wrapped local arrangement is still distinctive enough to pin down one orientation, one toroidal offset, and one color polarity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    base = _pattern_mask()
    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20):
                for c in range(20):
                    if np.array_equal(_torus_crop(pattern, r, c, h, w), visible):
                        matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return _paint(pattern, fg, bg)


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng

    base = _pattern_mask()
    pattern = rng.choice(_dihedral_variants(base))
    fg, bg = rng.sample(_ARC_COLORS, 2)

    for _ in range(100):
        h = rng.randint(8, 16)
        w = rng.randint(8, 12)
        wrap_rows, wrap_cols = rng.choice(((True, False), (False, True), (True, True)))
        r = _choose_offset(rng, h, wrap_rows)
        c = _choose_offset(rng, w, wrap_cols)
        assert r + h > 20 or c + w > 20

        inp = _paint(_torus_crop(pattern, r, c, h, w), fg, bg)
        out = _paint(pattern, fg, bg)
        if np.array_equal(main(inp), out):
            return inp

    raise RuntimeError("failed to sample a valid toroidal crop")
