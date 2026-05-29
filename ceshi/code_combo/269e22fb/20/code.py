# Combo variant of 269e22fb (index 20)
# combo_spec: D6/3+D17/0
# Name: 环面平移变体空间

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for combo D6/3 + D17/0
# ---------------------------------------
# 1. Concrete behavior trace
#    - Original task:
#      choose one of 8 dihedral orientations of the fixed 20x20 binary motif, take a planar
#      in-bounds crop at offset (r, c), and ask the solver to reconstruct the full oriented motif.
#    - D17/0 alone:
#      the crop may wrap across the frame edge, so the solver must compare against toroidal
#      windows indexed by all 20x20 start offsets.
#    - D6/3 adds cyclic translations as part of the hidden variant family itself.
#      On a torus these two freedoms collapse:
#      `torus_crop(pattern, r, c, h, w) == roll(pattern, (-r, -c))[:h, :w]`.
#      Example: the rows [14,15,16,17,18,19,0,1,2,3] and cols [9..18] seen from the canonical
#      torus are exactly the top-left 10x10 crop of the full output after a toroidal shift
#      by (14, 9). This is the core fused behavior: the old crop offset becomes the recovered
#      output translation.
#    - Therefore the generator must not randomize both a hidden translation and an independent
#      crop origin. That would create many equivalent explanations for one input. The clean fused
#      formulation is:
#        1. pick an oriented motif,
#        2. pick a toroidal shift,
#        3. roll the full 20x20 output by that shift,
#        4. expose only its top-left h x w rectangle as input.
#
# 2. Complexity calibration
#    - Original planar search space: at most 8 orientations x 13 row offsets x 9 col offsets x
#      2 color polarities = 1872 hypotheses.
#    - Fused search space: 8 orientations x 20 row shifts x 20 col shifts x 2 polarities = 6400
#      hypotheses.
#    - Exhaustive checks on this fixed motif show that every wrapped crop in the original size
#      family h in [8,16], w in [8,12] is still uniquely identifying even in the enlarged
#      orientation+translation space, so the generator can keep the same crop sizes while forcing
#      at least one wrapped axis to make the torus visible.
#
# 3. Implicit assumption audit
#    - `h <= 20` and `w <= 20` still hold because a toroidal window cannot exceed one full loop.
#    - The original planar loops `range(20-h+1)` / `range(20-w+1)` are invalid now. Matching must
#      enumerate all 20x20 torus starts because wraparound is legal.
#    - The original slice `pattern[r:r+h, c:c+w]` is invalid under wraparound and must become
#      modular indexing.
#    - Returning the unshifted matched orientation is no longer correct: the matched torus offset
#      now defines the output translation and must be applied with `np.roll`.
#    - Color polarity logic still works because inputs remain exactly two-colored.
#
# 4. Placement legality and cleanliness
#    - There is still only one object: the shared 20x20 motif. No overlap constraints arise.
#    - Legal generation reduces to choosing h, w in range and a torus shift where `r + h > 20`
#      or `c + w > 20`, so the visible top-left crop crosses a seam of the underlying canonical
#      motif.
#    - The input stays visually clean: one two-color rectangle, no extra markers, no stacked
#      annotations. The only new visual cue is the seam created by toroidal wrap.
#
# 5. Visual observability and inferability
#    - Humans can see that the visible rectangle may splice opposite edges of the motif because
#      contiguous runs in the crop align only if the hidden pattern wraps.
#    - The output is not merely the canonical orientation from D17/0: it is a translated full
#      motif. Since this motif has no non-trivial toroidal symmetry, any non-zero recovered shift
#      changes the 20x20 output, making the D6/3 contribution observable across examples.
#    - The generator samples row-wrap only, col-wrap only, or both, so examples repeatedly expose
#      the torus seam while varying the recovered translation in a small, interpretable set of
#      cases rather than adding unrelated freedoms.


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


def _translate(mask: np.ndarray, r: int, c: int) -> np.ndarray:
    return np.roll(mask, shift=(-int(r), -int(c)), axis=(0, 1))


def _paint(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _choose_offset(rng: random.Random, size: int, wrap_axis: bool) -> int:
    if wrap_axis:
        return rng.randrange(20 - size + 1, 20)
    return rng.randrange(0, 20 - size + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the exact translated 20x20 torus motif from a wrapped two-color crop.

    Input structure:
    - The grid is a two-color rectangle cut from the top-left corner of one hidden 20x20 motif state.
    - That hidden state is some rotation/reflection of the canonical motif, followed by a toroidal row/column translation.

    Rule:
    - Treat each input color as the candidate foreground of the binary motif.
    - Compare the visible mask against every toroidal offset of every dihedral orientation of the canonical 20x20 pattern.
    - When one torus offset matches, convert that offset into the corresponding full translated motif and repaint it with the recovered foreground/background colors.

    Key insight:
    - On a torus, the crop start and the hidden global translation are the same degree of freedom: matching a wrapped window at offset (r, c) tells you both which orientation is present and how the full 20x20 output must be cyclically shifted.
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
                        matches.append((int(fg), int(bg), _translate(pattern, r, c)))

    assert len(matches) == 1
    fg, bg, shifted_pattern = matches[0]
    return _paint(shifted_pattern, fg, bg)


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
        shifted = _translate(pattern, r, c)
        inp = _paint(shifted[:h, :w], fg, bg)
        out = _paint(shifted, fg, bg)

        assert r + h > 20 or c + w > 20
        if np.array_equal(main(inp), out):
            return inp

    raise RuntimeError("failed to sample a valid torus-translation crop")
