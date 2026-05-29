# Variant of eee78d87
# Dimension: D2 — 源对象的形状集合 (src)
# Change: 4种固定的3x3轮廓字形：plus(.#./#11/.#.)、H(#.#/###/#.#)、X(#.#/.#./#.#)... → 字形扩展为4x4或5x5尺寸
"""
Design analysis:
  concrete_spec:
    input_grid: fixed 10x10 orange canvas so both 4x4 and 5x5 motifs fit with position freedom
    source_object:
      count: exactly 1
      bbox: exactly 4x4 or 5x5
      color: exactly 1 non-orange color chosen from {1,2,3,4,5,6,8}
      families:
        - plus: expand coarse 3x3 plus by row/col bands [1, size-2, 1]
        - H: expand coarse 3x3 H by the same bands
        - X: expand coarse 3x3 X by the same bands
        - 4corners: expand coarse 3x3 four-corners by the same bands
    output_grid: fixed 28x28 canvas, still formed as 3*input_size-2
    lattice_rule:
      coarse_phase_tiles:
        plus:      ... / .## / .##
        H:         ... / #.. / #..
        X:         .## / #.. / #..
        4corners:  ### / #.. / #..
      expansion: enlarge the chosen coarse phase tile with the same [1, size-2, 1] bands, so period = motif size
    center_stamp:
      content: original motif tiled 2x2
      color: recolor foreground to maroon
      size: 8x8 for size-4 motifs, 10x10 for size-5 motifs
      placement: centered in the 28x28 output

  behavior_trace:
    original_path:
      input_bbox: 3x3
      decode_step: hard-coded pixel tests on a 3x3 glyph
      output_period: always 3
    variant_path:
      input_bbox: 4x4 or 5x5
      decode_step: identify which enlarged family matches exactly, then expand that family's coarse phase tile to the same size
      output_period: 4 or 5, matching the detected bbox size
    concrete_example:
      size4_plus:
        motif: .##. / #### / #### / .##.
        phase: .... / .### / .### / .###
      difference: the original solver would reject the bbox immediately, while the variant uses bbox size as a meaningful control variable

  complexity_calibration:
    branches: 4 families x 2 sizes = 8 visual codewords, enough to make the D2 change visible without creating an open-ended symbol language
    repeated_support: family semantics stay identical across both sizes, so a solver can align examples by shape family while noticing period changes

  implicit_assumption_audit:
    - `g.shape == (6, 6)` no longer holds; replace with fixed 10x10 so 5x5 glyphs fit.
    - `motif.shape == (3, 3)` no longer holds; solver must accept 4x4 and 5x5 only.
    - `_decode_phase_tile()` previously assumed specific 3x3 pixels; now it must work from family+size.
    - `center = np.tile(motif, (2, 2))` is still valid, but center placement must tolerate 8x8 or 10x10 stamps.
    - `out_shape = 3*g.shape - 2` remains valid after enlarging the input grid to 10x10, producing a clean 28x28 canvas.

  placement_legality:
    - generator samples top-left positions uniformly from all placements that keep the full 4x4/5x5 bbox inside 10x10
    - only one colored object is drawn, so overlap ambiguity cannot occur
    - all four families touch the outer bbox extremes, so `bbox()` always recovers the intended size exactly

  visual_observability:
    - size 4 and size 5 motifs differ by obvious stroke thickness and by the size of the central maroon stamp
    - the orange/black lattice visibly changes period from 4 to 5, which is observable in a few examples without relying on hidden ids

  visual_cleanliness:
    - there are only three output colors (black, orange, maroon), as in the original task
    - the maroon stamp occupies one centered block, while the lattice remains readable around it

  color_inference:
    - input foreground color is irrelevant and stays arbitrary-but-singleton, matching the original role of color as a carrier for the coded glyph only

  generator_validation:
    - every sampled input is oracle-checked with `main()`
    - the intended variant metric is bbox size / lattice period, which the generator explicitly varies between 4 and 5

  main_assumptions_after_change:
    - the non-orange pixels must match one of the exact enlarged prototypes
    - the foreground bbox must be square and its side must be 4 or 5
    - `generate_input()` guarantees these conditions by constructing motifs from the same prototype table used by the solver
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H, GRID_W = 10, 10
OUTPUT_H, OUTPUT_W = 3 * GRID_H - 2, 3 * GRID_W - 2
GLYPH_SIZES = (4, 5)
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

COARSE_GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

COARSE_PHASES = {
    "plus": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    "H": np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "X": np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "4corners": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
}


def _band_sizes(size: int) -> tuple[int, int, int]:
    size = int(size)
    assert size in GLYPH_SIZES
    return 1, size - 2, 1


def _expand_coarse(pattern: np.ndarray, size: int) -> np.ndarray:
    bands = np.array(_band_sizes(size), dtype=int)
    expanded = np.repeat(np.repeat(np.asarray(pattern, dtype=bool), bands, axis=0), bands, axis=1)
    assert expanded.shape == (size, size)
    return expanded


EXPANDED_GLYPHS = {
    (name, size): _expand_coarse(pattern, size)
    for name, pattern in COARSE_GLYPHS.items()
    for size in GLYPH_SIZES
}

EXPANDED_PHASES = {
    (name, size): _expand_coarse(pattern, size)
    for name, pattern in COARSE_PHASES.items()
    for size in GLYPH_SIZES
}


def _identify_glyph(motif: np.ndarray) -> tuple[str, int]:
    motif = np.asarray(motif, dtype=bool)
    assert motif.ndim == 2
    assert motif.shape[0] == motif.shape[1]
    size = int(motif.shape[0])
    assert size in GLYPH_SIZES

    for (name, glyph_size), glyph in EXPANDED_GLYPHS.items():
        if glyph_size == size and np.array_equal(motif, glyph):
            return name, size
    raise AssertionError("Unrecognized enlarged glyph")


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode a 4x4 or 5x5 enlarged contour glyph into the matching period phase mask."""
    glyph_name, size = _identify_glyph(motif)
    return EXPANDED_PHASES[(glyph_name, size)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 4x4 or 5x5 enlarged contour glyph into a larger-period orange lattice and stamp the glyph family back in maroon at the center.

    Input structure:
    - The grid is 10x10 with orange background and exactly one non-orange color.
    - The colored pixels have a tight square bbox of size 4x4 or 5x5.
    - That bbox matches one of four enlarged contour families: plus, H, X, or four corners.

    Rule:
    - Identify the glyph family and its bbox size.
    - Expand that family's coarse phase code to a period-4 or period-5 orange/black tile using the same banding as the input glyph.
    - Fill a black 28x28 canvas by repeating that phase tile, then place a 2x2 maroon copy of the original enlarged glyph at the canvas center.

    Key insight:
    - The input object is an enlarged codeword rather than a literal picture to preserve; the family selects the phase layout, while the bbox size controls the lattice period and the scale of the central stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    glyph_name, motif_size = _identify_glyph(motif)

    phase = EXPANDED_PHASES[(glyph_name, motif_size)]
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((OUTPUT_H, OUTPUT_W), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


GLYPHS = EXPANDED_GLYPHS


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(COARSE_GLYPHS.keys())))
    motif_size = int(rng.choice(GLYPH_SIZES))
    motif = GLYPHS[(glyph_name, motif_size)]
    fg = int(rng.choice(FG_COLORS))

    r0 = int(rng.integers(0, GRID_H - motif_size + 1))
    c0 = int(rng.integers(0, GRID_W - motif_size + 1))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + motif_size, c0 : c0 + motif_size][motif] = fg

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 variant")
