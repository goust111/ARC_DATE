# Variant of eee78d87
# Dimension: D2 — 源对象的形状集合 (src)
# Change: 4种固定的3x3轮廓字形：plus(.#./#11/.#.)、H(#.#/###/#.#)、X(#.#/.#./#.#)... → 字形为非对称图案

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis:
# variant:
#   preserved_structure:
#     - input remains a 6x6 orange field with exactly one 3x3 colored motif
#     - output remains a 16x16 orange/black period-3 lattice with a centered 6x6 maroon stamp
#     - the stamp is still the original 3x3 motif tiled 2x2
#   D2_change:
#     - replace the old symmetric contour family with four fixed asymmetric 3x3 glyphs
#     - decode rule changes from case-by-case contour logic to an oriented transform:
#       phase_tile = rotate_clockwise(invert(motif))
#   glyph_catalog:
#     L:
#       100
#       100
#       111
#     hook:
#       111
#       100
#       110
#     step:
#       110
#       010
#       001
#     snake:
#       100
#       110
#       011
#   behavior_trace:
#     - original logic expected either zero corners (plus) or four corners (H/X/4corners).
#       An input like L = 100/100/111 would fail immediately because its corners mix 1s and 0s.
#     - new logic crops the same 3x3 mask, verifies that it is one of the asymmetric glyphs,
#       inverts it to 011/011/000, rotates clockwise to 000/011/011, then tiles that phase.
#       The central maroon stamp still shows the original L orientation, so the relation between
#       source glyph and lattice is visually traceable.
#   complexity_calibration:
#     - there are only four glyph classes, which keeps the hypothesis space small enough for ARC
#     - all four glyphs have full 3x3 bbox support and no rotation/reflection symmetry
#     - their transformed phase tiles are also distinct and asymmetric, so orientation changes
#       alter the lattice in a visible way instead of collapsing to the same pattern
#   hidden_assumptions_review:
#     - g.shape == (6, 6): still valid and preserved by the generator
#     - background is orange and there is one non-orange color: unchanged, still enforced
#     - bbox(fg) is not None and crop is exactly 3x3: preserved by using glyphs that touch all
#       four bbox sides and by placing exactly one motif
#     - the old corner-based assertions are no longer valid; they are replaced by exact matching
#       against the new glyph set before decoding
#     - center stamping can still overwrite the lattice safely because the maroon layer is the
#       final write into the centered 6x6 region
#   placement_legality:
#     - motif size is still 3x3, so legal top-left positions remain rows/cols 0..3
#     - only one object is placed, so overlap questions do not arise in the input
#     - the solver only assumes a single tight 3x3 motif; the generator guarantees that
#   visual_observability:
#     - the repeated lattice is now orientation-sensitive because the phase tile is asymmetric
#     - the 2x2 maroon stamp exposes the source glyph directly, letting a human compare stamp
#       versus lattice and infer the rotate+invert relationship across a few examples
#   visual_cleanliness:
#     - the task still uses three output colors only: black, orange, maroon
#     - there is no extra decoration; the only layered region is the centered stamp, which stays
#       legible because maroon cleanly overwrites orange/black cells
#   color_inference:
#     - the input foreground color is still irrelevant except for locating the motif
#     - no branch depends on color id ordering or hidden numeric properties
#   post-change_main_assumptions:
#     - every accepted motif must match one of the four catalog entries exactly
#     - generator coverage should exercise all four glyph branches so the new orientation logic
#       does not degenerate into a single repeated case


def _has_nontrivial_symmetry(motif: np.ndarray) -> bool:
    """Return True if the 3x3 motif equals any non-identity rotation or reflection."""
    return any(
        np.array_equal(motif, other)
        for other in (
            np.rot90(motif, 1),
            np.rot90(motif, 2),
            np.rot90(motif, 3),
            np.fliplr(motif),
            np.flipud(motif),
        )
    )


GLYPHS = {
    "L": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    "hook": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    "step": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    "snake": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
}

for _glyph_name, _glyph in GLYPHS.items():
    assert _glyph.shape == (3, 3)
    assert bool(_glyph[0].any() and _glyph[2].any() and _glyph[:, 0].any() and _glyph[:, 2].any())
    assert not _has_nontrivial_symmetry(_glyph), _glyph_name


def _motif_name(motif: np.ndarray) -> str | None:
    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    return None


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode an allowed asymmetric 3x3 glyph into its period-3 orange phase mask."""
    assert motif.shape == (3, 3)
    assert _motif_name(motif) is not None
    return np.rot90(~motif, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Interpret the colored 3x3 asymmetric glyph as an oriented phase code, then build the lattice-and-stamp output.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels occupy one tight 3x3 bounding box.
    - That 3x3 mask is one of four fixed asymmetric glyphs, so its facing direction matters.

    Rule:
    - Crop the 3x3 glyph and verify it belongs to the asymmetric catalog.
    - Form the period-3 phase tile by inverting the glyph's filled/empty cells and rotating the result 90 degrees clockwise.
    - Repeat that orange/black phase tile across the 16x16 canvas, then stamp the original glyph tiled 2x2 in maroon at the center.

    Key insight:
    - The small source object is an oriented code: because both the glyph and its decoded phase tile are asymmetric, changing the glyph orientation changes the repeated lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)
    assert _motif_name(motif) is not None

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

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
    raise RuntimeError("Failed to generate input for eee78d87 D2 variant")
