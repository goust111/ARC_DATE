# Combo variant of eee78d87 (index 2)
# combo_spec: D1/2+D10/1
# Name: 正负反转解码+格栅正负反转

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for combo D1/2 + D10/1
# ---------------------------------------
# a) Concrete behavior and trace
#    - Preserved scaffold:
#      input remains a 6x6 orange board with exactly one non-orange 3x3 contour glyph,
#      chosen from plus / H / X / 4corners; output remains a 16x16 periodic lattice with
#      a centered 6x6 maroon 2x2 stamp of that glyph.
#    - Original pipeline:
#      glyph mask -> hand-coded phase bool -> colorize with True=orange, False=black.
#      For plus, that old phase tile was:
#          000
#          011
#          011
#      so the repeated field showed an offset 2x2 orange block.
#    - Fused pipeline:
#      1. D1/2 replaces the hand-coded phase tile with the glyph mask itself.
#         For plus, the phase bool becomes:
#             010
#             111
#             010
#      2. D10/1 then swaps lattice color roles, so phase=True becomes black and
#         phase=False becomes orange. The final colored 3x3 tile for plus is:
#             707
#             000
#             707
#      The two inversions do not cancel, because the first inversion changes which cells
#      are marked by the phase mask, while the second inversion changes how that mask is
#      colorized. The result is a new lattice rather than a disguised copy of the original.
# b) Complexity calibration
#    - The decode rule is simpler than the original, but still bounded to 4 known glyph
#      classes. That keeps the hypothesis space small enough for ARC while avoiding an
#      unrestricted "copy any bitmap" task.
#    - The visual change is global: every repeated 3x3 cell in the 16x16 lattice flips to
#      the complement-colored direct glyph mask. This is strong enough to infer from a few
#      examples and clearly distinct from both the original task and each single variant.
# c) Implicit-assumption audit
#    - `g.shape == (6, 6)`: still required; generator preserves it.
#    - `bg == Color.ORANGE` and exactly one extra color: unchanged and still required.
#    - `bbox(fg)` exists and the cropped bbox is exactly 3x3: unchanged, so generation must
#      still place exactly one glyph and nothing else.
#    - The original `_decode_phase_tile` relied on contour-specific corner/edge assertions.
#      Those assumptions are removed. New risk: a direct-copy decoder would silently accept
#      arbitrary 3x3 noise, so `main()` now explicitly checks membership in the 4-glyph
#      catalog.
#    - The color swap assumes the output lattice still uses only black/orange outside the
#      stamp. That remains true because the repeated tile is still boolean.
#    - The center overwrite assumes maroon can be painted on top of either lattice color
#      without ambiguity. That remains valid because the center is a dedicated overlay.
# d) Placement legality
#    - Generator still chooses `(r0, c0)` in `0..3`, so the 3x3 motif stays inside 6x6.
#    - Only one glyph is placed, so there are no overlap constraints beyond staying in-bounds.
#    - Because generation draws only from the fixed glyph catalog, the new template check in
#      `main()` is always satisfied.
# e) Visual observability
#    - A human can recover the rule from 2-4 examples by comparing the input contour with the
#      repeated outer lattice: black blocks now occur exactly where the input glyph has pixels,
#      repeated modulo 3, while orange occupies the complementary cells.
#    - This inference uses visible shape repetition only; it does not rely on numeric color ids.
# f) Visual cleanliness
#    - The output still has two clean layers:
#      black/orange periodic lattice everywhere, plus a maroon 2x2 stamp only in the centered
#      6x6 window. Most of the lattice remains unobscured, so the repeated rule stays legible.
# g) Color-change inferability
#    - Only three output colors appear: orange for lattice lines/background complement, black
#      for the direct glyph-derived phase cells, and maroon for the center stamp.
#    - The color roles are structurally anchored by repetition and by the separate central
#      overlay, so the orange/black inversion is inferable rather than arbitrary.


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_NAMES = tuple(GLYPHS.keys())
KNOWN_GLYPHS = tuple(GLYPHS.values())


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Use the glyph mask itself as the fused variant's boolean phase tile."""
    assert motif.shape == (3, 3)
    return motif.astype(bool, copy=True)


def _decode_original_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Original task decoder, kept only for non-degeneration checks."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edge_centers = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)
    phase = np.zeros((3, 3), dtype=bool)

    if not bool(corners.any()):
        assert bool(motif[1, 1])
        assert bool(edge_centers.all())
        phase[1:, 1:] = True
        return phase

    assert bool(corners.all())
    assert not bool(motif[0, 1])
    assert not bool(motif[2, 1])
    assert bool(motif[1, 0]) == bool(motif[1, 2])

    phase[1:, 0] = True
    if bool(motif[1, 0]):
        assert bool(motif[1, 1])
    else:
        phase[0, 1:] = True

    if not bool(motif[1, 1]):
        phase[0, 0] = True

    return phase


def _is_known_glyph(motif: np.ndarray) -> bool:
    return any(np.array_equal(motif, glyph) for glyph in KNOWN_GLYPHS)


def _center_slices(out_shape: tuple[int, int], motif_shape: tuple[int, int]) -> tuple[slice, slice]:
    r0 = (out_shape[0] - motif_shape[0]) // 2
    c0 = (out_shape[1] - motif_shape[1]) // 2
    return slice(r0, r0 + motif_shape[0]), slice(c0, c0 + motif_shape[1])


def _render_output_from_phase(motif: np.ndarray, phase: np.ndarray, *, invert_colors: bool) -> np.ndarray:
    phase_colors = np.where(phase, Color.BLACK, Color.ORANGE) if invert_colors else np.where(
        phase, Color.ORANGE, Color.BLACK
    )
    out_shape = (3 * GRID_H - 2, 3 * GRID_W - 2)
    out = tile_fill(out_shape, phase_colors.astype(int))

    center = np.tile(motif, (2, 2))
    rs, cs = _center_slices(out.shape, center.shape)
    out[rs, cs][center] = Color.MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinterpret the colored 3x3 contour glyph as a directly copied phase mask, invert the lattice color roles, and stamp the glyph in maroon at the center.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one additional color.
    - All non-orange cells belong to one contour glyph inside a tight 3x3 bounding box.
    - The glyph must be one of four allowed templates: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 glyph itself as the period-3 boolean phase tile instead of applying the original contour decoder.
    - Repeat that tile across a 16x16 canvas, but assign black to phase=True cells and orange to phase=False cells, producing the swapped-polarity lattice.
    - Tile the original 3x3 glyph 2x2 into the centered 6x6 region and recolor the glyph pixels to maroon.

    Key insight:
    - The clue is still a code for a periodic background, but the fused variant changes both the codebook and the color polarity: the glyph shape now directly marks the repeated phase cells, and those marked cells become black rather than orange.
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
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)
    assert _is_known_glyph(motif)

    phase = _decode_phase_tile(motif)
    return _render_output_from_phase(motif, phase, invert_colors=True)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(GLYPH_NAMES))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        combo_out = main(inp)
    except Exception:
        return None

    # Guard against silent collapse to the original or either single-dimension variant.
    original_out = _render_output_from_phase(motif, _decode_original_phase_tile(motif), invert_colors=False)
    d1_only_out = _render_output_from_phase(motif, motif, invert_colors=False)
    d10_only_out = _render_output_from_phase(motif, _decode_original_phase_tile(motif), invert_colors=True)
    assert np.any(combo_out != original_out)
    assert np.any(combo_out != d1_only_out)
    assert np.any(combo_out != d10_only_out)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant D1/2+D10/1")
