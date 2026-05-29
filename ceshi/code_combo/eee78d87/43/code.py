# Combo variant of eee78d87 (index 43)
# combo_spec: D1/2+D4/0+D10/1
# Name: 反转解码+保留原色+格栅反转

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The input format stays the same as the base task: a 6x6 orange board with exactly
#      one 3x3 contour glyph in a single non-orange foreground color. The glyph is one of
#      plus / H / X / 4corners and can appear anywhere that keeps its 3x3 bbox inside the
#      6x6 frame.
#    - The fused output still has two layers on a 16x16 board: a period-3 tiled lattice
#      over the full canvas plus a centered 6x6 stamp formed by tiling the 3x3 glyph 2x2.
#    - D1/2 changes the decode stage: instead of translating each glyph to a hand-written
#      phase tile, we now use the glyph mask itself as the boolean phase tile.
#    - D10/1 then flips the lattice coloring: phase=True cells become black(0), while
#      phase=False cells become orange(7). Because D1/2 already made phase=True coincide
#      with glyph pixels, the repeated background becomes the photographic negative of the
#      glyph mask itself.
#    - D4/0 changes only the centered stamp color: motif pixels in the 6x6 center keep the
#      original foreground color from the input rather than being recolored to maroon(9).
#    - Example with the plus glyph:
#        motif mask         = [[0,1,0],[1,1,1],[0,1,0]]
#        original phase     = [[0,0,0],[0,1,1],[0,1,1]]
#        fused phase        = [[0,1,0],[1,1,1],[0,1,0]]
#        fused color tile   = [[7,0,7],[0,0,0],[7,0,7]]
#      So the original task produced orange blocks decoded from the contour and then a
#      maroon center, whereas the combo produces black cross-shaped holes on an orange
#      lattice and overlays the center cross in the input color.
# b) Complexity calibration
#    - The task is not over-simplified because the 4 allowed glyph classes still yield 4
#      distinct repeated lattices, and the lattice now visibly tracks the glyph geometry.
#    - The rule remains constrained: there are still only 4 admissible glyphs, one input
#      foreground color, one fixed 16x16 output size, and one centered overlay region.
#    - D4/0 does not add uncontrolled color freedom: each sample still uses exactly one
#      non-orange input color, so the stamp color is easy to identify from the input and
#      remains repeated four times in the centered 2x2 tiling.
# c) Implicit-assumption audit
#    - `g.shape == (6, 6)` is still required and preserved by the generator.
#    - `bg == Color.ORANGE` and exactly one non-background color are still required.
#    - `bbox(fg)` must exist and `crop_bbox(...).shape == (3, 3)` still hold because the
#      task still has exactly one tight 3x3 glyph.
#    - The base solver's `_decode_phase_tile` relied on contour-specific corner/edge
#      assertions. Those assumptions are removed because the fused decode is direct. To
#      avoid silently accepting arbitrary 3x3 noise, `main()` now asserts that the cropped
#      mask matches one of the 4 known glyph templates.
#    - The stamp-color assignment now depends on reading the single input foreground color.
#      This introduces a new assumption: that there is exactly one foreground color. The
#      generator guarantees that by painting all glyph pixels with one chosen color.
#    - Overlay safety is unchanged: the centered 6x6 region simply overwrites lattice cells
#      at motif positions, and because the stamp color is never orange, it remains visually
#      separable from the lattice.
# d) Placement legality
#    - The generator samples the 3x3 top-left position from rows/cols 0..3, so the motif
#      always fits inside the 6x6 input.
#    - Only one object is placed, so overlap in the input is impossible.
#    - The output stamp is always centered as a 6x6 block on the 16x16 canvas, which is
#      valid because the canvas size is fixed at `(3 * 6 - 2, 3 * 6 - 2) = (16, 16)`.
# e) Visual observability
#    - A human can infer the fusion from a few examples because the repeated lattice now
#      directly mirrors the input glyph geometry, but with black where the glyph pixels
#      were and orange elsewhere. This is visible over almost the entire 16x16 board.
#    - The center also preserves the input color instead of switching to maroon, giving an
#      immediate cross-example cue that the stamp color is copied from the source object.
#    - No hidden ordering by numeric color id is used; the only color-dependent step is the
#      visibly repeated reuse of the unique input foreground color.
# f) Visual cleanliness
#    - The output remains easy to parse: an orange/black periodic field plus one centered
#      colored stamp. The overlay occupies only the central 6x6 region, leaving most of the
#      lattice unobscured for rule discovery.
#    - Even when the foreground color is dark (for example blue or maroon is excluded), the
#      stamp still differs from both lattice colors because generator colors exclude 0 and 7.
# g) Change inferability
#    - Each sample uses exactly 3 colors in the output: black lattice cells, orange lattice
#      cells, and the copied input foreground color for the center stamp. This is actually
#      cleaner than a generic color-preservation variant that might introduce more colors.
#    - The color-preservation rule is strongly signposted: the center stamp is the only
#      region using the non-lattice color, and its shape is a literal 2x2 tiling of the
#      input glyph, so the mapping is spatially explicit rather than arbitrary.


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
    """Use the glyph mask itself as the phase tile."""
    assert motif.shape == (3, 3)
    return motif.astype(bool, copy=True)


def _decode_original_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Reference implementation of the base-task contour decoding."""
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


def _render_output_from_components(
    motif: np.ndarray,
    stamp_color: int,
    *,
    direct_decode: bool,
    swapped_lattice_colors: bool,
) -> np.ndarray:
    """Render the 16x16 output under a selectable decode/color policy."""
    phase = _decode_phase_tile(motif) if direct_decode else _decode_original_phase_tile(motif)
    if swapped_lattice_colors:
        phase_colors = np.where(phase, Color.BLACK, Color.ORANGE).astype(int)
    else:
        phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)

    out_shape = (3 * GRID_H - 2, 3 * GRID_W - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = int(stamp_color)
    return out


def _extract_motif_and_color(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the cropped 3x3 glyph mask and its unique foreground color."""
    bg = int(get_bg(grid))
    assert bg == Color.ORANGE

    fg_colors = colors_present(grid, exclude={bg})
    assert len(fg_colors) == 1
    stamp_color = int(fg_colors[0])

    fg = grid != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)
    assert _is_known_glyph(motif)
    return motif, stamp_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose phase comes directly from the glyph, whose orange/black roles are swapped, and whose center stamp keeps the input foreground color.

    Input structure:
    - The grid is 6x6 with orange background and exactly one additional color.
    - The non-orange pixels form a single contour glyph inside a tight 3x3 bounding box.
    - The glyph is one of the four allowed templates: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 glyph mask itself as the period-3 phase tile instead of using the original contour-to-phase decoder.
    - Repeat that tile across a 16x16 canvas, but color glyph-mask positions black and complementary positions orange so the lattice polarity is inverted.
    - Tile the original 3x3 glyph 2x2 into the centered 6x6 region and overwrite its true pixels with the original input foreground color.

    Key insight:
    - The same small contour now controls the background in a literal but color-inverted way: its shape repeats as black cells across the lattice, while the centered overlay preserves the glyph's source color instead of converting it to a fixed highlight color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    motif, stamp_color = _extract_motif_and_color(g)
    return _render_output_from_components(
        motif,
        stamp_color,
        direct_decode=True,
        swapped_lattice_colors=True,
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(GLYPH_NAMES))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    # Non-degeneration guard: the fused behavior must differ from the base task and from
    # outputs where any one of the three component changes is omitted.
    combo_like = _render_output_from_components(
        motif,
        fg,
        direct_decode=True,
        swapped_lattice_colors=True,
    )
    assert np.array_equal(out, combo_like)

    original_like = _render_output_from_components(
        motif,
        Color.MAROON,
        direct_decode=False,
        swapped_lattice_colors=False,
    )
    d1_only_like = _render_output_from_components(
        motif,
        Color.MAROON,
        direct_decode=True,
        swapped_lattice_colors=False,
    )
    d4_only_like = _render_output_from_components(
        motif,
        fg,
        direct_decode=False,
        swapped_lattice_colors=False,
    )
    d10_only_like = _render_output_from_components(
        motif,
        Color.MAROON,
        direct_decode=False,
        swapped_lattice_colors=True,
    )

    assert np.any(out != original_like)
    assert np.any(out != d1_only_like)
    assert np.any(out != d4_only_like)
    assert np.any(out != d10_only_like)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 43")
