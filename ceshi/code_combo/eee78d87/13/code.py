# Combo variant of eee78d87 (index 13)
# combo_spec: D6/0+D13/0
# Name: 先粗后细双层铺贴+可变周期

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo D6/0 + D13/0 variant
# ---------------------------------------------------
# a) Concrete behavior and trace
#    - Preserved scaffold:
#      input is still a 6x6 orange field carrying exactly one colored 3x3 contour glyph;
#      output is still 16x16 and still ends with a centered 6x6 maroon 2x2 stamp of that glyph.
#    - Combined change:
#      the single original lattice is split into two tiled passes.
#      1. A coarse orange/black phase tile covers the whole canvas.
#      2. A finer orange/black phase tile is then union-overlaid only in a centered 10x10 window.
#      3. The maroon stamp is written last on top.
#    - Variable-period part:
#      each glyph class selects two periods, one large and one small:
#        plus      -> coarse 4, fine 2
#        H         -> coarse 5, fine 3
#        X         -> coarse 6, fine 4
#        4corners  -> coarse 7, fine 5
#      This guarantees the first layer is always visually coarser than the second.
#    - Phase-shape split:
#      the coarse tile keeps the seed task's contour-family phase logic, but drawn at the glyph's
#      coarse period. The fine tile uses a different shape family derived from the glyph's inner
#      cross structure: main diagonal / center column / center row / anti-diagonal.
#    - Example trace:
#      for H, the original task produced one fixed period-3 lattice everywhere.
#      Here, H first creates a period-5 background with orange only in the left column below the
#      first row, then adds a period-3 center-window overlay with a vertical center stripe, then
#      stamps the 6x6 maroon H pattern. The center region therefore has two visible rhythms.
#
# b) Complexity calibration
#    - Four glyph classes remain the only semantic categories, so the rule is still finite and
#      learnable from few samples.
#    - Period choices are small tables rather than free numeric parameters, avoiding brittle
#      inference while still making density changes obvious.
#    - The detail window is 10x10 rather than 6x6 or 8x8 so fine periods up to 5 still show at
#      least two repeats and do not collapse into a nearly non-periodic patch.
#    - Generator cycles glyph types over consecutive calls so small train sets are more likely to
#      expose multiple period pairs instead of repeating one class.
#
# c) Implicit-assumption audit against the original main()
#    - `g.shape == (6, 6)` still holds and stays asserted.
#    - `bg == Color.ORANGE` and `len(colors_present(...)) == 1` still hold and stay asserted.
#    - `motif.shape == (3, 3)` still holds; the generator still places exactly one tight 3x3 glyph.
#    - Original `_decode_phase_tile(motif)` assumed a single fixed 3x3 tile. That is invalid here:
#      both the period and the number of tiled passes now vary, so it is replaced by separate
#      coarse and fine decoders.
#    - Original `out_shape = 3 * input - 2` happened to equal 16x16. The fixed output size remains
#      valid, but it is now independent of either period table.
#    - Original one-shot `tile_fill` is insufficient because the center window must receive a second
#      overlay without disturbing the outer background.
#    - Original final maroon overwrite remains valid and must still happen last.
#
# d) Placement legality
#    - The generator samples only the four legal 3x3 contour glyphs, all of which already have a
#      tight bbox equal to the full 3x3 patch.
#    - The glyph top-left is still sampled from rows/cols 0..3, so input placement never exceeds
#      the 6x6 boundary.
#    - The detail window is fixed to rows/cols 3..12 and the 6x6 maroon stamp sits at rows/cols
#      5..10, so both overlays are centered and remain in-bounds.
#
# e) Visual observability
#    - Humans can read the rule from 2-4 examples because the outer ring of the 16x16 canvas shows
#      only the coarse period, while the centered 10x10 window reveals the added finer rhythm.
#    - The period mapping depends on glyph class, a visible shape attribute, never on the input
#      color id or position.
#    - Cycling glyph classes improves the chance that a small train split contains different period
#      pairs, making the D6 component observable rather than hidden.
#
# f) Visual cleanliness
#    - Only three visual layers exist: coarse orange/black background, fine orange/black center
#      overlay, maroon stamp. The two tiled layers share the same orange/black palette and combine
#      by union, so the image stays legible instead of becoming multicolor clutter.
#    - Restricting the fine layer to a centered window preserves a clean separation between
#      "background rhythm" and "detail rhythm".
#
# g) Change inferability
#    - No extra output colors are introduced, so there is no hidden color remapping burden.
#    - The key structural clues are repeated in the same places across samples:
#      outer area -> coarse period/shape, center window -> fine period/shape, innermost 6x6 ->
#      maroon stamp copied from the input glyph.
#    - Generator rejects motifs whose fine layer adds too few new orange pixels, preventing silent
#      regression to a single-layer variable-period task.
#
# Post-change input assumptions to re-check
#    - The solver now assumes the foreground glyph is exactly one of the four seed contour forms.
#    - It also assumes the fine overlay must add visible orange inside the detail window for the
#      training distribution; `generate_input()` enforces this with a minimum added-pixel threshold.

GRID_H = 6
GRID_W = 6
OUT_SHAPE = (3 * GRID_H - 2, 3 * GRID_W - 2)
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
DETAIL_R0 = 3
DETAIL_C0 = 3
DETAIL_SIZE = 10
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
MIN_DETAIL_GAIN = 12

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
COARSE_PERIODS = {"plus": 4, "H": 5, "X": 6, "4corners": 7}
FINE_PERIODS = {"plus": 2, "H": 3, "X": 4, "4corners": 5}

_GLYPH_ORDER = tuple(GLYPHS.keys())
_GEN_RNG = np.random.default_rng()
_GLYPH_QUEUE: list[str] = []


def _classify_glyph(motif: np.ndarray) -> str:
    """Classify the 3x3 contour glyph into one of the four seed forms."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edges = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)
    center = bool(motif[1, 1])

    if not bool(corners.any()):
        assert center
        assert bool(edges.all())
        return "plus"

    assert bool(corners.all())
    if bool(motif[1, 0]):
        assert bool(motif[1, 2])
        assert center
        assert not bool(motif[0, 1])
        assert not bool(motif[2, 1])
        return "H"

    assert not bool(edges.any())
    return "X" if center else "4corners"


def _decode_coarse_period_and_tile(motif: np.ndarray) -> tuple[int, np.ndarray]:
    """Use the glyph's broad contour class to produce the large background tile."""
    glyph = _classify_glyph(motif)
    period = COARSE_PERIODS[glyph]
    phase = np.zeros((period, period), dtype=bool)

    if glyph == "plus":
        phase[1:, 1:] = True
    else:
        phase[1:, 0] = True
        if glyph == "X":
            phase[0, 1:] = True
        elif glyph == "4corners":
            phase[0, :] = True

    return period, phase


def _decode_fine_period_and_tile(motif: np.ndarray) -> tuple[int, np.ndarray]:
    """Use the glyph's inner cross signature to produce the smaller detail tile."""
    glyph = _classify_glyph(motif)
    period = FINE_PERIODS[glyph]
    phase = np.zeros((period, period), dtype=bool)
    idx = np.arange(period)

    if glyph == "plus":
        phase[idx, idx] = True
    elif glyph == "H":
        phase[:, period // 2] = True
    elif glyph == "X":
        phase[period // 2, :] = True
    else:
        phase[idx, period - 1 - idx] = True

    return period, phase


def _compose_tiled_layers(motif: np.ndarray) -> tuple[np.ndarray, int]:
    _coarse_period, coarse = _decode_coarse_period_and_tile(motif)
    _fine_period, fine = _decode_fine_period_and_tile(motif)

    coarse_mask = tile_fill(OUT_SHAPE, coarse).astype(bool)
    fine_mask = tile_fill((DETAIL_SIZE, DETAIL_SIZE), fine).astype(bool)

    region = coarse_mask[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE].copy()
    added_mask = fine_mask & ~region
    region |= fine_mask
    coarse_mask[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE] = region

    out = np.where(coarse_mask, Color.ORANGE, Color.BLACK).astype(int)
    return out, int(added_mask.sum())


def _stamp_center(out: np.ndarray, motif: np.ndarray) -> np.ndarray:
    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    stamped = out.copy()
    stamped[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return stamped


def _detail_gain(motif: np.ndarray) -> int:
    _, gain = _compose_tiled_layers(motif)
    return gain


def _next_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_QUEUE
    if not _GLYPH_QUEUE:
        _GLYPH_QUEUE = list(rng.permutation(_GLYPH_ORDER))
    return _GLYPH_QUEUE.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one 3x3 contour glyph into a two-scale tiled scene whose background and center detail use different period tables, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange cells form a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of four allowed seed forms: plus, H, X, or four corners.

    Rule:
    - Classify the glyph type and use that class to choose a large background period and a smaller detail period.
    - Decode a coarse orange/black phase tile from the glyph's broad contour family, repeat it across the full 16x16 canvas, then decode a finer orange/black tile from the glyph's inner cross signature and union-overlay it only in the centered 10x10 window.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon inside the centered 6x6 region.

    Key insight:
    - The input glyph now carries two synchronized codes at once: its outer contour selects the coarse repeating background, while its inner structure selects a second, denser center overlay. The glyph class also determines both repeat periods, so different glyph types visibly change the spacing of both layers.
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

    out, _ = _compose_tiled_layers(motif)
    return _stamp_center(out, motif)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[_next_glyph_name(rng)]
    if _detail_gain(motif) < MIN_DETAIL_GAIN:
        return None

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
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(_GEN_RNG)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for combo eee78d87/13")
