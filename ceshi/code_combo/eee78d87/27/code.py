# Combo variant of eee78d87 (index 27)
# combo_spec: D4/1+D8/1
# Name: 栗色交叉处显示+背景色由字形色决定

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - The 3x3 contour glyph vocabulary is unchanged: plus / H / X / 4corners.
#    - D8/1 changes the input parsing premise: the 6x6 field is no longer always
#      orange. Instead the lone glyph color chooses the background color through
#      a small fixed mapping:
#        blue -> teal, red -> yellow, green -> pink, grey -> orange.
#    - D4/1 changes the center stamp: the 2x2 tiled glyph is not recolored in
#      full anymore; maroon appears only where tiled-glyph pixels also lie on
#      the 3-step seam rows or seam columns of the centered 6x6 stamp box.
#    - Example trace with a blue plus:
#      original task: blue plus on orange background -> orange/black lattice ->
#      all 20 plus pixels in the centered 6x6 tiled plus become maroon.
#      combo task: blue plus on teal background -> teal/black lattice ->
#      only the 8 plus pixels lying on local seam rows/cols {0, 3} become
#      maroon; the other 12 plus pixels stay as teal/black lattice pixels.
#    - Net difference: one change affects how the input background is read, the
#      other affects how much of the center stamp is visible.
# b) Complexity calibration
#    - The color rule uses only four foreground choices and four paired
#      background colors, so the relation is structured rather than arbitrary.
#    - The center AND rule is visible for every glyph: seam-overlap maroon counts
#      are plus=8, H=16, X=12, 4corners=12, all strictly between 0 and the full
#      tiled-glyph pixel count. This avoids collapse to the base task or to an
#      empty stamp.
#    - The two changes act on different stages, so neither masks the other:
#      background variation is obvious in the input, reduced maroon is obvious
#      in the output center.
# c) Implicit assumptions audit against the original solver
#    - `g.shape == (6, 6)` still holds unchanged.
#    - `bg == Color.ORANGE` is no longer valid and must be replaced by dominant
#      color detection plus mapping validation.
#    - `len(colors_present(g, exclude={bg})) == 1` still holds because the input
#      still contains exactly one monochrome glyph and no distractors.
#    - `bbox(g != bg)` still isolates the glyph because the background remains
#      uniform even though its color is no longer fixed.
#    - `_decode_phase_tile` stays valid because the 3x3 contour geometry itself
#      is unchanged.
#    - `phase_colors = np.where(phase, bg, Color.BLACK)` now uses the inferred
#      mapped background color, so the repeated lattice inherits that variable
#      background instead of always using orange.
#    - The new `center & seam_mask` step assumes each supported glyph intersects
#      the seam mask but is not contained in it; the count check above and the
#      generator validation guarantee this.
# d) Placement legality
#    - The glyph still occupies one 3x3 bbox fully inside the 6x6 input, so
#      top-left positions remain r0,c0 in [0, 3].
#    - There are no extra objects, so nothing can overlap the glyph.
#    - The centered 6x6 output stamp remains rows 5..10 and cols 5..10 inside
#      the 16x16 lattice, exactly as in the base task.
# e) Visual observability
#    - A human sees that the background can be teal / yellow / pink / orange
#      depending on the glyph color, so "orange is special" is no longer a safe
#      assumption.
#    - In the output, the center maroon is visibly sparser than the base task
#      because only seam crossings light up; the rest of the center remains part
#      of the lattice.
# f) Visual cleanliness
#    - The input still contains only one compact object on a flat field, so the
#      variable background does not create clutter.
#    - The output still uses only three colors at once: black, the chosen
#      background-derived lattice color, and maroon. This keeps the center AND
#      effect legible.
# g) Change inferability
#    - The color mapping is intentionally small and repeated over many pixels:
#      the chosen background fills almost the whole 6x6 input and also drives
#      the non-black lattice color in the 16x16 output.
#    - The seam-intersection rule has a stable geometric cue: rows/cols at the
#      3-step block boundaries inside the 6x6 center box. Across examples, the
#      maroon pattern is always a subset of the tiled glyph, never an unrelated
#      shape.
# Post-change solver assumptions and generator guarantees
#    - The only non-background color is one of {blue, red, green, grey}.
#    - The dominant background is exactly the mapped partner of that glyph color.
#    - The non-background pixels form one tight 3x3 glyph from the known set.
#    - `generate_input()` validates each candidate with `main()` and rejects any
#      candidate whose center maroon count is empty or equals the full glyph.

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

BG_BY_GLYPH_COLOR = {
    Color.BLUE: Color.TEAL,
    Color.RED: Color.YELLOW,
    Color.GREEN: Color.PINK,
    Color.GREY: Color.ORANGE,
}
FG_COLORS = list(BG_BY_GLYPH_COLOR.keys())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 lattice phase mask."""
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


def _lattice_seam_mask(shape: tuple[int, int]) -> np.ndarray:
    """Rows/cols at the 3-step block seams inside the centered 6x6 stamp box."""
    rows, cols = np.indices(shape)
    return ((rows % 3) == 0) | ((cols % 3) == 0)


def _extract_motif_and_bg(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Recover the 3x3 glyph mask and validate the glyph-color -> bg-color rule."""
    observed_bg = int(get_bg(grid))
    fg_colors = colors_present(grid, exclude={observed_bg})
    assert len(fg_colors) == 1

    glyph_color = int(fg_colors[0])
    assert glyph_color in BG_BY_GLYPH_COLOR
    expected_bg = int(BG_BY_GLYPH_COLOR[glyph_color])
    assert observed_bg == expected_bg

    fg_mask = grid == glyph_color
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)
    return motif, expected_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a color-coded 3x3 contour glyph from a non-fixed background, expand it into a 16x16 periodic lattice, and show maroon only at center seam/glyph crossings.

    Input structure:
    - The input is a 6x6 uniform background whose color is chosen by the glyph color rather than being fixed.
    - One monochrome 3x3 contour glyph appears in a tight bounding box and uses one color from the task's small glyph-color palette.
    - The glyph shape is still one of the four contour codes: plus, H, X, or four corners.

    Rule:
    - Detect the dominant background color, recover the lone glyph color, and validate that the two obey the task's glyph-color -> background-color mapping.
    - Crop the 3x3 glyph mask, decode it into the same period-3 phase tile as the base task, and repeat that tile across the 16x16 output using the inferred background color and black.
    - Tile the glyph 2x2 into the centered 6x6 box, but recolor to maroon only where those tiled glyph pixels also lie on the box's 3-step seam rows or seam columns.

    Key insight:
    - Two independent cues must be combined: color now tells you which background/lattice color is active, while the center mark is no longer the whole glyph but the boolean intersection of the tiled glyph with the periodic seam scaffold.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    motif, bg = _extract_motif_and_bg(g)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    maroon_mask = center & _lattice_seam_mask(center.shape)

    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][maroon_mask] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
    fg = int(rng.choice(FG_COLORS))
    bg = int(BG_BY_GLYPH_COLOR[fg])
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    center_box = out[5:11, 5:11]
    full_glyph_count = int(np.tile(motif, (2, 2)).sum())
    maroon_count = int((center_box == Color.MAROON).sum())
    if not (0 < maroon_count < full_glyph_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo D4/1+D8/1")
