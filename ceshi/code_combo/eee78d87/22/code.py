# Combo variant of eee78d87 (index 22)
# combo_spec: D8/2+D15/1
# Name: 背景纹理+加粗格栅线

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, tile_fill

# Design analysis for combo D8/2 + D15/1
# a) Concrete behavior and trace
#    - Seed task:
#      a 6x6 orange board contains one tight 3x3 contour glyph. The solver can
#      take `g != orange`, crop its bbox, decode the 3x3 glyph, and tile the
#      resulting 3x3 orange/black phase mask over a 16x16 output.
#    - Fused input:
#      the 6x6 board is first filled with a regular two-color texture
#      (orange + one texture color), chosen from vertical stripes, horizontal
#      stripes, or checkerboard, with period 2 and phase 0/1. A third color
#      then overwrites the `True` cells of one 3x3 glyph.
#      Example: a checkerboard background might place teal on every even parity
#      cell. A plus glyph in blue at rows 2-4, cols 1-3 changes only the five
#      plus cells; the four `False` cells inside that 3x3 box keep the original
#      checkerboard colors. Old logic `bbox(g != orange)` now covers almost the
#      whole board and fails to isolate the glyph.
#    - Fused output:
#      the same glyph decoder is used, but the carrier lattice no longer uses
#      a 3x3 tile with 1-pixel black lines. Instead the decoded phase is
#      thickened into a 4x4 tile by duplicating the "line residue" index 0
#      along both axes via `new_to_old = [0, 0, 1, 2]`. This turns every former
#      1-pixel grid line into a 2-pixel line while preserving the relative
#      phase pattern of the decoded glyph. The centered 6x6 maroon 2x2 stamp
#      remains unchanged.
# b) Complexity calibration
#    - The texture family is limited to three visually obvious periodic layouts
#      with only two phases, so a human can spot the repeating background from a
#      few examples.
#    - The glyph still comes from the original four-shape catalog. No new glyph
#      taxonomy is introduced, which keeps the core symbolic mapping stable.
#    - Texture and output-line changes are independent: input complexity is
#      "find the contour against regular texture", output complexity is "same
#      decoded phase, but with thicker grid lines and period 4 instead of 3".
# c) Implicit assumption review against the seed solver
#    - `bg = get_bg(g)` was safe before because every non-orange cell belonged
#      to the glyph. That is broken: orange is still present, but non-orange no
#      longer means "motif".
#    - `len(colors_present(g, exclude={bg})) == 1` is broken because textured
#      backgrounds add a second non-orange color.
#    - `bbox(g != bg)` is broken because the periodic texture spans most of the
#      board.
#    - `crop_bbox(...).shape == (3, 3)` is no longer guaranteed from a raw
#      foreground mask; it must be recovered after subtracting the correct
#      texture hypothesis.
#    - `tile_fill(..., phase_colors)` with a 3x3 tile encodes line width 1.
#      D15/1 replaces this with a 4x4 thickened tile, so the seed output fill
#      line must be changed.
# d) Placement legality
#    - The generator always places the 3x3 glyph wholly inside the 6x6 board.
#    - The texture is defined on the whole board first; glyph overwrite touches
#      only its own `True` cells and cannot go out of bounds.
#    - Texture color and glyph color are forced distinct, so the overwrite is
#      always visible.
# e) Visual observability
#    - In the input, large regions outside the glyph show a clear repeating
#      stripe/checkerboard texture, while the glyph remains the only tight 3x3
#      monochrome contour object.
#    - In the output, black runs appear in 2-pixel bands. Across examples this
#      is visibly different from the seed task's 1-pixel lattice, while the
#      maroon center stamp still anchors the correspondence to the input glyph.
# f) Visual cleanliness
#    - Only one extra texture color is introduced, and the background families
#      are low-frequency and regular rather than noisy. This keeps the glyph
#      readable and the board uncluttered.
#    - The output still has just two layers: the thick black/orange lattice and
#      the maroon center stamp.
# g) Change inferability
#    - There is no arbitrary color remapping. Orange keeps its carrier role,
#      maroon keeps its center-stamp role, and the input's two non-orange colors
#      separate into "regular texture" vs "glyph overwrite" through structure.
# Post-change solver assumptions and generator guarantees
#    - Exactly one texture hypothesis from the known family set explains the
#      board such that the mismatch mask is a 3x3 monochrome glyph from the
#      original catalog.
#    - The mismatch color is unique and distinct from the texture color.
#    - The generator validates every sample with `main()`, rejecting ambiguous
#      placements or degenerate texture/glyph combinations.

GRID_H, GRID_W = 6, 6
OUTPUT_SHAPE = (16, 16)
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
TEXTURE_KINDS = ("vstripe", "hstripe", "checker")
TEXTURE_PHASES = (0, 1)
MAX_RETRIES = 400

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_MASKS = tuple(GLYPHS.values())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the seed task's phase mask."""
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


def _build_texture(shape: tuple[int, int], texture_color: int, kind: str, phase: int) -> np.ndarray:
    """Construct one of the allowed orange-based background textures."""
    h, w = shape
    rr, cc = np.indices((h, w))
    if kind == "vstripe":
        mask = (cc + phase) % 2 == 0
    elif kind == "hstripe":
        mask = (rr + phase) % 2 == 0
    elif kind == "checker":
        mask = (rr + cc + phase) % 2 == 0
    else:
        raise ValueError(f"unknown texture kind: {kind}")
    return np.where(mask, texture_color, Color.ORANGE).astype(int)


def _extract_motif(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique 3x3 glyph by subtracting a valid texture hypothesis.

    A hypothesis is valid iff:
    - it uses one non-orange color as the periodic texture color,
    - every mismatch against that texture has one replacement color,
    - the mismatch bbox is exactly 3x3,
    - that 3x3 mismatch mask matches one of the seed glyphs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    candidates: list[np.ndarray] = []
    non_orange = colors_present(g, exclude={Color.ORANGE})
    assert len(non_orange) >= 2

    for texture_color in non_orange:
        for kind in TEXTURE_KINDS:
            for phase in TEXTURE_PHASES:
                texture = _build_texture(g.shape, texture_color, kind, phase)
                delta = g != texture
                if not bool(delta.any()):
                    continue

                mismatch_colors = colors_present(g[delta], exclude={Color.ORANGE})
                if len(mismatch_colors) != 1:
                    continue
                if mismatch_colors[0] == texture_color:
                    continue

                b = bbox(delta)
                if b is None:
                    continue
                motif = crop_bbox(delta, b).astype(bool)
                if motif.shape != (MOTIF_SIZE, MOTIF_SIZE):
                    continue
                if any(np.array_equal(motif, glyph) for glyph in GLYPH_MASKS):
                    candidates.append(motif)

    assert len(candidates) == 1
    return candidates[0]


def _thicken_phase_tile(phase: np.ndarray) -> np.ndarray:
    """
    Convert the seed 3x3 phase tile into a period-4 tile with 2-pixel line width.

    Index 0 in the seed tile represents the 1-pixel grid line residue. D15/1
    duplicates that residue along each axis, so the new tile samples the old
    one with mapping [0, 0, 1, 2].
    """
    assert phase.shape == (3, 3)
    new_to_old = np.array([0, 0, 1, 2], dtype=int)
    return phase[new_to_old][:, new_to_old]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a 3x3 contour glyph from a textured 6x6 board, then render its decoded lattice with thicker black grid lines and the usual centered maroon stamp.

    Input structure:
    - The grid is 6x6 and contains an orange-based period-2 background texture, using one additional repeating texture color.
    - A single 3x3 contour glyph from the original catalog is overlaid in a third color; only the glyph's `True` cells overwrite the texture, so the 3x3 box still contains background-colored holes.
    - The texture is one of three regular families: vertical stripes, horizontal stripes, or checkerboard.

    Rule:
    - Identify the unique texture hypothesis whose mismatches form exactly one valid 3x3 contour glyph.
    - Decode that recovered glyph into the same 3x3 phase mask as in the seed task, then thicken the mask into a 4x4 orange/black tile by duplicating the line residue along both axes.
    - Tile that 4x4 carrier across the 16x16 output and place a 2x2 maroon tiling of the recovered glyph in the centered 6x6 region.

    Key insight:
    - The input's extra non-orange pixels are not arbitrary noise; they form a regular texture that must be modeled and subtracted first. Once the hidden glyph is recovered, it controls the same symbolic phase as before, but the rendered lattice uses 2-pixel grid lines instead of 1-pixel lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    motif = _extract_motif(g)
    phase = _decode_phase_tile(motif)
    thick_phase = _thicken_phase_tile(phase)
    phase_colors = np.where(thick_phase, Color.ORANGE, Color.BLACK).astype(int)
    out = tile_fill(OUTPUT_SHAPE, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    texture_color = int(rng.choice([c for c in FG_COLORS if c != fg]))
    texture_kind = str(rng.choice(TEXTURE_KINDS))
    texture_phase = int(rng.choice(TEXTURE_PHASES))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = _build_texture((GRID_H, GRID_W), texture_color, texture_kind, texture_phase)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    # Keep the background visibly textured outside the glyph box.
    outside = np.ones((GRID_H, GRID_W), dtype=bool)
    outside[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE] = False
    assert np.count_nonzero(inp[outside] == texture_color) > 0

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
    raise RuntimeError("Failed to generate input for eee78d87 combo D8/2+D15/1")
