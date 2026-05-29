# Combo variant of eee78d87 (index 3)
# combo_spec: D2/1+D4/0
# Name: 双色字形+保留原色印章
#
# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - The input is still a 6x6 orange field with one tight 3x3 coded patch, but that patch now uses exactly two
#      non-orange colors instead of one.
#    - One color occupies exactly one canonical contour glyph from the original task (plus / H / X / 4corners).
#      That layer alone is structural: it decodes the orange/black period-3 lattice exactly as before.
#    - The other color occupies a non-empty subset of the phase-background cells inside the same 3x3 patch. Those
#      cells are decorative: they do not affect the lattice, but they do get tiled into the centered 6x6 stamp.
#    - Concrete trace: if the structural color forms plus and the decoration color marks two corners, the original
#      solver would fail because it collapses both colors into one boolean mask. The fused rule instead isolates the
#      plus mask, builds the same lower-right 2x2 orange phase tile, and then places a 2x2 center stamp where both
#      the plus cells and the corner cells keep their original input colors.
# b) Complexity calibration
#    - The solver still only needs to distinguish among four structural glyph categories, so the global behavior
#      remains compact and learnable.
#    - Decoration is limited to a small non-empty subset of the structural background cells, which makes the second
#      color visible in the center stamp without turning the patch into an arbitrary free-form picture.
#    - Preserving the structural color in the center stamp makes the fusion visible: the output now uses four colors
#      (black, orange, structural color, decoration color) instead of the original three.
# c) Hidden-assumption audit against the original main()
#    - `len(colors_present(...)) == 1` is invalid now; the fused input must contain exactly two non-background colors.
#    - Boolean collapse of the motif is invalid; it destroys the distinction between the structural layer and the
#      decoration layer.
#    - Recoloring all center glyph cells to maroon is invalid under D4/0; both layers must retain their original
#      input colors in the centered 2x2 stamp.
#    - Decoding is only well-defined if exactly one of the two color masks is a canonical glyph. The generator must
#      enforce this, otherwise the solver would face ambiguous structural-role assignment.
# d) Placement legality
#    - The canonical phase glyph already spans a full 3x3 bbox, so the two-color union always keeps the required 3x3
#      bbox regardless of which extra decoration cells are chosen.
#    - Decoration is sampled only from cells not occupied by the phase glyph, so the two layers never overlap.
#    - The 3x3 patch is still placed with top-left coordinates in [0, 3] x [0, 3], so it always fits inside 6x6.
# e) Visual observability
#    - Across examples, one input color reappears globally only indirectly through the orange lattice and locally as
#      the main repeated center glyph, while the second input color appears only in the center ornament pattern.
#    - This separation is visible from shape roles, not from color ids: the human solver can identify the structural
#      layer because it is the only one matching one of the four familiar contour glyphs.
# f) Visual cleanliness
#    - The orange/black lattice remains unchanged outside the center, so the global field stays easy to parse.
#    - The two center layers are disjoint by construction, preventing overwrite ambiguity and keeping the 6x6 stamp
#      legible even with four total colors in the output.
# g) Change inferability
#    - The fusion introduces only one additional retained color relative to the original task, and it is anchored to a
#      repeated 2x2 ornament pattern in the center.
#    - The preserved structural color is equally inferable because it occupies the exact canonical glyph shape inside
#      the center stamp, making the "keep original colors" rule observable from a few examples.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CODE_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_DECOR_CELLS = 3
MAX_RETRIES = 300


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 orange phase mask."""
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


def _glyph_name(mask: np.ndarray) -> str | None:
    for name, glyph in GLYPHS.items():
        if np.array_equal(mask, glyph):
            return name
    return None


def _split_layers(motif: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, int]:
    """Return the canonical phase layer and the decoration layer from the 3x3 two-color code patch."""
    assert motif.shape == (3, 3)

    cols = colors_present(motif, exclude={Color.ORANGE})
    assert len(cols) == 2
    assert all(int(col) not in {Color.BLACK, Color.ORANGE, Color.MAROON} for col in cols)

    masks = [(int(col), motif == int(col)) for col in cols]
    phase_candidates = [(col, mask) for col, mask in masks if _glyph_name(mask) is not None]
    assert len(phase_candidates) == 1

    phase_color, phase_mask = phase_candidates[0]
    deco_color = next(col for col, _mask in masks if col != phase_color)
    deco_mask = motif == deco_color

    assert bool(deco_mask.any())
    assert not bool(np.any(phase_mask & deco_mask))
    assert _glyph_name(deco_mask) is None
    return phase_mask.astype(bool), int(phase_color), deco_mask.astype(bool), int(deco_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-color 3x3 glyph patch into the usual orange/black lattice, then stamp both layers at the center
    without recoloring either one.

    Input structure:
    - The grid is 6x6 with orange background and exactly two distinct non-orange colors.
    - The non-orange cells have a tight 3x3 bbox.
    - Inside that bbox, one color forms exactly one canonical contour glyph, while the other color occupies extra
      cells that lie in the canonical glyph's background.

    Rule:
    - Separate the two non-orange layers by color.
    - Use the canonical-glyph layer to decode the period-3 orange/black lattice and tile it across a 16x16 canvas.
    - Tile both 3x3 layers 2x2 into the centered 6x6 region, preserving the original input color of each layer.

    Key insight:
    - The two colors have different roles: one color is structural and determines the global lattice, but both colors
      survive locally in the center because the stamp now keeps the original glyph colors instead of recoloring them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 2

    b = bbox(g != bg)
    assert b is not None
    motif = crop_bbox(g, b)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase_mask, phase_color, deco_mask, deco_color = _split_layers(motif)
    phase = _decode_phase_tile(phase_mask)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center_phase = np.tile(phase_mask, (2, 2))
    center_deco = np.tile(deco_mask, (2, 2))
    r0 = (out.shape[0] - center_phase.shape[0]) // 2
    c0 = (out.shape[1] - center_phase.shape[1]) // 2
    center = out[r0 : r0 + center_phase.shape[0], c0 : c0 + center_phase.shape[1]]
    center[center_phase] = phase_color
    center[center_deco] = deco_color
    return out


def _sample_deco_mask(rng: np.random.Generator, phase_mask: np.ndarray) -> np.ndarray:
    """Choose a small non-empty decoration subset from phase-background cells."""
    slots = np.argwhere(~phase_mask)
    assert len(slots) > 0
    k = int(rng.integers(1, min(MAX_DECOR_CELLS, len(slots)) + 1))
    choice = slots[rng.choice(len(slots), size=k, replace=False)]
    deco = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    deco[choice[:, 0], choice[:, 1]] = True
    assert _glyph_name(deco) is None
    return deco


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    phase_mask = GLYPHS[str(rng.choice(tuple(GLYPHS.keys())))]
    phase_color, deco_color = map(int, rng.choice(CODE_COLORS, size=2, replace=False))
    deco_mask = _sample_deco_mask(rng, phase_mask)
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    patch = inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    patch[phase_mask] = phase_color
    patch[deco_mask] = deco_color

    try:
        out = main(inp)
    except Exception:
        return None

    assert len(colors_present(out)) == 4
    assert Color.MAROON not in colors_present(out)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant D2/1+D4/0")
