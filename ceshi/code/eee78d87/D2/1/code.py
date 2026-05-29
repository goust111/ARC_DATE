# Variant of eee78d87
# Dimension: D2 — 源对象的形状集合 (src)
# Change: 4种固定的3x3轮廓字形：plus(.#./#11/.#.)、H(#.#/###/#.#)、X(#.#/.#./#.#)... → 字形使用双色编码
#
# Design analysis for the D2 variant
# a) Concretization and behavior trace
#    - Input stays 6x6 with orange background. The active object still fits in a 3x3 bbox, but it now contains
#      exactly two non-orange colors.
#    - One color is the phase layer: its 3x3 occupancy must be exactly one canonical glyph from the original task
#      (plus / H / X / 4corners). This layer alone determines the orange/black period-3 lattice.
#    - The other color is the decoration layer: it occupies a non-empty subset of cells that are empty in the
#      phase layer. Those cells do not affect the lattice; they only appear as an extra tiled layer in the center
#      6x6 stamp, using their original color.
#    - Concrete trace: if the phase color forms plus and the decoration color marks two corners, the original task
#      would have treated the entire non-orange union as one mask and failed to decode. The variant instead isolates
#      the plus mask, builds the same lower-right 2x2 orange phase tile as before, and adds the corner-color cells
#      into the centered 2x2-tiled stamp.
# b) Complexity calibration
#    - The key freedom is limited: only 4 phase glyph categories remain, and the decoration layer is restricted to a
#      small non-empty subset of phase-background cells. This keeps the new rule visible without turning the task into
#      unconstrained arbitrary pixel copying.
#    - The generator caps decoration size at 3 cells. That gives repeated support in the 2x2 center stamp while
#      keeping the maroon phase glyph legible.
# c) Hidden-assumption audit against the original main()
#    - `len(colors_present(...)) == 1` is no longer valid: the variant requires exactly 2 non-background colors.
#    - `fg = g != bg` followed by boolean cropping is no longer valid: boolean collapse destroys the distinction
#      between the phase layer and the decoration layer.
#    - Reusing the whole boolean motif as the center stamp would incorrectly recolor decoration cells to maroon.
#      The new solver must tile the phase layer and decoration layer separately.
#    - Decoding is only safe if exactly one of the two color masks matches a canonical glyph; the generator enforces
#      this by drawing decoration only on phase-background cells and limiting its size so it cannot itself become one
#      of the canonical glyphs.
# d) Placement legality
#    - The phase glyph alone already spans a tight 3x3 bbox, so the combined two-color object also has a 3x3 bbox.
#    - Decoration cells are sampled only from positions not used by the phase glyph, so there is no overlap and no
#      out-of-bounds risk.
#    - Both colors are chosen from non-special ARC colors excluding black, orange, and maroon; they are distinct.
# e) Visual observability
#    - Across examples, one input color controls the global orange lattice and the maroon core glyph, while the other
#      input color reappears only inside the centered 6x6 stamp. This is directly visible from a few examples because
#      the decoration pattern is spatially repeated 2x2.
#    - The rule does not depend on color ids or hidden ordering. It depends only on which color mask matches one of
#      the four visible canonical glyph shapes.
# f) Visual cleanliness
#    - Only one extra layer is added, and it is confined to the center stamp. The lattice outside the center stays
#      the original orange/black structure, so the output remains easy to parse.
#    - The maroon phase stamp and the decoration layer are disjoint by construction, avoiding ambiguous overpainting.
# g) Color-change inferability
#    - There are only two input non-background colors. The decoration color is preserved in the output center, while
#      the phase color is interpreted structurally: its shape becomes the orange lattice code plus a maroon stamp.
#    - This gives a strong structural cue for each color's role without introducing many independent color mappings.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


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


def _split_layers(motif: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Identify the canonical phase layer and the extra decoration layer inside the 3x3 code patch."""
    assert motif.shape == (3, 3)

    cols = colors_present(motif, exclude={Color.ORANGE})
    assert len(cols) == 2
    assert all(col not in {Color.BLACK, Color.ORANGE, Color.MAROON} for col in cols)

    masks = [(int(col), motif == int(col)) for col in cols]
    phase_candidates = [(col, mask) for col, mask in masks if _glyph_name(mask) is not None]
    assert len(phase_candidates) == 1

    _phase_color, phase_mask = phase_candidates[0]
    deco_color = next(col for col, _mask in masks if col != _phase_color)
    deco_mask = motif == deco_color

    assert bool(deco_mask.any())
    assert not bool(np.any(phase_mask & deco_mask))
    assert _glyph_name(deco_mask) is None
    return phase_mask.astype(bool), deco_mask.astype(bool), int(deco_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-color 3x3 code patch into the usual phase lattice, then stamp both encoded layers at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly two distinct non-orange colors.
    - The non-orange cells have a tight 3x3 bbox.
    - Within that bbox, one color forms exactly one canonical contour glyph, while the other color marks extra cells
      that lie outside that glyph.

    Rule:
    - Separate the two non-orange layers by color.
    - Use the color mask that matches a canonical glyph to decode the period-3 orange/black lattice.
    - Tile the canonical glyph 2x2 in maroon at the center, and tile the other color's mask 2x2 as an additional
      decoration layer using that same input color.

    Key insight:
    - The 3x3 patch is now a dual-purpose code: one color is structural and drives the global lattice, while the
      other color is local and only survives as a repeated ornament in the centered stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 2

    b = bbox(g != bg)
    assert b is not None
    motif = crop_bbox(g, b)
    assert motif.shape == (3, 3)

    phase_mask, deco_mask, deco_color = _split_layers(motif)
    phase = _decode_phase_tile(phase_mask)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center_phase = np.tile(phase_mask, (2, 2))
    center_deco = np.tile(deco_mask, (2, 2))
    r0 = (out.shape[0] - center_phase.shape[0]) // 2
    c0 = (out.shape[1] - center_phase.shape[1]) // 2
    out[r0 : r0 + center_phase.shape[0], c0 : c0 + center_phase.shape[1]][center_phase] = Color.MAROON
    out[r0 : r0 + center_deco.shape[0], c0 : c0 + center_deco.shape[1]][center_deco] = deco_color
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CODE_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_DECOR_CELLS = 3
MAX_RETRIES = 300


def _sample_deco_mask(rng: np.random.Generator, phase_mask: np.ndarray) -> np.ndarray:
    slots = np.argwhere(~phase_mask)
    assert len(slots) > 0
    k = int(rng.integers(1, min(MAX_DECOR_CELLS, len(slots)) + 1))
    choice = slots[rng.choice(len(slots), size=k, replace=False)]
    deco = np.zeros((3, 3), dtype=bool)
    deco[choice[:, 0], choice[:, 1]] = True
    assert _glyph_name(deco) is None
    return deco


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    phase_mask = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    phase_color, deco_color = map(int, rng.choice(CODE_COLORS, size=2, replace=False))
    deco_mask = _sample_deco_mask(rng, phase_mask)
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    patch = inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    patch[phase_mask] = phase_color
    patch[deco_mask] = deco_color

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
