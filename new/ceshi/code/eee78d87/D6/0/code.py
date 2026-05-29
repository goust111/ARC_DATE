# Variant of eee78d87
# Dimension: D6 — 铺贴周期 (meta)
# Change: 固定周期3（3x3相位瓦片），每个瓦片单元为3x3像素... → 周期由字形类型编码决定

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input stays 6x6 with orange background and one 3x3 contour glyph in a single non-orange color.
#    - Output stays 16x16 and still contains:
#      1. a black/orange periodic lattice over the full canvas
#      2. a centered 6x6 maroon stamp made by tiling the original 3x3 glyph 2x2
#    - The changed part is the lattice period. Instead of always using a 3x3 phase tile, the glyph type
#      selects the tile side length:
#        plus -> 2, H -> 4, X -> 5, 4corners -> 6
#    - The phase geometry is the same contour-dependent shape family as the original task, but drawn on
#      a p x p tile:
#        plus      -> lower-right block          : phase[1:, 1:] = True
#        H         -> left vertical bar          : phase[1:, 0] = True
#        X         -> top bar except corner + left bar below
#        4corners  -> full top bar + left bar below
#    - Example trace:
#      original H: decode to a fixed 3x3 tile with orange only in the lower two cells of column 0.
#      variant H : decode to period 4, then draw the same relation on a 4x4 tile, yielding a taller and
#                  more separated lattice. The center 6x6 maroon H-stamp is unchanged.
#
# b) Complexity calibration
#    - Four periods {2,4,5,6} are enough to create visibly different lattice densities without making the
#      rule depend on many cases.
#    - Avoiding period 3 prevents accidental regression to the original task on some inputs.
#    - The phase shape still depends on the same four glyph classes, so the task remains close to the seed.
#
# c) Implicit-assumption audit against the original main()
#    - `motif.shape == (3, 3)` still holds: the encoded object is unchanged.
#    - `_decode_phase_tile(motif)` used to guarantee a 3x3 result; this must change to return variable-size
#      tiles, otherwise `tile_fill` would silently keep the old behavior.
#    - `out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)` produced the fixed 16x16 canvas. This does
#      not depend on the period, so it remains valid and preserves the original framing.
#    - The center-stamp placement uses the output size and the 6x6 tiled glyph only; it is still correct for
#      all periods because the stamp is overlaid after the lattice is drawn.
#
# d) Placement legality
#    - Generator still places exactly one 3x3 motif at top-left positions 0..3 in each axis, so it never
#      crosses the 6x6 boundary and cannot overlap any other object because there are no others.
#    - Repeated generator calls are intentionally diversified with a shuffled four-glyph cycle so a small
#      train set is more likely to expose the period mapping.
#    - Every generated input is validated by `main()` before being returned.
#
# e) Visual observability
#    - The period change is easy to see from 2-4 examples: plus gives a dense 2-step rhythm, H is much
#      coarser at 4, X coarser still at 5, and 4corners at 6.
#    - The rule uses visible shape structure only; it does not depend on color ids or input position.
#
# f) Visual cleanliness
#    - Only two layers exist in the output: the periodic orange/black lattice and the central maroon stamp.
#      This matches the seed task and keeps the image readable even when the lattice period is large.
#
# g) Color-rule audit
#    - No new color mapping is introduced. Input non-orange color is ignored except as the glyph carrier.
#      Output still uses only black, orange, and maroon.
#
# Post-change input assumptions to re-check
#    - The solver assumes the non-orange foreground is exactly one of the four known 3x3 contour glyphs.
#    - The generator guarantees this by sampling only from `GLYPHS`.
#    - The period logic assumes the original structural distinctions:
#      plus has no corners; H/X/4corners have all corners; H alone has both side arms; X alone has center
#      without side arms; 4corners alone has neither center nor side arms.


def _decode_period_and_phase_tile(motif: np.ndarray) -> tuple[int, np.ndarray]:
    """Decode the contour glyph into both its lattice period and its period-sized orange phase mask."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edge_centers = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)

    if not bool(corners.any()):
        assert bool(motif[1, 1])
        assert bool(edge_centers.all())
        period = 2
        phase = np.zeros((period, period), dtype=bool)
        phase[1:, 1:] = True
        return period, phase

    assert bool(corners.all())
    assert not bool(motif[0, 1])
    assert not bool(motif[2, 1])
    assert bool(motif[1, 0]) == bool(motif[1, 2])

    if bool(motif[1, 0]):
        assert bool(motif[1, 1])
        period = 4
        phase = np.zeros((period, period), dtype=bool)
        phase[1:, 0] = True
        return period, phase

    period = 5 if bool(motif[1, 1]) else 6
    phase = np.zeros((period, period), dtype=bool)
    phase[1:, 0] = True
    phase[0, 1 if bool(motif[1, 1]) else 0 :] = True
    return period, phase


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 3x3 contour glyph as both a phase-shape code and a period code, then build the repeated lattice with the glyph stamped in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels form a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of four allowed forms: plus, H, X, or four corners.

    Rule:
    - Decode the glyph type from its 3x3 occupancy pattern.
    - Convert that glyph into a period-sized orange/black phase tile, where the period depends on the glyph type.
    - Fill the fixed 16x16 output canvas by repeating that tile.
    - Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon inside the centered 6x6 region.

    Key insight:
    - The input object now carries two pieces of information at once: which orange phase shape repeats and how large the repeat cycle is. The changed lattice spacing is the visible signal of the new rule.
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

    _period, phase = _decode_period_and_phase_tile(motif)
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

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
_GLYPH_ORDER = tuple(GLYPHS.keys())
_GEN_RNG = np.random.default_rng()
_GLYPH_QUEUE: list[str] = []


def _next_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_QUEUE
    if not _GLYPH_QUEUE:
        _GLYPH_QUEUE = list(rng.permutation(_GLYPH_ORDER))
    return _GLYPH_QUEUE.pop()


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[_next_glyph_name(rng)]
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
    raise RuntimeError("Failed to generate input for eee78d87")
