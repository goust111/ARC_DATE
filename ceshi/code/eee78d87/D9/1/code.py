# Variant of eee78d87
# Dimension: D9 — 中心印章的放置位置 (dst)
# Change: 印章固定放置在16x16画布正中心(5,5)-(10,10) → 印章放置在画布某个角落

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis for D9
# - Concrete behavior:
#   The input stays identical to the source task: a 6x6 orange grid contains one
#   3x3 contour glyph in a single non-orange color. The output still has the same
#   16x16 black/orange period-3 lattice decoded from that glyph. The only changed
#   structure is the 6x6 maroon stamp made by tiling the glyph 2x2: instead of
#   always occupying rows 5..10 and cols 5..10, it is anchored to one of the four
#   corners. We use a bijection so the location is inferable from shape type alone:
#   plus -> top-left, X -> top-right, 4corners -> bottom-left, H -> bottom-right.
# - Behavior trace:
#   In the source task, a plus glyph always produced the same centered maroon
#   plus-stamp. In this variant, the same plus still decodes to the same lattice,
#   but its 6x6 maroon overlay starts at (0,0). Likewise an H still decodes to the
#   same lattice as before, but its stamp starts at (10,10). The changed step is
#   only the stamp destination calculation.
# - Complexity calibration:
#   The control variable has four visible states, matching the four glyph classes.
#   This is enough to make the change obvious without introducing extra freedoms.
#   A one-to-one shape/corner mapping also allows elimination if three glyphs are
#   seen in training and the fourth appears at test time.
# - Hidden assumptions in the original main():
#   `center = np.tile(motif, (2, 2))` still holds because the stamp content is
#   unchanged. The old `(out.shape - center.shape)//2` assumption no longer holds;
#   it is replaced by a glyph-dependent corner anchor. All other assertions about
#   background color, single foreground color, tight 3x3 bbox, and valid contour
#   family remain valid and are intentionally preserved.
# - Placement legality:
#   The 6x6 stamp fits exactly in each corner of a 16x16 canvas at top-left
#   coordinates (0,0), (0,10), (10,0), or (10,10). No out-of-bounds writes occur.
#   Overlap with the orange/black lattice is intentional and safe because maroon is
#   simply painted onto the stamp's true cells, just as in the original center case.
# - Visual observability and cleanliness:
#   The maroon overlay is large and isolated, so moving it from center to a corner
#   is immediately visible. Because there is still only one overlay region per
#   output, the image stays tidy; the lattice remains readable elsewhere.
# - Color inference:
#   No color-id-dependent logic is introduced. The branch depends only on the
#   visible glyph shape.
# - Generator guarantees:
#   `main()` now assumes the foreground pattern is exactly one of the four known
#   glyphs so it can choose a corner. The generator enforces that by constructing
#   inputs from the same glyph catalog. It also cycles through glyph names across
#   successive calls so small train sets are more likely to expose multiple corners.


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

STAMP_CORNERS = {
    "plus": (0, 0),
    "X": (0, 1),
    "4corners": (1, 0),
    "H": (1, 1),
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


def _glyph_name(motif: np.ndarray) -> str:
    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    raise AssertionError("Unknown contour glyph")


def _corner_top_left(out_shape: tuple[int, int], stamp_shape: tuple[int, int], glyph_name: str) -> tuple[int, int]:
    corner_r, corner_c = STAMP_CORNERS[glyph_name]
    r0 = 0 if corner_r == 0 else out_shape[0] - stamp_shape[0]
    c0 = 0 if corner_c == 0 else out_shape[1] - stamp_shape[1]
    return r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at a glyph-specific corner.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other visible color.
    - The non-orange cells form a tight 3x3 bounding box containing one contour glyph.
    - The glyph family is unchanged: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the source task.
    - Repeat that phase tile across a 16x16 black canvas to build the orange/black lattice.
    - Tile the input glyph 2x2 into a 6x6 maroon stamp, then place that stamp in a corner chosen by glyph type rather than at the center.

    Key insight:
    - The input object still serves as a code for two output attributes: its contour determines the lattice phase, and that same contour class now also chooses which corner receives the maroon stamp.
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

    glyph_name = _glyph_name(motif)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    stamp = np.tile(motif, (2, 2))
    r0, c0 = _corner_top_left(out.shape, stamp.shape, glyph_name)
    out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]][stamp] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
GLYPH_ORDER = ["plus", "H", "X", "4corners"]
_GLYPH_COUNTER = 0


def _choose_glyph_name(rng: np.random.Generator) -> str:
    global _GLYPH_COUNTER
    # Cycle through all contour classes before repeating so small example sets
    # expose the corner-placement rule more reliably than pure iid sampling.
    offset = int(rng.integers(0, len(GLYPH_ORDER))) if _GLYPH_COUNTER == 0 else 0
    name = GLYPH_ORDER[(_GLYPH_COUNTER + offset) % len(GLYPH_ORDER)]
    _GLYPH_COUNTER += 1
    return name


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = _choose_glyph_name(rng)
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
    raise RuntimeError("Failed to generate input for eee78d87 D9 variant")
