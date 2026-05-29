# Combo variant of eee78d87 (index 1)
# combo_spec: D1/0+D14/0
# Name: 字形颜色双重语义：相位偏移+印章色

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill
except ImportError:
    ROOT_DIR = Path("/home/lixinyu/snap/new")
    SOLVER_DIR = ROOT_DIR / "solvers"
    for extra_path in (ROOT_DIR, SOLVER_DIR):
        if str(extra_path) not in sys.path:
            sys.path.append(str(extra_path))
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# a) Concretization and trace:
#    The input stays physically identical to the base task: one 3x3 contour glyph
#    in a 6x6 orange board. The fusion changes how the single visible glyph color
#    is interpreted.
#    Example with a blue plus:
#    - Base task: plus -> decode one fixed 3x3 orange/black tile, tile it with
#      phase (0,0), then stamp the 2x2 center copy in maroon.
#    - D1/0 alone: blue also chooses a shifted lattice phase, e.g. (0,0) here or
#      a different offset for another color, while the center remains maroon.
#    - D14/0 alone: the lattice stays shape-only, but the center stamp color is
#      recolored according to the glyph color instead of always maroon.
#    - This combo does both at once: blue selects both the lattice offset and the
#      center stamp recolor. The silhouette still controls the repeated 3x3 tile
#      content, so shape and color are both necessary to determine the output.
#
# b) Complexity calibration:
#    Use exactly three foreground colors: blue, red, green. That is enough to make
#    a learnable color->offset table and a parallel color->stamp-color table, while
#    staying small enough that 2-4 training examples can plausibly expose most of
#    the rule. The stamp targets are chosen to be visually distinct from orange and
#    black, and at most one case reuses maroon.
#
# c) Implicit assumption audit against the original main():
#    - `len(colors_present(...)) == 1` still holds, because the scene still has one
#      glyph color only.
#    - `fg = g != bg` is no longer sufficient semantically by itself: the boolean
#      mask still finds the motif shape, but the unique foreground color must also
#      be preserved because it drives two downstream decisions.
#    - `tile_fill(out_shape, phase_colors)` assumed zero phase. This becomes
#      `tile_fill(..., offset=color_to_offset[fg_color])`.
#    - `... = Color.MAROON` assumed a fixed stamp color. This becomes a color-driven
#      recolor map.
#    - There is no conflict between the two new semantics because both consume the
#      same unique foreground color and affect disjoint output regions: the lattice
#      everywhere, and the centered 6x6 stamp window.
#
# d) Placement legality:
#    The generator still places exactly one 3x3 glyph at a top-left in 0..3, so the
#    glyph never leaves the 6x6 board. No other objects exist, so overlap concerns do
#    not change from the base task.
#
# e) Visual observability:
#    The phase offset changes the whole 16x16 periodic background and is therefore
#    visible outside the central stamp. The recolor target changes only the 6x6
#    center stamp. Because these two effects live in different regions, a human can
#    compare examples and notice that the same input color consistently drives both.
#
# f) Visual cleanliness:
#    The output remains a clean three-layer scene: black canvas, orange lattice, one
#    solid-color center stamp. There is no new overlap clutter beyond the original
#    stamp-over-lattice overwrite.
#
# g) Inference of the color change:
#    The color mapping is intentionally tiny and explicit:
#    blue(1)  -> phase (0,0), stamp yellow(4)
#    red(2)   -> phase (1,0), stamp maroon(9)
#    green(3) -> phase (0,1), stamp pink(6)
#    No case matches the exact original behavior (zero shift + maroon stamp), so
#    generated examples cannot silently collapse back to the base task.
#
# h) New solver assumptions and generator guarantees:
#    - The foreground color must be one of the three keys in the semantic tables.
#    - The cropped foreground mask must still be one of the four legal contour glyphs.
#    - A non-zero phase shift must visibly alter the lattice for the chosen glyph.
#      `_gen_attempt()` checks this, so any future edit to the tables cannot quietly
#      introduce a hidden degenerate branch.


PHASE_SHIFT_BY_COLOR = {
    Color.BLUE: (0, 0),
    Color.RED: (1, 0),
    Color.GREEN: (0, 1),
}

STAMP_COLOR_BY_FG = {
    Color.BLUE: Color.YELLOW,
    Color.RED: Color.MAROON,
    Color.GREEN: Color.PINK,
}

assert set(PHASE_SHIFT_BY_COLOR) == set(STAMP_COLOR_BY_FG)


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


def _semantics_for_color(color: int) -> tuple[tuple[int, int], int]:
    assert color in PHASE_SHIFT_BY_COLOR
    return PHASE_SHIFT_BY_COLOR[color], STAMP_COLOR_BY_FG[color]


def _extract_motif_and_color(grid: np.ndarray) -> tuple[np.ndarray, int]:
    bg = int(get_bg(grid))
    assert bg == Color.ORANGE

    fg_colors = colors_present(grid, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])
    assert fg_color in PHASE_SHIFT_BY_COLOR

    fg_mask = grid == fg_color
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)
    return motif, fg_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one colored 3x3 contour glyph into a periodic orange/black lattice, with the glyph color simultaneously selecting the lattice phase and the center stamp recolor.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one non-orange foreground color.
    - The foreground pixels form one tight 3x3 contour glyph whose shape is plus, H, X, or four corners.
    - The glyph color is restricted to blue, red, or green, and that visible color is part of the code.

    Rule:
    - Read the 3x3 contour shape to recover the period-3 orange phase tile.
    - Read the glyph color to choose two semantics at once: a phase offset for the tiled lattice and a recolor target for the centered 2x2 stamp.
    - Tile the decoded phase over a black 16x16 canvas using the chosen offset, then tile the glyph mask 2x2 in the centered 6x6 region and paint those cells with the color-specific stamp color.

    Key insight:
    - The motif now carries split meaning across two attributes: shape tells you which periodic tile to repeat, while foreground color tells you both where that period starts and what final color the enlarged center stamp should use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    motif, fg_color = _extract_motif_and_color(g)
    phase_offset, stamp_color = _semantics_for_color(fg_color)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors, offset=phase_offset)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = stamp_color
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = list(PHASE_SHIFT_BY_COLOR.keys())
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
    fg_color = int(rng.choice(FG_COLORS))
    phase_offset, stamp_color = _semantics_for_color(fg_color)
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg_color

    try:
        out = main(inp)
    except Exception:
        return None

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    shifted = tile_fill(out.shape, phase_colors, offset=phase_offset)
    zero_phase = tile_fill(out.shape, phase_colors, offset=(0, 0))

    # Guard against accidental collapse back to the base behavior:
    # either the lattice phase must differ from zero phase, or the stamp color must
    # differ from maroon. With the current tables this is true for every color.
    assert (not np.array_equal(shifted, zero_phase)) or stamp_color != Color.MAROON

    center_mask = np.tile(motif, (2, 2))
    center_r0 = (out.shape[0] - center_mask.shape[0]) // 2
    center_c0 = (out.shape[1] - center_mask.shape[1]) // 2
    center_patch = out[center_r0 : center_r0 + center_mask.shape[0], center_c0 : center_c0 + center_mask.shape[1]]
    assert set(map(int, np.unique(center_patch[center_mask]))) == {stamp_color}
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 1")
