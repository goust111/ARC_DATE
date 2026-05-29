# Combo variant of eee78d87 (index 26)
# combo_spec: D1/0+D16/0
# Name: 旋转指令编码+颜色相位偏移

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the fused D1/0 + D16/0 variant
# ---------------------------------------------------
# a) Concrete behavior and step trace
# - Seed task:
#   - Read one 3x3 boolean contour glyph, decode it to one period-3 orange/black tile,
#     repeat that tile over 16x16, then place a centered 2x2 maroon stamp of the glyph.
# - D1/0 contribution:
#   - The visible foreground color now matters. We keep only three colors:
#     red -> phase offset (0, 0), blue -> (1, 0), green -> (0, 1).
# - D16/0 contribution:
#   - The input glyph itself becomes orientation-sensitive. Each semantic glyph family has
#     one canonical asymmetric 3x3 form; rotating that form by 0/1/2/3 quarter-turns
#     encodes a whole-canvas rotation of 0/90/180/270 degrees after the lattice is built.
# - Concrete trace:
#   - Suppose the input is the blue, left-facing rotation of the X-family glyph.
#   - The X-family identity still chooses the same phase tile as the seed task's X.
#   - Blue changes the tiling start from offset (0, 0) to (1, 0), shifting the orange/black
#     lattice before any rotation happens.
#   - The left-facing asymmetry says "rotate the finished lattice by one `np.rot90` step".
#   - Finally, the same observed asymmetric glyph is tiled 2x2 in maroon at the center.
# - Difference from the seed:
#   - The seed task could throw away both visible color and orientation. This fused variant
#     cannot: shape chooses tile content, color chooses phase origin, and asymmetry chooses
#     the lattice's final orientation.
#
# b) Complexity calibration
# - The phase offset uses exactly three color branches, which is enough to make color visibly
#   relevant without creating a hard-to-infer many-color lookup.
# - Rotation uses four quarter-turn states because that is the natural small discrete family
#   for a square 3x3 glyph and a square 16x16 output.
# - To avoid ambiguity, the generator does not reuse the seed's fully symmetric glyphs.
#   Instead it uses four asymmetric "derived contour" families whose 16 rotated instances are
#   all unique, so every input motif has one unambiguous (family, rotation) decode.
# - To avoid behavioral collapse in the generated data, the generator also avoids the neutral
#   red offset and the zero-rotation code. The solver still supports them, but emitted samples
#   use only visibly active color shifts and visibly active rotations.
#
# c) Implicit-assumption review of the original main()
# - `colors_present(..., exclude={bg}) == 1` still holds, but the old code's next step
#   `fg = g != bg` was no longer enough. The fused solver must preserve both the mask and
#   the unique foreground color because color now controls phase.
# - The old `_decode_phase_tile(motif)` assumed the raw cropped boolean motif was one of the
#   four seed glyphs. That assumption is broken here because the cropped motif is asymmetric.
#   The new solver first normalizes the motif against the 16 allowed oriented codes, recovers
#   the underlying base family, then runs the seed decoder on that base family.
# - The old lattice fill had no rotation stage. The fused rule inserts a deterministic
#   `np.rot90` after `tile_fill(...)` and before the maroon center stamp is painted.
# - The center stamp logic is still safe because the stamp uses the observed oriented motif,
#   not the normalized base family, so the input's asymmetry remains visually present.
#
# d) Placement legality
# - The generator still places exactly one 3x3 motif into a 6x6 orange field with top-left
#   position in rows/cols 0..3, so the bbox is tight, in-bounds, and non-overlapping.
# - All asymmetric codes remain 3x3, so the output stamp still fits the unchanged 6x6 center.
#
# e) Visual observability
# - The output lattice changes in two globally visible ways:
#   - color shifts the phase of the whole orange/black tiling
#   - motif orientation rotates that whole lattice
# - The centered maroon stamp stays in the input orientation, which gives the human solver a
#   stable visual handle for reading the orientation cue from the small glyph.
# - The generator enforces blue/green colors and non-zero rotations so every sampled training
#   pair exhibits both effects instead of occasionally landing on a visually neutral code.
#
# f) Visual cleanliness
# - The composition remains simple: one periodic black/orange background plus one centered
#   maroon overlay. No extra objects or auxiliary markers are introduced.
# - Only three output colors remain in use: black, orange, and maroon.
#
# g) Post-test degeneration metrics
# - Metric 1: `rotation_effect_pixels` compares the rotated lattice with the same lattice
#   before rotation (ignoring the maroon stamp). It must be positive whenever the sampled
#   rotation code is non-zero, otherwise D16/0 has collapsed.
# - Metric 2: `phase_effect_pixels` compares the chosen color offset with offset (0, 0)
#   under the same base family and rotation. It must be positive for blue/green samples,
#   otherwise D1/0 has collapsed back to a color-agnostic lattice.
#
# h) New solver assumptions the generator now guarantees
# - The solver accepts red/blue/green, but generated inputs use only blue/green so the color
#   branch always changes the lattice relative to the seed's zero-offset case.
# - The cropped 3x3 mask is always one of the 16 legal oriented codes, so family recovery and
#   rotation decoding never face ambiguous matches.
# - Generated rotations are always in {1, 2, 3}, so the rotation branch is always exercised.
# - Because the output is always 16x16, `np.rot90` preserves shape and the centered 6x6 stamp
#   placement remains valid for every decoded rotation.


PHASE_SHIFT_BY_COLOR = {
    Color.RED: (0, 0),
    Color.BLUE: (1, 0),
    Color.GREEN: (0, 1),
}


BASE_GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


CANONICAL_ORIENTED_GLYPHS = {
    # Base plus with one extra corner pixel.
    "plus": np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    # Base H with one missing corner pixel.
    "H": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=bool),
    # Base X with one extra edge-center pixel.
    "X": np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    # Base four-corners with one extra edge-center pixel.
    "4corners": np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


GLYPH_NAMES = tuple(BASE_GLYPHS.keys())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode one of the seed task's 3x3 contour families into its period-3 orange phase mask."""
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


def _phase_offset_for_color(color: int) -> tuple[int, int]:
    assert color in PHASE_SHIFT_BY_COLOR
    return PHASE_SHIFT_BY_COLOR[color]


def _decode_family_and_rotation(motif: np.ndarray) -> tuple[str, int]:
    """Recover the base family and the quarter-turn count encoded by the asymmetric glyph."""
    assert motif.shape == (3, 3)
    for name, canonical in CANONICAL_ORIENTED_GLYPHS.items():
        for rot_k in range(4):
            if np.array_equal(motif, np.rot90(canonical, rot_k)):
                return name, rot_k
    raise AssertionError("expected one of the 16 legal oriented glyph codes")


def _phase_canvas(base_name: str, fg_color: int, rot_k: int, out_shape: tuple[int, int]) -> np.ndarray:
    phase = _decode_phase_tile(BASE_GLYPHS[base_name])
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    lattice = tile_fill(out_shape, phase_colors, offset=_phase_offset_for_color(fg_color))
    return np.rot90(lattice, rot_k)


def _oriented_glyph(base_name: str, rot_k: int) -> np.ndarray:
    return np.rot90(CANONICAL_ORIENTED_GLYPHS[base_name], rot_k)


def _combo_is_observable(base_name: str, fg_color: int, rot_k: int) -> bool:
    """True when both the color offset and the rotation visibly change the lattice."""
    actual = _phase_canvas(base_name, fg_color, rot_k, (16, 16))
    no_rotation = _phase_canvas(base_name, fg_color, 0, (16, 16))
    neutral_color = _phase_canvas(base_name, Color.RED, rot_k, (16, 16))
    return bool(np.any(actual != no_rotation) and np.any(actual != neutral_color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one colored asymmetric 3x3 glyph into a phase-shifted periodic lattice, then rotate that lattice and stamp the glyph at the center.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one non-orange foreground color.
    - The foreground pixels occupy a tight 3x3 bounding box.
    - That 3x3 motif is one of four asymmetric glyph families in one of four quarter-turn orientations, and its visible color is red, blue, or green.

    Rule:
    - Normalize the asymmetric motif to recover its base glyph family; that family selects the same period-3 orange phase tile as in the seed task.
    - Read the foreground color as a phase offset: red -> (0, 0), blue -> (1, 0), green -> (0, 1), and tile the corresponding orange/black phase over a 16x16 canvas.
    - Read the motif's orientation as a whole-canvas `np.rot90` count, rotate the tiled lattice by that amount, then tile the observed oriented motif 2x2 and recolor its true cells to maroon in the centered 6x6 region.

    Key insight:
    - The small input glyph now carries three independent pieces of information at once: family chooses the repeated tile content, color shifts the tile's phase, and asymmetric direction rotates the completed lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])
    assert fg_color in PHASE_SHIFT_BY_COLOR

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    base_name, rot_k = _decode_family_and_rotation(motif)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = _phase_canvas(base_name, fg_color, rot_k, out_shape)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (combo) ---------------

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
GEN_FG_COLORS = [Color.BLUE, Color.GREEN]
GEN_ROTATIONS = (1, 2, 3)
MAX_RETRIES = 300


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base_name = str(rng.choice(GLYPH_NAMES))
    fg_color = int(rng.choice(GEN_FG_COLORS))
    rot_k = int(rng.choice(GEN_ROTATIONS))
    assert _combo_is_observable(base_name, fg_color, rot_k)
    motif = _oriented_glyph(base_name, rot_k)
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg_color

    try:
        out = main(inp)
    except Exception:
        return None

    assert out.shape == (16, 16)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant D1/0+D16/0")
