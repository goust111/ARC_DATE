# Combo variant of eee78d87 (index 20)
# combo_spec: D1/1+D2/2+D4/3
# Name: 旋转解码+旋转印章+非对称字形

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "solvers" / "arc_utils.py").exists()
)
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis:
# variant:
#   preserved_structure:
#     - input is still a 6x6 orange field with exactly one non-orange 3x3 glyph
#     - output is still a 16x16 orange/black period-3 lattice with a centered 6x6 maroon overlay
#     - foreground input color still serves only to reveal the glyph mask; it does not affect decoding
#   fused_changes:
#     - D2/2 replaces the old symmetric contour family with four asymmetric canonical glyphs
#     - D1/1 allows any 90-degree rotation of those glyphs in the input and requires the solver to
#       recover the canonical glyph identity plus the observed rotation before decoding
#     - D4/3 replaces the old 2x2 repeat stamp with a 2x2 rotation mosaic:
#       TL=observed motif, TR=clockwise 90, BR=180, BL=counterclockwise 90
#   behavior_trace:
#     example:
#       canonical_L:
#         100
#         100
#         111
#       observed_input:
#         111
#         100
#         100
#       original_solver:
#         - fails because it expects either zero active corners (plus) or four active corners
#         - this asymmetric mixed-corner shape violates the old contour assertions immediately
#       fused_solver:
#         - matches the observed 3x3 mask to canonical glyph "L" rotated clockwise 90 degrees
#         - decodes the canonical phase tile as rotate_clockwise(invert(canonical_L)) = 000/011/011
#         - rotates that phase tile by the same 90 degrees, yielding 000/110/110 for lattice tiling
#         - writes the centered 6x6 stamp as [obs | cw90(obs); ccw90(obs) | rot180(obs)]
#       visible_difference:
#         - both the repeated lattice and the maroon center now expose orientation, so a rotated
#           input does not collapse back to the same output as its canonical form
#   complexity_calibration:
#     - only four glyph families are used, each with four rotations, so there are 16 distinct
#       observable motif states without creating an arbitrary large lookup table
#     - every canonical glyph has a unique rotation orbit and no reflection/rotation symmetry,
#       preventing ambiguous matches when recovering orientation
#     - the phase rule is systematic instead of arbitrary: canonical_phase = rotate_clockwise(~glyph)
#       and the recovered input rotation is applied afterward, which keeps the concept inferable
#   hidden_assumptions_review:
#     - g.shape == (6, 6) remains valid and is preserved by the generator
#     - background must still be orange and there must be exactly one non-orange color
#     - bbox(fg) must be exactly 3x3; generator only uses glyphs that touch all four bbox sides,
#       and rotation preserves that property
#     - old corner-based contour assertions no longer hold; they are replaced by exact membership
#       in the precomputed catalog of 16 oriented asymmetric glyphs
#     - center stamping no longer assumes a plain np.tile(motif, (2, 2)); the 6x6 overlay is now
#       assembled quadrant-by-quadrant so each copy has a distinct rotation
#   placement_legality:
#     - motif size is unchanged at 3x3, so legal top-left positions stay in rows/cols 0..3
#     - there is still exactly one object in the input, so overlap constraints are trivial
#     - all rotated glyphs keep a tight 3x3 bbox, so no sampled rotation can shrink or spill
#   visual_observability:
#     - the asymmetric glyph alone makes orientation visible in the input
#     - the lattice orientation changes with the same rotation, so examples can show that rotating
#       the source object rotates the orange phase structure rather than switching to a new color
#     - the maroon center reinforces the rule by displaying a four-quadrant rotation cycle derived
#       from the observed motif instead of a plain repeated block
#   visual_cleanliness:
#     - only black, orange, and maroon appear in the output, preserving the original palette discipline
#     - the lattice and the stamp remain separated by role: the lattice fills the whole board, while
#       the maroon overwrite is confined to the central 6x6 square
#   post-change_main_assumptions:
#     - every accepted motif must belong to exactly one oriented-glyph catalog entry
#     - the observed motif orientation is the anchor for both decoded lattice rotation and stamp rotation
#     - generator coverage must include non-zero rotations often enough that the D1 behavior is not latent


def _has_nontrivial_symmetry(motif: np.ndarray) -> bool:
    """Return True iff motif matches a non-identity rotation or reflection of itself."""
    return any(
        np.array_equal(motif, other)
        for other in (
            np.rot90(motif, 1),
            np.rot90(motif, 2),
            np.rot90(motif, 3),
            np.fliplr(motif),
            np.flipud(motif),
        )
    )


def _touches_all_bbox_sides(motif: np.ndarray) -> bool:
    return bool(motif[0].any() and motif[2].any() and motif[:, 0].any() and motif[:, 2].any())


GLYPHS = {
    "L": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    "hook": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    "step": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    "snake": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
}

for _glyph_name, _glyph in GLYPHS.items():
    assert _glyph.shape == (3, 3)
    assert _touches_all_bbox_sides(_glyph), _glyph_name
    assert not _has_nontrivial_symmetry(_glyph), _glyph_name


BASE_PHASES = {name: np.rot90(~glyph, -1) for name, glyph in GLYPHS.items()}
ORIENTED_GLYPHS: dict[bytes, tuple[str, int]] = {}
for _glyph_name, _glyph in GLYPHS.items():
    for _k_ccw in range(4):
        _oriented = np.rot90(_glyph, _k_ccw)
        _key = _oriented.tobytes()
        assert _key not in ORIENTED_GLYPHS
        ORIENTED_GLYPHS[_key] = (_glyph_name, _k_ccw)

assert len(ORIENTED_GLYPHS) == 16


def _identify_glyph_and_rotation(motif: np.ndarray) -> tuple[str, int] | None:
    """Return (canonical glyph name, CCW rotation count) for an observed 3x3 motif."""
    assert motif.shape == (3, 3)
    return ORIENTED_GLYPHS.get(np.asarray(motif, dtype=bool).tobytes())


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode a rotated asymmetric glyph by canonical matching, then re-apply the same rotation."""
    match = _identify_glyph_and_rotation(motif)
    assert match is not None
    glyph_name, k_ccw = match
    return np.rot90(BASE_PHASES[glyph_name], k_ccw)


def _build_rotating_stamp(motif: np.ndarray) -> np.ndarray:
    """Assemble the central 6x6 stamp from four rotated 3x3 copies of the observed motif."""
    assert motif.shape == (3, 3)
    stamp = np.zeros((6, 6), dtype=bool)
    stamp[:3, :3] = motif
    stamp[:3, 3:] = np.rot90(motif, -1)
    stamp[3:, :3] = np.rot90(motif, 1)
    stamp[3:, 3:] = np.rot90(motif, 2)
    return stamp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a rotated asymmetric 3x3 glyph into an oriented period-3 lattice, then place a rotating maroon stamp at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels occupy one tight 3x3 bounding box.
    - That 3x3 mask is one of four asymmetric glyphs, and it may appear in any 90-degree rotation.

    Rule:
    - Crop the 3x3 motif, identify which canonical glyph it is and how much it has been rotated.
    - Decode the canonical glyph into its standard phase tile, then rotate that phase tile by the same amount before tiling the 16x16 orange/black lattice.
    - In the centered 6x6 region, place four maroon copies of the observed motif arranged as a rotation cycle: 0 degrees, clockwise 90, 180, and counterclockwise 90.

    Key insight:
    - The source object is an orientation-carrying code twice over: its facing direction rotates the repeated lattice, and the center stamp explicitly displays the four-step rotation orbit derived from that same observed orientation.
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
    assert _identify_glyph_and_rotation(motif) is not None

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = _build_rotating_stamp(motif)
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


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = np.rot90(GLYPHS[glyph_name], int(rng.integers(0, 4)))
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 20")
