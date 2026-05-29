# Combo variant of eee78d87 (index 34)
# combo_spec: D7/1+D16/0
# Name: 角落标记定位+旋转指令编码
from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   The base task had one 3x3 contour glyph somewhere inside a 6x6 orange grid. Its exact
#   bitmap directly selected one of four 3x3 orange/black phase tiles, and the solver could
#   find the glyph by taking the bbox of all non-orange cells.
#   This combo changes two different stages at once:
#   1) source localization now comes from a detached corner marker pixel;
#   2) symbol decoding is split into "family" and "orientation":
#      - glyph family chooses which phase tile to use;
#      - glyph orientation chooses how much to rotate the finished 16x16 output.
#   Concretely, suppose the encoded glyph top-left is (0, 3). The marker then sits in the
#   opposite corner block at (4, 1), so bbox(all non-orange) is no longer a tight 3x3 crop.
#   If the observed 3x3 glyph is a 90-degree rotation of the canonical "X-family" directed
#   glyph, the solver must decode:
#      marker (4, 1) -> read motif at (0, 3)
#      rotated glyph -> family = X-phase, rotation = 90 degrees CCW
#      build canonical X output first, then rotate the whole 16x16 canvas by the same amount
#   The difference from the base task is therefore not just where the motif sits, but also
#   that the motif's facing direction becomes an instruction rather than mere appearance.
# complexity_calibration:
#   The task still has only 16 legal motif positions and 4 phase families, so the search space
#   remains small. The extra rotation parameter adds only four orientations, but it is visible:
#   the maroon center stamp and the orange/black lattice both rotate together.
#   Each family uses a 3x3 glyph whose four quarter-turns are visually distinct and whose
#   rotation orbit does not collide with the other families, avoiding ambiguous decoding.
# implicit_assumption_audit:
#   The original solver assumed exactly one non-orange color; the combo needs two because the
#   singleton marker color must differ from the glyph color.
#   The original bbox-based crop no longer works because the detached marker expands the bbox.
#   The original phase decoder assumed the glyph bitmap itself was the full phase code. That is
#   no longer valid: now the solver must first normalize the observed glyph back to its unique
#   canonical family, then separately reuse the recovered rotation as a transform on the output.
#   The center stamp also changes subtly: stamping the observed rotated glyph and then rotating
#   again would double-apply the instruction. The correct logic is to stamp the canonical glyph
#   and rotate the full canvas once at the end.
# placement_legality:
#   The motif top-left still ranges over rows/cols 0..3, so a 3x3 crop always fits in 6x6.
#   The corner marker uses the same opposite-corner 2x2 scheme as D7/1, which guarantees the
#   marker is disjoint from the motif for every legal motif position.
# visual_observability:
#   Across examples, humans can see two independent cues:
#   - the detached singleton in a corner block tells where to look;
#   - the motif's facing direction matches the final output rotation.
#   No hidden color ordering is used. The marker meaning depends only on visible geometry, and
#   the rotation code depends only on visible orientation of the 3x3 glyph.
# visual_cleanliness:
#   The input still contains only two objects: one 3x3 glyph and one singleton marker.
#   The output still uses the original three-color palette (black, orange, maroon), so the new
#   rule adds structure without introducing clutter or extra overlapping overlays.
# change_inferability:
#   Rotation is only inferred from clearly asymmetric glyphs. This avoids the impossible case
#   where a symmetric plus or X shape would fail to reveal a unique angle. The generator also
#   samples non-zero rotations frequently enough that the rotation channel is actually present
#   in the produced distribution rather than degenerating into the D7-only variant.
# post_edit_assumptions:
#   main() now assumes exactly one singleton non-orange color exists and that the 3x3 window
#   decoded from the marker contains only background plus one glyph color.
#   It also assumes the observed glyph matches exactly one rotated canonical family. The
#   generator enforces these conditions by construction: distinct marker/glyph colors, opposite
#   corner placement, and glyph sampling only from the predefined asymmetric family library.


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CTRL_BLOCK = 2
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
OUTPUT_SHAPE = (3 * GRID_H - 2, 3 * GRID_W - 2)

# Each family owns one canonical asymmetric 3x3 glyph.
# Rotating the glyph changes only the output rotation instruction, not the phase family.
CANONICAL_GLYPHS = {
    "plus": np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool),
    "H": np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    "X": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    "4corners": np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
}

# Canonical family -> same four phase tiles as the original puzzle.
PHASE_TILES = {
    "plus": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    "H": np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "X": np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    "4corners": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
}


def _decode_marker_position(marker: tuple[int, int]) -> tuple[int, int]:
    """Map the detached corner marker back to the glyph top-left position."""
    r, c = map(int, marker)

    if 0 <= r < CTRL_BLOCK and 0 <= c < CTRL_BLOCK:
        return r + 2, c + 2
    if 0 <= r < CTRL_BLOCK and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r + 2, c - 4
    if GRID_H - CTRL_BLOCK <= r < GRID_H and 0 <= c < CTRL_BLOCK:
        return r - 4, c + 2
    if GRID_H - CTRL_BLOCK <= r < GRID_H and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r - 4, c - 4
    raise AssertionError("control pixel must lie in a 2x2 corner block")


def _encode_marker_position(r0: int, c0: int) -> tuple[int, int]:
    """Inverse of `_decode_marker_position` used by the generator."""
    r0 = int(r0)
    c0 = int(c0)

    if r0 < 2 and c0 < 2:
        return r0 + 4, c0 + 4
    if r0 < 2 and c0 >= 2:
        return r0 + 4, c0 - 2
    if r0 >= 2 and c0 < 2:
        return r0 - 2, c0 + 4
    return r0 - 2, c0 - 2


def _decode_glyph_family_and_rotation(motif: np.ndarray) -> tuple[str, int]:
    """Return (family_name, rot_k) for a uniquely matched rotated canonical glyph."""
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    matches: list[tuple[str, int]] = []
    for family, canonical in CANONICAL_GLYPHS.items():
        for rot_k in range(4):
            if np.array_equal(motif, np.rot90(canonical, rot_k)):
                matches.append((family, int(rot_k)))

    assert len(matches) == 1, "observed glyph must match exactly one rotated canonical family"
    return matches[0]


def _build_canonical_output(family: str) -> np.ndarray:
    """Construct the unrotated output for a canonical glyph family."""
    phase = PHASE_TILES[family]
    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    out = tile_fill(OUTPUT_SHAPE, phase_colors)

    center = np.tile(CANONICAL_GLYPHS[family], (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _sample_rotation(rng: np.random.Generator) -> int:
    """Prefer visible non-zero rotations so the fusion signal appears often in samples."""
    if float(rng.random()) < 0.2:
        return 0
    return int(rng.integers(1, 4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a corner-located directed glyph into a periodic lattice, then rotate the finished canvas by the glyph's facing direction.

    Input structure:
    - The grid is 6x6 with orange background, one multi-pixel glyph color, and one singleton control color.
    - The singleton control pixel sits in a 2x2 corner block and encodes the top-left position of the real 3x3 glyph.
    - The glyph itself is one of four asymmetric family templates seen in some quarter-turn orientation.

    Rule:
    - Find the singleton corner marker and decode it into the glyph's 3x3 source window.
    - Match that 3x3 bitmap against the rotated canonical glyph library to recover both the glyph family and its rotation.
    - Build the same 16x16 orange/black lattice and centered 2x2 maroon stamp as the canonical family would produce, then rotate the entire finished canvas by the recovered quarter-turn.

    Key insight:
    - The two cues do different jobs: the detached corner pixel tells where the glyph is, while the glyph's own orientation tells how much to rotate the final output. The observed motif must therefore be normalized before decoding the family.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    nonbg_colors = colors_present(g, exclude={bg})
    assert len(nonbg_colors) == 2

    ctrl_candidates = [int(col) for col in nonbg_colors if int(np.count_nonzero(g == int(col))) == 1]
    assert len(ctrl_candidates) == 1
    ctrl = int(ctrl_candidates[0])
    fg = int(next(col for col in nonbg_colors if int(col) != ctrl))

    marker_cells = np.argwhere(g == ctrl)
    assert marker_cells.shape == (1, 2)
    marker = tuple(map(int, marker_cells[0]))

    r0, c0 = _decode_marker_position(marker)
    motif_patch = g[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    assert motif_patch.shape == (MOTIF_SIZE, MOTIF_SIZE)
    assert not bool(np.any(motif_patch == ctrl))
    assert set(map(int, np.unique(motif_patch))) <= {bg, fg}

    motif = motif_patch == fg
    family, rot_k = _decode_glyph_family_and_rotation(motif)
    out = _build_canonical_output(family)
    return np.rot90(out, int(rot_k))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    family = str(rng.choice(list(CANONICAL_GLYPHS.keys())))
    rot_k = _sample_rotation(rng)
    motif = np.rot90(CANONICAL_GLYPHS[family], int(rot_k))

    fg = int(rng.choice(FG_COLORS))
    ctrl_choices = [c for c in FG_COLORS if int(c) != fg]
    ctrl = int(rng.choice(ctrl_choices))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))
    mr, mc = _encode_marker_position(r0, c0)

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg
    assert inp[mr, mc] == Color.ORANGE
    inp[mr, mc] = ctrl

    try:
        solved = main(inp)
    except Exception:
        return None

    assert solved.shape == OUTPUT_SHAPE
    if rot_k != 0:
        assert not np.array_equal(solved, _build_canonical_output(family))
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 34")
