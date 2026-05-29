# Combo variant of eee78d87 (index 10)
# combo_spec: D1/1+D2/2
# Name: 旋转解码+非对称字形

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis:
# variant:
#   preserved_structure:
#     - input stays a 6x6 orange field containing exactly one colored 3x3 motif
#     - output stays a 16x16 black/orange period-3 lattice with a centered 6x6 maroon stamp
#     - the stamp still uses a 2x2 tiling of the observed 3x3 motif
#   fusion_rule:
#     - D2/2 replaces the old symmetric plus/H/X/4corners family with asymmetric 3x3 glyphs
#     - D1/1 adds a second control variable: the glyph may appear in any 90-degree rotation
#     - decoding is therefore two-stage: identify canonical glyph class + recover its rotation,
#       then fetch that class's canonical phase tile and rotate the phase tile by the same amount
#   concrete_trace:
#     - original task: an input contour like plus was already orientation-invariant, so there was
#       no meaningful notion of "which way it faces"; the solver only needed contour topology
#     - combo task example: observed motif = rot90(L, 3) = 111/100/100
#       1. match it against the catalog and recover (name='L', turns_ccw=3)
#       2. take canonical L phase 110/100/001
#       3. rotate that phase tile by the same 3 quarter-turns to get 001/100/110
#       4. tile the rotated phase across 16x16, then stamp the observed 111/100/100 motif in maroon
#     - difference: the lattice now depends jointly on glyph identity and on the recovered rotation;
#       the same glyph class in a different orientation produces a rotated lattice
#   complexity_calibration:
#     - keep only four canonical glyph classes and four rotations each, for 16 oriented states total
#     - all glyphs and all canonical phase tiles are asymmetric under non-identity rotations, so the
#       orientation signal is visible instead of collapsing to fewer cases
#     - the orbits of the 16 oriented glyph states and the 16 oriented phase states are asserted to
#       be unique, avoiding accidental collisions that would make examples ambiguous
#   hidden_assumptions_review:
#     - g.shape == (6, 6): unchanged and guaranteed by the generator
#     - background is orange and only one non-orange input color exists: unchanged and enforced
#     - bbox(fg) must be exactly 3x3: preserved by using full-support glyphs that touch every side
#     - old corner-pattern assertions are invalid after replacing the glyph family; they are replaced
#       by exact catalog matching over rotated asymmetric glyphs
#     - rotation now matters, so solver correctness depends on canonicalizing orientation before
#       choosing a phase tile; the generator guarantees every motif belongs to exactly one orbit state
#   placement_legality:
#     - motif size is still 3x3, so legal top-left positions remain rows/cols 0..3
#     - there is still exactly one object, so no overlap constraints beyond in-bounds placement exist
#     - the centered 6x6 maroon stamp is unchanged and safely overwrites the tiled lattice last
#   visual_observability:
#     - a human sees the oriented glyph twice: once in the input and again as the 2x2 maroon stamp
#     - the surrounding orange/black lattice rotates with the same facing direction, making the
#       orientation effect legible across only a few examples
#     - because the phase catalog is explicit and class-dependent, different glyph classes can share
#       the same spatial role while still producing clearly different repeating textures
#   visual_cleanliness:
#     - only three colors appear in the output: black, orange, maroon
#     - there is one repeated texture layer plus one centered stamp layer, so the scene stays readable
#   color_inference:
#     - the input foreground color still only marks the motif location; decode never depends on color id
#   fusion_metric:
#     - the generator should cover multiple non-zero rotations; with uniform sampling, about 75% of
#       samples should have turns_ccw != 0, which confirms the orientation variable is actually active
#     - over a moderate sample, the number of distinct (glyph, rotation) states and distinct phase
#       tiles should both exceed the four-class baseline, otherwise the combo would be visually degenerate
#   post-change_main_assumptions:
#     - every accepted motif must match exactly one rotated catalog glyph
#     - every matched glyph class must have exactly one canonical phase tile
#     - rotating that phase tile by the recovered orientation must still produce a 3x3 tile


def _has_nontrivial_symmetry(motif: np.ndarray) -> bool:
    """Return True if a 3x3 pattern equals any non-identity rotation or flip."""
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


GLYPHS = {
    "L": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    "hook": np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    "step": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    "snake": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
}

PHASE_TILES = {
    "L": np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
    "hook": np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=bool),
    "step": np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=bool),
    "snake": np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
}


def _assert_catalog_integrity() -> None:
    glyph_states: set[tuple[int, ...]] = set()
    phase_states: set[tuple[int, ...]] = set()

    assert GLYPHS.keys() == PHASE_TILES.keys()
    for name, glyph in GLYPHS.items():
        phase = PHASE_TILES[name]
        assert glyph.shape == (3, 3)
        assert phase.shape == (3, 3)
        assert bool(glyph[0].any() and glyph[2].any() and glyph[:, 0].any() and glyph[:, 2].any())
        assert not _has_nontrivial_symmetry(glyph), name
        assert not _has_nontrivial_symmetry(phase), f"{name}_phase"

        for turns_ccw in range(4):
            glyph_state = tuple(np.rot90(glyph, turns_ccw).astype(int).ravel())
            phase_state = tuple(np.rot90(phase, turns_ccw).astype(int).ravel())
            assert glyph_state not in glyph_states, (name, turns_ccw, "glyph collision")
            assert phase_state not in phase_states, (name, turns_ccw, "phase collision")
            glyph_states.add(glyph_state)
            phase_states.add(phase_state)


_assert_catalog_integrity()


def _match_glyph(motif: np.ndarray) -> tuple[str, int] | None:
    """Return (canonical_name, turns_ccw) if the motif matches a rotated catalog glyph."""
    assert motif.shape == (3, 3)
    match: tuple[str, int] | None = None
    for name, glyph in GLYPHS.items():
        for turns_ccw in range(4):
            if np.array_equal(motif, np.rot90(glyph, turns_ccw)):
                assert match is None
                match = (name, turns_ccw)
    return match


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode an oriented asymmetric glyph by matching its class and rotating the class phase."""
    match = _match_glyph(motif)
    assert match is not None
    name, turns_ccw = match
    return np.rot90(PHASE_TILES[name], turns_ccw)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode an oriented asymmetric 3x3 glyph into a rotated period-3 lattice, then stamp the same glyph at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels form one tight 3x3 bounding box.
    - That 3x3 mask is one of four asymmetric glyph classes, but it may appear in any 0/90/180/270-degree rotation.

    Rule:
    - Crop the 3x3 motif and identify both its canonical glyph class and its rotation.
    - Look up the canonical 3x3 orange phase tile for that class, then rotate the phase tile by the same amount as the observed glyph.
    - Repeat the rotated phase tile over the 16x16 black canvas, then tile the observed motif 2x2 and recolor its true cells to maroon in the centered 6x6 region.

    Key insight:
    - The source object is an orientation-coded symbol rather than a plain shape: the solver must recover which asymmetric glyph it is and which way it faces, because both factors jointly determine the repeated lattice.
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
    assert _match_glyph(motif) is not None

    phase = _decode_phase_tile(motif)
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


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    name = str(rng.choice(list(GLYPHS.keys())))
    turns_ccw = int(rng.integers(0, 4))
    motif = np.rot90(GLYPHS[name], turns_ccw)
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
