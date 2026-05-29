# Combo variant of eee78d87 (index 9)
# combo_spec: D2/2+D4/2
# Name: 非对称字形+镜像翻转印章

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis:
# variant:
#   preserved_structure:
#     - input stays a 6x6 orange field with one colored 3x3 motif in a tight bbox
#     - output stays a 16x16 orange/black period-3 lattice with a centered 6x6 maroon overlay
#     - the overlay still derives from the same 3x3 motif, so source and center remain linked
#   fusion_rule:
#     - D2/2 replaces the old symmetric contour family with four fixed asymmetric glyphs
#     - D4/2 replaces the old plain 2x2 repetition with a mirrored 2x2 assembly:
#       top-left = motif
#       top-right = horizontal flip
#       bottom-left = vertical flip
#       bottom-right = 180-degree rotation
#     - the lattice decode remains a deterministic oriented transform of the exact 3x3 mask:
#       phase_tile = rotate_clockwise(invert(motif))
#   concrete_trace:
#     - example motif L = 100/100/111
#     - original task would reject it because the corner pattern is neither all-zero nor all-one
#     - fused task accepts it, decodes phase as rot90(~L, -1) = 000/011/011, and tiles that
#       asymmetric phase across the 16x16 canvas
#     - original center stamp for a 2x2 repeat would be:
#       100100
#       100100
#       111111
#       100100
#       100100
#       111111
#     - fused center stamp becomes:
#       100001
#       100001
#       111111
#       111111
#       100001
#       100001
#     - difference: the lattice is already orientation-sensitive because the phase is asymmetric,
#       and the center now exposes horizontal, vertical, and 180-degree transforms of the motif
#   complexity_calibration:
#     - the glyph catalog is kept at four entries, matching the original task's low branch count
#     - each glyph has full 3x3 bbox support and no non-identity rotation/reflection symmetry
#     - because of that asymmetry, the mirror assembly is visually nontrivial instead of collapsing
#       to the old repeated stamp; this avoids degeneration while keeping the hypothesis space small
#   hidden_assumptions_review:
#     - g.shape == (6, 6): preserved by the generator
#     - orange is still the unique background and there is exactly one non-orange color: preserved
#     - bbox(fg) must exist and crop to 3x3: preserved by placing exactly one 3x3 glyph
#     - old corner-based decode assumptions are invalid for asymmetric motifs and are replaced by
#       exact catalog matching before decode
#     - center placement assumes a 6x6 stamp fits exactly in the middle of the 16x16 output;
#       preserved because each quadrant is still one 3x3 copy of the motif
#     - mirror logic assumes the motif is asymmetric enough that flips are observable; enforced by
#       rejecting any glyph with rotational or reflection symmetry
#   placement_legality:
#     - motif size remains 3x3, so valid top-left positions stay in rows/cols 0..3
#     - there is still exactly one placed object in the input, so overlap cannot occur
#     - the 6x6 center stamp overwrites the lattice only after the lattice is complete, so no
#       partial interactions need to be resolved
#   visual_observability:
#     - across a few examples, the repeated lattice differs with the chosen asymmetric glyph,
#       making the decode visibly orientation-sensitive
#     - the central 6x6 region now shows four related but different subcopies, so a viewer can
#       infer that the source motif is mirrored rather than merely repeated
#     - the rule uses geometry only; no branch depends on numeric color ids or hidden ordering
#   visual_cleanliness:
#     - the output still uses only black, orange, and maroon
#     - the mirrored stamp occupies one compact centered region and cleanly overwrites the lattice,
#       so the two layers remain easy to separate by color and position
#   fusion_metric_plan:
#     - to detect degeneration after implementation, measure the Hamming distance between the
#       mirrored 6x6 stamp and the old plain np.tile(motif, (2, 2)) stamp
#     - every valid asymmetric glyph should give a strictly positive distance, proving that D4/2 is
#       active on generated samples instead of silently reverting to the old center behavior
#   post-change_main_assumptions:
#     - every accepted motif must match one catalog entry exactly
#     - every catalog entry must be asymmetric so the mirror assembly remains informative
#     - generate_input must sample the full catalog often enough that the asymmetric decode does not
#       collapse into a single repeated example family


def _has_nontrivial_symmetry(motif: np.ndarray) -> bool:
    """Return True when a 3x3 mask matches a non-identity rotation or reflection."""
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


def _make_mirror_stamp(motif: np.ndarray) -> np.ndarray:
    """Build the 6x6 mirrored 2x2 stamp from one asymmetric 3x3 motif."""
    top = np.hstack((motif, np.fliplr(motif)))
    bottom = np.hstack((np.flipud(motif), np.rot90(motif, 2)))
    return np.vstack((top, bottom))


for _glyph_name, _glyph in GLYPHS.items():
    assert _glyph.shape == (3, 3)
    assert bool(_glyph[0].any() and _glyph[2].any() and _glyph[:, 0].any() and _glyph[:, 2].any())
    assert not _has_nontrivial_symmetry(_glyph), _glyph_name
    assert not np.array_equal(_make_mirror_stamp(_glyph), np.tile(_glyph, (2, 2))), _glyph_name


def _motif_name(motif: np.ndarray) -> str | None:
    for name, glyph in GLYPHS.items():
        if np.array_equal(motif, glyph):
            return name
    return None


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode an allowed asymmetric 3x3 glyph into its period-3 orange phase mask."""
    assert motif.shape == (3, 3)
    assert _motif_name(motif) is not None
    return np.rot90(~motif, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the colored 3x3 asymmetric glyph as an oriented phase code, then render a mirrored maroon stamp at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange cells form one tight 3x3 bounding box.
    - That 3x3 mask is one of four fixed asymmetric glyphs, so flips and rotations change its appearance.

    Rule:
    - Crop the 3x3 glyph and verify it belongs to the asymmetric catalog.
    - Convert it into the orange/black period-3 phase tile by inverting the mask and rotating the result 90 degrees clockwise.
    - Tile that asymmetric phase across the 16x16 canvas.
    - In the centered 6x6 region, place four transformed maroon copies of the glyph: original, horizontal flip, vertical flip, and 180-degree rotation.

    Key insight:
    - The same asymmetric 3x3 source controls both layers: it decodes the lattice orientation, and its lack of symmetry makes the mirrored 2x2 center assembly visibly different from a plain repetition.
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
    assert _motif_name(motif) is not None

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = _make_mirror_stamp(motif)
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
    motif_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
        center = _make_mirror_stamp(motif)
        plain = np.tile(motif, (2, 2))
        cr0 = (out.shape[0] - center.shape[0]) // 2
        cc0 = (out.shape[1] - center.shape[1]) // 2
        observed = out[cr0 : cr0 + 6, cc0 : cc0 + 6] == Color.MAROON
        assert np.array_equal(observed, center)
        assert int(np.count_nonzero(center != plain)) > 0
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 9")
