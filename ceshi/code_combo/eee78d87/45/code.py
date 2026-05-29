# Combo variant of eee78d87 (index 45)
# combo_spec: D3/1+D11/1
# Name: 多印章分布+单次不平铺小印章

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, paste, tile_fill


# Design analysis for combo D3/1 + D11/1
# ---------------------------------------
# fused_rule:
#   preserved:
#     - input stays a 6x6 orange field containing exactly one non-orange 3x3 contour glyph
#     - the glyph family and the phase-decoding logic stay unchanged
#     - output is still a 16x16 orange/black lattice derived from the decoded 3x3 phase tile
#   changed:
#     - D3/1 removes the old "single centered stamp" assumption and requires several regularly spaced stamp sites
#     - D11/1 removes the old 2x2 tiling of the motif; each stamp must remain one original-size 3x3 glyph
#     - combining them yields four 3x3 maroon glyph copies at quadrant-like slots:
#       (1,1), (1,12), (12,1), (12,12)
# behavior_trace:
#   example_glyph: plus = [[0,1,0],[1,1,1],[0,1,0]]
#   source_task:
#     - decode plus -> phase tile with orange in the lower-right 2x2 corner
#     - repeat that 3x3 phase tile across 16x16
#     - enlarge the maroon event by tiling the glyph 2x2 into one centered 6x6 block
#   fused_task:
#     - decode plus in exactly the same way
#     - repeat the same phase lattice across the full 16x16 canvas
#     - place the original 3x3 plus once per slot at four evenly spaced corner-side sites
#   visible_difference:
#     - maroon no longer forms one large central 6x6 object
#     - instead the output shows four small, clearly separated 3x3 marks over the same lattice
# complexity_calibration:
#   - four slots are enough to make "multiple stamp regions" obvious in 2-3 examples
#   - keeping the stamp at native 3x3 size makes the D11 change visible without adding extra scaling cases
#   - the rule still has low branching: one fixed slot lattice, one fixed stamp size, same four glyph choices
# implicit_assumption_review:
#   - get_bg / colors_present still assume orange dominates and there is exactly one foreground color; generator preserves that
#   - bbox/crop_bbox still assume the glyph occupies a tight 3x3 box; generator preserves that
#   - _decode_phase_tile still assumes the glyph is one of the four contour codes; generator samples only those
#   - the original center-stamp code assumed a single 6x6 placement computed from output centering; that assumption is removed
#   - the fused placement code now assumes:
#       1. output size is 16x16
#       2. stamp size is exactly 3x3
#       3. two slot origins per axis fit with a 1-cell border
#     these hold because output shape is still 3*6-2 and motif size is still 3
# placement_legality:
#   - origins [1,12] on each axis give occupied spans [1,3] and [12,14]
#   - every 3x3 stamp stays in bounds, stamps never overlap, and row/col 0 and 15 remain a clean border
# visual_observability:
#   - the four-slot arrangement is human-visible without reading color ids or hidden metadata
#   - the large empty center strongly distinguishes this combo from the original centered 6x6 stamp behavior
# visual_cleanliness:
#   - maroon overlays only the orange/black lattice, never another maroon stamp
#   - the 8-cell central gutter keeps the four marks visually separate from one another
# fusion_metric_plan:
#   - maroon pixel count is not enough because four 3x3 native stamps can match the original 2x2-tiled stamp count
#   - a useful anti-degeneration metric is the maroon union bbox:
#       source task -> always 6x6 centered bbox
#       fused task  -> always 14x14 corner-spanning bbox
#   - later validation should check that generated samples consistently produce the fused bbox, proving the combo behavior occurs


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


def _stamp_origins(axis_len: int, stamp_len: int) -> list[int]:
    """Return the two evenly spaced native-size stamp origins on one axis."""
    axis_len = int(axis_len)
    stamp_len = int(stamp_len)
    outer_margin = 1
    step = axis_len - stamp_len - 2 * outer_margin
    assert step > stamp_len
    origins = [outer_margin, outer_margin + step]
    assert origins[1] + stamp_len == axis_len - outer_margin
    return origins


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a 16x16 periodic lattice with several small maroon glyph stamps.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange pixels occupy a tight 3x3 bounding box.
    - That 3x3 object is one of four contour glyphs: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange phase tile as in the base task.
    - Fill a black 16x16 canvas by repeating that orange/black phase tile.
    - Recolor the original 3x3 glyph to maroon and place one native-size copy at each site of a regular 2x2 slot lattice.

    Key insight:
    - The input glyph still serves as a phase code for the full orange lattice, but its output copy behavior has changed: it is no longer enlarged into one center block and is instead distributed as several equally spaced 3x3 maroon marks.
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

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    stamp = np.where(motif, Color.MAROON, Color.BLACK).astype(int)
    row_origins = _stamp_origins(out.shape[0], stamp.shape[0])
    col_origins = _stamp_origins(out.shape[1], stamp.shape[1])
    for r0 in row_origins:
        for c0 in col_origins:
            paste(out, stamp, r0, c0, transparent=Color.BLACK)

    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
        maroon_mask = out == Color.MAROON
        maroon_box = bbox(maroon_mask)
        assert maroon_box == (1, 1, 14, 14)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo D3/1+D11/1")
