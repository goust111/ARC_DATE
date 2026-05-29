# Variant of eee78d87
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 单一流程：提取字形 -> 解码相位瓦片 -> 铺满画布 -> 放置中心印章，无迭代
# ... → 先铺贴后根据规则擦除部分区域

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis (pre-implementation, condensed YAML-style notes)
# variant_goal:
#   keep:
#     - 6x6 orange input with one 3x3 contour glyph in one non-orange color
#     - original phase decoding for the 16x16 orange/black lattice
#     - centered 6x6 maroon stamp made by 2x2 tiling the 3x3 glyph
#   change:
#     - add an explicit second stage after full tiling:
#       build the full lattice first, then erase everything outside a glyph-driven window
# concrete_behavior:
#   - The output is still a 16x16 grid, but the orange lattice no longer fills the full rectangle.
#   - Two window families are used so the erase stage is visible:
#     - cross window: keep rows 5..10 or cols 5..10
#     - diamond window: keep cells with Manhattan distance <= 8.5 from the canvas center
#   - The branch is chosen from a visible glyph property, not a color id:
#     - plus / H have side pixels in the middle row (motif[1,0] == motif[1,2] == 1) -> cross
#     - X / 4corners do not -> diamond
# step_trace:
#   original:
#     - extract motif -> decode phase tile -> tile full 16x16 lattice -> stamp center
#   variant:
#     - extract motif -> decode phase tile -> tile full 16x16 lattice
#       -> choose window from motif structure -> erase outside window to black
#       -> stamp the centered maroon 2x2 motif
#   difference:
#     - the new output depends on two motif signals:
#       one for phase decoding, one for aperture selection
# complexity_check:
#   - only two erase branches are used, so samples remain inferable from 2-4 examples
#   - both branches have clean, large silhouettes; they are easy to distinguish by eye
#   - the branch split is balanced across the four original glyphs (2 cross, 2 diamond)
# implicit_assumptions_review:
#   - bbox(fg) must still be exactly 3x3; generator preserves that.
#   - _decode_phase_tile still expects one of the original four contour glyphs; generator preserves that.
#   - the erase stage assumes the centered 6x6 stamp should remain legible, so both windows fully contain it.
#   - stamping must happen after erasing; otherwise the erase step would clip maroon pixels and blur the two-stage logic.
# placement_legality:
#   - input placement is unchanged from the original: one 3x3 glyph at any top-left in 0..3, 0..3.
#   - no extra objects are added, so no overlap risks are introduced.
# visual_observability:
#   - cross vs diamond is derived from visible shape geometry, not hidden metadata.
#   - the output keeps only black/orange/maroon, so the new effect is silhouette-based and clean.
# cleanliness:
#   - only two layers remain in the lattice region (orange on black), plus the central maroon stamp.
#   - no competing markers are placed near the border, so the aperture shape reads clearly.


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


def _window_kind(motif: np.ndarray) -> str:
    """Orthogonal side arms (plus/H) keep a cross; diagonal/corner-only glyphs keep a diamond."""
    assert motif.shape == (3, 3)
    return "cross" if bool(motif[1, 0]) else "diamond"


def _window_mask(shape: tuple[int, int], kind: str) -> np.ndarray:
    """Return the keep-mask for the erase stage."""
    h, w = shape
    rr, cc = np.indices((h, w))
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0

    if kind == "cross":
        return (np.abs(rr - center_r) <= 2.5) | (np.abs(cc - center_c) <= 2.5)
    if kind == "diamond":
        return (np.abs(rr - center_r) + np.abs(cc - center_c)) <= 8.5
    raise ValueError(f"unknown window kind: {kind}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a periodic lattice, then cut that lattice back to a glyph-selected window before stamping the center.

    Input structure:
    - The grid is 6x6 with orange background and one non-orange color.
    - The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the period-3 orange phase tile.
    - Fill a black 16x16 canvas by repeating that orange phase tile everywhere.
    - Classify the glyph: plus/H keep a cross-shaped window, X/four-corners keep a diamond-shaped window.
    - Erase the tiled lattice outside that window, then tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The same small glyph now controls two stages: it still decodes the lattice phase, and its visible arm structure also decides which large aperture survives the erase pass.
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

    tiled = tile_fill(out_shape, phase_colors)
    keep = _window_mask(out_shape, _window_kind(motif))
    out = np.where(keep, tiled, Color.BLACK).astype(int)

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


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
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
    raise RuntimeError("Failed to generate input for eee78d87 D13 variant")
