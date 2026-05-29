# Variant of eee78d87
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 单一流程：提取字形 -> 解码相位瓦片 -> 铺满画布 -> 放置中心印章，无迭代... → 两步铺贴（先粗后细）

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# Design analysis
# ---------------
# spec:
#   input_grid: 6x6
#   output_grid: 16x16
#   background: orange(7)
#   object:
#     count: 1
#     bbox: exactly 3x3
#     color: any single non-{black, orange, maroon}
#   preserved_from_original:
#     - single colored 3x3 motif on orange background
#     - centered 6x6 maroon stamp made by tiling the input motif 2x2
#     - output uses black/orange/maroon only
#
# variant_design:
#   core_change:
#     - The motif is now read in two passes.
#     - The four corner cells encode a coarse 4x4 background tile.
#     - The four edge-center cells encode a fine 2x2 detail tile.
#     - The center cell is always on; it is structural, not a code bit.
#   coarse_pass:
#     - Fill the full 16x16 canvas by tiling the decoded 4x4 orange/black tile.
#   fine_pass:
#     - Tile the decoded 2x2 orange/black tile only over the centered 8x8 region.
#     - Overlay rule is union in orange-space: detail orange turns center-region black
#       cells orange, but never erases existing orange from the coarse pass.
#   final_pass:
#     - As in the original task, tile the 3x3 motif 2x2 and recolor its true cells to
#       maroon in the centered 6x6 region.
#
# concretization_and_trace:
#   example_input_motif:
#     - corners = main diagonal -> coarse tile has orange in the top-left and bottom-right
#       2x2 quadrants of each 4x4 period.
#     - edges = left/right pair -> fine tile is a 2x2 vertical stripe pattern.
#   original_flow:
#     - one decoder -> one global periodic lattice -> center stamp.
#   variant_flow:
#     - decode corners -> large 4x4 repeated background
#     - decode edges -> smaller 2x2 repeated detail inside rows/cols 4..11
#     - union the two layers in that center window
#     - stamp the maroon 6x6 motif last
#   visible_difference:
#     - Outside the center 8x8, only the coarse large-block lattice appears.
#     - Inside the center 8x8, a finer periodic texture is added on top.
#
# complexity_calibration:
#   - Corner codes are limited to four patterns and edge codes are limited to four
#     patterns. This keeps the rule finite and learnable while still giving repeated
#     evidence across samples.
#   - The two code groups occupy disjoint visible positions in the 3x3 motif, so the
#     input exposes which part controls which output layer.
#   - Generator rejects samples where the detail layer adds no orange cells, preventing
#     regression back to a single-layer task.
#
# implicit_assumption_review_of_original_main:
#   - `len(colors_present(...)) == 1` still holds and is kept.
#   - `motif.shape == (3, 3)` still holds and is kept; the generator always builds a
#     tight full-span 3x3 motif.
#   - Original `_decode_phase_tile(motif)` assumed corners/edges/center were one coupled
#     code; that assumption is invalid now, so it is replaced by two independent decoders.
#   - Original `out_shape = 3*input - 2` relied on a period-3 fill. The new design uses a
#     fixed 16x16 canvas because the coarse period is 4 and the fine period is 2.
#   - Original single `tile_fill` is insufficient; the new solver must compose two tiled
#     layers in sequence before stamping maroon.
#   - Original final maroon overwrite remains valid and must stay last so the stamp is not
#     partially hidden by the fine overlay.
#
# placement_legality:
#   - Every allowed corner code already touches top/bottom/left/right borders of the 3x3
#     bbox, so the motif remains exactly 3x3 even when all edge-center bits are off.
#   - The input motif is still placed at any top-left position in a 6x6 grid where a 3x3
#     box fits: rows/cols 0..3.
#   - The detail region is fixed to rows/cols 4..11 (an 8x8 square), and the maroon stamp
#     is fixed to rows/cols 5..10, so all overlays stay centered, non-out-of-bounds, and
#     visually separated.
#
# observability_tidiness_and_color_inference:
#   - No sorting or mapping depends on hidden color ids; colors only distinguish background,
#     motif, and output layers.
#   - The output is visually clean: coarse background uses large 2x2 blocks inside a 4x4
#     period, fine detail uses 2x2 repetition only in the center, and maroon remains the
#     topmost central marker.
#   - Humans can infer the rule from few examples because the outer area isolates the corner
#     code effect, while the center 8x8 isolates the edge-code effect.

OUT_SHAPE = (16, 16)
DETAIL_R0 = 4
DETAIL_C0 = 4
DETAIL_SIZE = 8
GRID_H = 6
GRID_W = 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
MIN_DETAIL_GAIN = 8

_BLOCK2 = np.ones((2, 2), dtype=bool)

# Corner order: TL, TR, BL, BR.
COARSE_QUADRANTS = {
    "diag_main": np.array([[1, 0], [0, 1]], dtype=bool),
    "diag_anti": np.array([[0, 1], [1, 0]], dtype=bool),
    "missing_tl": np.array([[0, 1], [1, 1]], dtype=bool),
    "missing_br": np.array([[1, 1], [1, 0]], dtype=bool),
}
COARSE_TILES = {
    tuple(int(x) for x in quadrants.reshape(-1)): np.kron(quadrants, _BLOCK2).astype(bool)
    for quadrants in COARSE_QUADRANTS.values()
}
CORNER_CODES = tuple(COARSE_TILES.keys())

# Edge order: top, left, right, bottom.
DETAIL_TILES = {
    (0, 0, 0, 0): np.array([[1, 0], [0, 0]], dtype=bool),
    (0, 1, 1, 0): np.array([[1, 0], [1, 0]], dtype=bool),
    (1, 0, 0, 1): np.array([[1, 1], [0, 0]], dtype=bool),
    (1, 1, 1, 1): np.array([[1, 0], [0, 1]], dtype=bool),
}
EDGE_CODES = tuple(DETAIL_TILES.keys())


def _decode_coarse_tile(motif: np.ndarray) -> np.ndarray:
    """Read the motif corners as a coarse 4x4 tile code."""
    assert motif.shape == (3, 3)
    corners = tuple(int(x) for x in [motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]])
    assert corners in COARSE_TILES
    return COARSE_TILES[corners]


def _decode_detail_tile(motif: np.ndarray) -> np.ndarray:
    """Read the motif edge centers as a fine 2x2 tile code."""
    assert motif.shape == (3, 3)
    edges = tuple(int(x) for x in [motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]])
    assert edges in DETAIL_TILES
    return DETAIL_TILES[edges]


def _compose_tiled_layers(motif: np.ndarray) -> tuple[np.ndarray, int]:
    coarse = _decode_coarse_tile(motif)
    detail = _decode_detail_tile(motif)

    base = tile_fill(OUT_SHAPE, coarse).astype(bool)
    detail_mask = tile_fill((DETAIL_SIZE, DETAIL_SIZE), detail).astype(bool)

    region = base[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE].copy()
    added_mask = detail_mask & ~region
    region |= detail_mask
    base[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE] = region

    out = np.where(base, Color.ORANGE, Color.BLACK).astype(int)
    return out, int(added_mask.sum())


def _stamp_center(out: np.ndarray, motif: np.ndarray) -> np.ndarray:
    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    stamped = out.copy()
    stamped[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return stamped


def _detail_gain(motif: np.ndarray) -> int:
    _, gain = _compose_tiled_layers(motif)
    return gain


def _assemble_motif(corners: tuple[int, int, int, int], edges: tuple[int, int, int, int]) -> np.ndarray:
    motif = np.zeros((3, 3), dtype=bool)
    motif[1, 1] = True

    motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2] = [bool(x) for x in corners]
    motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1] = [bool(x) for x in edges]

    assert bbox(motif) == (0, 0, 2, 2)
    return motif


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one 3x3 glyph into a two-stage tiled scene: a coarse background lattice, then a finer central overlay, then a maroon center stamp.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange cells form a tight 3x3 motif.
    - The motif uses its four corners as one code and its four edge-centers as a second code; the center cell is always present.

    Rule:
    - Read the corner pattern and convert it into a 4x4 orange/black coarse tile, repeated across the whole 16x16 output.
    - Read the edge-center pattern and convert it into a 2x2 orange/black detail tile, repeated only in the centered 8x8 region; detail orange is overlaid by union, so it adds fine orange pixels without deleting coarse orange.
    - Tile the original 3x3 motif 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

    Key insight:
    - The single input glyph is a dual code. Its corners control the large-scale background periodicity, while its edge-centers control a second finer periodic layer that appears only near the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)
    assert bool(motif[1, 1])

    out, _ = _compose_tiled_layers(motif)
    return _stamp_center(out, motif)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    corners = CORNER_CODES[int(rng.integers(0, len(CORNER_CODES)))]
    edges = EDGE_CODES[int(rng.integers(0, len(EDGE_CODES)))]
    motif = _assemble_motif(corners, edges)

    if _detail_gain(motif) < MIN_DETAIL_GAIN:
        return None

    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    detail_window = out[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE]
    if np.count_nonzero(detail_window == Color.ORANGE) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D13 variant")
