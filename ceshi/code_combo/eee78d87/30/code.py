# Combo variant of eee78d87 (index 30)
# combo_spec: D2/1+D13/0
# Name: 双色字形+先粗后细双层铺贴
#
# Design analysis for the combo variant
# a) Concretization and behavior trace
#    - Input remains a 6x6 orange grid with one tight 3x3 active patch, but the patch now uses exactly two
#      non-orange colors instead of one.
#    - The corner-color layer is the coarse code: it occupies corners only, and its 4-bit corner pattern decodes
#      to one 4x4 orange/black tile that repeats across the entire 16x16 output.
#    - The center/edge-color layer is the fine code: it always includes the center cell and may include edge-centers;
#      its 4-bit edge pattern decodes to one 2x2 orange/black tile that repeats only inside the centered 8x8 window.
#    - Concrete trace: if the corner color marks TL+BR, the coarse background becomes a large 4x4 diagonal checker of
#      2x2 orange blocks. If the second color marks center+left+right edges, the fine code becomes a vertical 2x2
#      stripe tile, repeated only in rows/cols 4..11 and union-overlaid on top of the coarse background.
#    - The original task had one boolean glyph and one lattice. This combo first splits colors into two structural
#      roles, then performs two different decodes, then composes the layers in order before stamping the center.
# b) Complexity calibration
#    - The coarse code has four discrete corner patterns and the fine code has four discrete edge patterns. This keeps
#      the hypothesis class small enough to learn while still producing visibly different outputs.
#    - The two code layers are disjoint and spatially meaningful: corners drive the large outer texture, while the
#      center+edges drive only the finer central texture. This makes the fusion visible from a few examples.
#    - The generator rejects cases where the fine layer adds too little or adds only under the maroon stamp, preventing
#      collapse back to an effectively single-layer coarse task.
# c) Hidden-assumption audit against the original main()
#    - `len(colors_present(...)) == 1` no longer holds; the combo requires exactly two non-background colors.
#    - Boolean-collapsing the whole motif is invalid because it would erase which cells belong to the coarse code and
#      which belong to the fine code.
#    - The original `_decode_phase_tile(motif)` is no longer the right decoder: this combo needs one corner decoder
#      for the coarse layer and one edge decoder for the fine layer.
#    - The original output size formula `3 * input - 2` happens to be 16 here, but the reason changes: the canvas is
#      fixed because the coarse period is 4 and the fine overlay is an 8x8 centered window.
#    - The original center stamp recolored one layer. Here the maroon stamp must use the union of both color layers,
#      because both colors are structural parts of the encoded glyph.
# d) Placement legality
#    - Coarse masks occupy corners only, and every allowed coarse code spans the full 3x3 bbox through opposite-side
#      corner usage, so the combined object always has a valid tight 3x3 bbox.
#    - Fine masks occupy the center plus edge-centers only, so the two colors never overlap.
#    - The motif can still be placed at any top-left position in rows/cols 0..3 of the 6x6 input.
#    - The fine overlay window is fixed to rows/cols 4..11 and the maroon stamp to rows/cols 5..10, so all overlays
#      stay in bounds and the ring around the stamp can visibly expose the second tiling pass.
# e) Visual observability
#    - A solver can infer the rule from pixels alone: one input color appears only at corners and correlates with the
#      outer large-period texture, while the other appears at the center/cross positions and correlates with the extra
#      fine texture near the middle.
#    - No step depends on color id ordering. Roles are assigned by visible geometry: corners mean coarse, center+edges
#      mean fine.
# f) Visual cleanliness
#    - Only three output colors are used: black, orange, maroon. The two input colors are not copied through, which
#      keeps the final scene easy to parse.
#    - The fine overlay is confined to the centered 8x8 window, and the maroon stamp is drawn last in the smaller 6x6
#      window, so there are only two controlled overlap zones rather than many competing layers.
# g) Change inferability
#    - The color change rule is structural rather than arbitrary: both input colors disappear as colors and become two
#      encoded orange/black tilings plus one maroon union stamp.
#    - The generator enforces positive fine-layer gain outside the maroon stamp, so the second code color always has an
#      observable consequence beyond merely existing in the input.
# h) New implicit assumptions introduced by the combo
#    - Exactly one input color must satisfy the coarse-mask grammar and exactly one must satisfy the fine-mask grammar.
#      `generate_input()` guarantees this by construction.
#    - The fine mask must include the center cell; otherwise the zero-edge fine code would be visually absent and color
#      separation would become ambiguous. The generator always includes that center cell for the fine layer.
#    - The fine overlay is composed by orange-union over the coarse layer, never subtraction. The generator checks that
#      this union changes the scene in the visible ring around the center stamp.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


GRID_H = 6
GRID_W = 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
OUT_SHAPE = (16, 16)
DETAIL_R0 = 4
DETAIL_C0 = 4
DETAIL_SIZE = 8
STAMP_R0 = 5
STAMP_C0 = 5
STAMP_SIZE = 6
CODE_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300
MIN_DETAIL_GAIN = 8
MIN_RING_GAIN = 4

_BLOCK2 = np.ones((2, 2), dtype=bool)

# Corner order: TL, TR, BL, BR.
COARSE_TILES = {
    (1, 0, 0, 1): np.kron(np.array([[1, 0], [0, 1]], dtype=bool), _BLOCK2).astype(bool),
    (0, 1, 1, 0): np.kron(np.array([[0, 1], [1, 0]], dtype=bool), _BLOCK2).astype(bool),
    (0, 1, 1, 1): np.kron(np.array([[0, 1], [1, 1]], dtype=bool), _BLOCK2).astype(bool),
    (1, 0, 1, 1): np.kron(np.array([[1, 0], [1, 1]], dtype=bool), _BLOCK2).astype(bool),
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


def _corner_code(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.shape != (3, 3):
        return None
    if bool(mask[1, 1]):
        return None
    if bool(mask[0, 1] or mask[1, 0] or mask[1, 2] or mask[2, 1]):
        return None
    code = tuple(int(mask[r, c]) for r, c in ((0, 0), (0, 2), (2, 0), (2, 2)))
    return code if code in COARSE_TILES else None


def _edge_code(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.shape != (3, 3):
        return None
    if not bool(mask[1, 1]):
        return None
    if bool(mask[0, 0] or mask[0, 2] or mask[2, 0] or mask[2, 2]):
        return None
    code = tuple(int(mask[r, c]) for r, c in ((0, 1), (1, 0), (1, 2), (2, 1)))
    return code if code in DETAIL_TILES else None


def _decode_coarse_tile(mask: np.ndarray) -> np.ndarray:
    """Decode the corner-only color layer into the 4x4 coarse phase tile."""
    code = _corner_code(mask)
    assert code is not None
    return COARSE_TILES[code]


def _decode_detail_tile(mask: np.ndarray) -> np.ndarray:
    """Decode the center/edge color layer into the 2x2 fine phase tile."""
    code = _edge_code(mask)
    assert code is not None
    return DETAIL_TILES[code]


def _split_layers(motif: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate the two input colors into the coarse corner layer and fine center/edge layer."""
    assert motif.shape == (3, 3)

    cols = colors_present(motif, exclude={Color.ORANGE})
    assert len(cols) == 2
    assert all(int(col) not in {Color.BLACK, Color.ORANGE, Color.MAROON} for col in cols)

    masks = [(int(col), motif == int(col)) for col in cols]
    coarse_candidates = [mask for _col, mask in masks if _corner_code(mask) is not None]
    fine_candidates = [mask for _col, mask in masks if _edge_code(mask) is not None]

    assert len(coarse_candidates) == 1
    assert len(fine_candidates) == 1

    coarse_mask = coarse_candidates[0].astype(bool)
    fine_mask = fine_candidates[0].astype(bool)
    assert not bool(np.any(coarse_mask & fine_mask))
    assert bbox(coarse_mask | fine_mask) == (0, 0, 2, 2)
    return coarse_mask, fine_mask


def _compose_tiled_layers(coarse_mask: np.ndarray, fine_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    coarse_tile = _decode_coarse_tile(coarse_mask)
    detail_tile = _decode_detail_tile(fine_mask)

    base = tile_fill(OUT_SHAPE, coarse_tile).astype(bool)
    detail_mask = tile_fill((DETAIL_SIZE, DETAIL_SIZE), detail_tile).astype(bool)

    region = base[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE].copy()
    added_local = detail_mask & ~region
    region |= detail_mask
    base[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE] = region

    added_full = np.zeros(OUT_SHAPE, dtype=bool)
    added_full[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE] = added_local

    stamp_window = np.zeros(OUT_SHAPE, dtype=bool)
    stamp_window[STAMP_R0 : STAMP_R0 + STAMP_SIZE, STAMP_C0 : STAMP_C0 + STAMP_SIZE] = True

    out = np.where(base, Color.ORANGE, Color.BLACK).astype(int)
    return out, int(added_full.sum()), int(np.count_nonzero(added_full & ~stamp_window))


def _stamp_center(scene: np.ndarray, stamp_mask: np.ndarray) -> np.ndarray:
    center = np.tile(stamp_mask, (2, 2))
    stamped = scene.copy()
    stamped[STAMP_R0 : STAMP_R0 + STAMP_SIZE, STAMP_C0 : STAMP_C0 + STAMP_SIZE][center] = Color.MAROON
    return stamped


def _fusion_metrics(grid: np.ndarray) -> dict[str, int | tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    b = bbox(g != Color.ORANGE)
    assert b is not None
    motif = crop_bbox(g, b)
    coarse_mask, fine_mask = _split_layers(motif)
    _scene, detail_gain, ring_gain = _compose_tiled_layers(coarse_mask, fine_mask)
    return {
        "coarse_code": _corner_code(coarse_mask),
        "detail_code": _edge_code(fine_mask),
        "detail_gain": detail_gain,
        "ring_gain": ring_gain,
    }


def _assemble_masks(
    corners: tuple[int, int, int, int], edges: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    coarse_mask = np.zeros((3, 3), dtype=bool)
    fine_mask = np.zeros((3, 3), dtype=bool)

    coarse_mask[0, 0], coarse_mask[0, 2], coarse_mask[2, 0], coarse_mask[2, 2] = [bool(x) for x in corners]
    fine_mask[1, 1] = True
    fine_mask[0, 1], fine_mask[1, 0], fine_mask[1, 2], fine_mask[2, 1] = [bool(x) for x in edges]

    assert _corner_code(coarse_mask) == corners
    assert _edge_code(fine_mask) == edges
    assert not bool(np.any(coarse_mask & fine_mask))
    assert bbox(coarse_mask | fine_mask) == (0, 0, 2, 2)
    return coarse_mask, fine_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-color 3x3 code patch into a coarse global lattice plus a finer centered overlay, then stamp the full glyph union in maroon.

    Input structure:
    - The grid is 6x6 with orange background and exactly two distinct non-orange colors.
    - The non-orange cells have a tight 3x3 bbox.
    - One color appears only on corners and forms a valid coarse corner code; the other appears on the center cell and optional edge-centers and forms a valid fine code.

    Rule:
    - Split the 3x3 patch by color and identify the corner-only layer as the coarse code and the center/edge layer as the fine code.
    - Decode the coarse layer into a 4x4 orange/black tile and repeat it across the full 16x16 canvas.
    - Decode the fine layer into a 2x2 orange/black tile and repeat it only in the centered 8x8 region, union-overlaid on top of the coarse background; then tile the union of both input layers 2x2 and recolor it to maroon in the centered 6x6 stamp.

    Key insight:
    - The two input colors are not decorative copies. They are two independent structural codes packed into one 3x3 glyph: corners control the large-period background, while the center-cross color controls a second finer lattice that only appears near the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 2

    b = bbox(g != bg)
    assert b is not None
    motif = crop_bbox(g, b)
    assert motif.shape == (3, 3)

    coarse_mask, fine_mask = _split_layers(motif)
    scene, _detail_gain, _ring_gain = _compose_tiled_layers(coarse_mask, fine_mask)
    return _stamp_center(scene, coarse_mask | fine_mask)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    corners = CORNER_CODES[int(rng.integers(0, len(CORNER_CODES)))]
    edges = EDGE_CODES[int(rng.integers(0, len(EDGE_CODES)))]
    coarse_mask, fine_mask = _assemble_masks(corners, edges)

    _scene, detail_gain, ring_gain = _compose_tiled_layers(coarse_mask, fine_mask)
    if detail_gain < MIN_DETAIL_GAIN or ring_gain < MIN_RING_GAIN:
        return None

    coarse_color, fine_color = map(int, rng.choice(CODE_COLORS, size=2, replace=False))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    patch = inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    patch[coarse_mask] = coarse_color
    patch[fine_mask] = fine_color

    try:
        out = main(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    detail_window = out[DETAIL_R0 : DETAIL_R0 + DETAIL_SIZE, DETAIL_C0 : DETAIL_C0 + DETAIL_SIZE]
    if np.count_nonzero(detail_window == Color.ORANGE) == 0:
        return None
    if int(metrics["detail_gain"]) < MIN_DETAIL_GAIN or int(metrics["ring_gain"]) < MIN_RING_GAIN:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 30")
