# Combo variant of eee78d87 (index 21)
# combo_spec: D7/2+D14/2
# Name: 多个字形副本+多色多层

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D7/2 + D14/2 combo
# a) Concrete behavior:
#    Original: one 3x3 contour glyph in one color -> decode one phase tile ->
#    repeat one orange/black lattice -> stamp that glyph in maroon at the center.
#    Combo: two 3x3 bboxes appear in the 6x6 input, but they are copies of the
#    same glyph shape and differ only by color and position. Each color still forms
#    its own lattice layer, yet identical shapes would decode to identical phase
#    masks. To keep D14's "two independent layers" visible, the combo uses each
#    bbox top-left as the layer's period-3 phase offset. Example: a blue H at (0,0)
#    and a red H at (1,3) decode to the same H phase tile, but the blue lattice is
#    aligned with offset (0,0) while the red lattice is aligned with offset (1,3);
#    these shifted lattices overlap, and the later bbox in row-major order overwrites
#    the earlier one on shared cells. The center stamp stays a single 2x2 maroon
#    copy of the unique glyph, because the two inputs encode one shape, not two
#    different glyph identities.
# b) Complexity calibration:
#    If the two copies were allowed to share the same offset modulo 3, the shifted
#    layers would collapse into the same lattice and one color would disappear.
#    The generator therefore samples only position pairs whose modulo-3 offsets yield
#    both overlap and non-overlap in the 16x16 output, so examples visibly demonstrate
#    both "same glyph copied twice" and "colored layers stacked with priority".
# c) Implicit assumptions broken from the original main():
#    - `len(colors_present(...)) == 1` is false; the combo requires exactly 2 colors.
#    - A single union mask is no longer valid; extraction must happen per color.
#    - One global bbox is no longer valid; each color needs its own tight 3x3 bbox.
#    - The original output color was always orange; now each layer keeps its input color.
#    - The original decoding ignored source position; that would make identical copies
#      degenerate into one hidden layer, so the combo explicitly uses bbox position as
#      the phase offset.
#    - The original center stamp came from one motif; here it still does, but only after
#      asserting that both extracted motifs are exactly the same glyph.
# d) Placement legality:
#    The generator uses two non-overlapping 3x3 bboxes fully inside the 6x6 canvas.
#    Each bbox contains only orange plus its own foreground color, so solver-side
#    per-color cropping is well-defined and cannot mix the two copies.
# e) Visual observability:
#    Priority is determined by row-major bbox order, which humans can inspect directly.
#    The offset rule is also human-visible because it depends only on each glyph copy's
#    location, not on hidden numeric color ids or any external ordering.
# f) Visual cleanliness:
#    The output still uses only black, two layer colors, and one centered maroon stamp.
#    Both lattices share one periodic scaffold, so the picture stays structured rather
#    than becoming a clutter of unrelated objects.
# g) Change inferability:
#    Colors keep a simple structural role: "this color owns this shifted lattice layer".
#    There are only two colors, both reappear unchanged in the output, and their visible
#    overlaps are constrained so a solver can infer that color and position jointly decide
#    each layer instead of guessing an arbitrary remapping.


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 phase mask."""
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


GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
PHASES = {name: _decode_phase_tile(motif) for name, motif in GLYPHS.items()}


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
OUT_SHAPE = (3 * GRID_H - 2, 3 * GRID_W - 2)
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_from_topleft(r0: int, c0: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + MOTIF_SIZE - 1, c0 + MOTIF_SIZE - 1


def _is_valid_glyph(motif: np.ndarray) -> bool:
    return any(np.array_equal(motif, glyph) for glyph in GLYPHS.values())


def _sort_key(layer: dict[str, object]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = layer["bbox"]
    return int(r0), int(c0)


def _tiled_phase_mask(phase: np.ndarray, offset: tuple[int, int], shape: tuple[int, int] = OUT_SHAPE) -> np.ndarray:
    return tile_fill(shape, phase.astype(int), offset=offset).astype(bool)


def _extract_layers(g: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2

    layers: list[dict[str, object]] = []
    for color in fg_colors:
        mask = g == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        assert (r1 - r0 + 1, c1 - c0 + 1) == (MOTIF_SIZE, MOTIF_SIZE)

        cropped = crop_bbox(g, b)
        assert np.isin(cropped, [bg, color]).all()
        motif = cropped == color
        assert _is_valid_glyph(motif)

        layers.append(
            {
                "color": int(color),
                "bbox": b,
                "offset": (int(r0), int(c0)),
                "motif": motif,
                "phase": _decode_phase_tile(motif),
            }
        )

    assert not _boxes_overlap(layers[0]["bbox"], layers[1]["bbox"])
    assert np.array_equal(layers[0]["motif"], layers[1]["motif"])
    layers.sort(key=_sort_key)
    return layers[0]["motif"], layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two colored copies of the same contour glyph into shifted lattice layers, then place the shared glyph as a centered maroon seal.

    Input structure:
    - The grid is 6x6 with orange background and exactly two non-orange colors.
    - Each color occupies its own tight 3x3 bbox and forms a valid contour glyph.
    - The two 3x3 glyphs are identical in shape, do not overlap, and their row-major order defines draw priority.

    Rule:
    - Decode each 3x3 glyph copy into the task's period-3 phase tile.
    - Repeat each decoded tile across a black 16x16 canvas using that copy's own input color, but align the repetition with the copy's bbox top-left position as the phase offset.
    - Composite the two shifted lattices so the later bbox in row-major order overwrites the earlier one on overlap, then stamp the shared glyph tiled 2x2 in maroon inside the centered 6x6 window.

    Key insight:
    - The two inputs are not two different glyph identities; they are two colored instances of one glyph, and the only way both layers stay informative is to combine shape decoding with the visibly different source positions that shift the repeated phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    motif, layers = _extract_layers(g)

    out = np.full(OUT_SHAPE, Color.BLACK, dtype=int)
    for layer in layers:
        phase_colors = np.where(layer["phase"], layer["color"], Color.BLACK).astype(int)
        tiled = tile_fill(OUT_SHAPE, phase_colors, offset=layer["offset"])
        mask = tiled != Color.BLACK
        out[mask] = tiled[mask]

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


POSITIONS = [(r, c) for r in range(MOTIF_POS_MAX) for c in range(MOTIF_POS_MAX)]
VALID_ORDERED_POSITION_PAIRS = [
    (p1, p2)
    for i, p1 in enumerate(POSITIONS)
    for p2 in POSITIONS[i + 1 :]
    if not _boxes_overlap(_bbox_from_topleft(*p1), _bbox_from_topleft(*p2))
]


VISIBLE_SAME_GLYPH_SETUPS: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
for name, phase in PHASES.items():
    for low_pos, high_pos in VALID_ORDERED_POSITION_PAIRS:
        low_mask = _tiled_phase_mask(phase, low_pos)
        high_mask = _tiled_phase_mask(phase, high_pos)
        overlap = int((low_mask & high_mask).sum())
        low_only = int((low_mask & ~high_mask).sum())
        if overlap > 0 and low_only > 0:
            VISIBLE_SAME_GLYPH_SETUPS.append((name, low_pos, high_pos))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name, low_pos, high_pos = VISIBLE_SAME_GLYPH_SETUPS[int(rng.integers(len(VISIBLE_SAME_GLYPH_SETUPS)))]
    low_color, high_color = map(int, rng.choice(FG_COLORS, size=2, replace=False))
    motif = GLYPHS[glyph_name]

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    low_r0, low_c0 = low_pos
    high_r0, high_c0 = high_pos
    inp[low_r0 : low_r0 + MOTIF_SIZE, low_c0 : low_c0 + MOTIF_SIZE][motif] = low_color
    inp[high_r0 : high_r0 + MOTIF_SIZE, high_c0 : high_c0 + MOTIF_SIZE][motif] = high_color

    try:
        out = main(inp)
    except Exception:
        return None

    out_colors = set(map(int, np.unique(out)))
    if not {low_color, high_color, Color.MAROON}.issubset(out_colors):
        return None

    phase = PHASES[glyph_name]
    low_mask = _tiled_phase_mask(phase, low_pos)
    high_mask = _tiled_phase_mask(phase, high_pos)
    if not bool((low_mask & high_mask).any()):
        return None
    if not bool((low_mask & ~high_mask).any()):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 21")
