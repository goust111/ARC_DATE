# Combo variant of eee78d87 (index 37)
# combo_spec: D5/2+D6/1
# Name: 非正方形输出+行列不同周期
#
# Design analysis for this combo variant
# -------------------------------------
# This fusion keeps the original "decode one 3x3 contour glyph into a periodic
# orange/black lattice, then stamp a maroon 2x2 copy at the center" backbone,
# but it breaks two core assumptions at once:
# - the output is no longer the fixed square `3*6-2 = 16`,
# - the repeated lattice no longer uses one shared period in both axes.
#
# a) Concrete behavior trace
# - Original seed:
#   - infer the 3x3 contour glyph,
#   - decode one 3x3 phase tile,
#   - tile it on a fixed 16x16 square canvas,
#   - place the 6x6 maroon stamp in the center.
# - Fused variant:
#   - still infer the same 3x3 contour glyph and decode the same 3x3 phase tile,
#   - read the glyph's middle column occupancy count `mid_col` and middle row
#     occupancy count `mid_row`,
#   - use `mid_col` to choose output height and vertical period,
#   - use `mid_row` to choose output width and horizontal period,
#   - resize the decoded 3x3 phase tile to a `period_r x period_c` rectangle and
#     repeat that rectangular phase tile across the whole output,
#   - keep the centered 6x6 maroon 2x2 stamp.
# - Example for the H glyph:
#   - `mid_col = 1`, `mid_row = 3`,
#   - original output would be 16x16 with period (3,3),
#   - combo output is 12x15 with period (3,5),
#   - the same contour stamp still appears in maroon at the center.
#
# b) Complexity calibration
# - The four glyphs already provide three visible values for `mid_row` and
#   `mid_col` ({0,1,3}), so we can expose several size/period regimes without
#   inventing extra markers in the input.
# - The axis rules are intentionally asymmetric:
#   - height = `10 + 2*mid_col`,
#   - width  = `9 + 2*mid_row`,
#   - vertical period map `{3:2, 1:3, 0:4}`,
#   - horizontal period map `{3:5, 1:2, 0:3}`.
#   This guarantees `height != width` and `period_r != period_c` for every
#   allowed glyph, so generated samples cannot silently collapse back to the
#   original square / equal-period behavior.
# - The number of regimes stays small: only four glyph classes and three count
#   values per axis, so the task remains learnable from a few examples.
#
# c) Hidden-assumption audit against the original main()
# - `assert g.shape == (6, 6)`: still valid; the input format is unchanged.
# - `bg == ORANGE` and "exactly one non-orange color": still valid; generation
#   keeps the same palette discipline.
# - `motif.shape == (3, 3)`: still valid; the object remains a tight 3x3 glyph.
# - Original `out_shape = (3*h-2, 3*w-2)` is no longer valid. The new solver
#   must derive `out_h` and `out_w` separately from glyph structure.
# - Original `tile_fill(..., phase_colors)` assumed the tile itself fixed the
#   period as 3x3. Broken by D6/1. We now construct a resized rectangular phase
#   tile whose shape directly equals `(period_r, period_c)`.
# - Center-stamp placement previously relied on even square symmetry. With odd
#   widths / heights it must instead use floor-centered placement in each axis.
#
# d) Placement legality
# - The input still contains just one 3x3 glyph inside a 6x6 orange field, so
#   there are no object-overlap risks beyond keeping the bbox in-bounds.
# - The output size formulas always yield at least 9x10, so the centered 6x6
#   maroon stamp always fits with non-negative start indices.
# - Because the stamp is written after the lattice, visual overlap is deliberate
#   and exactly mirrors the original task structure.
#
# e) Visual observability
# - A human sees that different glyphs produce different rectangle aspect ratios
#   and different row-vs-column repeat beats in the orange/black lattice.
# - The center stamp stays unchanged across variants, which isolates the new
#   information in the background lattice and frame size instead of mixing many
#   simultaneous changes in the same region.
#
# f) Visual cleanliness
# - The lattice remains a single repeated orange/black texture, and the maroon
#   stamp remains a single compact 6x6 block at the center.
# - No extra symbols, legends, or palette expansions are introduced, so the
#   output is still easy to parse visually despite the extra asymmetry.
#
# g) Change inferability / non-degeneracy target
# - The generator samples all four glyph classes, ensuring coverage of multiple
#   `(height,width,period_r,period_c)` combinations.
# - A sample is considered non-degenerate when:
#   - `out.shape[0] != out.shape[1]`,
#   - `period_r != period_c`,
#   - and the combo output differs from the original 16x16 period-3 output for
#     the same input. The post-edit verification script checks these properties.
#
# h) New solver assumptions introduced here and enforced by the generator
# - Only the four canonical contour glyphs are valid; `_decode_phase_tile()`
#   still uses that family-specific logic.
# - `mid_row` and `mid_col` must lie in `{0,1,3}`. This is automatic for the
#   allowed glyph family and is asserted through the lookup tables.
# - Exactly one tight 3x3 glyph exists in the input. The generator still builds
#   exactly one such object with a single foreground color.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[4]
    for rel in ("arc2_solver_select", "solvers"):
        cand = ROOT / rel
        if cand.exists():
            sys.path.append(str(cand))
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the seed task's base phase mask."""
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


def _infer_axis_params(motif: np.ndarray) -> tuple[int, int, int, int]:
    mid_row = int(motif[1, :].sum())
    mid_col = int(motif[:, 1].sum())

    height = 10 + 2 * mid_col
    width = 9 + 2 * mid_row
    period_r = {3: 2, 1: 3, 0: 4}[mid_col]
    period_c = {3: 5, 1: 2, 0: 3}[mid_row]

    assert height != width
    assert period_r != period_c
    return int(height), int(width), int(period_r), int(period_c)


def _resize_phase_tile(phase: np.ndarray, period_r: int, period_c: int) -> np.ndarray:
    row_idx = (np.arange(int(period_r)) * phase.shape[0]) // int(period_r)
    col_idx = (np.arange(int(period_c)) * phase.shape[1]) // int(period_c)
    return phase[row_idx][:, col_idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into an asymmetric rectangular lattice and stamp a maroon 2x2 copy at the center.

    Input structure:
    - The input is still a 6x6 orange field containing one non-orange contour glyph.
    - That glyph occupies a tight 3x3 bounding box and is one of the same four
      canonical contour forms: plus, H, X, or four corners.
    - The glyph uses one arbitrary foreground color that is neither black,
      orange, nor maroon.

    Rule:
    - Decode the contour glyph into the original task's 3x3 orange/black phase mask.
    - Read the glyph's middle-column occupancy to determine the output height and
      the vertical tiling period, and read its middle-row occupancy to determine
      the output width and the horizontal tiling period.
    - Stretch the decoded phase mask into a `period_r x period_c` rectangular
      phase tile, repeat it across the whole output, then place a 2x2 tiling of
      the original glyph as a centered 6x6 maroon stamp.

    Key insight:
    - The contour glyph is now a two-axis code: its row structure controls the
      horizontal beat and width, while its column structure controls the
      vertical beat and height.
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

    out_h, out_w, period_r, period_c = _infer_axis_params(motif)
    base_phase = _decode_phase_tile(motif)
    phase = _resize_phase_tile(base_phase, period_r, period_c)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out = tile_fill((out_h, out_w), phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

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
_GLYPH_ORDER = ["plus", "H", "X", "4corners"]
_GEN_COUNTER = 0


def _build_input(motif: np.ndarray, fg: int, r0: int, c0: int) -> np.ndarray:
    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg
    return inp


def _gen_attempt(rng: np.random.Generator, glyph_name: str | None = None) -> np.ndarray | None:
    if glyph_name is None:
        glyph_name = str(rng.choice(_GLYPH_ORDER))
    motif = GLYPHS[str(glyph_name)]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))
    inp = _build_input(motif, fg, r0, c0)

    try:
        out = main(inp)
        period_r, period_c = _infer_axis_params(motif)[2:]
        assert out.shape[0] != out.shape[1]
        assert period_r != period_c
        assert out.shape[0] >= 10 and out.shape[1] >= 9
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    rng = np.random.default_rng()
    start = int(_GEN_COUNTER % len(_GLYPH_ORDER))
    _GEN_COUNTER += 1

    for step in range(len(_GLYPH_ORDER)):
        glyph_name = _GLYPH_ORDER[(start + step) % len(_GLYPH_ORDER)]
        g = _gen_attempt(rng, glyph_name=glyph_name)
        if g is not None:
            return g

    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo D5/2+D6/1")
