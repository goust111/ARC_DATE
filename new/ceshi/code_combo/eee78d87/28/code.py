# Combo variant of eee78d87 (index 28)
# combo_spec: D1/0+D4/2+D10/0
# Name: 颜色偏移+镜像印章+格栅传色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# concrete behavior trace:
# - Base task pipeline:
#   1. Crop one 3x3 contour glyph from a 6x6 orange field.
#   2. Decode that boolean glyph into one asymmetric 3x3 phase tile.
#   3. Tile orange/black phase colors over a 16x16 canvas with zero offset.
#   4. Put a centered 6x6 maroon stamp made from a plain 2x2 copy of the input glyph.
# - Combo pipeline:
#   1. The same 3x3 contour still selects the phase tile content.
#   2. The visible glyph color now has two roles at once:
#      red -> offset (0,0), blue -> offset (1,0), green -> offset (0,1),
#      and that same color replaces orange as the lattice fill color.
#   3. The center no longer uses a plain 2x2 copy. It uses a 2x2 mirror layout:
#      top-left = seed, top-right = horizontal flip, bottom-left = vertical flip,
#      bottom-right = 180-degree rotation.
# - Example trace:
#   a blue plus still decodes to the same 3x3 phase boolean pattern as before,
#   but the tiled lattice becomes blue/black instead of orange/black, shifted down
#   by one row, and the centered maroon stamp is built from mirrored copies of the
#   decoded 3x3 seed rather than four identical copies.
#
# complexity calibration:
# - The color-to-offset rule is intentionally capped at three visible colors so a
#   human can infer the mapping from 2-3 training pairs.
# - The lattice color transfer is global and therefore visually obvious.
# - The mirror rule is confined to the center 6x6 box, keeping the scene clean
#   while still producing a local structure distinct from plain repetition.
#
# hidden-assumption review:
# - `bg == ORANGE` remains valid and is preserved by the generator.
# - `len(colors_present(exclude={bg})) == 1` is still required, but that color can
#   no longer be discarded after forming `g != bg`; it controls both offset and fill.
# - `_decode_phase_tile(motif)` remains correct because the contour catalog is unchanged.
# - The original `center = np.tile(motif, (2, 2))` line is no longer safe to reuse:
#   all four original contour glyphs are mirror-symmetric, so literal flipping of the
#   raw glyph would collapse back to the old center stamp and make D4/2 invisible.
#   To keep the reflection signal observable, the mirrored stamp uses the decoded 3x3
#   phase seed, which is asymmetric for every allowed contour.
#
# placement legality:
# - Input placement is still one 3x3 glyph with top-left row/col in 0..3, so the
#   bbox is tight, unique, and always in bounds.
# - The output stamp is always 6x6 and is centered inside the fixed 16x16 canvas,
#   so no boundary or overlap issues are introduced beyond the intentional overwrite.
#
# visual observability and cleanliness:
# - Outside the center, the whole lattice advertises both D1 and D10: color selects
#   phase offset and also becomes the non-black lattice color.
# - Inside the center, maroon is the only overlay color and stays confined to one
#   clean 6x6 block. Because the mirrored seed is asymmetric, the four quadrants are
#   visibly related by reflection rather than being four identical copies.
#
# generator guarantees:
# - Foreground colors are restricted to {red, blue, green} so the offset mapping is
#   repeated across a tiny, inferable category set.
# - Successive generated examples cycle through those three colors to improve support
#   when only a few training pairs are sampled.


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
OUT_SHAPE = (3 * GRID_H - 2, 3 * GRID_W - 2)
MAX_RETRIES = 300

PHASE_SHIFT_BY_COLOR = {
    Color.RED: (0, 0),
    Color.BLUE: (1, 0),
    Color.GREEN: (0, 1),
}

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}

GEN_COLOR_ORDER = [Color.RED, Color.BLUE, Color.GREEN]
GEN_GLYPH_ORDER = ["plus", "H", "X", "4corners"]
_GEN_COUNTER = 0


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


def _phase_offset_for_color(color: int) -> tuple[int, int]:
    color = int(color)
    assert color in PHASE_SHIFT_BY_COLOR
    return PHASE_SHIFT_BY_COLOR[color]


def _mirror_quadrants(seed: np.ndarray) -> np.ndarray:
    """Assemble a 6x6 mirror collage from one 3x3 seed."""
    assert seed.shape == (3, 3)
    return np.block(
        [
            [seed, np.fliplr(seed)],
            [np.flipud(seed), np.rot90(seed, 2)],
        ]
    )


def _stamp_seed_from_phase(phase: np.ndarray) -> np.ndarray:
    """Use the decoded phase seed so the mirror rule stays visible for symmetric glyphs."""
    assert phase.shape == (3, 3)
    return phase


def _build_output(motif: np.ndarray, fg_color: int, *, mirror_stamp: bool = True) -> np.ndarray:
    phase = _decode_phase_tile(motif)
    phase_offset = _phase_offset_for_color(fg_color)
    phase_colors = np.where(phase, int(fg_color), Color.BLACK).astype(int)
    out = tile_fill(OUT_SHAPE, phase_colors, offset=phase_offset)

    seed = _stamp_seed_from_phase(phase)
    stamp_mask = _mirror_quadrants(seed) if mirror_stamp else np.tile(seed, (2, 2))
    r0 = (out.shape[0] - stamp_mask.shape[0]) // 2
    c0 = (out.shape[1] - stamp_mask.shape[1]) // 2
    out[r0 : r0 + stamp_mask.shape[0], c0 : c0 + stamp_mask.shape[1]][stamp_mask] = Color.MAROON
    return out


def _build_original_like_output(motif: np.ndarray) -> np.ndarray:
    phase = _decode_phase_tile(motif)
    out = tile_fill(OUT_SHAPE, np.where(phase, Color.ORANGE, Color.BLACK).astype(int))
    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one colored 3x3 contour glyph into a color-transferred phase lattice, then place a mirrored maroon stamp at the center.

    Input structure:
    - The input is a 6x6 grid with orange background and exactly one non-orange foreground color.
    - The foreground pixels occupy one tight 3x3 bounding box containing a contour glyph.
    - The contour is one of plus, H, X, or four corners, and the foreground color is red, blue, or green.

    Rule:
    - Read the 3x3 contour to recover the task's 3x3 phase seed.
    - Use the glyph color twice: it becomes the lattice fill color, and it selects the tiling offset
      (red -> (0,0), blue -> (1,0), green -> (0,1)).
    - Repeat the colored phase seed over the 16x16 black canvas, then place a centered 6x6 maroon mirror-collage
      built from that 3x3 seed: original, horizontal flip, vertical flip, and 180-degree rotation.

    Key insight:
    - The small colored motif now carries two independent signals. Shape determines the repeated 3x3 seed, while color
      determines both where that seed starts in the lattice and which non-black color the lattice uses; the center stamp
      reveals the decoded seed again through mirror composition rather than plain duplication.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])
    assert fg_color in PHASE_SHIFT_BY_COLOR

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    return _build_output(motif, fg_color, mirror_stamp=True)


def _gen_attempt(
    rng: np.random.Generator,
    *,
    glyph_name: str,
    fg_color: int,
) -> np.ndarray | None:
    motif = GLYPHS[glyph_name]
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = int(fg_color)

    try:
        out = main(inp)
    except Exception:
        return None

    phase = _decode_phase_tile(motif)
    mirrored_stamp = _mirror_quadrants(_stamp_seed_from_phase(phase))
    plain_stamp = np.tile(_stamp_seed_from_phase(phase), (2, 2))
    if np.array_equal(mirrored_stamp, plain_stamp):
        return None

    no_mirror = _build_output(motif, fg_color, mirror_stamp=False)
    original_like = _build_original_like_output(motif)
    assert np.any(out != no_mirror)
    assert np.any(out != original_like)
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = np.random.default_rng()
    for attempt in range(MAX_RETRIES):
        glyph_name = GEN_GLYPH_ORDER[(_GEN_COUNTER + attempt) % len(GEN_GLYPH_ORDER)]
        fg_color = GEN_COLOR_ORDER[(_GEN_COUNTER + attempt) % len(GEN_COLOR_ORDER)]
        g = _gen_attempt(rng, glyph_name=glyph_name, fg_color=fg_color)
        if g is not None:
            _GEN_COUNTER += 1
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 28")
