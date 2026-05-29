# Combo variant of eee78d87 (index 17)
# combo_spec: D4/4+D10/0
# Name: 颜色映射印章+格栅填充传色
"""
Design analysis:
  variant_core:
    - The input geometry stays the same as the base task: a 6x6 orange field contains one
      3x3 contour glyph in a single non-orange foreground color.
    - The glyph shape still decodes a 3x3 phase tile and that tile still repeats across a
      16x16 canvas. The structural code is unchanged; only the color roles are fused.
    - Fusion rule:
      1. D10/0 side: every phase=True lattice cell uses the input glyph color directly,
         with phase=False remaining black.
      2. D4/4 side: the centered 2x2 glyph stamp no longer uses fixed maroon; its color
         is chosen by a separate mapping from the input glyph color.
    - To keep the mapping human-inferable from a few examples, the generator uses a small
      visible three-color cycle instead of a hidden full-palette arithmetic rule:
      blue -> red, red -> green, green -> blue.

  behavior_trace:
    - Concrete example: a blue plus glyph in the input.
    - Base task:
      - plus decodes to the same 3x3 phase mask as before;
      - the repeated lattice is orange/black;
      - the center 6x6 stamp is maroon.
    - Fused variant:
      - the same plus mask is used, so the repeated geometry is unchanged;
      - the lattice becomes blue/black because phase=True cells inherit the glyph color;
      - the center stamp becomes red because blue maps to red in the color cycle.
    - Difference summary:
      - shape still controls which cells belong to the phase;
      - the lattice color is a direct transfer of input color;
      - the stamp color is a transformed transfer of the same input color.

  complexity_calibration:
    - The task still has only four glyph shapes and one centered stamp placement, so the
      structural search burden does not grow.
    - The fusion is visually stronger than either single change alone because the whole
      16x16 field and the center stamp now both depend on input color, but via different
      rules that stay easy to compare.
    - Restricting foreground colors to {blue, red, green} prevents an arbitrary-looking
      many-to-many recolor table and gives training samples a realistic chance to repeat
      the same color cycle enough for humans to infer it.

  implicit_assumptions_review:
    - `get_bg(g) == ORANGE` still holds: the clue still sits on an orange input background.
    - `len(colors_present(g, exclude={bg})) == 1` still holds and is now more important,
      because the unique foreground color drives both output color channels.
    - `motif = crop_bbox(g != bg, b)` still works because phase decoding depends only on
      the glyph mask, not on the specific color value.
    - `_decode_phase_tile(motif)` is unchanged because neither D4/4 nor D10/0 alters the
      contour vocabulary.
    - The original `np.where(phase, bg, BLACK)` assumption is invalid after fusion: the
      lattice no longer uses orange as its positive fill color, so this must become
      `np.where(phase, fg_color, BLACK)`.
    - The original fixed center overwrite with maroon is also invalid: the stamp color now
      depends on the foreground color and must come from an explicit mapping.
    - New assumption introduced by the mapping: the foreground color must be one of the
      mapping keys. The generator enforces that by sampling only from the cycle domain.

  placement_legality:
    - The 3x3 motif still uses top-left coordinates in 0..3, so it always fits inside the
      6x6 input with no overlap issues.
    - The center stamp is still a 2x2 tiling of the 3x3 motif, occupying the fixed centered
      6x6 window of the 16x16 output. No new placement branch is needed.

  visual_observability:
    - A human can see two separate color facts at once:
      1. the repeated lattice copies the input glyph color exactly;
      2. the center stamp consistently changes to the next color in a small cycle.
    - Because the mapped stamp color is always different from the lattice color in the
      chosen three-color cycle, the center overlay remains easy to segment by eye.
    - The rule uses visible color correspondences between examples, not hidden ordering of
      color ids such as "take numeric value +1" over the full palette.

  cleanliness_and_inference:
    - Each output uses only three colors: black, the direct-transfer lattice color, and the
      mapped stamp color. That keeps the image legible even though two color rules coexist.
    - The center region is still the only overlap area, but the stamp sits on top of the
      lattice with a distinct color, so the layered structure stays readable.
    - The generator additionally checks that sampled inputs do not collapse back to either
      single variant: outputs must differ from both the fixed-maroon lattice-transfer case
      and the mapped-stamp original-lattice case.

  generator_guarantees:
    - Foreground colors are sampled only from the mapping domain {blue, red, green}.
    - All three colors have distinct mapped targets, and none map to themselves, so both
      fused behaviors remain active in every valid sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


STAMP_COLOR_MAP = {
    Color.BLUE: Color.RED,
    Color.RED: Color.GREEN,
    Color.GREEN: Color.BLUE,
}


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


def _stamp_color_from_fg(fg_color: int) -> int:
    """Map the glyph foreground color to the center stamp color."""
    fg_color = int(fg_color)
    assert fg_color in STAMP_COLOR_MAP
    return int(STAMP_COLOR_MAP[fg_color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice whose filled phase cells inherit the glyph color, then stamp the motif at the center using a mapped color.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other visible color.
    - The non-orange pixels occupy a tight 3x3 bounding box and form one contour glyph.
    - The glyph is one of four code shapes: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 contour glyph as the same period-3 phase tile used in the base task.
    - Repeat that tile across a 16x16 canvas, coloring phase=True cells with the glyph's own foreground color and phase=False cells black.
    - Tile the original 3x3 glyph 2x2 into the centered 6x6 window and recolor its true pixels with the mapped stamp color associated with the input foreground color.

    Key insight:
    - The input glyph carries two independent clues at once: its shape chooses the periodic geometry, while its color drives two different output color channels, one copied directly into the lattice and one transformed for the center stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = int(fg_colors[0])
    stamp_color = _stamp_color_from_fg(fg_color)

    fg_mask = g != bg
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, fg_color, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    stamp = out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]]
    stamp[center] = stamp_color
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [Color.BLUE, Color.RED, Color.GREEN]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _center_slice(out_shape: tuple[int, int], center_shape: tuple[int, int]) -> tuple[slice, slice]:
    """Return the centered 2D slice for placing the tiled glyph stamp."""
    r0 = (out_shape[0] - center_shape[0]) // 2
    c0 = (out_shape[1] - center_shape[1]) // 2
    return slice(r0, r0 + center_shape[0]), slice(c0, c0 + center_shape[1])


def _build_single_variant_baselines(
    motif: np.ndarray,
    fg_color: int,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two relevant single-variant outputs to reject fusion collapse."""
    phase = _decode_phase_tile(motif)
    center = np.tile(motif, (2, 2))
    rs, cs = _center_slice(out_shape, center.shape)

    d10_only = tile_fill(out_shape, np.where(phase, fg_color, Color.BLACK).astype(int))
    d10_only[rs, cs][center] = Color.MAROON

    d4_only = tile_fill(out_shape, np.where(phase, Color.ORANGE, Color.BLACK).astype(int))
    d4_only[rs, cs][center] = _stamp_color_from_fg(fg_color)
    return d10_only, d4_only


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg_color = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg_color

    try:
        out = main(inp)
        d10_only, d4_only = _build_single_variant_baselines(motif, fg_color, out.shape)
        assert np.any(out != d10_only)
        assert np.any(out != d4_only)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant 17")
