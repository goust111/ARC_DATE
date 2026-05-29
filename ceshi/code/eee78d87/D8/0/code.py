# Variant of eee78d87
# Dimension: D8 — 输入网格的背景结构 (meta)
# Change: 6x6纯橙色(7)背景，字形是唯一的非橙色内容... → 背景包含干扰像素需过滤

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill

# Design analysis for the D8 variant
# a) Concrete behavior and trace
#    - The true object is still exactly one 3x3 monochrome contour glyph:
#      plus / H / X / 4corners.
#    - The new input also contains 2-4 distractor pixels of other colors,
#      scattered outside the glyph's 3x3 bbox.
#    - Example trace:
#      original logic: `fg = g != bg; bbox(fg)` assumes every non-orange pixel
#      belongs to the glyph. If a distractor sits at (0, 5) while the glyph is
#      at rows 2-4, cols 1-3, the global non-bg bbox becomes 5x5 and the old
#      `motif.shape == (3, 3)` assertion fails.
#      new logic: scan every 3x3 window, keep only windows whose non-orange
#      pixels are all one color and exactly match one known glyph mask. The
#      distractor pixels never satisfy this structural test, so the unique
#      glyph window is recovered and the output rule stays unchanged.
# b) Complexity calibration
#    - Distractors are sparse single pixels, not extra glyph-like objects.
#      This keeps the core rule focused on anomaly filtering instead of adding
#      a second object taxonomy.
#    - There are always at least 2 distractors, so the input is visibly no
#      longer "one clean glyph on orange".
#    - Distractor colors are distinct from the glyph color, which prevents a
#      trivial global single-color extraction but keeps the visual rule easy to
#      infer from a few examples: ignore stray colored noise, keep the only
#      valid contour code.
# c) Old implicit assumptions that break
#    - `len(colors_present(g, exclude={bg})) == 1` no longer holds.
#    - `bbox(g != bg)` no longer isolates the glyph.
#    - `crop_bbox(...).shape == (3, 3)` no longer follows from the input.
#    - The phase decoding itself is still valid once the correct 3x3 motif has
#      been detected, so `_decode_phase_tile` can remain unchanged.
# d) Placement legality
#    - The generator places the 3x3 glyph fully inside the 6x6 grid.
#    - Distractors are chosen from cells strictly outside that 3x3 bbox, so
#      they never overwrite the glyph and never change its local 3x3 window.
#    - Distinct distractor colors guarantee that no distractor subset can form
#      a fake monochrome glyph window.
# e) Visual observability
#    - Across examples, the glyph stays a tight 3x3 monochrome contour cluster
#      while the extra pixels are isolated and ignored by the output.
#      A human can see that the output depends on the contour code only, not on
#      the scattered background anomalies.
# f) Visual cleanliness
#    - Using isolated distractors avoids clutter and preserves a clear focal
#      object. The output remains identical in style to the original task.
# g) Color inferability
#    - No new color mapping is introduced. Input distractor colors are purely
#      "irrelevant marker" colors, and output colors remain black/orange/maroon.
# Post-change solver assumptions and how the generator guarantees them
#    - There is exactly one 3x3 window whose non-orange pixels use one color and
#      match one of the four glyph masks.
#    - The background color is still orange and dominant.
#    - The generator validates each sample by running `main`, so accidental
#      violations of uniqueness are rejected.

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_MASKS = tuple(GLYPHS.values())


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


def _find_motif(grid: np.ndarray, bg: int) -> np.ndarray:
    """Locate the unique 3x3 monochrome glyph window amid colored distractors."""
    candidates: list[np.ndarray] = []
    h, w = grid.shape
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            window = grid[r0 : r0 + 3, c0 : c0 + 3]
            cols = colors_present(window, exclude={bg})
            if len(cols) != 1:
                continue
            mask = window == cols[0]
            if any(np.array_equal(mask, glyph) for glyph in GLYPH_MASKS):
                candidates.append(mask.astype(bool))
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the lone valid 3x3 contour code from a noisy orange field, then build the same 16x16 phase lattice and centered maroon stamp as in the base task.

    Input structure:
    - The grid is 6x6 with orange as the dominant background.
    - One 3x3 window contains a monochrome contour glyph in a non-orange color.
    - Additional non-orange distractor pixels of other colors may appear elsewhere in the background and do not belong to the glyph.

    Rule:
    - Search the 6x6 grid for the unique 3x3 window whose non-orange pixels are all one color and form one of the task's four contour glyphs.
    - Decode that recovered glyph into the corresponding period-3 orange/black tile and repeat it over the 16x16 canvas.
    - Tile the recovered 3x3 glyph 2x2 and stamp its true cells in maroon inside the centered 6x6 region.

    Key insight:
    - The output ignores stray colored anomalies; only the unique structurally valid 3x3 contour glyph acts as the codebook entry for both the lattice phase and the central stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    motif = _find_motif(g, bg)
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
DISTRACTOR_COUNTS = [2, 3, 4]
MAX_RETRIES = 300


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    outside_bbox = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if not (r0 <= r < r0 + MOTIF_SIZE and c0 <= c < c0 + MOTIF_SIZE)
    ]
    distractor_count = int(rng.choice(DISTRACTOR_COUNTS))
    cells = rng.choice(len(outside_bbox), size=distractor_count, replace=False)
    noise_colors = rng.choice([c for c in FG_COLORS if c != fg], size=distractor_count, replace=False)
    for idx, color in zip(cells, noise_colors):
        rr, cc = outside_bbox[int(idx)]
        inp[rr, cc] = int(color)

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
    raise RuntimeError("Failed to generate input for eee78d87 D8 variant")
