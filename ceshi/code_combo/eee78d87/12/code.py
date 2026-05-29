# Combo variant of eee78d87 (index 12)
# combo_spec: D8/0+D12/0
# Name: 背景干扰+可变输入尺寸

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill

# Design analysis for the D8/0 + D12/0 combo variant
# a) Concrete behavior and trace
#    - The preserved object is still exactly one 3x3 monochrome contour glyph:
#      plus / H / X / 4corners.
#    - The changed input is now an orange N x N square with N in 4..9, and the
#      background also contains a few scattered non-orange distractor pixels.
#    - Example trace:
#      original logic on a clean 6x6 input does `bbox(g != bg)` and directly
#      crops the only non-orange object, which works because every non-orange
#      cell belongs to the glyph and the bbox is exactly 3x3.
#      combo logic on a noisy 4x4 or 8x8 input cannot do that: one stray pixel
#      outside the glyph expands the global non-orange bbox beyond 3x3, and the
#      old fixed-size assertion `g.shape == (6, 6)` also fails when N != 6.
#      The repaired logic first infers N from the actual square input, then
#      scans every 3x3 window and keeps only the unique window whose non-orange
#      pixels use one color and match one of the four legal glyph masks.
#      After that recovery step, the output rule is unchanged except that the
#      canvas size becomes (3N-2) x (3N-2) instead of a hard-coded 16x16.
# b) Complexity calibration
#    - Only two freedoms are added, and both are visually direct:
#      input side length varies, and sparse noise appears in the orange field.
#    - Distractors stay as isolated single pixels rather than larger objects, so
#      the task remains about filtering anomalies plus reading the contour code,
#      not about inventing a new object taxonomy.
#    - The generator prefers 2-4 distractors when the canvas is roomy enough,
#      but falls back to 1 on tiny dense cases such as a 4x4 grid carrying the
#      7-pixel H glyph. This keeps the noise visible without breaking the idea
#      that orange is still the background color.
# c) Old implicit assumptions that break
#    - `assert g.shape == (6, 6)` is invalid once N varies.
#    - `len(colors_present(g, exclude={bg})) == 1` is invalid because noise
#      adds extra colors.
#    - `bbox(g != bg)` no longer isolates the motif, so cropping the global
#      non-background bbox is no longer sound.
#    - `out_shape = (3 * h - 2, 3 * w - 2)` remains valid and becomes the main
#      size rule that exposes the D12 change.
#    - `_decode_phase_tile` still applies exactly once the true 3x3 glyph mask
#      has been recovered, so the phase-decoding branch logic can stay intact.
# d) Placement legality
#    - The glyph top-left is sampled from `0..N-3`, so its 3x3 bbox always fits.
#    - Distractors are sampled only from cells outside that bbox, so they never
#      overwrite glyph pixels and always force the global non-orange bbox to be
#      larger than 3x3.
#    - Distractor colors are all distinct from the glyph color and from each
#      other, preventing fake monochrome 3x3 windows from being assembled out of
#      noise. `main()` is still run as an oracle check before accepting a sample.
# e) Visual observability
#    - Humans can see both changes from a few examples:
#      larger or smaller input squares produce outputs of different side lengths,
#      and stray colored pixels are ignored unless they participate in the only
#      valid 3x3 contour window.
#    - The relation between sizes is easy to compare across examples because the
#      central maroon 6x6 stamp stays constant while the orange/black lattice
#      expands according to the input side length.
# f) Visual cleanliness
#    - Noise stays outside the motif bbox and remains sparse, so the 3x3 glyph
#      is still the obvious focal object rather than being buried in clutter.
#    - The output keeps the original two-layer composition: periodic lattice in
#      the full canvas and one centered maroon stamp on top.
# g) Color inferability
#    - No new color mapping is introduced. Input distractor colors are irrelevant
#      markers only; the output still uses exactly black, orange, and maroon.
# New solver assumptions and how the generator guarantees them
#    - The input is square with side length in 4..9.
#    - Orange remains the most common color even on small noisy cases.
#    - There is exactly one 3x3 window whose non-orange cells are monochrome and
#      match a legal glyph mask.
#    - The generator enforces these constraints by construction and rejects any
#      accidental counterexample by validating with `main()`.

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_MASKS = tuple(GLYPHS.values())
GLYPH_NAMES = tuple(GLYPHS)


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
    """Locate the unique legal 3x3 contour glyph inside a noisy variable-size grid."""
    candidates: list[np.ndarray] = []
    h, w = map(int, grid.shape)
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
    Recover the lone valid 3x3 contour code from a noisy variable-size orange square, then expand it into the corresponding periodic lattice and centered maroon stamp.

    Input structure:
    - The grid is square, orange-background, and its side length N can vary from one example to another.
    - Exactly one 3x3 window contains a monochrome contour glyph in a non-orange color.
    - Additional non-orange distractor pixels may be scattered elsewhere in the orange field and do not belong to the glyph.

    Rule:
    - Search the whole N x N grid for the unique 3x3 window whose non-orange cells are all one color and match one of the four legal contour glyphs.
    - Decode that recovered glyph into the corresponding 3x3 orange/black phase tile and repeat it over an output canvas of size (3N-2) x (3N-2).
    - Tile the recovered 3x3 glyph 2x2 into a 6x6 mask and recolor its true cells to maroon at the exact center of the output.

    Key insight:
    - The useful signal is the only structurally valid 3x3 contour code, not the whole set of colored pixels; once noise is filtered out, the same glyph controls both the lattice phase and the centered stamp, while the input side length controls the output size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h == w
    assert 4 <= h <= 9

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    motif = _find_motif(g, bg)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * h - 2, 3 * w - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 9
MOTIF_SIZE = 3
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
PREFERRED_DISTRACTOR_COUNTS = (2, 3, 4, 1)
MAX_RETRIES = 300


def _allowed_distractor_counts(n: int, motif: np.ndarray) -> list[int]:
    """Return visible distractor counts that still keep orange as the unique background color."""
    motif_pixels = int(motif.sum())
    outside_cells = n * n - MOTIF_SIZE * MOTIF_SIZE
    max_by_color_pool = len(FG_COLORS) - 1
    max_by_bg_majority = n * n - 2 * motif_pixels - 1
    limit = min(outside_cells, max_by_color_pool, max_by_bg_majority)
    assert limit >= 1
    return [count for count in PREFERRED_DISTRACTOR_COUNTS if count <= limit]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    glyph_name = str(rng.choice(GLYPH_NAMES))
    motif = GLYPHS[glyph_name]
    fg = int(rng.choice(FG_COLORS))
    pos_max = n - MOTIF_SIZE + 1
    r0 = int(rng.integers(0, pos_max))
    c0 = int(rng.integers(0, pos_max))

    inp = np.full((n, n), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    outside_bbox = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if not (r0 <= r < r0 + MOTIF_SIZE and c0 <= c < c0 + MOTIF_SIZE)
    ]
    distractor_count = int(rng.choice(_allowed_distractor_counts(n, motif)))
    chosen = rng.choice(len(outside_bbox), size=distractor_count, replace=False)
    noise_palette = [color for color in FG_COLORS if color != fg]
    noise_colors = rng.choice(noise_palette, size=distractor_count, replace=False)
    for idx, color in zip(chosen, noise_colors):
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant D8/0+D12/0")
