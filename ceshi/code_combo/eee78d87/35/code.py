# Combo variant of eee78d87 (index 35)
# combo_spec: D7/2+D8/0
# Name: 背景干扰+多字形副本

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Base task: one monochrome 3x3 contour glyph sits on a clean 6x6 orange
#      field, so `bbox(g != bg)` isolates the motif directly.
#    - Combo task: the 6x6 field contains two copies of the same 3x3 glyph,
#      each copy monochrome but allowed to use a different color, plus a few
#      stray distractor pixels elsewhere.
#    - Example trace:
#      if copy A is a blue plus at (0,0), copy B is a red plus at (3,3), and
#      noise pixels sit at (0,5) and (5,0), then the old logic fails twice:
#      `len(colors_present(..., exclude={bg})) == 1` is false, and the bbox of
#      all non-orange cells expands far beyond 3x3.
#      The new solver instead scans all 3x3 windows and keeps only windows that
#      are monochrome-on-orange and exactly match one legal glyph mask. Among
#      these candidates, the true motif is the unique mask that appears twice.
# b) Complexity calibration
#    - I keep exactly two glyph copies, not an arbitrary count, so the repeated
#      pattern is visible from few examples and the inference burden stays low.
#    - Distractors are isolated pixels, not extra structured shapes, so the new
#      difficulty is source extraction rather than learning a second object type.
#    - Copy colors are forced to differ, and all distractors share one third
#      color. This makes the three semantic classes human-visible: glyph copy 1,
#      glyph copy 2, and background noise.
# c) Hidden assumptions broken from the base solver
#    - `len(colors_present(g, exclude={bg})) == 1` no longer holds because the
#      two copies and the distractors can use different colors.
#    - `bbox(g != bg)` no longer isolates one tight 3x3 motif.
#    - `crop_bbox(...).shape == (3,3)` is therefore invalid.
#    - What still survives is the glyph-to-phase decoding once the repeated 3x3
#      contour mask has been recovered.
# d) Placement legality
#    - The generator chooses two legal 3x3 top-left positions whose bboxes do
#      not overlap.
#    - Distractor cells are chosen strictly outside both 3x3 bboxes, so they do
#      not overwrite either copy.
#    - Samples are oracle-validated so there is exactly one repeated glyph mask
#      among all valid 3x3 windows.
# e) Visual observability
#    - Across examples, humans can spot that two separated 3x3 contour clusters
#      share the same shape even when their colors differ.
#    - The output ignores isolated anomalies and depends on the repeated shape,
#      which is a visible geometric cue rather than a color-id trick.
# f) Visual cleanliness
#    - The grid contains only two compact 3x3 objects plus 2-3 isolated dots.
#      This is busier than the base task but still sparse and legible.
# g) Color inferability
#    - No new output recoloring rule is introduced: outputs still use only
#      black/orange/maroon. Input colors only partition candidate objects from
#      each other; shape repetition determines which pixels matter.
# Post-change solver assumptions and generator guarantees
#    - Exactly one glyph shape appears in exactly two monochrome 3x3 windows.
#    - Any distractor-only window fails the glyph-mask test.
#    - The orange background remains the dominant color, so `get_bg` is stable.


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
DISTRACTOR_COUNTS = [2, 3]
MAX_RETRIES = 400

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_MASKS = tuple(GLYPHS.values())
LEGAL_TOP_LEFTS = [(r0, c0) for r0 in range(MOTIF_POS_MAX) for c0 in range(MOTIF_POS_MAX)]


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


def _bboxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + MOTIF_SIZE <= br or br + MOTIF_SIZE <= ar or ac + MOTIF_SIZE <= bc or bc + MOTIF_SIZE <= ac)


def _iter_candidate_windows(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, np.ndarray]]:
    """Return all monochrome 3x3 windows that exactly realize a legal glyph mask."""
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    h, w = grid.shape
    for r0 in range(h - MOTIF_SIZE + 1):
        for c0 in range(w - MOTIF_SIZE + 1):
            window = grid[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
            cols = colors_present(window, exclude={bg})
            if len(cols) != 1:
                continue
            color = int(cols[0])
            mask = window == color
            if any(np.array_equal(mask, glyph) for glyph in GLYPH_MASKS):
                candidates.append((r0, c0, color, mask.astype(bool)))
    return candidates


def _find_repeated_motif(grid: np.ndarray, bg: int) -> np.ndarray:
    """Recover the unique glyph shape that appears in exactly two valid windows."""
    groups: dict[bytes, list[tuple[int, int, int, np.ndarray]]] = {}
    for candidate in _iter_candidate_windows(grid, bg):
        key = candidate[3].tobytes()
        groups.setdefault(key, []).append(candidate)

    repeated = [items for items in groups.values() if len(items) == 2]
    assert len(repeated) == 1

    items = repeated[0]
    anchors = {(int(r0), int(c0)) for r0, c0, _color, _mask in items}
    assert len(anchors) == 2
    motif = items[0][3]
    assert np.array_equal(motif, items[1][3])
    return motif


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the repeated 3x3 contour code from a noisy 6x6 orange grid, then render the same 16x16 periodic lattice and central maroon stamp as in the base task.

    Input structure:
    - The grid is 6x6 with orange as the dominant background.
    - Two separate 3x3 windows contain the same contour glyph shape, though the two copies may use different non-orange colors.
    - Additional isolated non-orange distractor pixels may appear elsewhere and do not belong to either glyph copy.

    Rule:
    - Scan the grid for 3x3 windows whose non-orange cells are monochrome and form one of the four legal contour glyphs.
    - Identify the unique glyph mask that appears twice; that repeated shape is the true code and all other colored pixels are irrelevant noise.
    - Decode that mask into the period-3 orange/black tile, repeat it over the 16x16 canvas, then stamp a 2x2 maroon tiling of the glyph into the centered 6x6 region.

    Key insight:
    - Neither raw color nor the overall non-orange bounding box reveals the answer anymore; the reliable signal is the shape that repeats exactly twice among the noisy candidate windows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    motif = _find_repeated_motif(g, bg)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[glyph_name]

    pos1 = LEGAL_TOP_LEFTS[int(rng.integers(len(LEGAL_TOP_LEFTS)))]
    disjoint_positions = [pos for pos in LEGAL_TOP_LEFTS if not _bboxes_overlap(pos1, pos)]
    if not disjoint_positions:
        return None
    pos2 = disjoint_positions[int(rng.integers(len(disjoint_positions)))]

    fg1, fg2 = map(int, rng.choice(FG_COLORS, size=2, replace=False))
    noise_palette = [c for c in FG_COLORS if c not in {fg1, fg2}]
    if not noise_palette:
        return None
    noise_color = int(rng.choice(noise_palette))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    r1, c1 = pos1
    r2, c2 = pos2
    inp[r1 : r1 + MOTIF_SIZE, c1 : c1 + MOTIF_SIZE][motif] = fg1
    inp[r2 : r2 + MOTIF_SIZE, c2 : c2 + MOTIF_SIZE][motif] = fg2

    protected = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if (r1 <= r < r1 + MOTIF_SIZE and c1 <= c < c1 + MOTIF_SIZE)
        or (r2 <= r < r2 + MOTIF_SIZE and c2 <= c < c2 + MOTIF_SIZE)
    }
    free_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if (r, c) not in protected]

    distractor_count = int(rng.choice(DISTRACTOR_COUNTS))
    if len(free_cells) < distractor_count:
        return None
    chosen = rng.choice(len(free_cells), size=distractor_count, replace=False)
    for idx in chosen:
        rr, cc = free_cells[int(idx)]
        inp[rr, cc] = noise_color

    try:
        recovered = _find_repeated_motif(inp, Color.ORANGE)
        assert np.array_equal(recovered, motif)
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
