# Combo variant of eee78d87 (index 8)
# combo_spec: D7/1+D8/0
# Name: 角落标记定位+背景干扰像素

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, tile_fill


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete fused behavior and trace
#    - Base task: one monochrome 3x3 contour glyph sits on a clean 6x6 orange field.
#      The old solver can take the bbox of all non-orange pixels, crop a 3x3 mask,
#      and decode it directly.
#    - D7 alone breaks bbox-localization by adding one detached corner control pixel.
#      Example: a glyph at top-left (0,3) gets a marker at (4,1), so the bbox of all
#      non-orange cells is no longer 3x3; the solver must decode the corner marker to
#      recover the glyph location.
#    - D8 alone breaks the clean-background assumption by adding stray non-orange
#      pixels. A solver can still recover the glyph by scanning all 3x3 windows for
#      the unique monochrome contour mask.
#    - Fused version here makes both modifications interact:
#      1. one singleton control pixel still lives in the opposite 2x2 corner block,
#         so location must come from geometry rather than a global bbox;
#      2. distractors are split into two roles:
#         - at least one distractor lands inside the decoded 3x3 window but only on a
#           background cell of the contour, so the window is no longer monochrome;
#         - at least one more distractor sits elsewhere in the orange field.
#      Example trace:
#        real glyph X at rows 1-3, cols 0-2 in blue
#        marker at (5,4) in red -> decodes top-left (1,0)
#        patch noise at (2,0)? illegal because that would overwrite the glyph, so
#        generator instead picks one of the contour-false cells such as (1,1) or (3,1)
#        and paints it green.
#        Old base logic fails because all non-orange pixels no longer form one tight 3x3
#        object. D7-only logic fails because the decoded 3x3 patch contains extra colors.
#        D8-only scan fails because there is no monochrome glyph window anymore.
#        The fused solver succeeds by using the marker to choose the window, then
#        selecting the unique color inside that window whose support matches a legal glyph.
# b) Complexity calibration
#    - The scene still uses only one true glyph, one control pixel, and 2-4 singleton
#      distractors. This keeps the input legible on a 6x6 grid.
#    - At least one distractor is inside the decoded 3x3 patch and at least one is
#      outside it, so the visual difference from both single variants is visible.
#    - The corner-marker encoding still has only 16 legal positions and four corner
#      branches; the task stays small enough for ARC-style inference.
# c) Implicit assumption audit against the base solver
#    - `len(colors_present(g, exclude={bg})) == 1` no longer holds: glyph, marker, and
#      distractors may all use different colors.
#    - `bbox(g != bg)` no longer isolates the glyph because the marker and outside noise
#      enlarge the bbox.
#    - Even after decoding the marker, the old assumption that the 3x3 patch uses only
#      `{bg, fg}` is false, because background cells in that patch may contain noise.
#    - Therefore the solver must:
#      1. identify the unique non-orange corner-block marker,
#      2. decode its coordinates into a 3x3 window,
#      3. search colors inside that window for the unique legal contour mask.
# d) Placement legality
#    - The marker is always placed in the opposite 2x2 corner block, which is disjoint
#      from the 3x3 glyph window for every legal top-left position.
#    - In-patch distractors are sampled only from cells where the chosen glyph mask is
#      false, so they never overwrite the contour itself.
#    - Outside distractors are sampled outside both the 3x3 glyph bbox and all 2x2
#      corner blocks, so they never collide with the marker and remain visually separate.
# e) Visual observability
#    - Across examples, the only non-orange pixel in a corner block is visibly special
#      and acts as a locator. Inside the indicated 3x3 region, one repeated color still
#      traces a known contour while off-color singletons are irregular clutter.
#    - No hidden ordering by numeric color ids is used; all deductions are geometric or
#      based on visible monochrome-shape consistency.
# f) Visual cleanliness
#    - Noise is capped at 2-4 single pixels, not extra multi-pixel objects.
#    - Corner blocks are reserved for the marker, which keeps the location cue easy to
#      separate from the rest of the clutter.
# g) Change inferability
#    - Output colors remain the original black/orange/maroon trio, so the new difficulty
#      is entirely in parsing the input.
#    - The true glyph is identified by shape consistency within the marker-decoded 3x3
#      window, not by any fixed input color mapping.
# Post-change solver assumptions and how the generator enforces them
#    - Exactly one non-orange cell appears across the four 2x2 corner blocks.
#    - The decoded 3x3 patch contains exactly one color whose support equals one of the
#      four legal glyph masks.
#    - At least one distractor is inside that patch and at least one distractor is
#      outside the glyph bbox, so neither the D7-only nor the D8-only single-variant
#      strategy is sufficient.
#    - The generator validates these invariants and rejects samples where a naive
#      markerless 3x3 scan would still recover a unique monochrome glyph.


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
CTRL_BLOCK = 2
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
DISTRACTOR_COUNTS = [2, 3, 4]
MAX_RETRIES = 300

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


def _decode_marker_position(marker: tuple[int, int]) -> tuple[int, int]:
    """Map the detached corner marker back to the glyph top-left position."""
    r, c = map(int, marker)

    if 0 <= r < CTRL_BLOCK and 0 <= c < CTRL_BLOCK:
        return r + 2, c + 2
    if 0 <= r < CTRL_BLOCK and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r + 2, c - 4
    if GRID_H - CTRL_BLOCK <= r < GRID_H and 0 <= c < CTRL_BLOCK:
        return r - 4, c + 2
    if GRID_H - CTRL_BLOCK <= r < GRID_H and GRID_W - CTRL_BLOCK <= c < GRID_W:
        return r - 4, c - 4
    raise AssertionError("Control pixel must lie in a 2x2 corner block")


def _encode_marker_position(r0: int, c0: int) -> tuple[int, int]:
    """Inverse of `_decode_marker_position` used by the generator."""
    r0 = int(r0)
    c0 = int(c0)

    if r0 < 2 and c0 < 2:
        return r0 + 4, c0 + 4
    if r0 < 2 and c0 >= 2:
        return r0 + 4, c0 - 2
    if r0 >= 2 and c0 < 2:
        return r0 - 2, c0 + 4
    return r0 - 2, c0 - 2


def _corner_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask for the four 2x2 corner blocks."""
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[:CTRL_BLOCK, :CTRL_BLOCK] = True
    mask[:CTRL_BLOCK, w - CTRL_BLOCK :] = True
    mask[h - CTRL_BLOCK :, :CTRL_BLOCK] = True
    mask[h - CTRL_BLOCK :, w - CTRL_BLOCK :] = True
    return mask


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, int]:
    """Find the unique non-background control pixel reserved to the corner blocks."""
    corner_nonbg = np.argwhere((grid != bg) & _corner_mask(grid.shape))
    assert corner_nonbg.shape == (1, 2)
    return tuple(map(int, corner_nonbg[0]))


def _matching_glyph_masks(window: np.ndarray, bg: int) -> list[np.ndarray]:
    """Return all legal glyph masks induced by single colors in the 3x3 patch."""
    matches: list[np.ndarray] = []
    for col in colors_present(window, exclude={bg}):
        mask = window == int(col)
        if any(np.array_equal(mask, glyph) for glyph in GLYPH_MASKS):
            matches.append(mask.astype(bool))
    return matches


def _extract_motif(window: np.ndarray, bg: int) -> np.ndarray:
    """Recover the unique legal contour glyph from the noisy decoded 3x3 patch."""
    assert window.shape == (MOTIF_SIZE, MOTIF_SIZE)
    matches = _matching_glyph_masks(window, bg)
    assert len(matches) == 1
    return matches[0]


def _count_markerless_scan_candidates(grid: np.ndarray, bg: int) -> int:
    """Count D8-style monochrome 3x3 glyph windows; combo inputs force this to be non-unique."""
    candidates = 0
    h, w = grid.shape
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            window = grid[r0 : r0 + 3, c0 : c0 + 3]
            cols = colors_present(window, exclude={bg})
            if len(cols) != 1:
                continue
            mask = window == int(cols[0])
            if any(np.array_equal(mask, glyph) for glyph in GLYPH_MASKS):
                candidates += 1
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a contour glyph from a marker-addressed noisy 3x3 patch, then render the same 16x16 lattice and centered maroon stamp as the base task.

    Input structure:
    - The grid is 6x6 with orange as the dominant background.
    - Exactly one detached non-orange pixel appears in a 2x2 corner block and serves as a location marker for the glyph's 3x3 window.
    - The decoded 3x3 window contains the real monochrome contour glyph plus extra distractor pixels on some of its background cells, and more distractor pixels may appear elsewhere in the orange field.

    Rule:
    - Find the unique corner-block marker and decode its coordinates into the glyph's 3x3 location.
    - Inside that 3x3 patch, identify the unique color whose support matches one of the four legal contour glyph masks, ignoring the other noise pixels.
    - Decode that recovered glyph into the period-3 orange/black tile, repeat it over the 16x16 output, and stamp a 2x2 maroon tiling of the glyph in the centered 6x6 region.

    Key insight:
    - Neither the global non-orange bbox nor a markerless search for a clean monochrome 3x3 window is reliable anymore. The corner pixel tells you where to look, and the contour shape inside that addressed patch tells you which color is the true glyph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE

    marker = _find_marker(g, bg)
    r0, c0 = _decode_marker_position(marker)
    window = g[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    assert window.shape == (MOTIF_SIZE, MOTIF_SIZE)

    motif = _extract_motif(window, bg)
    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r_center = (out.shape[0] - center.shape[0]) // 2
    c_center = (out.shape[1] - center.shape[1]) // 2
    out[r_center : r_center + center.shape[0], c_center : c_center + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    fg = int(rng.choice(FG_COLORS))
    ctrl_choices = [c for c in FG_COLORS if int(c) != fg]
    ctrl = int(rng.choice(ctrl_choices))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))
    mr, mc = _encode_marker_position(r0, c0)

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg
    assert inp[mr, mc] == Color.ORANGE
    inp[mr, mc] = ctrl

    false_cells = [
        (r0 + dr, c0 + dc)
        for dr in range(MOTIF_SIZE)
        for dc in range(MOTIF_SIZE)
        if not bool(motif[dr, dc])
    ]
    distractor_total = int(rng.choice(DISTRACTOR_COUNTS))
    in_patch_count = int(rng.integers(1, min(len(false_cells), distractor_total - 1) + 1))
    out_patch_count = distractor_total - in_patch_count
    assert out_patch_count >= 1

    noise_pool = [c for c in FG_COLORS if int(c) not in {fg, ctrl}]
    rng.shuffle(noise_pool)
    noise_colors = noise_pool[:distractor_total]

    in_patch_pick = rng.choice(len(false_cells), size=in_patch_count, replace=False)
    for idx, color in zip(in_patch_pick, noise_colors[:in_patch_count]):
        rr, cc = false_cells[int(idx)]
        inp[rr, cc] = int(color)

    corner_cells = _corner_mask(inp.shape)
    outside_cells = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if not (r0 <= r < r0 + MOTIF_SIZE and c0 <= c < c0 + MOTIF_SIZE)
        and not bool(corner_cells[r, c])
    ]
    out_patch_pick = rng.choice(len(outside_cells), size=out_patch_count, replace=False)
    for idx, color in zip(out_patch_pick, noise_colors[in_patch_count:]):
        rr, cc = outside_cells[int(idx)]
        inp[rr, cc] = int(color)

    try:
        solved = main(inp)
    except Exception:
        return None

    window = inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    if len(_matching_glyph_masks(window, Color.ORANGE)) != 1:
        return None
    if _count_markerless_scan_candidates(inp, Color.ORANGE) != 0:
        return None

    corner_nonbg = int(np.count_nonzero((inp != Color.ORANGE) & _corner_mask(inp.shape)))
    if corner_nonbg != 1:
        return None

    assert solved.shape == (16, 16)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for combo variant eee78d87/8")
