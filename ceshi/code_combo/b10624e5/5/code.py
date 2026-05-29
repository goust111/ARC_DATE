# Combo variant of b10624e5 (index 5)
# combo_spec: D4/3+D11/0
# Name: 分形式缩放 + 迭代传播
#
# Design analysis for this combo variant
# -------------------------------------
# spec_like:
#   preserved_structure:
#     - Grid stays 27x27 with bg=4 and one full divider row+col of color 1 at index 13.
#     - There is still exactly one red square anchor per quadrant, and only the TL quadrant
#       contains a visible source motif in the input.
#     - The source motif is still attached to the TL red anchor and all non-TL quadrants start
#       from anchor-only inputs.
#   fused_rule_concretized:
#     - The visible TL patch is a 2x2 logical L-shape expanded into 2x2 solid blocks:
#           logical0 =
#             [2, A]
#             [B, 4]
#       where 2 is the anchor cell, A/B are two distinct non-bg colors, and 4 is background.
#     - Old task:
#         TL -> every target quadrant independently, using one direct scale+mirror from TL.
#     - New fused task:
#         round 1: TL -> TR. Mirror horizontally, then replace every foreground logical cell
#                  by a whole mirrored copy of the logical motif.
#         round 2: TR result -> BR. The entire logical result of round 1 becomes the new source,
#                  is mirrored vertically, and is fractal-substituted again.
#         round 3: BR result -> BL. The logical result of round 2 becomes the new source,
#                  is mirrored horizontally, and is fractal-substituted again.
#       Each round finally scales one logical cell to the target anchor size of that destination.
#   concrete_small_example:
#     - Start from logical0 = [[2, 7], [5, 4]].
#     - After TL -> TR:
#         reflect H => [[7, 2], [4, 5]]
#         fractal substitute => a 4x4 logical pattern whose anchor sits at its top-right corner.
#     - After TR -> BR:
#         the whole 4x4 TR logical pattern is reflected V and fractal-substituted again,
#         producing a 16x16 logical pattern anchored at the bottom-right corner.
#     - After BR -> BL:
#         the whole 16x16 BR logical pattern is reflected H and fractal-substituted again,
#         producing a 256x256 logical pattern anchored at the bottom-left corner.
#       In the 13x13 ARC viewport only a corner window is visible, which is exactly the intended
#       nested chain effect: later quadrants show deeper self-similarity than earlier ones.
#   complexity_calibration:
#     - A literal freeform source patch would explode visually after two chained fractal rounds.
#       Restricting the source to a 2x2 logical L-shape keeps the recursion interpretable.
#     - Non-TL anchor sizes are limited to {1, 2}. This preserves visible size cues without
#       making the deeper BR/BL windows so coarse that the nested structure disappears.
#     - BL always uses size 1 so the deepest stage still exposes several logical cells inside
#       the 13x13 viewport instead of turning into a few oversized blocks.
#   implicit_assumptions_checked:
#     - Original solver assumed each transformed patch contained a unique red region. That breaks
#       immediately after one fractal round because many red copies appear. The fix is to track an
#       explicit designated anchor cell through every reflection and substitution step.
#     - Original solver scaled the visible TL pixels directly. That is invalid here because later
#       rounds must use the previous logical result as the next source. The solver therefore
#       compresses TL back to its logical cells once, then performs the whole chain symbolically.
#     - Original solver treated all non-TL quadrants independently. That also breaks: BR now
#       depends on TR, and BL depends on BR, so a fixed propagation order is part of the rule.
#   placement_legality:
#     - TL's 4x4 source patch is always placed fully inside the TL quadrant.
#     - TR/BR/BL anchors are pushed toward the outer corners that match the propagated anchor
#       corner (TR top-right, BR bottom-right, BL bottom-left). This ensures the clipped window
#       reveals the correct local part of the huge recursive patch and prevents cross-quadrant overlap.
#     - Stamping still skips divider cells and preserves existing red anchor pixels.
#   observability_and_cleanliness:
#     - The three destination quadrants are no longer parallel copies: they show three different
#       recursion depths, which is visible even when anchor sizes match.
#     - Using only two non-red colors keeps the repeated structure legible. The missing bottom-right
#       cell in the base L-shape guarantees obvious gaps, so the result does not resemble plain scaling.
#     - Anchors remain isolated and the recursive windows live in separate quadrants, avoiding clutter.
#   new_solver_assumptions_audited:
#     - TL motif must be tiled by uniform 2x2 blocks and contain exactly one designated red cell.
#       The generator guarantees this by constructing the source from a fixed 2x2 logical template.
#     - The propagation order is TL -> TR -> BR -> BL. The generator matches that order exactly and
#       only places one anchor per quadrant, so quadrant identification is unambiguous.
#     - Later source states are represented logically rather than re-read from the rasterized output.
#       This is intentional: it is the only robust way to preserve the designated anchor through
#       multi-red fractal stages while still matching the fused rule.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_CHOICES = [1, 2]
MAX_RETRIES = 300
ANCHOR_JITTER = 2
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
CHAIN = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0))]


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _reflect_state(
    logical: np.ndarray,
    anchor_cell: tuple[int, int],
    flip_ud: bool,
    flip_lr: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect a logical source and move its designated anchor cell with it."""
    out = logical
    ar, ac = anchor_cell
    h, w = logical.shape
    if flip_lr:
        out = np.fliplr(out)
        ac = w - 1 - ac
    if flip_ud:
        out = np.flipud(out)
        ar = h - 1 - ar
    return out, (int(ar), int(ac))


def _compress_uniform_blocks(
    patch: np.ndarray,
    block: int,
    anchor_r_px: int,
    anchor_c_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Recover a logical grid from a patch built from solid block x block cells.

    The explicit anchor is passed in pixel coordinates and converted to a logical cell.
    """
    h, w = patch.shape
    assert h % block == 0 and w % block == 0, "source patch must be tiled by uniform blocks"
    logical = np.empty((h // block, w // block), dtype=int)
    for r in range(logical.shape[0]):
        for c in range(logical.shape[1]):
            cell = patch[r * block : (r + 1) * block, c * block : (c + 1) * block]
            vals = np.unique(cell)
            assert vals.size == 1, "each source logical cell must be a solid block"
            logical[r, c] = int(vals[0])
    assert anchor_r_px % block == 0 and anchor_c_px % block == 0, "anchor must align to logical cell boundaries"
    return logical, (int(anchor_r_px // block), int(anchor_c_px // block))


def _fractal_substitute(
    logical: np.ndarray,
    anchor_cell: tuple[int, int],
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Replace every non-bg logical cell with a whole copy of the logical source.

    The designated anchor in the new logical grid is the anchor inside the outer copy
    placed at the previous designated anchor cell.
    """
    h, w = logical.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        rr = slice(r * h, (r + 1) * h)
        for c in range(w):
            if logical[r, c] == bg:
                continue
            cc = slice(c * w, (c + 1) * w)
            out[rr, cc] = logical
    ar, ac = anchor_cell
    return out, (int(ar * h + ar), int(ac * w + ac))


def _scale_logical(logical: np.ndarray, cell_size: int) -> np.ndarray:
    """Render a logical grid as square cell_size x cell_size blocks."""
    assert cell_size > 0, "cell size must be positive"
    return np.repeat(np.repeat(logical, cell_size, axis=0), cell_size, axis=1)


def _stamp(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    anchor_r_in_patch: int,
    anchor_c_in_patch: int,
    mid_r: int,
    mid_c: int,
    bg: int,
    quad_bounds: tuple[int, int, int, int],
) -> None:
    """Stamp non-bg pixels from a transformed patch, aligned by its designated anchor."""
    place_r0 = int(anchor_r - anchor_r_in_patch)
    place_c0 = int(anchor_c - anchor_c_in_patch)
    src_rr, src_cc = np.where(patch_s != bg)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    q_r0, q_r1, q_c0, q_c1 = quad_bounds
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < out.shape[0])
        & (tgt_cc >= 0)
        & (tgt_cc < out.shape[1])
        & (tgt_rr >= q_r0)
        & (tgt_rr <= q_r1)
        & (tgt_cc >= q_c0)
        & (tgt_cc <= q_c1)
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _extract_source_patch(
    g: np.ndarray,
    mid_r: int,
    mid_c: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract the TL visible motif patch and the top-left pixel of its explicit red anchor block."""
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = bbox_coords(np.asarray(pts))
            break
    assert patch_bb is not None, "top-left quadrant must contain a motif component with the red anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size == SOURCE_ANCHOR_SIZE * SOURCE_ANCHOR_SIZE, "TL source must contain exactly one 2x2 red anchor"
    anchor_r_px = int(rr_red.min())
    anchor_c_px = int(cc_red.min())
    assert rr_red.max() - rr_red.min() + 1 == SOURCE_ANCHOR_SIZE
    assert cc_red.max() - cc_red.min() + 1 == SOURCE_ANCHOR_SIZE
    return patch, (anchor_r_px, anchor_c_px)


def _read_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Return one square red anchor per quadrant as {(qb, qr): (r0, c0, size)}."""
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size_h = r1 - r0 + 1
        size_w = c1 - c0 + 1
        assert size_h == size_w, "anchor blocks must be square"
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size_h))
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "must have one red anchor in every quadrant"
    assert anchors[(0, 0)][2] == SOURCE_ANCHOR_SIZE, "top-left red anchor must remain 2x2"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the top-left motif around the quadrants as a chained sequence of reflected fractal copies.

    Input structure:
    - A single non-background divider row and column split the 27x27 grid into four 13x13 quadrants.
    - Every quadrant contains one red square anchor block; the TL anchor is always 2x2.
    - Only the TL quadrant shows a colored motif, and that motif is drawn as uniform 2x2 logical cells.

    Rule:
    - Compress the TL patch back to its logical grid and keep track of which logical red cell is the explicit anchor.
    - Follow a fixed propagation chain: TL -> TR -> BR -> BL. At each step, reflect the current logical source
      according to the source/target quadrant relation, then replace every foreground logical cell with a whole copy
      of that reflected logical source.
    - Render the new logical result at the destination anchor size, align the propagated anchor copy to the
      destination red block, and stamp all non-background pixels while preserving divider cells and existing anchors.

    Key insight: The destination quadrants are no longer independent. Each new quadrant uses the previous quadrant's
    whole logical result as its source, so recursion depth accumulates along the chain while anchor size only sets
    the visible cell size of the current stage.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _read_anchors(g, mid_r, mid_c)

    patch, (anchor_r_px, anchor_c_px) = _extract_source_patch(g, mid_r, mid_c, bg)
    logical, anchor_cell = _compress_uniform_blocks(
        patch,
        SOURCE_ANCHOR_SIZE,
        anchor_r_px,
        anchor_c_px,
    )
    assert logical.shape == (2, 2), "the fused variant expects a 2x2 logical TL motif"
    assert logical[anchor_cell] == RED, "the designated TL anchor cell must be red"

    current_logical = logical
    current_anchor_cell = anchor_cell
    current_quad = (0, 0)
    for src_quad, dst_quad in CHAIN:
        assert src_quad == current_quad, "propagation order must be TL -> TR -> BR -> BL"
        flip_ud = src_quad[0] != dst_quad[0]
        flip_lr = src_quad[1] != dst_quad[1]
        reflected, reflected_anchor = _reflect_state(current_logical, current_anchor_cell, flip_ud, flip_lr)
        current_logical, current_anchor_cell = _fractal_substitute(reflected, reflected_anchor, bg)

        tr0, tc0, target_size = anchors[dst_quad]
        patch_s = _scale_logical(current_logical, target_size)
        ar0 = current_anchor_cell[0] * target_size
        ac0 = current_anchor_cell[1] * target_size
        _stamp(out, patch_s, tr0, tc0, ar0, ac0, mid_r, mid_c, bg, QUADS[dst_quad])
        current_quad = dst_quad

    return out


def _make_logical_patch() -> np.ndarray:
    """Create the fixed 2x2 L-shape source motif with two distinct non-red colors."""
    c1, c2 = random.sample(COLOR_POOL, 2)
    return np.array([[RED, c1], [c2, BG]], dtype=int)


def _choose_target_sizes() -> dict[tuple[int, int], int]:
    """Later rounds are kept small so deep recursion remains visible inside a 13x13 quadrant."""
    return {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.choice(TARGET_SIZE_CHOICES),
        (1, 1): random.choice(TARGET_SIZE_CHOICES),
        (1, 0): 1,
    }


def _choose_corner_anchor(qb: int, qr: int, size: int) -> tuple[int, int]:
    """Place a target anchor near the outer corner that matches the propagated anchor corner."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    if (qb, qr) == (0, 1):
        r0 = random.randint(q_r0, min(q_r0 + ANCHOR_JITTER, q_r1 - size + 1))
        c0 = random.randint(max(q_c0, q_c1 - size + 1 - ANCHOR_JITTER), q_c1 - size + 1)
        return r0, c0
    if (qb, qr) == (1, 1):
        r0 = random.randint(max(q_r0, q_r1 - size + 1 - ANCHOR_JITTER), q_r1 - size + 1)
        c0 = random.randint(max(q_c0, q_c1 - size + 1 - ANCHOR_JITTER), q_c1 - size + 1)
        return r0, c0
    if (qb, qr) == (1, 0):
        r0 = random.randint(max(q_r0, q_r1 - size + 1 - ANCHOR_JITTER), q_r1 - size + 1)
        c0 = random.randint(q_c0, min(q_c0 + ANCHOR_JITTER, q_c1 - size + 1))
        return r0, c0
    raise ValueError("corner anchor placement is only used for non-TL quadrants")


def _build_chain_states(
    logical0: np.ndarray,
    anchor0: tuple[int, int],
    bg: int,
) -> dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]]:
    """Return the logical state for each destination quadrant in propagation order."""
    states: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]] = {}
    current_logical = logical0
    current_anchor = anchor0
    current_quad = (0, 0)
    for src_quad, dst_quad in CHAIN:
        assert src_quad == current_quad
        flip_ud = src_quad[0] != dst_quad[0]
        flip_lr = src_quad[1] != dst_quad[1]
        reflected, reflected_anchor = _reflect_state(current_logical, current_anchor, flip_ud, flip_lr)
        current_logical, current_anchor = _fractal_substitute(reflected, reflected_anchor, bg)
        states[dst_quad] = (current_logical, current_anchor)
        current_quad = dst_quad
    return states


def _gen_attempt() -> np.ndarray | None:
    logical = _make_logical_patch()
    source_patch = _scale_logical(logical, SOURCE_ANCHOR_SIZE)
    source_anchor_cell = (0, 0)
    source_anchor_px = (0, 0)

    states = _build_chain_states(logical, source_anchor_cell, BG)
    target_sizes = _choose_target_sizes()

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - source_patch.shape[0])
    tl_c0 = random.randint(0, MID - source_patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + source_anchor_px[0], tl_c0 + source_anchor_px[1])
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        anchors[key] = _choose_corner_anchor(key[0], key[1], size)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, source_patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)

    for key, (logical_state, anchor_cell) in states.items():
        size = target_sizes[key]
        patch_s = _scale_logical(logical_state, size)
        ar0 = anchor_cell[0] * size
        ac0 = anchor_cell[1] * size
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], ar0, ac0, MID, MID, BG, QUADS[key])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
