# Variant of b10624e5
# Dimension: D4 — 变换：缩放规则 (transform)
# Change: 缩放因子 = 目标红色锚块尺寸 / 源红色锚块尺寸，图案每个像素按比例放大为 factor x factor 的色块... → 分形式缩放：图案中的每个像素递归替换为整个图案

# Design analysis for this D4 variant
# -----------------------------------
# spec_like:
#   preserved_structure:
#     - 27x27 grid, background 4, full divider row+col of one non-bg color at index 13.
#     - Exactly one red square anchor per quadrant; TL still provides the only visible source motif.
#     - Right quadrants still mirror left-right; bottom quadrants still mirror up-down.
#   core_D4_change:
#     - The old task resized the TL motif directly with nearest-neighbor interpolation.
#     - The new task first compresses the TL motif into a 2x2 logical pattern whose cells are 2x2 blocks.
#     - In each target quadrant, every non-bg logical cell is replaced by one full reflected copy
#       of that logical pattern, scaled so one logical cell becomes a target_anchor_size x target_anchor_size block.
#     - This yields a one-step self-similar tiling. It is "recursive" in the ARC sense because the
#       whole motif reappears inside each active cell, but we stop after one explicit substitution layer
#       so the result stays readable inside a 13x13 quadrant.
#   concrete_pixel_behavior:
#     - Source motif: a 2x2 logical L-shape with exactly 3 foreground cells, one of them red.
#     - Visible TL patch: each logical cell is shown as a 2x2 solid block, so the TL red anchor is still 2x2.
#     - Example:
#         logical =
#           [2, 7]
#           [5, 4(bg)]
#       old scaling to anchor size 3:
#         would make a 6x6 blocky enlargement of the same 2x2 arrangement.
#       new fractal tiling to anchor size 3:
#         each of the three foreground cells becomes a full 6x6 copy of the reflected logical motif,
#         so the target quadrant shows three copies arranged in the same L-shape, each copy itself
#         containing the original color layout.
#   complexity_calibration:
#     - A larger logical motif (for example 3x3) would produce h^2 x w^2 growth after one substitution
#       and become excessively clipped. Restricting the logical motif to 2x2 keeps target outputs at
#       sizes 4, 8, and 12 when target anchors are 1, 2, and 3, respectively.
#     - Using exactly three non-TL anchor sizes {1,2,3} in each sample ensures the control variable
#       is repeated and visually comparable across quadrants.
#   implicit_assumptions_checked:
#     - Original solver assumed any scaled patch still had one red bbox aligned to the target anchor.
#       That fails here because the fractal motif contains many red copies. The new solver instead aligns
#       a canonical red copy: the one inside the outer copy placed at the source red logical cell.
#     - Original solver used arbitrary nearest-neighbor scaling on the visible patch. That is invalid
#       here because the visible TL patch is only an encoded 2x2-block representation of the true logical motif.
#       The new solver explicitly compresses 2x2 uniform blocks back to a logical grid before transforming.
#     - Original generator allowed larger freeform patches; that would explode under fractal tiling.
#       The new generator guarantees a tiny block-coherent source patch.
#   placement_legality:
#     - Target fractal patches are chosen so their full bounding boxes fit their quadrants.
#     - Anchors are sampled from ranges that satisfy both anchor-square fit and whole-fractal fit.
#     - Stamping still skips divider cells and existing red cells, so the explicit target anchor remains intact.
#   observability_and_cleanliness:
#     - The L-shaped 2x2 logical motif guarantees visible holes/gaps, so the result is not mistaken for
#       plain enlargement.
#     - Distinct non-red colors make reflections and repeated copies easy to see.
#     - Only one substitution layer is used; more layers would overcrowd the 13x13 quadrants and obscure the rule.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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


def _reflect(img: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = img
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _compress_uniform_blocks(patch: np.ndarray, block: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Recover the logical source motif from a visible patch made of uniform block x block cells.

    Returns the logical grid and the logical coordinates of the unique red cell.
    """
    h, w = patch.shape
    assert h % block == 0 and w % block == 0, "source patch must be tiled by uniform blocks"
    logical = np.empty((h // block, w // block), dtype=int)
    for r in range(logical.shape[0]):
        for c in range(logical.shape[1]):
            cell = patch[r * block : (r + 1) * block, c * block : (c + 1) * block]
            vals = np.unique(cell)
            assert vals.size == 1, "each source cell must be a solid block"
            logical[r, c] = int(vals[0])
    red_cells = np.argwhere(logical == RED)
    assert red_cells.shape == (1, 2), "logical source motif must contain exactly one red cell"
    ar, ac = map(int, red_cells[0])
    return logical, (ar, ac)


def _fractal_patch(logical: np.ndarray, cell_size: int, bg: int) -> tuple[np.ndarray, int, int]:
    """
    Replace every non-bg logical cell by one full scaled copy of the logical motif.

    The returned anchor is the canonical red copy inside the outer copy positioned at the
    source red logical cell.
    """
    assert cell_size > 0, "cell size must be positive"
    h, w = logical.shape
    base = np.repeat(np.repeat(logical, cell_size, axis=0), cell_size, axis=1)
    tile_h, tile_w = base.shape
    out = np.full((h * tile_h, w * tile_w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if logical[r, c] == bg:
                continue
            paste(out, base, r * tile_h, c * tile_w)
    red_cells = np.argwhere(logical == RED)
    assert red_cells.shape == (1, 2), "logical motif must contain exactly one red cell"
    ar, ac = map(int, red_cells[0])
    anchor_r = ar * tile_h + ar * cell_size
    anchor_c = ac * tile_w + ac * cell_size
    return out, int(anchor_r), int(anchor_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif across quadrants using reflected fractal tiling keyed to red anchors.

    Input structure:
    - A full non-background divider row and column split the grid into four quadrants.
    - Each quadrant contains one red square anchor block; the top-left anchor is still 2x2.
    - The top-left quadrant contains the only visible colored motif, drawn as solid 2x2 blocks.

    Rule:
    - Recover the top-left motif as a small logical grid by compressing those 2x2 blocks.
    - Reflect that logical motif for the destination quadrant, then replace each of its foreground cells
      with a full scaled copy of the logical motif, where one logical cell has the size of the target red anchor.
    - Align the canonical red copy in that self-similar patch to the quadrant's red anchor and stamp all non-bg pixels.

    Key insight: The quadrant still determines mirror direction, but anchor size now controls the cell size of a
    repeated whole-motif tiling rather than ordinary pixel enlargement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

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
    _, _, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left red anchor must remain 2x2"

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

    logical, _ = _compress_uniform_blocks(patch, SOURCE_ANCHOR_SIZE)

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        logical_t = _reflect(logical, qb, qr)
        patch_t, ar0, ac0 = _fractal_patch(logical_t, tL, bg)
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_t != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_t[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

    return out


# --------------- generator (fractal_tiling) ---------------

BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZES = [1, 2, 3]
MAX_RETRIES = 300
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}


def _make_logical_patch() -> np.ndarray:
    """Return a 2x2 logical L-shape with one red cell and two distinct extra colors."""
    logical = np.full((2, 2), BG, dtype=int)
    missing_idx = random.randrange(4)
    cells = [(r, c) for r in range(2) for c in range(2) if r * 2 + c != missing_idx]
    red_cell = random.choice(cells)
    extra_cells = [cell for cell in cells if cell != red_cell]
    colors = random.sample(COLOR_POOL, 2)
    logical[red_cell] = RED
    logical[extra_cells[0]] = colors[0]
    logical[extra_cells[1]] = colors[1]
    return logical


def _source_patch(logical: np.ndarray) -> tuple[np.ndarray, int, int]:
    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    red_cells = np.argwhere(logical == RED)
    ar, ac = map(int, red_cells[0])
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    anchor_r_in_patch: int,
    anchor_c_in_patch: int,
) -> tuple[int, int]:
    """Pick an anchor location so the full fractal patch stays inside the quadrant whenever possible."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    fit_r0 = q_r0 + anchor_r_in_patch
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + anchor_r_in_patch
    fit_c0 = q_c0 + anchor_c_in_patch
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + anchor_c_in_patch

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, ar0: int, ac0: int) -> None:
    """Stamp non-bg pixels from a fractal patch, aligned by its canonical anchor."""
    place_r0 = anchor_r - ar0
    place_c0 = anchor_c - ac0
    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    logical = _make_logical_patch()
    patch, anchor_r_in_patch, anchor_c_in_patch = _source_patch(logical)

    perm = random.sample(TARGET_SIZES, len(TARGET_SIZES))
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): perm[0],
        (1, 0): perm[1],
        (1, 1): perm[2],
    }

    fractals: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        logical_t = _reflect(logical, key[0], key[1])
        fractals[key] = _fractal_patch(logical_t, size, BG)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, (patch_s, ar0, ac0) in fractals.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, ar0, ac0)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, (patch_s, ar0, ac0) in fractals.items():
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], ar0, ac0)

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
    raise RuntimeError("failed to generate a valid b10624e5 D4 fractal-tiling input")
