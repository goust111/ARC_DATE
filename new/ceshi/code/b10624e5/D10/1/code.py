# Variant of b10624e5
# Dimension: D10 — 锚块颜色的角色 (meta)
# Change: 锚块统一为红色，颜色仅用于标识锚块身份，不编码其他信息
#   -> 锚块颜色决定输出图案颜色；目标象限中的复制图案整体重着色为该象限锚块颜色

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D10 variant
# ----------------------------------
# spec:
#   grid:
#     keep the original 27x27 layout, bg=4, divider color=1, divider crossing at (13,13)
#   anchors:
#     still exactly one solid NxN anchor square per quadrant
#     TL anchor size stays 2x2
#     non-TL anchor sizes stay in 1..4
#     change only the role of anchor color:
#       original task: all anchors are red and color carries no extra meaning
#       variant: each anchor has its own color and each target motif is repainted
#       entirely to that target anchor color
#   source motif:
#     still a single 4-connected non-bg component in TL containing the TL anchor
#     still scaled by target_size / source_size and reflected by quadrant
#     source motif remains multi-color so the recoloring effect is visually obvious
#
# a) concretization and behavior trace:
#   - Input:
#       TL shows a multi-color motif attached to a 2x2 anchor square.
#       TR/BL/BR each contain only one colored square anchor, and these anchors may
#       have different colors.
#   - Output:
#       The TL motif is copied to the other quadrants with the same reflection+scale
#       logic as before, but every copied non-bg pixel takes the target anchor color.
#   - Small trace:
#       Suppose the TL patch has colors {pink, teal, green, TL-anchor=orange}.
#       In the old task, the TR copy kept {pink, teal, green, red}.
#       In this variant, if the TR anchor is blue, then every visible TR motif pixel,
#       including the aligned anchor region, becomes blue. Geometry is preserved;
#       only color handling changes.
#
# b) complexity calibration:
#   - The variant needs more than "different anchor colors exist"; the recoloring must
#     be inferable. The generator therefore forces the three target anchors to use
#     distinct colors and keeps the TL motif multi-color.
#   - The color rule is still low-complexity: one local cause, one local effect.
#     No hidden ordering or palette arithmetic is introduced.
#
# c) hidden-assumption review against the original solver:
#   - Original code hard-coded color 2 for anchor detection, source extraction,
#     scaled-anchor alignment, and "do not overwrite red" stamping.
#   - Those assumptions are invalid now because anchor colors vary by quadrant.
#   - New solver assumptions:
#       1. Each anchor color is used only on its anchor square.
#       2. Non-anchor motif colors do not themselves form a single filled square
#          component, so anchor discovery is unambiguous.
#       3. The TL anchor color appears inside the TL source component.
#     The generator constructs inputs to satisfy all three.
#
# d) placement legality:
#   - Non-TL quadrants are still empty except for the anchor, so copied motifs cannot
#     clash with unrelated objects.
#   - Anchor placement still prefers full-fit positions, with clipping tolerated only
#     at borders/dividers exactly as in the original task.
#   - Source colors avoid divider/bg and avoid reusing anchor colors, which keeps
#     anchor identification stable.
#
# e) visual observability:
#   - The TL motif is intentionally multi-color.
#   - The target anchors are intentionally different colors in the same example.
#   - The source shape is required to be non-trivially asymmetric, so at least one
#     reflection is visually meaningful even after monochrome repainting.
#
# f) visual cleanliness:
#   - Each target quadrant contains exactly one anchor and one copied motif.
#   - Because the copied motif becomes monochrome per quadrant, the output is easier
#     to parse than a cluttered multi-color overlay.
#
# g) color-change inferability:
#   - The recoloring rule is local and repeated three times per sample:
#       anchor color in quadrant -> copied motif color in that quadrant.
#   - No invisible color-id ordering is used. Only visible spatial grouping
#     (quadrant anchor paired with adjacent copied motif) explains the color change.


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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting the original task's fractional scales."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _find_anchor_blocks(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """
    Detect one monochrome square anchor per quadrant.

    Returns:
      {(qb, qr): (r0, c0, size, color)}

    Generator guarantees that anchor colors are not reused elsewhere and that no
    non-anchor motif color forms a single filled square component.
    """
    div = int(g[mid_r, mid_c])
    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, g.flatten())) - {bg, div}):
        mask = g == color
        comps = components_mask(mask, connectivity=4)
        if len(comps) != 1:
            continue
        pts = comps[0]
        r0, c0, r1, c1 = bbox_coords(pts)
        if r0 <= mid_r <= r1 or c0 <= mid_c <= c1:
            continue
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        if size_r != size_c:
            continue
        if len(pts) != size_r * size_c:
            continue
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one anchor block"
        anchors[key] = (int(r0), int(c0), int(size_r), int(color))
    assert len(anchors) == 4, "grid must contain exactly four anchor blocks"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif across quadrants, but recolor each copy from its local anchor.

    Input structure:
    - The grid is split into four quadrants by one full-row/full-column divider of a
      single non-background color.
    - Each quadrant contains one solid square anchor block; anchor sizes vary by quadrant
      and anchor colors are allowed to differ.
    - The top-left quadrant also contains one multi-color motif attached to its 2x2 anchor.

    Rule:
    - Extract the full TL motif component that contains the TL anchor.
    - For every other quadrant, reflect the motif according to quadrant position and scale
      it by the ratio of target-anchor size to TL-anchor size.
    - Align the scaled anchor region onto the target anchor, then paint every copied
      non-background pixel using the target anchor's color instead of the source colors.

    Key insight:
    - Anchor size still determines scale and quadrant still determines reflection, but
      anchor color now determines the copied motif's final color in that quadrant.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchor_blocks(g, bg, mid_r, mid_c)

    assert (0, 0) in anchors, "top-left quadrant must contain an anchor block"
    _sr0, _sc0, sL, source_anchor_color = anchors[(0, 0)]
    assert sL > 0, "top-left anchor must have positive size"

    # Extract the TL non-bg component that contains the TL anchor color.
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(int(tl[r, c]) == source_anchor_color for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with the anchor color"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    # Paste a scaled/reflected version of the TL patch in every non-TL quadrant,
    # recoloring the entire copied motif to that quadrant's anchor color.
    for (qb, qr), (tr0, tc0, tL, target_color) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_anchor, cc_anchor = np.where(patch_s == source_anchor_color)
        assert rr_anchor.size > 0, "scaled source patch must contain anchor pixels"
        place_r0 = tr0 - int(rr_anchor.min())
        place_c0 = tc0 - int(cc_anchor.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            out[rr_t, cc_t] = target_color

    return out


# --------------- generator (output-first D10 recoloring variant) ---------------

# --- Grid constants ---
BG = 4
DIV = 1
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# --- Anchor / palette parameters ---
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
ANCHOR_COLOR_POOL = [0, 2, 3, 5, 6, 7, 8, 9]

# --- Source motif parameters ---
LOGICAL_DIMS = [(2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.7
FILL_RATIO_MAX = 1.0
SOURCE_EXTRA_COLORS_MIN = 2
SOURCE_EXTRA_COLORS_MAX = 4

# --- Generation retry limit ---
MAX_RETRIES = 400


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _is_square_component(arr: np.ndarray, color: int) -> bool:
    """True iff all pixels of `color` form one filled square component."""
    comps = components_mask(arr == color, connectivity=4)
    if len(comps) != 1:
        return False
    pts = comps[0]
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and len(pts) == h * w


def _assign_logical_colors(
    filled: set[tuple[int, int]],
    anchor_cell: tuple[int, int],
    h: int,
    w: int,
    anchor_color: int,
    extra_palette: list[int],
) -> np.ndarray | None:
    """Color a logical motif while preventing non-anchor colors from masquerading as anchors."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_anchor = [cell for cell in filled if cell != anchor_cell]
    max_extra = min(SOURCE_EXTRA_COLORS_MAX, len(extra_palette), len(non_anchor) // 2)
    if max_extra < SOURCE_EXTRA_COLORS_MIN:
        return None

    for _ in range(60):
        n_extra = random.randint(SOURCE_EXTRA_COLORS_MIN, max_extra)
        counts = [2] * n_extra
        leftover = len(non_anchor) - 2 * n_extra
        while leftover > 0:
            counts[random.randrange(n_extra)] += 1
            leftover -= 1

        random.shuffle(non_anchor)
        random.shuffle(extra_palette)
        logical = np.full((h, w), BG, dtype=int)
        logical[anchor_cell] = anchor_color

        idx = 0
        used_colors = extra_palette[:n_extra]
        for color, count in zip(used_colors, counts, strict=False):
            for cell in non_anchor[idx : idx + count]:
                logical[cell] = color
            idx += count

        patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
        if any(_is_square_component(patch, color) for color in used_colors):
            continue
        mask = logical != BG
        if np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
            continue
        return logical
    return None


def _source_patch(anchor_color: int, forbidden_colors: set[int]) -> tuple[np.ndarray, int, int] | None:
    """Build a TL motif patch as a 2x-upscaled logical pattern."""
    extra_palette = [c for c in ANCHOR_COLOR_POOL if c not in forbidden_colors]
    if len(extra_palette) < SOURCE_EXTRA_COLORS_MIN:
        return None

    lh, lw = random.choice(LOGICAL_DIMS)
    n_cells = lh * lw
    min_fill = max(5, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))

    logical = _assign_logical_colors(filled, (ar, ac), lh, lw, anchor_color, extra_palette[:])
    if logical is None:
        return None

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Reflect and scale the source patch for a given quadrant and target anchor size."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    source_anchor_color: int,
) -> tuple[int, int]:
    """Pick an anchor position, preferring full-fit placements for the whole scaled motif."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == source_anchor_color)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp_recolored(
    out: np.ndarray,
    patch_s: np.ndarray,
    source_anchor_color: int,
    target_color: int,
    anchor_r: int,
    anchor_c: int,
) -> None:
    """Stamp the scaled patch, recoloring every non-bg pixel to the target anchor color."""
    rr_anchor, cc_anchor = np.where(patch_s == source_anchor_color)
    place_r0 = anchor_r - int(rr_anchor.min())
    place_c0 = anchor_c - int(cc_anchor.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0) & (tgt_rr < GRID_SIZE) &
        (tgt_cc >= 0) & (tgt_cc < GRID_SIZE) &
        (tgt_rr != MID) & (tgt_cc != MID)
    )
    out[tgt_rr[valid], tgt_cc[valid]] = target_color


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    anchor_colors = random.sample(ANCHOR_COLOR_POOL, 4)
    anchor_by_quad = {
        (0, 0): anchor_colors[0],
        (0, 1): anchor_colors[1],
        (1, 0): anchor_colors[2],
        (1, 1): anchor_colors[3],
    }

    source = _source_patch(anchor_by_quad[(0, 0)], set(anchor_colors))
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, anchor_by_quad[(0, 0)])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), anchor_by_quad[key], dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp_recolored(out, patch_s, anchor_by_quad[(0, 0)], anchor_by_quad[key], *anchors[key])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), anchor_by_quad[key], dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D10 input")
