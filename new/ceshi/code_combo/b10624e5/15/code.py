# Combo variant of b10624e5 (index 15)
# combo_spec: D6/1+D10/1
# Name: 锚块颜色决定输出图案颜色 + 源图案被重新着色

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D6/1 + D10/1 on b10624e5
# ----------------------------------------------------
# spec anchor:
#   - Keep the original 27x27 grid, one full divider row and one full divider col at
#     the same non-bg color, splitting the canvas into four 13x13 quadrants.
#   - Keep one solid square anchor block per quadrant.
#   - Keep the TL motif as a single 4-connected non-bg component containing the TL anchor.
#   - Keep the original geometry rule: copy TL motif to TR/BL/BR with quadrant-based
#     reflection and anchor-size-based nearest-neighbor scaling.
#
# fused color rule:
#   - D10/1: every non-TL copied motif is recolored to that quadrant's own anchor color.
#   - D6/1: the TL source motif is not preserved verbatim in the output; it is recolored too.
#   - To make the source recolor structural rather than arbitrary, TL is recolored to the
#     TR anchor color. This uses a visible quadrant relation: the top-row pair share color,
#     while BL and BR each use their own local anchor colors.
#
# a) concretization and behavior trace:
#   - Input:
#       TL contains a multi-color component with a 2x2 anchor square.
#       TR/BL/BR each contain only a colored square anchor, possibly in different colors.
#   - Old task:
#       TL stayed unchanged; TR/BL/BR received reflected+scaled copies that kept the
#       source colors.
#   - Fused task:
#       TL is repainted to one uniform color: the TR anchor color.
#       TR is reflected/scaled and painted uniformly with the TR anchor color.
#       BL and BR are reflected/scaled and painted uniformly with their own anchor colors.
#   - Small trace:
#       Suppose TL input uses {orange anchor, teal, pink, maroon}; TR anchor is green,
#       BL anchor is blue, BR anchor is red.
#       Output TL becomes all green on its non-bg cells, TR copy is all green,
#       BL copy is all blue, BR copy is all red.
#       Geometry is still governed by size+reflection; color is now governed by anchors,
#       with a special top-row linkage for the source.
#
# b) complexity calibration:
#   - If source recolor were arbitrary per sample, humans could not infer it from 2-3 pairs.
#     Fixing it to the TR anchor color keeps the rule low-entropy and spatially grounded.
#   - To make D10 visible, all four anchor colors are distinct in each generated sample.
#   - To make D6 visible, TL input stays multi-color and its output recolor must differ from
#     its input anchor color. The generator enforces both.
#
# c) hidden-assumption review:
#   - Original solver assumed anchors were always color 2. Broken: anchors now vary in color.
#     New solver therefore detects anchors by geometry: one monochrome filled square component
#     per quadrant, with colors unused elsewhere in the input.
#   - Original solver identified the source component by "contains red". Broken: source anchor
#     is no longer red. New solver identifies the TL component by the TL anchor color.
#   - Original target stamping preserved source colors and skipped existing red pixels.
#     Broken: copied targets must overwrite using target-anchor color, and TL must be repainted.
#   - New assumption introduced: TR anchor exists and provides the source recolor color.
#     Generator guarantees one anchor in every quadrant.
#
# d) placement legality:
#   - The generator still places the TL patch wholly within TL.
#   - Non-TL quadrants remain empty except for anchors in the input, so stamping cannot
#     collide with unrelated objects.
#   - Anchor placement prefers locations where the full scaled motif fits inside the quadrant,
#     while still allowing border clipping exactly like the base task.
#
# e) visual observability:
#   - TL input is multi-color, but TL output becomes monochrome; that makes D6 plainly visible.
#   - The three target anchors use different colors, so D10 is repeated three times per sample.
#   - The source shape is forced to be non-trivially asymmetric, so reflection remains visible
#     after monochrome recoloring.
#
# f) visual cleanliness:
#   - Each quadrant still contains one anchor-centered motif only.
#   - Recoloring whole motifs to one color per quadrant reduces clutter compared with keeping
#     multi-color copies everywhere.
#   - Source extra colors avoid anchor colors, preventing accidental fake anchors.
#
# g) change inferability:
#   - Color changes are explained only by visible structure: local anchor color for targets,
#     and the fixed top-row partner anchor for the source.
#   - The generator rejects degenerate samples where the fused output matches a D10-only
#     ablation or where target recoloring would be invisible on the same input.


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
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
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _find_anchor_blocks(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """
    Detect one monochrome filled square anchor per quadrant.

    Returns:
      {(qb, qr): (r0, c0, size, color)}
    """
    div = int(g[mid_r, mid_c])
    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, g.flatten())) - {bg, div}):
        comps = components_mask(g == color, connectivity=4)
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


def _extract_source_patch(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
    source_anchor_color: int,
) -> tuple[np.ndarray, int, int]:
    """Return (patch, patch_r0_in_grid, patch_c0_in_grid) for the TL component."""
    tl = g[0:mid_r, 0:mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(int(tl[r, c]) == source_anchor_color for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with the anchor color"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy(), int(pr0), int(pc0)


def _paint_patch(
    out: np.ndarray,
    patch: np.ndarray,
    place_r0: int,
    place_c0: int,
    bg: int,
    mid_r: int,
    mid_c: int,
    paint_color: int | None,
) -> None:
    """Paint all non-bg cells of a patch, optionally recoloring them to one color."""
    rr_p, cc_p = np.where(patch != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        out[rr_t, cc_t] = int(patch[rr0, cc0]) if paint_color is None else int(paint_color)


def _render_variant(
    grid: np.ndarray,
    *,
    recolor_targets: bool,
    recolor_source: bool,
) -> np.ndarray:
    """Shared renderer used by the solver and generator-side ablations."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchor_blocks(g, bg, mid_r, mid_c)

    assert (0, 0) in anchors and (0, 1) in anchors, "top-left and top-right anchors are required"
    _sr0, _sc0, sL, source_anchor_color = anchors[(0, 0)]
    source_recolor_color = anchors[(0, 1)][3]
    patch, pr0, pc0 = _extract_source_patch(g, bg, mid_r, mid_c, source_anchor_color)

    if recolor_source:
        _paint_patch(out, patch, pr0, pc0, bg, mid_r, mid_c, source_recolor_color)

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

        paint_color = target_color if recolor_targets else None
        _paint_patch(out, patch_s, place_r0, place_c0, bg, mid_r, mid_c, paint_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif across quadrants, but make both source and targets follow anchor colors.

    Input structure:
    - The grid is split into four quadrants by one full-row/full-column divider of a
      single non-background color.
    - Each quadrant contains one solid square anchor block; anchor sizes vary and anchor
      colors may differ across quadrants.
    - The top-left quadrant also contains one multi-color motif attached to its 2x2 anchor.

    Rule:
    - Extract the full TL motif component containing the TL anchor.
    - Recolor the TL source motif in the output to the top-right anchor's color.
    - For every other quadrant, reflect and scale the TL motif by the target/source
      anchor-size ratio, align the scaled anchor region to the target anchor, and paint
      every copied non-background pixel with that quadrant's own anchor color.

    Key insight:
    - Anchor size still gives the scale and quadrant still gives the reflection, but
      anchor color now controls all output colors: local anchor color for targets and
      the fixed top-row partner anchor color for the source.
    """
    return _render_variant(grid, recolor_targets=True, recolor_source=True)


# --------------- generator (combo D6/1 + D10/1) ---------------

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

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
ANCHOR_COLOR_POOL = [0, 2, 3, 5, 6, 7, 8, 9]

LOGICAL_DIMS = [(2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.7
FILL_RATIO_MAX = 1.0
SOURCE_EXTRA_COLORS_MIN = 2
SOURCE_EXTRA_COLORS_MAX = 4

MAX_RETRIES = 500


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
    """Color a logical motif while avoiding accidental fake anchors."""
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
    """Build the TL motif as a 2x-upscaled logical pattern."""
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
    """Pick an anchor position, preferring full-fit placements for the full scaled motif."""
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


def _build_input() -> np.ndarray | None:
    """Construct one candidate input grid satisfying the solver's assumptions."""
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
    if anchor_by_quad[(0, 0)] == anchor_by_quad[(0, 1)]:
        return None

    source = _source_patch(anchor_by_quad[(0, 0)], set(anchor_colors))
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], target_sizes[key])
        for key in ((0, 1), (1, 0), (1, 1))
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(
            key[0], key[1], target_sizes[key], patch_s, anchor_by_quad[(0, 0)]
        )

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key in ((0, 1), (1, 0), (1, 1)):
        ar, ac = anchors[key]
        size = target_sizes[key]
        paste(inp, np.full((size, size), anchor_by_quad[key], dtype=int), ar, ac)
    return inp


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated sample really exercises the fused behavior.

    Metrics:
      - source_change_pixels: TL non-bg cells changed by the combo rule.
      - delta_vs_d10_only: pixels changed by adding D6 on top of D10.
      - target_recolor_pixels: pixels changed by D10 relative to a color-preserving copy rule.
      - extra_pixels_in_targets: copied visible pixels outside the three target anchor squares.
    """
    g = np.asarray(grid, dtype=int)
    combo = _render_variant(g, recolor_targets=True, recolor_source=True)
    d10_only = _render_variant(g, recolor_targets=True, recolor_source=False)
    color_preserving = _render_variant(g, recolor_targets=False, recolor_source=False)

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchor_blocks(g, bg, mid_r, mid_c)
    source_anchor_color = anchors[(0, 0)][3]
    patch, pr0, pc0 = _extract_source_patch(g, bg, mid_r, mid_c, source_anchor_color)
    patch_mask = patch != bg

    tl_in = g[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]]
    tl_combo = combo[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]]
    source_change_pixels = int(np.count_nonzero(tl_combo[patch_mask] != tl_in[patch_mask]))

    extra_pixels = 0
    for key in ((0, 1), (1, 0), (1, 1)):
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        quadrant = combo[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        anchor_r0, anchor_c0, size, _acol = anchors[key]
        local_r0 = anchor_r0 - q_r0
        local_c0 = anchor_c0 - q_c0
        mask = quadrant != bg
        mask[local_r0 : local_r0 + size, local_c0 : local_c0 + size] = False
        extra_pixels += int(mask.sum())

    return {
        "source_change_pixels": source_change_pixels,
        "delta_vs_d10_only": int(np.count_nonzero(combo != d10_only)),
        "target_recolor_pixels": int(np.count_nonzero(combo != color_preserving)),
        "extra_pixels_in_targets": extra_pixels,
    }


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt with oracle and anti-degeneration checks."""
    inp = _build_input()
    if inp is None:
        return None

    combo = main(inp)
    metrics = fusion_metrics(inp)
    if metrics["source_change_pixels"] <= 0:
        return None
    if metrics["delta_vs_d10_only"] <= 0:
        return None
    if metrics["target_recolor_pixels"] <= 0:
        return None
    if metrics["extra_pixels_in_targets"] < 3:
        return None
    if np.array_equal(combo, _render_variant(inp, recolor_targets=True, recolor_source=False)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo D6/1+D10/1 input")
