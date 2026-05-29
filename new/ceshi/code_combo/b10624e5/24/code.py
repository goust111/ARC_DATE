# Combo variant of b10624e5 (index 24)
# combo_spec: D1/0+D5/0+D10/1
# Name: 多颜色锚块配对 + 颜色决定输出颜色 + 旋转替代反射

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the fused combo variant
# ---------------------------------------------------------------------------
# design:
#   overview:
#     original_task:
#       - TL had exactly one source component attached to one red 2x2 anchor.
#       - TR / BL / BR each had one red square anchor.
#       - Solver used quadrant for reflection and anchor-size ratio for scaling.
#     fused_goal:
#       - Replace the single TL source with a small catalog of 3 TL source components,
#         each keyed by its own anchor color.
#       - Replace reflection with quadrant-dependent rotation.
#       - After selecting and rotating the matched source, recolor the whole copied
#         target motif to the target anchor color.
#       - Color therefore has two simultaneous roles:
#           1. decide which TL source component is paired to a target quadrant
#           2. decide the final visible color of that target quadrant's output motif
#
#   a_concretization_and_trace:
#     concrete_layout:
#       - Grid stays 27x27 with divider row/col at index 13.
#       - TL contains exactly 3 disconnected source components.
#       - Each TL component is a 4x6 physical patch built by 2x-upscaling a 2x3 logical
#         polyomino with one missing corner. One logical cell is the colored 2x2 anchor;
#         the remaining logical cells use 2-3 extra colors.
#       - TR / BL / BR each contain one standalone NxN colored anchor square.
#     trace_example:
#       - Suppose TL has:
#           red anchor + hole-at-top-left motif,
#           green anchor + hole-at-bottom-left motif,
#           teal anchor + hole-at-top-right motif.
#       - Suppose TR contains a green 3x3 anchor.
#         Original task:
#           always picked the unique TL red motif, mirrored LR, kept source colors.
#         Fused task:
#           first matches green->green and selects the green TL source component,
#           then rotates it 90 degrees clockwise because TR is top-right,
#           scales by 3/2,
#           then paints every copied non-bg pixel green.
#       - Difference:
#           source identity is no longer fixed,
#           transform is no longer a mirror,
#           target colors no longer preserve the multi-color source palette.
#
#   b_complexity_calibration:
#     - Use exactly 3 source colors per puzzle because there are exactly 3 target quadrants.
#       This is enough to make pairing nontrivial without introducing ambiguous leftovers.
#     - Force the three target anchors to use those same three colors in a random permutation,
#       so quadrant location alone cannot predict which TL motif family is used.
#     - Keep source shapes in a small family of asymmetric 5-cell logical polyominoes
#       (2x3 minus one corner). This makes color-matched identity visible while keeping
#       the scene sparse and scale-safe.
#     - Keep TL source patches multi-color, but keep each copied target motif monochrome.
#       That makes the D10 recoloring effect easy to read.
#
#   c_implicit_assumptions_review:
#     original_main_assumptions_that_break:
#       - `components_mask(g == 2)` assumed all anchors were red.
#       - "the TL component containing red is the source" assumed only one source existed.
#       - `np.where(patch_s == 2)` assumed anchor alignment always used red.
#       - stamping source pixel values assumed target motifs preserved source colors.
#       - LR/UD flip branches assumed quadrant geometry was encoded only as reflection.
#     new_solver_assumptions:
#       - Every non-TL quadrant contains exactly one standalone square anchor.
#       - TL contains exactly one square source anchor for each target color.
#       - Extra motif colors never reuse the anchor colors, so TL source-anchor detection
#         can safely filter square components by "colors that also appear as targets".
#       - Each TL source anchor lies inside exactly one TL non-bg connected component.
#       - Rotating and resizing the 2x-upscaled logical patches preserves at least one
#         anchor-colored pixel for alignment.
#     generator_guarantees:
#       - Target colors are a permutation of the TL source-anchor colors.
#       - Extra motif colors are chosen from a palette disjoint from the chosen anchor colors.
#       - TL source components occupy separated preset slots.
#       - Source patches are built only from 2x-upscaled logical cells, so resize roundtrip
#         back to the transformed source patch is exact for target sizes 1..4.
#
#   d_placement_legality:
#     - TL uses three out of four preset 4x6 slots inside the 13x13 quadrant, so source
#       components never overlap and never touch the divider.
#     - Each target anchor is placed from the already-transformed, already-scaled patch
#       footprint. Full-fit placements are preferred; anchor-only fit is the fallback.
#     - Stamping clips at borders and skips the divider row/col, matching the original family.
#
#   e_visual_observability:
#     - D1 is visible because the three TL source shapes differ, while target anchor colors
#       are permuted across TR / BL / BR.
#     - D5 is visible because generator rejects samples where the rotated result matches the
#       old reflected result in both TR and BL.
#     - D10 is visible because TL sources are multi-color but every copied target motif
#       collapses to a single target-anchor color.
#
#   f_visual_cleanliness:
#     - Only TL is dense; each other quadrant still starts with a single anchor square.
#     - Monochrome target outputs reduce clutter despite multi-color TL exemplars.
#     - The chosen 4x6 source-patch footprint leaves background gaps between TL objects.
#
#   g_change_inferability:
#     - No hidden numeric ordering on color ids is used.
#     - The only color cue is visible equality:
#         target-anchor color == chosen TL source-anchor color == final copied motif color.
#     - Because the same color participates in both matching and recoloring, the user must
#       jointly infer object pairing, target color transfer, and rotation direction.


BG = 4
DIV = 1
GRID_SIZE = 27
MID = 13
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 500

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
TARGET_KEYS = [(0, 1), (1, 0), (1, 1)]

ROTATION_K = {
    (0, 0): 0,
    (0, 1): 3,  # 90 clockwise
    (1, 0): 1,  # 270 clockwise
    (1, 1): 2,  # 180
}

ANCHOR_COLOR_POOL = [0, 2, 3, 5, 6, 7, 8, 9]
HOLE_CORNERS = [(0, 0), (0, 2), (1, 0), (1, 2)]
SOURCE_SLOTS = [(0, 0), (0, 7), (8, 0), (8, 7)]


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


def _quadrant_key(mid_r: int, mid_c: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    assert r1 < mid_r or r0 > mid_r, "component must stay within one row half"
    assert c1 < mid_c or c0 > mid_c, "component must stay within one col half"
    rr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _transform_patch(patch: np.ndarray, qb: int, qr: int, *, use_rotation: bool) -> np.ndarray:
    if use_rotation:
        return np.rot90(patch, ROTATION_K[(qb, qr)])
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _find_square_parts(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> list[dict[str, int]]:
    """Collect all monochrome square connected components that live inside a single quadrant."""
    divider_color = int(g[mid_r, mid_c])
    parts: list[dict[str, int]] = []
    colors = sorted({int(v) for v in g.flatten()} - {int(bg), divider_color})
    for color in colors:
        for pts in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 <= mid_r <= r1 or c0 <= mid_c <= c1:
                continue
            size_h = r1 - r0 + 1
            size_w = c1 - c0 + 1
            if size_h != size_w:
                continue
            if len(pts) != size_h * size_w:
                continue
            qb, qr = _quadrant_key(mid_r, mid_c, int(r0), int(c0), int(r1), int(c1))
            parts.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(size_h),
                    "qb": int(qb),
                    "qr": int(qr),
                }
            )
    return parts


def _extract_catalog(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> tuple[dict[tuple[int, int], dict[str, int]], dict[int, tuple[np.ndarray, int]]]:
    """
    Parse the three target anchors and the TL source catalog.

    Returns:
      targets[(qb, qr)] -> {"color", "r0", "c0", "size", ...}
      source_patches[color] -> (patch, source_anchor_size)
    """
    square_parts = _find_square_parts(g, bg, mid_r, mid_c)

    targets: dict[tuple[int, int], dict[str, int]] = {}
    target_colors: set[int] = set()
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        if key == (0, 0):
            continue
        assert key not in targets, "each non-top-left quadrant must contain exactly one anchor square"
        targets[key] = part
        target_colors.add(int(part["color"]))
    assert set(targets) == set(TARGET_KEYS), "TR/BL/BR must each contain one target anchor"

    source_anchors: dict[int, dict[str, int]] = {}
    for part in square_parts:
        key = (int(part["qb"]), int(part["qr"]))
        if key != (0, 0):
            continue
        color = int(part["color"])
        if color not in target_colors:
            continue
        assert color not in source_anchors, "each target color must have exactly one TL source anchor"
        source_anchors[color] = part
    assert target_colors == set(source_anchors), "every target color must exist in TL"

    tl = g[0:mid_r, 0:mid_c]
    tl_entries: list[tuple[set[tuple[int, int]], tuple[int, int, int, int]]] = []
    for pts in components_mask(tl != bg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        tl_entries.append(({(int(r), int(c)) for r, c in pts}, (int(r0), int(c0), int(r1), int(c1))))

    source_patches: dict[int, tuple[np.ndarray, int]] = {}
    for color, anchor in source_anchors.items():
        anchor_pixel = (int(anchor["r0"]), int(anchor["c0"]))
        patch_bb = None
        for ptset, bb in tl_entries:
            if anchor_pixel in ptset:
                patch_bb = bb
                break
        assert patch_bb is not None, "each TL source anchor must belong to one non-bg component"
        pr0, pc0, pr1, pc1 = patch_bb
        patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        source_patches[int(color)] = (patch, int(anchor["size"]))

    return targets, source_patches


def _solve(grid: np.ndarray, *, use_rotation: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    targets, source_patches = _extract_catalog(g, bg, mid_r, mid_c)

    for (qb, qr), target in targets.items():
        color = int(target["color"])
        tr0 = int(target["r0"])
        tc0 = int(target["c0"])
        t_size = int(target["size"])
        patch, s_size = source_patches[color]

        patch_t = _transform_patch(patch, qb, qr, use_rotation=use_rotation)
        ratio = float(t_size) / float(s_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve the matched motif"

        rr_anchor, cc_anchor = np.where(patch_s == color)
        assert rr_anchor.size > 0, "scaled patch must retain the matched anchor color"
        place_r0 = tr0 - int(rr_anchor.min())
        place_c0 = tc0 - int(cc_anchor.min())

        src_rr, src_cc = np.where(patch_s != bg)
        tgt_rr = src_rr + place_r0
        tgt_cc = src_cc + place_c0
        valid = (
            (tgt_rr >= 0)
            & (tgt_rr < h)
            & (tgt_cc >= 0)
            & (tgt_cc < w)
            & (tgt_rr != mid_r)
            & (tgt_cc != mid_c)
        )
        out[tgt_rr[valid], tgt_cc[valid]] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a color-matched top-left source catalog into the other quadrants with rotation and recoloring.

    Input structure:
    - One full-row/full-column divider splits the board into four 13x13 quadrants.
    - The top-left quadrant contains several disconnected source components; each source component
      has its own colored square anchor plus a multi-color motif attached to that anchor.
    - Each non-top-left quadrant contains exactly one standalone square anchor, and its color
      matches exactly one of the top-left source anchors.

    Rule:
    - For each target quadrant, choose the top-left source component whose anchor color matches the
      target anchor color.
    - Rotate the matched source instead of reflecting it: top-right uses 90 degrees clockwise,
      bottom-left uses 270 degrees clockwise, and bottom-right uses 180 degrees.
    - Scale the rotated source by the ratio of target-anchor size to matched-source-anchor size,
      align the anchor-colored region to the target anchor, then paint every copied non-background
      pixel using the target anchor color.

    Key insight: Anchor color now simultaneously identifies which source motif to use and what color
    the copied target motif becomes, while quadrant location selects a rotation and anchor size selects scale.
    """
    return _solve(grid, use_rotation=True)


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    target_size: int,
    *,
    use_rotation: bool,
) -> np.ndarray:
    p = _transform_patch(patch, qb, qr, use_rotation=use_rotation)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _make_source_patch(anchor_color: int, hole: tuple[int, int], extra_palette: list[int]) -> tuple[np.ndarray, int, int]:
    """
    Build one TL source patch from a 2x3 logical polyomino.

    The physical patch is always 4x6, so scaling to target sizes 1..4 remains lossless
    after nearest-neighbor resize and roundtrip.
    """
    logical = np.full((2, 3), BG, dtype=int)
    filled = [(r, c) for r in range(2) for c in range(3) if (r, c) != hole]
    anchor_r, anchor_c = random.choice(filled)
    logical[anchor_r, anchor_c] = anchor_color

    other_cells = [(r, c) for r, c in filled if (r, c) != (anchor_r, anchor_c)]
    random.shuffle(other_cells)
    random.shuffle(extra_palette)
    n_extra = random.randint(2, min(3, len(extra_palette)))
    colors = extra_palette[:n_extra]
    for idx, (r, c) in enumerate(other_cells):
        logical[r, c] = colors[idx % len(colors)]

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, anchor_r * SOURCE_ANCHOR_SIZE, anchor_c * SOURCE_ANCHOR_SIZE


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    anchor_color: int,
) -> tuple[int, int]:
    """Pick an anchor position, preferring placements where the full transformed motif fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == anchor_color)
    anchor_r0, anchor_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + anchor_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + anchor_r0
    fit_c0 = q_c0 + anchor_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + anchor_c0

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
    """Align by the source anchor color, then paint every copied non-bg pixel to one target color."""
    rr_anchor, cc_anchor = np.where(patch_s == source_anchor_color)
    place_r0 = anchor_r - int(rr_anchor.min())
    place_c0 = anchor_c - int(cc_anchor.min())

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
    out[tgt_rr[valid], tgt_cc[valid]] = target_color


def _build_output(
    source_specs: dict[int, dict[str, object]],
    target_sizes: dict[tuple[int, int], int],
    target_colors: dict[tuple[int, int], int],
    target_anchors: dict[tuple[int, int], tuple[int, int]],
    *,
    use_rotation: bool,
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    for spec in source_specs.values():
        paste(out, spec["patch"], int(spec["slot_r0"]), int(spec["slot_c0"]))

    for key in TARGET_KEYS:
        ar, ac = target_anchors[key]
        size = target_sizes[key]
        color = target_colors[key]
        paste(out, np.full((size, size), color, dtype=int), ar, ac)

    for key in TARGET_KEYS:
        color = target_colors[key]
        patch = source_specs[color]["patch"]
        patch_s = _scaled_patch(patch, key[0], key[1], target_sizes[key], use_rotation=use_rotation)
        _stamp_recolored(out, patch_s, color, color, target_anchors[key][0], target_anchors[key][1])

    return out


def _build_input(
    out: np.ndarray,
    target_sizes: dict[tuple[int, int], int],
    target_colors: dict[tuple[int, int], int],
    target_anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    inp = out.copy()
    for key in TARGET_KEYS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = target_anchors[key]
        size = target_sizes[key]
        color = target_colors[key]
        paste(inp, np.full((size, size), color, dtype=int), ar, ac)
    return inp


def _rotation_diff_stats(rot_out: np.ndarray, ref_out: np.ndarray) -> dict[tuple[int, int], int]:
    stats: dict[tuple[int, int], int] = {}
    for key in TARGET_KEYS:
        r0, r1, c0, c1 = QUADS[key]
        q_rot = rot_out[r0 : r1 + 1, c0 : c1 + 1]
        q_ref = ref_out[r0 : r1 + 1, c0 : c1 + 1]
        stats[key] = int(np.count_nonzero(q_rot != q_ref))
    return stats


def _observable_rotation_variant(stats: dict[tuple[int, int], int]) -> bool:
    return stats[(0, 1)] > 0 or stats[(1, 0)] > 0


def _gen_attempt() -> np.ndarray | None:
    source_colors = random.sample(ANCHOR_COLOR_POOL, 3)
    holes = random.sample(HOLE_CORNERS, 3)
    slots = random.sample(SOURCE_SLOTS, 3)
    extra_palette = [c for c in ANCHOR_COLOR_POOL if c not in source_colors]

    source_specs: dict[int, dict[str, object]] = {}
    for color, hole, slot in zip(source_colors, holes, slots, strict=False):
        patch, anchor_r, anchor_c = _make_source_patch(color, hole, extra_palette[:])
        source_specs[color] = {
            "patch": patch,
            "anchor_r": int(anchor_r),
            "anchor_c": int(anchor_c),
            "slot_r0": int(slot[0]),
            "slot_c0": int(slot[1]),
        }

    target_sizes = {key: random.randint(*TARGET_SIZE_RANGE) for key in TARGET_KEYS}
    if all(target_sizes[key] == SOURCE_ANCHOR_SIZE for key in TARGET_KEYS):
        force_key = random.choice(TARGET_KEYS)
        target_sizes[force_key] = random.choice([1, 3, 4])

    permuted_colors = random.sample(source_colors, len(source_colors))
    target_colors = {key: color for key, color in zip(TARGET_KEYS, permuted_colors, strict=False)}

    target_anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for key in TARGET_KEYS:
        color = target_colors[key]
        patch_s = _scaled_patch(source_specs[color]["patch"], key[0], key[1], target_sizes[key], use_rotation=True)
        target_anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, color)

    out_rot = _build_output(
        source_specs,
        target_sizes,
        target_colors,
        target_anchors,
        use_rotation=True,
    )
    out_ref = _build_output(
        source_specs,
        target_sizes,
        target_colors,
        target_anchors,
        use_rotation=False,
    )

    if not _observable_rotation_variant(_rotation_diff_stats(out_rot, out_ref)):
        return None

    inp = _build_input(out_rot, target_sizes, target_colors, target_anchors)
    return inp if np.array_equal(main(inp), out_rot) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
