# Combo variant of b10624e5 (index 17)
# combo_spec: D1/3+D8/0
# Name: 连接器匹配配对 + 显式分割线划分

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   keep:
#     - one source motif in the top-left region, scaled to the other three regions by the
#       target/source red-anchor size ratio
#     - red square anchors remain the alignment reference for stamping scaled copies
#   fused_changes:
#     - D8/0: the quadrant split is no longer implicit or fixed at the center; one full row
#       and one full column of the same divider color explicitly partition the grid, and their
#       indices vary so the four regions can have different sizes
#     - D1/3: targets are no longer interpreted by quadrant identity; each non-TL anchor is
#       paired to the source through a repeated connector color that appears once near the TL
#       anchor and once near exactly one target anchor
#     - the connector shapes are L-like line components whose free tails run to cells adjacent
#       to the divider, so the divider is part of the visual parsing problem without being
#       overwritten by connector pixels
#   concrete_trace:
#     - original TR always implied LR flip because it sat on the right side of a fixed center
#     - here a TR anchor can instead receive no LR flip if its repeated connector color uses a
#       right-side TL contact, or an LR flip if that color touches the TL anchor on the left
#     - similarly, a BR anchor can receive either UD state depending on whether its own connector
#       touches the anchor from above or below; region position alone is insufficient
#     - the explicit divider must be found first because the same connector component also has a
#       tail that terminates next to that divider, and region membership is defined by the divider
#   complexity_calibration:
#     - exactly three connector colors are used, one per non-TL target, so pairing is repeated
#       and unambiguous but still visible to humans
#     - motif templates are asymmetric 2x-upscaled block patterns, making reflections observable
#     - divider indices vary within a restricted interior range so regions are unequal but still
#       large enough for anchors, connectors, and scaled motifs
#   implicit_assumptions_replaced:
#     - old code assumed reflection bits came directly from (top/bottom, left/right); that is no
#       longer valid and must be replaced by connector-side decoding
#     - old generator assumed fixed 13x13 quadrants; now every placement helper must derive legal
#       bounds from the detected divider row/column
#     - source extraction can no longer treat all TL non-background pixels as motif, because the TL
#       connector touches the same anchor and must be masked out by connector color first
#   placement_legality:
#     - TL patch placement keeps at least one row of clearance from the TL region's top/bottom so
#       left-encoded connectors can route to the divider without intersecting the motif
#     - target anchor placement enforces at least one free cell on the connector side (above/below)
#       and prefers positions where the scaled motif fits inside the target region
#     - connector colors are reserved and never appear in the motif or divider, avoiding ambiguity
#   visual_observability:
#     - humans can first detect the off-center full row/column divider, then notice three repeated
#       connector colors that each bind TL to one target
#     - connector-to-anchor contact side gives a small discrete transform code, while the connector
#       tails touching divider-adjacent cells make the coexistence of divider lines and connectors
#       visually explicit rather than incidental
#   cleanliness:
#     - divider pixels remain a single solid color and are never overwritten
#     - connectors stay in otherwise empty space around anchors, so motif, anchor, connector, and
#       divider remain visually separable instead of stacking on the same cells


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 500

CONNECTOR_COLORS = [5, 6, 7]
MOTIF_COLOR_POOL = [0, 3, 8, 9]
DIVIDER_RANGE = (9, 17)

MOTIF_TEMPLATES = [
    {
        "shape": (3, 3),
        "anchor": (1, 1),
        "cells": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    },
    {
        "shape": (3, 3),
        "anchor": (1, 1),
        "cells": [(0, 1), (0, 2), (1, 1), (2, 0), (2, 1)],
    },
    {
        "shape": (3, 3),
        "anchor": (1, 1),
        "cells": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    },
    {
        "shape": (3, 3),
        "anchor": (0, 1),
        "cells": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
    },
]


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider pair."""
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
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _quad_bounds(mid_r: int, mid_c: int, h: int, w: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _component_quadrants(pts: np.ndarray, mid_r: int, mid_c: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in pts.tolist():
        rr = int(r)
        cc = int(c)
        if rr == mid_r or cc == mid_c:
            continue
        out.add((0 if rr < mid_r else 1, 0 if cc < mid_c else 1))
    return out


def _touching_anchors(pts: np.ndarray, anchors: dict[tuple[int, int], tuple[int, int, int]]) -> list[tuple[int, int]]:
    coords = {(int(r), int(c)) for r, c in pts.tolist()}
    touched = []
    for key, (r0, c0, size) in anchors.items():
        hit = False
        for r, c in coords:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if r0 <= rr < r0 + size and c0 <= cc < c0 + size:
                    hit = True
                    break
            if hit:
                touched.append(key)
                break
    return touched


def _side_hits(pts: np.ndarray, anchor: tuple[int, int, int]) -> tuple[bool, bool, bool, bool]:
    r0, c0, size = anchor
    coords = {(int(r), int(c)) for r, c in pts.tolist()}
    left = any((r, c0 - 1) in coords for r in range(r0, r0 + size))
    right = any((r, c0 + size) in coords for r in range(r0, r0 + size))
    up = any((r0 - 1, c) in coords for c in range(c0, c0 + size))
    down = any((r0 + size, c) in coords for c in range(c0, c0 + size))
    return left, right, up, down


def _connector_pairs(
    g: np.ndarray,
    anchors: dict[tuple[int, int], tuple[int, int, int]],
    bg: int,
    mid_r: int,
    mid_c: int,
) -> tuple[dict[tuple[int, int], tuple[int, bool, bool]], set[int]]:
    pairs: dict[tuple[int, int], tuple[int, bool, bool]] = {}
    connector_colors: set[int] = set()
    forbidden = {int(bg), int(DIV), int(RED)}
    for color in sorted(set(map(int, g.flatten())) - forbidden):
        comps = components_mask(g == color, connectivity=4)
        if len(comps) != 2:
            continue
        info = []
        ok = True
        for pts in comps:
            touched = _touching_anchors(pts, anchors)
            qs = _component_quadrants(pts, mid_r, mid_c)
            if len(touched) != 1 or len(qs) != 1:
                ok = False
                break
            info.append((touched[0], qs.pop(), pts))
        if not ok:
            continue
        by_quad = {key: pts for key, _q, pts in info}
        if (0, 0) not in by_quad or len(by_quad) != 2:
            continue
        targets = [key for key in by_quad if key != (0, 0)]
        if len(targets) != 1:
            continue
        target_key = targets[0]
        src_pts = by_quad[(0, 0)]
        tgt_pts = by_quad[target_key]
        src_left, src_right, src_up, src_down = _side_hits(src_pts, anchors[(0, 0)])
        tgt_left, tgt_right, tgt_up, tgt_down = _side_hits(tgt_pts, anchors[target_key])
        assert src_left ^ src_right, "TL connector must touch exactly one horizontal side of the TL anchor"
        assert tgt_up ^ tgt_down, "target connector must touch exactly one vertical side of its anchor"
        hflip = bool(src_left)
        vflip = bool(tgt_down)
        assert target_key not in pairs, "each target quadrant must have exactly one connector color"
        pairs[target_key] = (int(color), hflip, vflip)
        connector_colors.add(int(color))
    assert set(pairs) == {(0, 1), (1, 0), (1, 1)}, "must find one connector-matched target in each non-TL region"
    return pairs, connector_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif into the other divider-defined regions using connector-matched transforms.

    Input structure:
    - One full non-background row and one full non-background column of the same color explicitly
      split the board into four regions; their crossing is not fixed at the center.
    - Each region contains one red square anchor, and the TL region also contains the full motif.
    - Three extra connector colors each appear twice: one component in TL and one component in a
      single non-TL region. These connector components end next to the divider but touch anchors.

    Rule:
    - Detect the divider row/column, then locate the four red anchors by region.
    - Extract the TL motif while ignoring connector colors.
    - Pair each non-TL anchor with TL by matching the repeated connector color. The TL connector's
      touch side gives the LR reflection bit, and the target connector's touch side gives the UD
      reflection bit.
    - Scale the motif by target-anchor-size / source-anchor-size, align the scaled red anchor to the
      target anchor, and stamp non-background pixels without overwriting divider cells or red anchors.

    Key insight: The explicit divider defines the regions, but it does not determine the transform.
    The repeated connector color selects the destination, and the connector-to-anchor contact sides
    encode the reflection independently of region position.
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
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in anchors, "each region must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one anchor in every divider-defined region"

    connector_map, connector_colors = _connector_pairs(g, anchors, bg, mid_r, mid_c)

    _sr0, _sc0, sL = anchors[(0, 0)]
    tl = g[0:mid_r, 0:mid_c]
    keep = (tl != bg) & (~np.isin(tl, list(connector_colors)))
    comps = components_mask(keep, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "TL region must contain a source motif attached to the TL anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0:pr1 + 1, pc0:pc1 + 1].copy()
    patch[np.isin(patch, list(connector_colors))] = bg

    for key, (_color, hflip, vflip) in connector_map.items():
        tr0, tc0, tL = anchors[key]
        patch_t = patch
        if hflip:
            patch_t = np.fliplr(patch_t)
        if vflip:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled motif must retain the red anchor"
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
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


def _build_source_patch() -> tuple[np.ndarray, int, int]:
    template = random.choice(MOTIF_TEMPLATES)
    h, w = template["shape"]
    ar, ac = template["anchor"]
    logical = np.full((h, w), BG, dtype=int)
    logical[ar, ac] = RED
    extras = [cell for cell in template["cells"] if cell != (ar, ac)]
    palette = MOTIF_COLOR_POOL[:]
    random.shuffle(palette)
    for idx, (r, c) in enumerate(extras):
        logical[r, c] = palette[idx]
    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _roundtrip_ok(patch: np.ndarray) -> bool:
    for hflip in (False, True):
        for vflip in (False, True):
            p = patch
            if hflip:
                p = np.fliplr(p)
            if vflip:
                p = np.flipud(p)
            for target_size in range(TARGET_SIZE_RANGE[0], TARGET_SIZE_RANGE[1] + 1):
                ratio = target_size / SOURCE_ANCHOR_SIZE
                new_h = max(1, int(round(p.shape[0] * ratio)))
                new_w = max(1, int(round(p.shape[1] * ratio)))
                q = _resize_nn(p, new_h, new_w)
                if not np.array_equal(_resize_nn(q, p.shape[0], p.shape[1]), p):
                    return False
    return True


def _scaled_patch(patch: np.ndarray, hflip: bool, vflip: bool, target_size: int) -> np.ndarray:
    p = patch
    if hflip:
        p = np.fliplr(p)
    if vflip:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _choose_transform_bits() -> dict[tuple[int, int], tuple[bool, bool]] | None:
    keys = [(0, 1), (1, 0), (1, 1)]
    original = {key: (bool(key[1]), bool(key[0])) for key in keys}
    for _ in range(100):
        bits = {key: (bool(random.getrandbits(1)), bool(random.getrandbits(1))) for key in keys}
        if all(bits[key] == original[key] for key in keys):
            continue
        # TL-side encoding only has two horizontal attachment lanes per side.
        hflip_count = sum(int(bits[key][0]) for key in keys)
        if hflip_count not in (1, 2):
            continue
        if len({bits[key] for key in keys}) < 2:
            continue
        return bits
    return None


def _source_slot_rows(bits: dict[tuple[int, int], tuple[bool, bool]]) -> dict[tuple[int, int], int]:
    used = {True: 0, False: 0}
    out = {}
    for key in [(0, 1), (1, 0), (1, 1)]:
        hflip = bits[key][0]
        idx = used[hflip]
        assert idx < SOURCE_ANCHOR_SIZE, "at most two connectors may use the same TL side"
        out[key] = idx
        used[hflip] += 1
    return out


def _choose_dividers() -> tuple[int, int]:
    return random.randint(*DIVIDER_RANGE), random.randint(*DIVIDER_RANGE)


def _choose_tl_top_left(
    patch: np.ndarray,
    anchor_r_in_patch: int,
    anchor_c_in_patch: int,
    mid_r: int,
    mid_c: int,
    any_hflip: bool,
) -> tuple[int, int]:
    min_r0 = 1
    max_r0 = mid_r - patch.shape[0] - 2
    min_c0 = 1 if any_hflip else 0
    max_c0 = min(mid_c - patch.shape[1] - 1, mid_c - 3 - anchor_c_in_patch)
    assert min_r0 <= max_r0 and min_c0 <= max_c0, "no legal TL placement for patch and connectors"
    return random.randint(min_r0, max_r0), random.randint(min_c0, max_c0)


def _choose_anchor(
    key: tuple[int, int],
    size: int,
    patch_s: np.ndarray,
    vflip: bool,
    mid_r: int,
    mid_c: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = _quad_bounds(mid_r, mid_c, h, w)[key]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1
    if vflip:
        anch_r1 = min(anch_r1, q_r1 - size)
    else:
        anch_r0 = max(anch_r0, q_r0 + 1)
    if anch_r0 > anch_r1:
        raise AssertionError("no legal row remains for the target connector direction")

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _source_connector(
    color: int,
    anchor_r: int,
    anchor_c: int,
    hflip: bool,
    slot: int,
    mid_r: int,
    mid_c: int,
) -> tuple[np.ndarray, int, int]:
    region_h = mid_r
    divider_col = mid_c - 1
    attach_row = anchor_r + slot
    canvas = np.full((region_h, mid_c), BG, dtype=int)
    if hflip:
        lane_row = 0 if slot == 0 else region_h - 1
        canvas[attach_row, 0:anchor_c] = color
        r0, r1 = sorted((lane_row, attach_row))
        canvas[r0:r1 + 1, 0] = color
        canvas[lane_row, 0:divider_col + 1] = color
    else:
        canvas[attach_row, anchor_c + SOURCE_ANCHOR_SIZE:divider_col + 1] = color
    pts = np.argwhere(canvas != BG)
    r0, c0, r1, c1 = bbox_coords(pts)
    return canvas[r0:r1 + 1, c0:c1 + 1].copy(), int(r0), int(c0)


def _target_connector(
    color: int,
    key: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    size: int,
    vflip: bool,
    mid_r: int,
    mid_c: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, int, int]:
    q_r0, q_r1, q_c0, q_c1 = _quad_bounds(mid_r, mid_c, h, w)[key]
    attach_col = anchor_c + random.randint(0, size - 1)
    divider_adj_col = q_c0 if key[1] == 1 else q_c1
    boundary_row = q_r1 if vflip else q_r0
    canvas = np.full((h, w), BG, dtype=int)
    if vflip:
        canvas[anchor_r + size:q_r1 + 1, attach_col] = color
    else:
        canvas[q_r0:anchor_r, attach_col] = color
    c0, c1 = sorted((attach_col, divider_adj_col))
    canvas[boundary_row, c0:c1 + 1] = color
    pts = np.argwhere(canvas != BG)
    r0, c0, r1, c1 = bbox_coords(pts)
    return canvas[r0:r1 + 1, c0:c1 + 1].copy(), int(r0), int(c0)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, mid_r: int, mid_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < out.shape[0])
        & (tgt_cc >= 0)
        & (tgt_cc < out.shape[1])
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    keep = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def _gen_attempt(return_meta: bool = False):
    mid_r, mid_c = _choose_dividers()
    bits = _choose_transform_bits()
    if bits is None:
        return None

    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    patch, anchor_r_in_patch, anchor_c_in_patch = _build_source_patch()
    if not _roundtrip_ok(patch):
        return None

    scaled = {key: _scaled_patch(patch, bits[key][0], bits[key][1], target_sizes[key]) for key in bits}

    try:
        tl_r0, tl_c0 = _choose_tl_top_left(
            patch, anchor_r_in_patch, anchor_c_in_patch, mid_r, mid_c, any(v[0] for v in bits.values())
        )
    except AssertionError:
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {(0, 0): (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)}
    try:
        for key, patch_s in scaled.items():
            anchors[key] = _choose_anchor(key, target_sizes[key], patch_s, bits[key][1], mid_r, mid_c, GRID_SIZE, GRID_SIZE)
    except AssertionError:
        return None

    slot_rows = _source_slot_rows(bits)
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    paste(out, patch, tl_r0, tl_c0)

    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])

    source_stubs = {}
    target_stubs = {}
    for idx, key in enumerate([(0, 1), (1, 0), (1, 1)]):
        color = CONNECTOR_COLORS[idx]
        hflip, vflip = bits[key]
        src_stub, src_r0, src_c0 = _source_connector(color, anchors[(0, 0)][0], anchors[(0, 0)][1], hflip, slot_rows[key], mid_r, mid_c)
        tgt_stub, tgt_r0, tgt_c0 = _target_connector(color, key, anchors[key][0], anchors[key][1], target_sizes[key], vflip, mid_r, mid_c, GRID_SIZE, GRID_SIZE)
        source_stubs[key] = (src_stub, src_r0, src_c0)
        target_stubs[key] = (tgt_stub, tgt_r0, tgt_c0)
        paste(out, src_stub, src_r0, src_c0, transparent=BG, overwrite=False)
        paste(out, tgt_stub, tgt_r0, tgt_c0, transparent=BG, overwrite=False)

    for key, patch_s in scaled.items():
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], mid_r, mid_c)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[mid_r, :] = DIV
    inp[:, mid_c] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, (stub, r0, c0) in source_stubs.items():
        paste(inp, stub, r0, c0, transparent=BG, overwrite=False)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])
        stub, r0, c0 = target_stubs[key]
        paste(inp, stub, r0, c0, transparent=BG, overwrite=False)

    if not np.array_equal(main(inp), out):
        return None

    if return_meta:
        return inp, {
            "mid_r": mid_r,
            "mid_c": mid_c,
            "bits": bits,
            "target_sizes": target_sizes,
            "anchors": anchors,
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
