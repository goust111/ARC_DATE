# Variant of b10624e5
# Dimension: D1 — 匹配条件：红色锚块颜色匹配 (match_cond)
# Change: 所有象限的锚块共享同一颜色（红色），通过锚块颜色相同进行配对；
#         现在改为通过跨象限重复出现的彩色连接器配对。每条连接器颜色在 TL 和某个目标象限各出现一次，
#         解题者需要先按连接器配对，再根据连接器端点朝向决定反射轴。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the connector-match variant
# ------------------------------------------------
# spec_variant:
#   keep:
#     - 27x27 grid, bg=4, divider row/col of color 1 at (13,13)
#     - one red square anchor per quadrant; TL anchor is still the source anchor
#     - target motifs are still scaled copies of the TL motif
#   change_core:
#     - original solver matched targets implicitly because every anchor had the same color
#       and reflection came from the target quadrant position
#     - new solver must first find 3 connector colors; each such color forms exactly two
#       straight stubs: one attached to the TL anchor, one attached to a non-TL anchor
#     - a shared connector color defines the source-target pair
#     - the TL stub is horizontal and encodes LR reflection:
#         left of TL anchor -> flip LR
#         right of TL anchor -> no LR flip
#     - the target stub is vertical and encodes UD reflection:
#         above target anchor -> no UD flip
#         below target anchor -> flip UD
#   concrete_trace:
#     - old logic for TR always did LR only because TR is on the right
#     - new logic can place a TR connector with TL-side on the right and target-side below
#       so the same TR anchor receives an UD-only or both-flips copy depending on the
#       connector stubs, not on quadrant identity
#   complexity_calibration:
#     - only 3 connector colors are used, one per non-TL quadrant
#     - each connector color appears exactly twice, so pairing is visible and unambiguous
#     - motif templates are asymmetric block patterns so different reflections are observable
#   implicit_assumptions_replaced:
#     - old code assumed every red anchor should be processed the same way apart from size
#       and quadrant; this is no longer valid
#     - source patch extraction can no longer use all non-bg pixels in TL because connectors
#       also touch the TL anchor, so connector colors must be excluded first
#   placement_legality:
#     - TL anchor is placed away from left/right borders so both left and right connector
#       stubs can exist
#     - each target anchor is placed with at least one free cell above or below, matching
#       its connector direction
#     - connector colors are reserved and never used inside the motif, preventing ambiguity
#   visual_observability:
#     - humans can match repeated connector colors across quadrants
#     - the two connector endpoint directions provide a compact 2-bit code for the transform
#     - at least one target per sample is forced to differ from the original quadrant-based
#       reflection, preventing silent degeneration back to the source task


BG = 4
DIV = 1
RED = 2
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
MAX_RETRIES = 400

CONNECTOR_COLORS = [5, 6, 7]
MOTIF_COLOR_POOL = [0, 3, 8, 9]

# Each template is a logical grid later upscaled by 2x in both axes.
# The anchor cell is red; its immediate left/right logical neighbors stay empty so the TL
# connector stubs always have room to touch the anchor without colliding with the motif.
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
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
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


def _is_straight_line(pts: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return (h == 1 or w == 1) and int(len(pts)) == h * w


def _component_quadrants(pts: np.ndarray, mid_r: int, mid_c: int) -> set[tuple[int, int]]:
    qs = set()
    for r, c in pts.tolist():
        rr = int(r)
        cc = int(c)
        if rr == mid_r or cc == mid_c:
            continue
        qs.add((0 if rr < mid_r else 1, 0 if cc < mid_c else 1))
    return qs


def _touching_anchors(
    pts: np.ndarray, anchors: dict[tuple[int, int], tuple[int, int, int]]
) -> list[tuple[int, int]]:
    touched = []
    coords = {(int(r), int(c)) for r, c in pts.tolist()}
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


def _hflip_from_source_connector(pts: np.ndarray, anchor: tuple[int, int, int]) -> bool:
    _r0, c0, size = anchor
    _pr0, pc0, _pr1, pc1 = bbox_coords(pts)
    if int(pc1) < int(c0):
        return True
    if int(pc0) > int(c0 + size - 1):
        return False
    raise AssertionError("source connector must lie strictly left or right of the TL anchor")


def _vflip_from_target_connector(pts: np.ndarray, anchor: tuple[int, int, int]) -> bool:
    r0, _c0, size = anchor
    pr0, _pc0, pr1, _pc1 = bbox_coords(pts)
    if int(pr1) < int(r0):
        return False
    if int(pr0) > int(r0 + size - 1):
        return True
    raise AssertionError("target connector must lie strictly above or below the target anchor")


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
        if not all(_is_straight_line(pts) for pts in comps):
            continue
        hit_info = []
        ok = True
        for pts in comps:
            touched = _touching_anchors(pts, anchors)
            qs = _component_quadrants(pts, mid_r, mid_c)
            if len(touched) != 1 or len(qs) != 1:
                ok = False
                break
            hit_info.append((touched[0], qs.pop(), pts))
        if not ok:
            continue
        by_quad = {key: pts for key, _q, pts in hit_info}
        if (0, 0) not in by_quad or len(by_quad) != 2:
            continue
        target_keys = [key for key in by_quad if key != (0, 0)]
        if len(target_keys) != 1:
            continue
        target_key = target_keys[0]
        src_pts = by_quad[(0, 0)]
        tgt_pts = by_quad[target_key]
        hflip = _hflip_from_source_connector(src_pts, anchors[(0, 0)])
        vflip = _vflip_from_target_connector(tgt_pts, anchors[target_key])
        assert target_key not in pairs, "each target quadrant must have exactly one connector color"
        pairs[target_key] = (int(color), bool(hflip), bool(vflip))
        connector_colors.add(int(color))
    assert set(pairs) == {(0, 1), (1, 0), (1, 1)}, "must find one connector-matched target in each non-TL quadrant"
    return pairs, connector_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across quadrants with scaling and connector-driven reflections.

    Input structure:
    - The grid is split into four quadrants by one full non-background divider row and column.
    - Each quadrant contains a red square anchor block; the TL quadrant also contains the full motif.
    - Three connector colors appear twice each: once as a horizontal stub touching the TL anchor,
      and once as a vertical stub touching one non-TL anchor.

    Rule:
    - Extract the TL motif while ignoring connector colors.
    - Pair each non-TL anchor with the TL source by matching the repeated connector color.
    - Scale the motif by the target/source anchor-size ratio, then reflect it using the connector
      directions: TL-side left stub means LR flip, target-side bottom stub means UD flip.
    - Align the scaled red region to the target anchor and stamp the transformed motif.

    Key insight: The repeated connector color identifies which anchor should receive the copy,
    and the two connector endpoint directions replace quadrant position as the reflection code.
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
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one anchor in every quadrant"

    connector_map, connector_colors = _connector_pairs(g, anchors, bg, mid_r, mid_c)

    sr0, sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

    tl = g[0:mid_r, 0:mid_c]
    keep = (tl != bg) & (~np.isin(tl, list(connector_colors)))
    comps = components_mask(keep, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "TL quadrant must contain a source motif attached to the TL anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
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
    cells = list(template["cells"])

    logical = np.full((h, w), BG, dtype=int)
    logical[ar, ac] = RED

    extras = [cell for cell in cells if cell != (ar, ac)]
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
    for _ in range(80):
        bits = {key: (bool(random.getrandbits(1)), bool(random.getrandbits(1))) for key in keys}
        h_count = sum(int(bits[key][0]) for key in keys)
        if h_count == 0 or h_count == 3:
            continue
        if all(bits[key] == (bool(key[1]), bool(key[0])) for key in keys):
            continue
        return bits
    return None


def _source_anchor_slots(bits: dict[tuple[int, int], tuple[bool, bool]]) -> dict[tuple[int, int], tuple[int, int]]:
    slots = {
        True: [(0, -1), (1, -1)],
        False: [(0, SOURCE_ANCHOR_SIZE), (1, SOURCE_ANCHOR_SIZE)],
    }
    used = {True: 0, False: 0}
    out = {}
    for key in [(0, 1), (1, 0), (1, 1)]:
        hflip = bits[key][0]
        idx = used[hflip]
        assert idx < len(slots[hflip]), "at most two connectors may use the same TL side"
        out[key] = slots[hflip][idx]
        used[hflip] += 1
    return out


def _choose_tl_top_left(
    patch: np.ndarray, anchor_r_in_patch: int, anchor_c_in_patch: int
) -> tuple[int, int]:
    min_r0 = 0
    max_r0 = MID - patch.shape[0]
    min_c0 = 1 if anchor_c_in_patch >= 1 else 0
    max_c0 = min(MID - patch.shape[1], 11 - anchor_c_in_patch)
    assert min_r0 <= max_r0 and min_c0 <= max_c0
    return random.randint(min_r0, max_r0), random.randint(min_c0, max_c0)


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray, vflip: bool) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
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


def _horizontal_stub(
    color: int, anchor_r: int, anchor_c: int, size: int, hflip: bool, slot_row: int
) -> tuple[np.ndarray, int, int]:
    row = anchor_r + slot_row
    if hflip:
        c0 = 0
        c1 = anchor_c - 1
    else:
        c0 = anchor_c + size
        c1 = QUADS[(0, 0)][3]
    assert c0 <= c1, "TL connector stub must have positive length"
    stub = np.full((1, c1 - c0 + 1), color, dtype=int)
    return stub, row, c0


def _vertical_stub(
    color: int, qb: int, qr: int, anchor_r: int, anchor_c: int, size: int, vflip: bool
) -> tuple[np.ndarray, int, int]:
    q_r0, q_r1, _q_c0, _q_c1 = QUADS[(qb, qr)]
    col = anchor_c + random.randint(0, size - 1)
    if vflip:
        r0 = anchor_r + size
        r1 = q_r1
    else:
        r0 = q_r0
        r1 = anchor_r - 1
    assert r0 <= r1, "target connector stub must have positive length"
    stub = np.full((r1 - r0 + 1, 1), color, dtype=int)
    return stub, r0, col


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

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
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    keep = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def _gen_attempt() -> np.ndarray | None:
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
    scaled = {
        key: _scaled_patch(patch, bits[key][0], bits[key][1], target_sizes[key])
        for key in bits
    }
    tl_r0, tl_c0 = _choose_tl_top_left(patch, anchor_r_in_patch, anchor_c_in_patch)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, bits[key][1])

    tl_anchor_r, tl_anchor_c = anchors[(0, 0)]
    source_slots = _source_anchor_slots(bits)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)

    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)

    target_stubs: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    source_stubs: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    for idx, key in enumerate([(0, 1), (1, 0), (1, 1)]):
        color = CONNECTOR_COLORS[idx]
        hflip, vflip = bits[key]
        slot_row, _slot_dc = source_slots[key]
        src_stub, src_r0, src_c0 = _horizontal_stub(color, tl_anchor_r, tl_anchor_c, SOURCE_ANCHOR_SIZE, hflip, slot_row)
        tgt_stub, tgt_r0, tgt_c0 = _vertical_stub(color, key[0], key[1], anchors[key][0], anchors[key][1], target_sizes[key], vflip)
        source_stubs[key] = (src_stub, src_r0, src_c0)
        target_stubs[key] = (tgt_stub, tgt_r0, tgt_c0)
        paste(out, src_stub, src_r0, src_c0)
        paste(out, tgt_stub, tgt_r0, tgt_c0)

    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, (stub, r0, c0) in source_stubs.items():
        paste(inp, stub, r0, c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
        stub, r0, c0 = target_stubs[key]
        paste(inp, stub, r0, c0)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-matched b10624e5 input")
