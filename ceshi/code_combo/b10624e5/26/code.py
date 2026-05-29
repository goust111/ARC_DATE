# Combo variant of b10624e5 (index 26)
# combo_spec: D1/3+D8/0+D12/1
# Name: 显式分割线 + 连接器匹配 + 箭头标记反射

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
# - Original task: the divider was always centered, every non-TL red anchor was
#   automatically a target, and mirror directions came directly from quadrant
#   position.
# - This combo changes three links in that chain at once:
#   1. The divider row/col is still explicit, but it can move away from the
#      center, so the solver must detect the actual split before talking about
#      quadrants.
#   2. Only non-TL anchors that share a repeated connector color with the TL
#      anchor are real targets. One non-TL quadrant is left as a decoy anchor
#      with no connector, so blindly copying into every quadrant is wrong.
#   3. Reflection is read from arrow glyphs inside the matched target quadrant:
#      right-arrow => LR flip, down-arrow => UD flip. Quadrant position no longer
#      determines reflection.
# - Concrete trace:
#   old TR behavior = "copy there because it is TR; apply LR because it is on the
#   right". New TR behavior can be:
#     * no connector at all -> decoy, leave only the red anchor;
#     * connector + only down-arrow -> copy there, but apply UD instead of LR;
#     * connector + both arrows -> copy there with both flips.
#   The same visible quadrant can therefore be inactive or active with different
#   mirrors, depending on connector presence and marker glyphs.
#
# b) Complexity calibration
# - The divider stays a single full row + full column, so segmentation is still a
#   compact first step.
# - Exactly two active targets are used in every sample. That is enough repeated
#   evidence for connector matching while guaranteeing at least one decoy anchor,
#   which prevents D1 from becoming a no-op.
# - Arrow vocabulary stays tiny: only two exact 3x3 glyphs, composing into three
#   transform cases {LR, UD, both}. This keeps D12 visible without exploding the
#   search space.
#
# c) Hidden-assumption audit of the original main()
# - Original `_find_dividers` logic survives, but all later code must stop
#   assuming `(13, 13)` or equal 13x13 quadrants.
# - Original "every red anchor except TL gets processed" is invalid because this
#   combo has an explicit decoy anchor. The solver must first discover which
#   anchors are connector-matched.
# - Original quadrant-based flip logic is invalid. Arrow markers now carry the
#   reflection bits, so spatial position can only identify a region, not a mirror.
# - Original `out = g.copy()` is also invalid because connector lines and marker
#   arrows are instruction artifacts. They are stripped before motifs are stamped.
# - Original TL patch extraction from all non-bg pixels is invalid because TL now
#   contains source-side connector stubs touching the anchor. The solver therefore
#   excludes detected connector colors when extracting the source motif.
#
# d) Placement legality
# - The generator chooses the divider away from the borders so all four explicit
#   regions are large enough for anchors, arrow glyphs, and at least one-cell
#   connector stubs.
# - TL motif templates reserve horizontal space beside the red anchor, so source
#   connector stubs can touch the anchor without overwriting motif pixels.
# - Active target anchors reserve a top band for arrows and at least one extra row
#   below for the vertical target-side connector.
#
# e) Visual observability
# - A human can read the task in three passes:
#   1. find the explicit divider and thus the four regions;
#   2. notice that only two non-TL anchors share repeated connector colors with TL;
#   3. read the arrows in those same quadrants to decide the reflection bits.
# - The generator rejects transform assignments that collapse back to the spatial
#   default, and the always-present decoy anchor makes connector matching visible
#   even if a reflected copy happens to resemble the original.
#
# f) Visual cleanliness
# - Colors are function-separated: divider=1, anchors=2, arrows=3, bg=4,
#   connectors in {5,6,7}, motif colors in {0,8,9}. This avoids ambiguous reuse.
# - Arrows live in a reserved top band of active target quadrants; connectors are
#   thin straight stubs; the final output removes both instruction types so the
#   copied motifs remain readable.
#
# g) Change inferability
# - No hidden color remapping is introduced. Colors only identify object roles.
# - The only nontrivial mapping is connector-color repetition, and it is supported
#   twice per active pair: once in TL and once in exactly one target quadrant.
#
# New solver-side assumptions introduced here and guaranteed by the generator
# - There are exactly four red square anchors, one per explicit region.
# - Exactly two non-TL anchors are active targets; they are the only non-TL
#   quadrants with both a connector color repeated in TL and at least one arrow.
# - Connector colors never appear in the motif, and marker color never appears in
#   the motif or connectors.
# - Every marker component is exactly one of the two glyph templates below.
# ---------------------------------------------------------------------------


BG = 4
DIV = 1
RED = 2
MARKER = 3
GRID_SIZE = 27
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 500
ACTIVE_TARGET_COUNT = 2
MIN_VARIANT_DIFF = 8

CONNECTOR_COLORS = [5, 6, 7]
MOTIF_COLOR_POOL = [0, 8, 9]
MID_CHOICES = list(range(10, 17))

RIGHT_ARROW = frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)})
DOWN_ARROW = frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)})
TARGET_KEYS = [(0, 1), (1, 0), (1, 1)]
TRANSFORM_OPTIONS = [(False, True), (True, False), (True, True)]  # (flip_ud, flip_lr)
SPATIAL_DEFAULTS = {
    (0, 1): (False, True),
    (1, 0): (True, False),
    (1, 1): (True, True),
}

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
    Return the unique full-row/full-col divider pair: (mid_row, mid_col).
    """
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching divider row/col pair"
    return matches[0]


def _quadrant_bounds(mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, GRID_SIZE - 1),
        (1, 0): (mid_r + 1, GRID_SIZE - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, GRID_SIZE - 1, mid_c + 1, GRID_SIZE - 1),
    }


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _transform_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool, target_size: int) -> np.ndarray:
    p = patch
    if flip_lr:
        p = np.fliplr(p)
    if flip_ud:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _normalize_component(pts: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(pts)
    return frozenset((int(r - r0), int(c - c0)) for r, c in pts)


def _is_straight_line(pts: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(pts)
    hh = int(r1 - r0 + 1)
    ww = int(c1 - c0 + 1)
    return (hh == 1 or ww == 1) and int(len(pts)) == hh * ww


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


def _connector_pairs(
    g: np.ndarray,
    anchors: dict[tuple[int, int], tuple[int, int, int]],
    bg: int,
    mid_r: int,
    mid_c: int,
) -> tuple[dict[tuple[int, int], int], set[int]]:
    pairs: dict[tuple[int, int], int] = {}
    connector_colors: set[int] = set()
    forbidden = {int(bg), int(DIV), int(RED), int(MARKER)}
    for color in sorted(set(map(int, g.ravel())) - forbidden):
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
        assert target_key not in pairs, "each active target must have exactly one connector color"
        pairs[target_key] = int(color)
        connector_colors.add(int(color))
    assert 1 <= len(pairs) <= 3, "must find at least one and at most three connector-matched targets"
    return pairs, connector_colors


def _read_marker_transforms(
    g: np.ndarray, mid_r: int, mid_c: int
) -> tuple[dict[tuple[int, int], tuple[bool, bool]], list[list[tuple[int, int]]]]:
    comps = components_mask(g == MARKER, connectivity=4)
    transforms: dict[tuple[int, int], list[bool]] = {}
    grouped: dict[tuple[int, int], int] = {}
    marker_pts = []
    for pts_arr in comps:
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key != (0, 0), "markers are only allowed outside TL"
        if key not in transforms:
            transforms[key] = [False, False]
            grouped[key] = 0
        shape = _normalize_component(pts_arr)
        if shape == RIGHT_ARROW:
            transforms[key][1] = True
        elif shape == DOWN_ARROW:
            transforms[key][0] = True
        else:
            raise AssertionError("unrecognized marker glyph")
        grouped[key] += 1
        marker_pts.append([(int(r), int(c)) for r, c in pts_arr])
    return {key: (bool(v[0]), bool(v[1])) for key, v in transforms.items()}, marker_pts


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int, connector_colors: set[int]) -> np.ndarray:
    tl = g[0:mid_r, 0:mid_c]
    keep = (tl != bg) & (~np.isin(tl, list(connector_colors)))
    comps = components_mask(keep, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "TL quadrant must contain a source motif attached to the red anchor"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    patch[np.isin(patch, list(connector_colors))] = bg
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif into connector-matched regions, scaling by anchor size and mirroring by arrow markers.

    Input structure:
    - One full non-background row and one full non-background column explicitly split the board into four regions; the split need not be centered.
    - Every region contains one red square anchor, but only some non-TL anchors are active targets.
    - Each active target shares a repeated connector color with a TL-side connector stub, and that same target region also contains one or two green arrow glyphs.

    Rule:
    - Detect the explicit divider, locate the four anchors, and identify which non-TL anchors are active by matching repeated connector colors back to TL.
    - Extract the TL motif while ignoring connector colors, then for each active target read its arrows: right-arrow means LR reflection, down-arrow means UD reflection.
    - Scale the transformed motif by the target/source anchor-size ratio, align the red area onto the matched target anchor, stamp it, and remove all connector/marker instruction pixels from the final grid.

    Key insight: The divider only tells you where the four regions are; connector colors decide which anchors participate at all, and arrow glyphs independently decide how each matched copy is reflected.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert len(red) == 4, "grid must contain exactly four red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in anchors, "each region must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one anchor in every explicit region"

    connector_map, connector_colors = _connector_pairs(g, anchors, bg, mid_r, mid_c)
    transforms, marker_components = _read_marker_transforms(g, mid_r, mid_c)
    assert set(transforms) == set(connector_map), "active targets must be exactly the connector-matched targets"

    out = g.copy()
    if connector_colors:
        out[np.isin(out, list(connector_colors))] = bg
    for pts in marker_components:
        for r, c in pts:
            out[r, c] = bg

    _sr0, _sc0, source_size = anchors[(0, 0)]
    assert source_size > 0, "TL anchor must have positive size"
    patch = _extract_source_patch(g, bg, mid_r, mid_c, connector_colors)

    for key in connector_map:
        tr0, tc0, target_size = anchors[key]
        flip_ud, flip_lr = transforms[key]
        patch_t = patch
        if flip_lr:
            patch_t = np.fliplr(patch_t)
        if flip_ud:
            patch_t = np.flipud(patch_t)
        patch_s = _transform_patch(patch, flip_ud, flip_lr, target_size)
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
        logical[r, c] = palette[idx % len(palette)]

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _roundtrip_ok(patch: np.ndarray) -> bool:
    for flip_ud, flip_lr in TRANSFORM_OPTIONS:
        transformed = patch
        if flip_lr:
            transformed = np.fliplr(transformed)
        if flip_ud:
            transformed = np.flipud(transformed)
        for target_size in range(TARGET_SIZE_RANGE[0], TARGET_SIZE_RANGE[1] + 1):
            scaled = _transform_patch(patch, flip_ud, flip_lr, target_size)
            if not np.array_equal(_resize_nn(scaled, transformed.shape[0], transformed.shape[1]), transformed):
                return False
    return True


def _is_asymmetric(patch: np.ndarray) -> bool:
    return (
        not np.array_equal(patch, np.fliplr(patch))
        and not np.array_equal(patch, np.flipud(patch))
        and not np.array_equal(patch, np.flipud(np.fliplr(patch)))
    )


def _choose_dividers() -> tuple[int, int]:
    for _ in range(80):
        mid_r = random.choice(MID_CHOICES)
        mid_c = random.choice(MID_CHOICES)
        if (mid_r, mid_c) == (13, 13):
            continue
        return mid_r, mid_c
    raise AssertionError("failed to choose non-central explicit divider")


def _choose_transforms(active_keys: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[bool, bool]] | None:
    transforms = {key: random.choice(TRANSFORM_OPTIONS) for key in active_keys}
    if len(set(transforms.values())) < len(active_keys):
        return None
    if all(transforms[key] == SPATIAL_DEFAULTS[key] for key in active_keys):
        return None
    return transforms


def _source_stub_slots(n: int) -> list[tuple[str, int]]:
    slots = [("left", 0), ("left", 1), ("right", 0), ("right", 1)]
    return random.sample(slots, n)


def _choose_tl_top_left(
    mid_r: int,
    mid_c: int,
    patch: np.ndarray,
    anchor_r_in_patch: int,
    anchor_c_in_patch: int,
    slots: list[tuple[str, int]],
) -> tuple[int, int] | None:
    min_r0 = 0
    max_r0 = mid_r - patch.shape[0]
    if max_r0 < min_r0:
        return None

    min_c0 = 0
    max_c0 = mid_c - patch.shape[1]
    if max_c0 < min_c0:
        return None

    if any(side == "left" for side, _row in slots):
        min_c0 = max(min_c0, 1 - anchor_c_in_patch)
    if any(side == "right" for side, _row in slots):
        max_c0 = min(max_c0, mid_c - SOURCE_ANCHOR_SIZE - 1 - anchor_c_in_patch)
    if min_c0 > max_c0:
        return None
    return random.randint(min_r0, max_r0), random.randint(min_c0, max_c0)


def _active_anchor_choice(
    bounds: tuple[int, int, int, int],
    size: int,
    patch_s: np.ndarray,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = bounds
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0 = q_r0 + 5
    anch_r1 = q_r1 - size
    anch_c0 = q_c0 + 4
    anch_c1 = q_c1 - size - 1
    if anch_r0 > anch_r1 or anch_c0 > anch_c1:
        return None

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _decoy_anchor_choice(bounds: tuple[int, int, int, int], size: int) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = bounds
    if q_r0 > q_r1 - size + 1 or q_c0 > q_c1 - size + 1:
        return None
    return random.randint(q_r0, q_r1 - size + 1), random.randint(q_c0, q_c1 - size + 1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, mid_r: int, mid_c: int) -> None:
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
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    keep = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def _paint_arrow(grid: np.ndarray, top: int, left: int, shape: frozenset[tuple[int, int]]) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = MARKER


def _place_markers(grid: np.ndarray, bounds: tuple[int, int, int, int], flip_ud: bool, flip_lr: bool) -> None:
    q_r0, _q_r1, q_c0, _q_c1 = bounds
    if flip_lr:
        _paint_arrow(grid, q_r0 + 1, q_c0 + 1, RIGHT_ARROW)
    if flip_ud:
        _paint_arrow(grid, q_r0 + 1, q_c0 + 5, DOWN_ARROW)


def _source_stub(
    color: int,
    bounds: tuple[int, int, int, int],
    anchor_r: int,
    anchor_c: int,
    side: str,
    row_offset: int,
) -> tuple[np.ndarray, int, int]:
    _q_r0, _q_r1, q_c0, q_c1 = bounds
    row = anchor_r + row_offset
    if side == "left":
        c0 = q_c0
        c1 = anchor_c - 1
    else:
        c0 = anchor_c + SOURCE_ANCHOR_SIZE
        c1 = q_c1
    assert c0 <= c1, "source connector stub must have positive length"
    return np.full((1, c1 - c0 + 1), color, dtype=int), row, c0


def _target_stub_side(bounds: tuple[int, int, int, int], anchor_c: int, size: int) -> str:
    _q_r0, _q_r1, q_c0, q_c1 = bounds
    can_left = anchor_c - 1 >= q_c0
    can_right = anchor_c + size <= q_c1
    assert can_left or can_right, "target anchor needs at least one horizontal side for a connector"
    if can_left and can_right:
        return random.choice(["left", "right"])
    return "left" if can_left else "right"


def _target_stub(
    color: int,
    bounds: tuple[int, int, int, int],
    anchor_r: int,
    anchor_c: int,
    size: int,
    side: str,
) -> tuple[np.ndarray, int, int]:
    _q_r0, q_r1, _q_c0, _q_c1 = bounds
    col = anchor_c - 1 if side == "left" else anchor_c + size
    r0 = anchor_r
    r1 = q_r1
    assert r0 <= r1, "target connector stub must have positive length"
    return np.full((r1 - r0 + 1, 1), color, dtype=int), r0, col


def _build_output(
    patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    mid_r: int,
    mid_c: int,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
    active_keys: list[tuple[int, int]],
    transforms: dict[tuple[int, int], tuple[bool, bool]],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key in active_keys:
        patch_s = _transform_patch(patch, transforms[key][0], transforms[key][1], target_sizes[key])
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], mid_r, mid_c)
    return out


def _gen_attempt() -> np.ndarray | None:
    mid_r, mid_c = _choose_dividers()
    quads = _quadrant_bounds(mid_r, mid_c)

    active_keys = sorted(random.sample(TARGET_KEYS, ACTIVE_TARGET_COUNT))
    transforms = _choose_transforms(active_keys)
    if transforms is None:
        return None

    target_sizes = {(0, 0): SOURCE_ANCHOR_SIZE}
    for key in TARGET_KEYS:
        target_sizes[key] = random.randint(*TARGET_SIZE_RANGE)

    patch, anchor_r_in_patch, anchor_c_in_patch = _build_source_patch()
    if not _roundtrip_ok(patch) or not _is_asymmetric(patch):
        return None

    scaled = {
        key: _transform_patch(patch, transforms[key][0], transforms[key][1], target_sizes[key])
        for key in active_keys
    }

    source_slots = _source_stub_slots(len(active_keys))
    tl_pos = _choose_tl_top_left(mid_r, mid_c, patch, anchor_r_in_patch, anchor_c_in_patch, source_slots)
    if tl_pos is None:
        return None
    tl_r0, tl_c0 = tl_pos

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key in TARGET_KEYS:
        if key in active_keys:
            pos = _active_anchor_choice(quads[key], target_sizes[key], scaled[key])
        else:
            pos = _decoy_anchor_choice(quads[key], target_sizes[key])
        if pos is None:
            return None
        anchors[key] = pos

    out = _build_output(patch, tl_r0, tl_c0, mid_r, mid_c, target_sizes, anchors, active_keys, transforms)
    old_out = _build_output(
        patch,
        tl_r0,
        tl_c0,
        mid_r,
        mid_c,
        target_sizes,
        anchors,
        TARGET_KEYS,
        SPATIAL_DEFAULTS,
    )
    if int(np.count_nonzero(out != old_out)) < MIN_VARIANT_DIFF:
        return None

    # Require each active target to reveal at least one non-red copied pixel.
    for key in active_keys:
        q_r0, q_r1, q_c0, q_c1 = quads[key]
        target_view = out[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        if not np.any((target_view != BG) & (target_view != RED)):
            return None

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[mid_r, :] = DIV
    inp[:, mid_c] = DIV
    paste(inp, patch, tl_r0, tl_c0)

    source_stubs: list[tuple[np.ndarray, int, int]] = []
    target_stubs: list[tuple[np.ndarray, int, int]] = []
    used_colors = random.sample(CONNECTOR_COLORS, len(active_keys))
    for color, key, slot in zip(used_colors, active_keys, source_slots, strict=False):
        side, row_offset = slot
        src_stub, src_r0, src_c0 = _source_stub(color, quads[(0, 0)], anchors[(0, 0)][0], anchors[(0, 0)][1], side, row_offset)
        source_stubs.append((src_stub, src_r0, src_c0))
        tgt_side = _target_stub_side(quads[key], anchors[key][1], target_sizes[key])
        tgt_stub, tgt_r0, tgt_c0 = _target_stub(color, quads[key], anchors[key][0], anchors[key][1], target_sizes[key], tgt_side)
        target_stubs.append((tgt_stub, tgt_r0, tgt_c0))

    for stub, r0, c0 in source_stubs:
        paste(inp, stub, r0, c0)

    for key, size in target_sizes.items():
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    for (stub, r0, c0), key in zip(target_stubs, active_keys, strict=False):
        paste(inp, stub, r0, c0)
        _place_markers(inp, quads[key], transforms[key][0], transforms[key][1])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
