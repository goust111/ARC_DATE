# Combo variant of 5dbc8537 (index 19)
# combo_spec: D1/0+D3/2+D4/0
# Name: 边缘色+洞内标记+颜色重映射三重颜色推理

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    The original task only needs rectangle size: a full rectangular sprite is
#    dropped into an equally sized bg hole inside a base-colored strip.
#
#    The combo variant turns that into a three-stage visible inference chain:
#    1. Edge color determines the target hole family.
#       Every hole has a 1-cell stripe on one exposed side, and every sprite has
#       the same cue color on its perimeter.
#    2. In-hole markers determine the exact slot within that family.
#       Hole interiors are mostly bg, but they show 1-2 colored marker pixels.
#       The sprite carries the same marker colors at the same relative offsets.
#    3. After placement, the sprite is recolored to the hole-edge cue color.
#       Output rectangles become solid cue-colored fills, so the source interior
#       colors are not preserved.
#
#    Small trace:
#    - Old: two 4x5 sprites and two 4x5 holes are interchangeable if sizes match.
#    - Combo: the two 4x5 sprites may both have blue borders, so edge color only
#      narrows the target to the two blue-striped holes. One hole shows a red
#      inner marker at (1,1), the other a yellow marker at (2,3). The sprite
#      with the red marker goes to the first hole, the yellow-marked sprite goes
#      to the second. After insertion, both rectangles are repainted solid blue.
#
# b) Complexity calibration
#    The generator explicitly prevents two common degeneracies:
#    - D1 must matter: at least one size group contains multiple edge cue colors.
#      This blocks a pure size-only solver.
#    - D3 must matter: at least one (h, w, cue) group contains multiple distinct
#      marker signatures. This blocks a pure size+cue solver.
#    Recoloring is also forced to be visible: sprite interiors use a body color
#    distinct from the cue, so the output differs from directly pasting input.
#
# c) Implicit-assumption audit against the original main()
#    - Original `hole_mask = dst != base` is invalid now because the template bbox
#      contains three kinds of non-base pixels: edge stripes, hole markers, and
#      the holes themselves.
#    - Original DFS over arbitrary all-hole windows is unnecessary and unsafe:
#      the rectangle identity is now recovered directly from the colored stripe
#      plus the marker signature.
#    - Original `out = full(base)` would erase stripes and markers. The combo
#      solver must start from the cropped template and overwrite only slot bboxes.
#    - Original raw paste preserved sprite colors. The combo solver instead
#      repaints the whole fitted rectangle to the slot's cue color.
#
# d) Placement legality
#    The generator keeps the original corridor-like stacked-rectangle template so
#    each row/column inside the strip contains at most one slot interval. This
#    makes stripe-based slot recovery reliable. Rectangles are kept at least 3x3
#    so they can hold interior markers without conflicting with the cue border.
#
# e) Visual observability
#    Humans can read the rule from a few examples because the three color roles
#    are visually distinct:
#    - stripe color on the template wall,
#    - marker color(s) inside the empty slot,
#    - source interior colors that disappear in the output.
#    The same size appears with multiple cues, and the same cue appears with
#    multiple marker signatures, so each reasoning layer has repeated support.
#
# f) Visual cleanliness
#    Cue stripes stay outside the slot, markers stay inside the slot, and source
#    sprites stay outside the template bbox. The three signal types do not stack
#    on top of each other in the same cells.
#
# g) New solver assumptions and matching generator guarantees
#    - Every sprite is a full rectangle with a uniform cue-colored perimeter.
#    - Every slot has exactly one straight cue stripe and at least one interior
#      marker pixel.
#    - Markers lie strictly inside the slot, not on the perimeter, so they do
#      not conflict with cue-stripe detection.
#    - Within each (h, w, cue) family, marker signatures are unique.
# ---------------------------------------------------------------------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 5, 8
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.34, 0.48
CORRIDOR_COVERAGE = 0.72
MAX_RECT_LEN = 5
MAX_RECT_WIDTH = 5
MIN_SHAPE_GROUPS = 3
MAX_RETRIES = 500


def _largest_non_bg_color(g: np.ndarray, bg: int) -> int:
    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))
    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    return int(base_candidates[0][1])


def _scan_right_until_base(dst: np.ndarray, row: int, c0: int, base: int) -> tuple[int, int]:
    H, W = map(int, dst.shape)
    assert 0 <= row < H and 0 <= c0 < W and int(dst[row, c0]) != int(base)
    c1 = int(c0)
    while c1 + 1 < W and int(dst[row, c1 + 1]) != int(base):
        c1 += 1
    return int(c0), int(c1)


def _scan_left_until_base(dst: np.ndarray, row: int, c1: int, base: int) -> tuple[int, int]:
    H, W = map(int, dst.shape)
    assert 0 <= row < H and 0 <= c1 < W and int(dst[row, c1]) != int(base)
    c0 = int(c1)
    while c0 - 1 >= 0 and int(dst[row, c0 - 1]) != int(base):
        c0 -= 1
    return int(c0), int(c1)


def _scan_down_until_base(dst: np.ndarray, r0: int, col: int, base: int) -> tuple[int, int]:
    H, W = map(int, dst.shape)
    assert 0 <= r0 < H and 0 <= col < W and int(dst[r0, col]) != int(base)
    r1 = int(r0)
    while r1 + 1 < H and int(dst[r1 + 1, col]) != int(base):
        r1 += 1
    return int(r0), int(r1)


def _scan_up_until_base(dst: np.ndarray, r1: int, col: int, base: int) -> tuple[int, int]:
    H, W = map(int, dst.shape)
    assert 0 <= r1 < H and 0 <= col < W and int(dst[r1, col]) != int(base)
    r0 = int(r1)
    while r0 - 1 >= 0 and int(dst[r0 - 1, col]) != int(base):
        r0 -= 1
    return int(r0), int(r1)


def _vertical_stripe_side(dst: np.ndarray, r: int, c: int, base: int) -> str | None:
    H, W = map(int, dst.shape)
    left = int(dst[r, c - 1]) if c - 1 >= 0 else int(base)
    right = int(dst[r, c + 1]) if c + 1 < W else int(base)
    if left == int(base) and right != int(base):
        return "left"
    if right == int(base) and left != int(base):
        return "right"
    return None


def _horizontal_stripe_side(dst: np.ndarray, r: int, c: int, base: int) -> str | None:
    H, W = map(int, dst.shape)
    up = int(dst[r - 1, c]) if r - 1 >= 0 else int(base)
    down = int(dst[r + 1, c]) if r + 1 < H else int(base)
    if up == int(base) and down != int(base):
        return "top"
    if down == int(base) and up != int(base):
        return "bottom"
    return None


def _marker_signature_from_slot(dst: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> tuple[tuple[int, int, int], ...]:
    r0, c0, r1, c1 = map(int, bbox)
    patch = dst[r0 : r1 + 1, c0 : c1 + 1]
    sig = [
        (int(rr), int(cc), int(patch[rr, cc]))
        for rr, cc in np.argwhere(patch != int(bg))
    ]
    sig.sort()
    assert sig
    return tuple(sig)


def _extract_slots(dst: np.ndarray, bg: int, base: int) -> list[dict]:
    H, W = map(int, dst.shape)
    vertical = H > W
    hint_mask = (dst != int(bg)) & (dst != int(base))
    seen = np.zeros((H, W), dtype=bool)
    slots: list[dict] = []

    for r, c in np.argwhere(hint_mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        cue = int(dst[rr, cc])
        if vertical:
            side = _vertical_stripe_side(dst, rr, cc, int(base))
            if side is None:
                continue
            r0 = int(rr)
            while r0 - 1 >= 0 and int(dst[r0 - 1, cc]) == cue and _vertical_stripe_side(dst, r0 - 1, cc, int(base)) == side:
                r0 -= 1
            r1 = int(rr)
            while r1 + 1 < H and int(dst[r1 + 1, cc]) == cue and _vertical_stripe_side(dst, r1 + 1, cc, int(base)) == side:
                r1 += 1
            seen[r0 : r1 + 1, cc] = True
            if side == "left":
                c0 = int(cc + 1)
                c0a, c1 = _scan_right_until_base(dst, r0, c0, int(base))
                assert c0a == c0
            else:
                c1 = int(cc - 1)
                c0, c1a = _scan_left_until_base(dst, r0, c1, int(base))
                assert c1a == c1
            for rrr in range(r0, r1 + 1):
                assert int(dst[rrr, cc]) == cue
                if side == "left":
                    got = _scan_right_until_base(dst, rrr, c0, int(base))
                else:
                    got = _scan_left_until_base(dst, rrr, c1, int(base))
                assert got == (c0, c1)
        else:
            side = _horizontal_stripe_side(dst, rr, cc, int(base))
            if side is None:
                continue
            c0 = int(cc)
            while c0 - 1 >= 0 and int(dst[rr, c0 - 1]) == cue and _horizontal_stripe_side(dst, rr, c0 - 1, int(base)) == side:
                c0 -= 1
            c1 = int(cc)
            while c1 + 1 < W and int(dst[rr, c1 + 1]) == cue and _horizontal_stripe_side(dst, rr, c1 + 1, int(base)) == side:
                c1 += 1
            seen[rr, c0 : c1 + 1] = True
            if side == "top":
                r0 = int(rr + 1)
                r0a, r1 = _scan_down_until_base(dst, r0, c0, int(base))
                assert r0a == r0
            else:
                r1 = int(rr - 1)
                r0, r1a = _scan_up_until_base(dst, r1, c0, int(base))
                assert r1a == r1
            for ccc in range(c0, c1 + 1):
                assert int(dst[rr, ccc]) == cue
                if side == "top":
                    got = _scan_down_until_base(dst, r0, ccc, int(base))
                else:
                    got = _scan_up_until_base(dst, r1, ccc, int(base))
                assert got == (r0, r1)

        bbox = (int(r0), int(c0), int(r1), int(c1))
        sig = _marker_signature_from_slot(dst, bbox, int(bg))
        slots.append(
            {
                "bbox": bbox,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "cue": int(cue),
                "markers": sig,
            }
        )

    assert slots
    slot_bboxes = [tuple(map(int, s["bbox"])) for s in slots]
    assert len(slot_bboxes) == len(set(slot_bboxes))
    return slots


def _sprite_key(patch: np.ndarray) -> tuple[int, int, int, tuple[tuple[int, int, int], ...]]:
    h, w = map(int, patch.shape)
    border = set(map(int, patch[0, :]))
    border.update(map(int, patch[h - 1, :]))
    border.update(map(int, patch[:, 0]))
    border.update(map(int, patch[:, w - 1]))
    assert len(border) == 1
    cue = int(next(iter(border)))
    assert h >= 3 and w >= 3
    interior = patch[1:-1, 1:-1]
    fill_color = int(Counter(map(int, interior.flatten())).most_common(1)[0][0])
    sig = [
        (int(rr + 1), int(cc + 1), int(interior[rr, cc]))
        for rr, cc in np.argwhere(interior != fill_color)
    ]
    sig.sort()
    assert sig
    return int(h), int(w), int(cue), tuple(sig)


def _recolor_to_cue(patch: np.ndarray, cue: int) -> np.ndarray:
    return np.full_like(patch, int(cue))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill stripe-marked cavities in a template strip by matching edge colors, aligning marker pixels, and then recoloring.

    Input structure:
    - The most common color is the background. One large non-background color forms the template strip.
    - Each destination slot is a rectangle inside that strip. It has a colored side stripe on the wall next to it and 1-2 colored marker pixels inside the otherwise bg interior.
    - Outside the strip are full rectangular sprites whose perimeter is uniformly colored and whose interior contains matching marker pixels.

    Rule:
    - The largest non-background connected component identifies the base strip and its tight output bbox.
    - Recover each slot from its wall stripe, read the cue color from that stripe, and read the marker signature from inside the slot.
    - Match each outside sprite to the unique slot with the same rectangle size, the same cue-colored perimeter, and the same marker signature.
    - After placing the sprite, repaint the entire rectangle to the slot's cue color and return the cropped template.

    Key insight: Size alone is insufficient, and even edge color alone is insufficient for some duplicates. The stripe color chooses the target family, the in-hole markers choose the exact slot, and that same stripe color becomes the final painted output color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _largest_non_bg_color(g, int(bg))

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

    slots = _extract_slots(dst, int(bg), int(base_col))
    slot_by_key: dict[tuple[int, int, int, tuple[tuple[int, int, int], ...]], tuple[int, int, int, int]] = {}
    for slot in slots:
        key = (int(slot["h"]), int(slot["w"]), int(slot["cue"]), slot["markers"])
        assert key not in slot_by_key
        slot_by_key[key] = tuple(map(int, slot["bbox"]))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, _pr1, _pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert bool(m.all())
        sprites.append({"patch": patch, "key": _sprite_key(patch)})

    assert len(sprites) == len(slots)
    sprite_keys = [sp["key"] for sp in sprites]
    assert len(sprite_keys) == len(set(sprite_keys))
    assert set(sprite_keys) == set(slot_by_key.keys())

    out = np.array(dst, copy=True)
    for sp in sprites:
        h, w, cue, _sig = sp["key"]
        rr0, cc0, _rr1, _cc1 = slot_by_key[sp["key"]]
        recolored = _recolor_to_cue(sp["patch"], int(cue))
        paste(out, recolored, int(rr0), int(cc0), mask=np.ones((h, w), dtype=bool), overwrite=True)

    return out


def _shape_groups(rects: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        groups.setdefault((int(h), int(w)), []).append(int(i))
    return groups


def _marker_candidates(h: int, w: int) -> list[tuple[int, int]]:
    pts = [
        (1, 1),
        (1, w - 2),
        (h - 2, 1),
        (h - 2, w - 2),
        (1, w // 2),
        (h // 2, 1),
        (h // 2, w - 2),
        (h - 2, w // 2),
        (h // 2, w // 2),
    ]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in pts:
        if 0 < p[0] < h - 1 and 0 < p[1] < w - 1 and p not in seen:
            seen.add(p)
            out.append((int(p[0]), int(p[1])))
    assert out
    return out


def _make_signature_pool(h: int, w: int, marker_palette: list[int]) -> list[tuple[tuple[int, int, int], ...]]:
    anchors = _marker_candidates(int(h), int(w))
    pool: set[tuple[tuple[int, int, int], ...]] = set()
    for r, c in anchors:
        for color in marker_palette:
            pool.add(((int(r), int(c), int(color)),))
    out = list(pool)
    random.shuffle(out)
    return out


def _split_group_counts(total: int) -> list[int] | None:
    max_groups = min(5, total - 2)
    if max_groups < MIN_SHAPE_GROUPS:
        return None
    for _ in range(100):
        n_groups = random.randint(MIN_SHAPE_GROUPS, max_groups)
        counts = [1] * n_groups
        fusion_idx = random.randrange(n_groups)
        counts[fusion_idx] = 3
        remaining = total - sum(counts)
        if remaining < 0:
            continue
        for _ in range(remaining):
            choices = [i for i, v in enumerate(counts) if v < 5]
            if not choices:
                break
            counts[random.choice(choices)] += 1
        if sum(counts) == total:
            random.shuffle(counts)
            return counts
    return None


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 3:
        return None
    full_area = int(long * thick)
    lo = int(np.ceil(HOLE_FRAC_LO * full_area))
    hi = int(np.floor(HOLE_FRAC_HI * full_area))
    min_len = max(3 * count, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1

    for _ in range(240):
        counts = _split_group_counts(count)
        if counts is None:
            return None
        shapes: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        ok = True
        for _cnt in counts:
            picked = None
            for _ in range(80):
                h = random.randint(3, MAX_RECT_LEN)
                w = random.randint(3, max_w)
                if (h - 2) * (w - 2) < 2:
                    continue
                shape = (int(h), int(w))
                if shape in used:
                    continue
                picked = shape
                break
            if picked is None:
                ok = False
                break
            used.add(picked)
            shapes.append(picked)
        if not ok:
            continue

        hole_len = sum(int(cnt) * int(shape[0]) for cnt, shape in zip(counts, shapes))
        area = sum(int(cnt) * int(shape[0]) * int(shape[1]) for cnt, shape in zip(counts, shapes))
        if not (min_len <= hole_len <= long - 1 and lo <= area <= hi):
            continue

        dims: list[tuple[int, int]] = []
        for cnt, shape in zip(counts, shapes):
            dims.extend([shape] * cnt)
        random.shuffle(dims)

        start = random.randint(0, long - hole_len)
        r = start
        prev_c = None
        prev_w = None
        rects: list[tuple[int, int, int, int]] = []
        for h, w in dims:
            c_hi = thick - w - 1
            if c_hi < 1:
                ok = False
                break
            if prev_c is None:
                c = random.randint(1, c_hi)
            else:
                c_lo = max(1, int(prev_c) - int(w) + 1)
                c_hi2 = min(c_hi, int(prev_c) + int(prev_w) - 1)
                if c_lo > c_hi2:
                    ok = False
                    break
                c = random.randint(c_lo, c_hi2)
            rects.append((int(r), int(c), int(h), int(w)))
            r += h
            prev_c, prev_w = c, w
        if ok:
            groups = _shape_groups(rects)
            if len(groups) >= MIN_SHAPE_GROUPS and any(len(ids) >= 3 for ids in groups.values()):
                return rects
    return None


def _assign_cues_and_sides(
    rects: list[tuple[int, int, int, int]],
    cue_palette: list[int],
    vertical: bool,
) -> tuple[list[int], list[str]] | None:
    groups = _shape_groups(rects)
    fusion_keys = [key for key, ids in groups.items() if len(ids) >= 3]
    if not fusion_keys:
        return None

    cues = [-1] * len(rects)
    fusion_key = random.choice(fusion_keys)
    ids = groups[fusion_key][:]
    random.shuffle(ids)
    primary, secondary = random.sample(cue_palette, 2)
    cues[ids[0]] = int(primary)
    cues[ids[1]] = int(primary)
    cues[ids[2]] = int(secondary)
    for idx in ids[3:]:
        cues[idx] = int(random.choice(cue_palette))

    for key, key_ids in groups.items():
        if key == fusion_key:
            continue
        if len(key_ids) >= 2 and random.random() < 0.65:
            a, b = random.sample(cue_palette, 2)
            order = key_ids[:]
            random.shuffle(order)
            for j, idx in enumerate(order):
                cues[idx] = int(a if j % 2 == 0 else b)
        else:
            cue = int(random.choice(cue_palette))
            for idx in key_ids:
                cues[idx] = cue

    assert all(c >= 0 for c in cues)
    side_opts = ["left", "right"] if vertical else ["top", "bottom"]
    order = sorted(range(len(rects)), key=lambda i: (rects[i][0], rects[i][1]))
    sides = [""] * len(rects)
    prev_idx = None
    for idx in order:
        opts = side_opts[:]
        if prev_idx is not None and int(cues[idx]) == int(cues[prev_idx]) and sides[prev_idx] in opts:
            opts.remove(sides[prev_idx])
        if not opts:
            return None
        sides[idx] = random.choice(opts)
        prev_idx = idx

    return [int(c) for c in cues], sides


def _build_pieces(
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    body_color: int,
    marker_palette: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, tuple[tuple[int, int, int], ...]]] | None:
    by_family: dict[tuple[int, int, int], list[int]] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        by_family.setdefault((int(h), int(w), int(cues[i])), []).append(int(i))

    pieces: dict[int, np.ndarray] = {}
    signatures: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for (h, w, cue), ids in by_family.items():
        pool = _make_signature_pool(int(h), int(w), marker_palette)
        if len(pool) < len(ids):
            return None
        chosen = pool[: len(ids)]
        for idx, sig in zip(ids, chosen):
            patch = np.full((h, w), int(cue), dtype=int)
            patch[1:-1, 1:-1] = int(body_color)
            for mr, mc, color in sig:
                patch[int(mr), int(mc)] = int(color)
            pieces[idx] = patch
            signatures[idx] = sig

    assert len(pieces) == len(rects)
    return pieces, signatures


def _render_template(
    dh: int,
    dw: int,
    bg: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    sides: list[str],
    signatures: dict[int, tuple[tuple[int, int, int], ...]],
) -> np.ndarray:
    template = np.full((dh, dw), int(base), dtype=int)
    vertical = dh > dw
    for i, (r, c, h, w) in enumerate(rects):
        template[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = int(bg)
        for mr, mc, color in signatures[i]:
            template[int(r) + int(mr), int(c) + int(mc)] = int(color)
        cue = int(cues[i])
        side = sides[i]
        if vertical and side == "left":
            template[int(r) : int(r) + int(h), int(c) - 1] = cue
        elif vertical:
            template[int(r) : int(r) + int(h), int(c) + int(w)] = cue
        elif side == "top":
            template[int(r) - 1, int(c) : int(c) + int(w)] = cue
        else:
            template[int(r) + int(h), int(c) : int(c) + int(w)] = cue
    return template


def _render_output(
    template: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
) -> np.ndarray:
    out = np.array(template, copy=True)
    for i, (r, c, h, w) in enumerate(rects):
        out[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = int(cues[i])
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    template: np.ndarray,
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    dh, dw = map(int, template.shape)
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = template

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(pieces.keys())
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=int(bg), padding=1, padding_connectivity=8)
        paste(inp, patch, int(r), int(c))
        paste(placement, patch, int(r), int(c))
    return inp


def _fusion_metrics_from_layout(
    rects: list[tuple[int, int, int, int]],
    cues: list[int],
    pieces: dict[int, np.ndarray],
) -> dict[str, int]:
    by_shape: dict[tuple[int, int], set[int]] = {}
    by_family: dict[tuple[int, int, int], set[tuple[tuple[int, int, int], ...]]] = {}
    recolor_changed = 0
    for i, (_r, _c, h, w) in enumerate(rects):
        by_shape.setdefault((int(h), int(w)), set()).add(int(cues[i]))
        sig = _sprite_key(pieces[i])[3]
        by_family.setdefault((int(h), int(w), int(cues[i])), set()).add(sig)
        if not np.array_equal(pieces[i], _recolor_to_cue(pieces[i], int(cues[i]))):
            recolor_changed += 1
    edge_groups = sum(1 for cueset in by_shape.values() if len(cueset) >= 2)
    marker_groups = sum(1 for sigs in by_family.values() if len(sigs) >= 2)
    return {
        "edge_groups": int(edge_groups),
        "marker_groups": int(marker_groups),
        "recolor_changed": int(recolor_changed),
    }


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base = _largest_non_bg_color(g, int(bg))
    base_coords = np.argwhere(g == int(base))
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    slots = _extract_slots(dst, int(bg), int(base))

    by_shape: dict[tuple[int, int], set[int]] = {}
    by_family: dict[tuple[int, int, int], set[tuple[tuple[int, int, int], ...]]] = {}
    for slot in slots:
        shape = (int(slot["h"]), int(slot["w"]))
        cue = int(slot["cue"])
        by_shape.setdefault(shape, set()).add(cue)
        by_family.setdefault((shape[0], shape[1], cue), set()).add(slot["markers"])

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    changed = 0
    for comp in sprite_comps:
        patch = crop_bbox(g, tuple(map(int, comp["bbox"])))
        _h, _w, cue, _sig = _sprite_key(patch)
        if not np.array_equal(patch, _recolor_to_cue(patch, int(cue))):
            changed += 1

    return {
        "edge_groups": int(sum(1 for v in by_shape.values() if len(v) >= 2)),
        "marker_groups": int(sum(1 for v in by_family.values() if len(v) >= 2)),
        "recolor_changed": int(changed),
    }


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    canon = _corridor_rects(long, thick, total)
    if canon is None:
        return None
    rects = canon if vertical else [(int(c), int(r), int(w), int(h)) for r, c, h, w in canon]
    groups = _shape_groups(rects)
    if len(groups) < MIN_SHAPE_GROUPS or not any(len(ids) >= 3 for ids in groups.values()):
        return None

    bg, base = random.sample(range(10), 2)
    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < 6:
        return None
    random.shuffle(available)
    cue_palette = available[:3]
    body_color = int(available[3])
    marker_palette = [int(c) for c in available[4:7]]
    marker_palette = [c for c in marker_palette if c != body_color]
    if len(marker_palette) < 2:
        return None

    assigned = _assign_cues_and_sides(rects, cue_palette, vertical)
    if assigned is None:
        return None
    cues, sides = assigned

    built = _build_pieces(rects, cues, int(body_color), marker_palette)
    if built is None:
        return None
    pieces, signatures = built

    metrics = _fusion_metrics_from_layout(rects, cues, pieces)
    if metrics["edge_groups"] < 1 or metrics["marker_groups"] < 1:
        return None
    if metrics["recolor_changed"] < max(3, len(rects) - 1):
        return None

    template = _render_template(dh, dw, int(bg), int(base), rects, cues, sides, signatures)
    out = _render_output(template, rects, cues)

    try:
        inp = _scatter_sprites(H, W, int(bg), int(base), tr0, tc0, template, pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
