# Combo variant of b10624e5 (index 38)
# combo_spec: D1/2+D9/3
# Name: 锚块数量配对 + 分散多色块锚块

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original: each quadrant had one connected red square; its side length set the scale.
#    Fused: each quadrant has one logical anchor group made of several separated red pixels.
#    Example: TL may contain three red pixels on a 3x3 diagonal, while BR contains four on a 4x4
#    diagonal. The solver must first aggregate "all red fragments in the quadrant" into one anchor
#    group, count the fragments (3 vs 4), then scale the TL motif by 4/3 before reflecting it into BR.
#    Treating each red fragment as an independent anchor would produce multiple false targets.
# b) Complexity calibration:
#    Counts are restricted to {2,3,4}. This is visibly different from the original connected squares,
#    but still small enough that humans can count by eye from 2-4 examples. Generator enforces at
#    least two distinct counts per sample so count-driven scaling is observable.
# c) Broken assumptions from the original solver:
#    - "one red connected component per quadrant" is no longer true;
#    - "anchor size = connected-square side length" is no longer true;
#    - "the red region inside the copied patch should be pasted onto targets" is no longer true.
#    The new solver instead unions all quadrant red pixels into one logical anchor group, uses the
#    red-pixel count as the scale key, and pastes only non-red motif pixels so the target's dispersed
#    red anchor remains the reference object.
# d) Placement legality:
#    The generator places exactly one anchor group per quadrant, keeps every scaled motif inside its
#    quadrant, and chooses anchor positions from the intersection of "anchor fits" and "whole motif fits".
# e) Visual observability:
#    All non-TL quadrants show only compact diagonal red fragment groups in the input; outputs add
#    mirrored motifs whose sizes clearly track the number of red fragments. The repeated-count cases
#    provide support that count, not absolute position, controls scale.
# f) Visual cleanliness:
#    Only one motif lives in each quadrant; copied ornaments do not overlap the divider or other
#    quadrants. Red anchors are kept untouched, so the reference marks stay easy to read.
# g) Inferability:
#    No color remapping is introduced. The only new latent variable is anchor-group fragment count,
#    and it is directly visible as a small cluster of separated red pixels.


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

COUNT_CHOICES = (2, 3, 4)
LOGICAL_DIMS = ((2, 2), (2, 3), (3, 2), (3, 3))
COLOR_POOL = [3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider pair."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = [(int(r), int(c)) for r, vr in full_rows for c, vc in full_cols if vr == vc]
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_pixels(count: int) -> list[tuple[int, int]]:
    return [(i, i) for i in range(int(count))]


def _paint_anchor_group(out: np.ndarray, top: int, left: int, count: int) -> None:
    for dr, dc in _anchor_pixels(count):
        out[int(top) + dr, int(left) + dc] = RED


def _quadrant_array(g: np.ndarray, key: tuple[int, int], mid_r: int, mid_c: int) -> np.ndarray:
    qb, qr = key
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r if qb == 0 else g.shape[0]
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c if qr == 0 else g.shape[1]
    return g[r0:r1, c0:c1]


def _quadrant_origin(key: tuple[int, int], mid_r: int, mid_c: int) -> tuple[int, int]:
    qb, qr = key
    return (0 if qb == 0 else mid_r + 1, 0 if qr == 0 else mid_c + 1)


def _detect_anchor_group(quad: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(quad == RED)
    assert coords.size > 0, "each quadrant must contain a red anchor group"
    r0, c0, r1, c1 = bbox_coords(coords)
    count = int(coords.shape[0])
    assert r1 - r0 + 1 == count and c1 - c0 + 1 == count, "anchor group bbox must match its fragment count"
    assert len(components_mask(quad == RED, connectivity=4)) == count, "anchor fragments must remain separated"
    return int(r0), int(c0), count


def _stamp_non_red(out: np.ndarray, patch: np.ndarray, top: int, left: int, mid_r: int, mid_c: int) -> None:
    h, w = out.shape
    rr, cc = np.where((patch != BG) & (patch != RED))
    for pr, pc in zip(rr.tolist(), cc.tolist(), strict=False):
        tr = int(top) + pr
        tc = int(left) + pc
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        if tr == mid_r or tc == mid_c:
            continue
        out[tr, tc] = int(patch[pr, pc])


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _scale_patch(patch: np.ndarray, source_count: int, target_count: int) -> np.ndarray:
    assert patch.shape[0] % source_count == 0 and patch.shape[1] % source_count == 0
    base_h = patch.shape[0] // source_count
    base_w = patch.shape[1] // source_count
    return _resize_nn(patch, base_h * target_count, base_w * target_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants using dispersed red anchor groups as the scale key.

    Input structure:
    - A single full-row/full-column divider splits the board into four 13x13 quadrants.
    - Every quadrant contains one logical anchor group made of several disconnected red pixels.
    - The top-left quadrant also contains the full colored motif; the other quadrants contain only their red anchor groups.

    Rule:
    - Treat all red pixels inside one quadrant as one anchor group rather than as independent anchors.
    - Count the red fragments in the top-left group and in each target group; this fragment count replaces the original square-size ratio.
    - Reflect the top-left motif left/right for right quadrants and up/down for bottom quadrants, scale it by target_count / source_count, align the scaled source anchor cell to the target group's bbox, and stamp only non-red motif pixels.

    Key insight:
    - The ambiguous red fragments must first be aggregated into one logical anchor group per quadrant.
    - Once grouped, the number of fragments encodes the scale, while quadrant position still encodes reflection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for key in QUADS:
        quad = _quadrant_array(g, key, mid_r, mid_c)
        ar0, ac0, count = _detect_anchor_group(quad)
        base_r, base_c = _quadrant_origin(key, mid_r, mid_c)
        anchors[key] = (base_r + ar0, base_c + ac0, count)

    tl = _quadrant_array(g, (0, 0), mid_r, mid_c)
    src_non_bg = np.argwhere(tl != bg)
    assert src_non_bg.size > 0, "top-left quadrant must contain the source motif"
    pr0, pc0, pr1, pc1 = bbox_coords(src_non_bg)
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    sr0, sc0, source_count = anchors[(0, 0)]
    sr0 -= pr0
    sc0 -= pc0
    assert sr0 % source_count == 0 and sc0 % source_count == 0, "source anchor must align to the logical cell grid"
    assert patch.shape[0] % source_count == 0 and patch.shape[1] % source_count == 0, "source patch must be count-aligned"

    for (qb, qr), (tr0, tc0, target_count) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = _reflect_patch(patch, qb, qr)
        patch_s = _scale_patch(patch_t, source_count, target_count)

        rr_red, cc_red = np.where(patch_t == RED)
        assert rr_red.size == source_count, "reflected source patch must preserve the source anchor fragments"
        src_anchor_r0 = int(rr_red.min())
        src_anchor_c0 = int(cc_red.min())
        assert src_anchor_r0 % source_count == 0 and src_anchor_c0 % source_count == 0

        scaled_anchor_r0 = (src_anchor_r0 // source_count) * target_count
        scaled_anchor_c0 = (src_anchor_c0 // source_count) * target_count
        place_r0 = tr0 - scaled_anchor_r0
        place_c0 = tc0 - scaled_anchor_c0
        _stamp_non_red(out, patch_s, place_r0, place_c0, mid_r, mid_c)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _make_source_patch(source_count: int) -> tuple[np.ndarray, int, int] | None:
    cell_h, cell_w = random.choice(LOGICAL_DIMS)
    anchor_cell = (random.randrange(cell_h), random.randrange(cell_w))
    n_fill = random.randint(3, cell_h * cell_w)
    filled = _grow_4conn(cell_h, cell_w, {anchor_cell}, n_fill)
    bbox = bbox_coords(np.array(list(filled), dtype=int))
    r0, c0, r1, c1 = bbox
    filled = {(r - r0, c - c0) for r, c in filled}
    anchor_cell = (anchor_cell[0] - r0, anchor_cell[1] - c0)
    logical_h = r1 - r0 + 1
    logical_w = c1 - c0 + 1

    non_anchor = [p for p in filled if p != anchor_cell]
    if len(non_anchor) < 2:
        return None
    n_colors = random.randint(2, min(4, len(non_anchor)))
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    chosen = pool[:n_colors]
    random.shuffle(non_anchor)

    logical = np.full((logical_h, logical_w), BG, dtype=int)
    for i, (r, c) in enumerate(non_anchor):
        logical[r, c] = chosen[i] if i < n_colors else random.choice(chosen)

    patch = np.full((logical_h * source_count, logical_w * source_count), BG, dtype=int)
    for r, c in non_anchor:
        block = np.full((source_count, source_count), int(logical[r, c]), dtype=int)
        paste(patch, block, r * source_count, c * source_count)

    anchor_r0 = anchor_cell[0] * source_count
    anchor_c0 = anchor_cell[1] * source_count
    for dr, dc in _anchor_pixels(source_count):
        patch[anchor_r0 + dr, anchor_c0 + dc] = RED

    if len(np.unique(patch[patch != BG])) < 3:
        return None
    return patch, anchor_r0, anchor_c0


def _choose_target_counts(source_count: int) -> dict[tuple[int, int], int]:
    values = list(COUNT_CHOICES)
    if random.random() < 0.7:
        repeat = random.choice(values)
        other_choices = [v for v in values if v != repeat]
        other = random.choice(other_choices)
        targets = [repeat, repeat, other]
    else:
        targets = random.sample(values, 3)
    random.shuffle(targets)
    if len(set(targets + [source_count])) < 2 or all(v == source_count for v in targets):
        return _choose_target_counts(source_count)
    return {(0, 0): source_count, (0, 1): targets[0], (1, 0): targets[1], (1, 1): targets[2]}


def _choose_anchor_top_left(
    qb: int,
    qr: int,
    patch_t: np.ndarray,
    patch_scaled: np.ndarray,
    source_count: int,
    target_count: int,
) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr_red, cc_red = np.where(patch_t == RED)
    src_anchor_r0 = int(rr_red.min())
    src_anchor_c0 = int(cc_red.min())
    scaled_anchor_r0 = (src_anchor_r0 // source_count) * target_count
    scaled_anchor_c0 = (src_anchor_c0 // source_count) * target_count

    r_lo = q_r0 + scaled_anchor_r0
    r_hi = q_r1 - patch_scaled.shape[0] + 1 + scaled_anchor_r0
    c_lo = q_c0 + scaled_anchor_c0
    c_hi = q_c1 - patch_scaled.shape[1] + 1 + scaled_anchor_c0
    if r_lo > r_hi or c_lo > c_hi:
        return q_r0, q_c0
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _build_scaled_patch(patch: np.ndarray, qb: int, qr: int, source_count: int, target_count: int) -> np.ndarray:
    return _scale_patch(_reflect_patch(patch, qb, qr), source_count, target_count)


def _output_from_layout(
    patch: np.ndarray,
    source_count: int,
    tl_patch_r0: int,
    tl_patch_c0: int,
    target_counts: dict[tuple[int, int], int],
    target_anchor_tops: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_patch_r0, tl_patch_c0)
    for key, count in target_counts.items():
        if key == (0, 0):
            continue
        top, left = target_anchor_tops[key]
        _paint_anchor_group(out, top, left, count)
        patch_t = _reflect_patch(patch, key[0], key[1])
        patch_s = _scale_patch(patch_t, source_count, count)
        rr_red, cc_red = np.where(patch_t == RED)
        src_anchor_r0 = int(rr_red.min())
        src_anchor_c0 = int(cc_red.min())
        place_r0 = top - (src_anchor_r0 // source_count) * count
        place_c0 = left - (src_anchor_c0 // source_count) * count
        _stamp_non_red(out, patch_s, place_r0, place_c0, MID, MID)
    return out


def _gen_attempt() -> np.ndarray | None:
    source_count = random.choice((2, 3))
    made = _make_source_patch(source_count)
    if made is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = made
    if patch.shape[0] > MID or patch.shape[1] > MID:
        return None

    target_counts = _choose_target_counts(source_count)

    tl_patch_r0 = random.randint(0, MID - patch.shape[0])
    tl_patch_c0 = random.randint(0, MID - patch.shape[1])

    target_anchor_tops: dict[tuple[int, int], tuple[int, int]] = {}
    for key, count in target_counts.items():
        if key == (0, 0):
            continue
        patch_t = _reflect_patch(patch, key[0], key[1])
        patch_s = _scale_patch(patch_t, source_count, count)
        if patch_s.shape[0] > MID or patch_s.shape[1] > MID:
            return None
        top, left = _choose_anchor_top_left(key[0], key[1], patch_t, patch_s, source_count, count)
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        if top + count - 1 > q_r1 or left + count - 1 > q_c1:
            return None
        target_anchor_tops[key] = (top, left)

    out = _output_from_layout(
        patch=patch,
        source_count=source_count,
        tl_patch_r0=tl_patch_r0,
        tl_patch_c0=tl_patch_c0,
        target_counts=target_counts,
        target_anchor_tops=target_anchor_tops,
    )

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_patch_r0, tl_patch_c0)
    for key, count in target_counts.items():
        if key == (0, 0):
            continue
        top, left = target_anchor_tops[key]
        _paint_anchor_group(inp, top, left, count)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused b10624e5 input")
