# Variant of b10624e5
# Dimension: D12 — 符号编码：图例或参数标记 (symbol_instructions)
# Change: 无符号编码——缩放因子和反射轴均从空间关系隐式推导... → 锚块边缘像素编码旋转角度

"""
Design analysis for the D12 variant.

spec:
  concrete_variant:
    preserved:
      - 27x27 grid, bg=4, divider row/col color=1 at index 13
      - exactly one red square anchor per quadrant
      - TL quadrant contains the full source motif attached to its 2x2 red anchor
      - target anchor size still encodes the scale ratio
    changed:
      - non-TL quadrants now contain a short instruction marker in addition to the red anchor
      - marker is a 1-cell-thick ray of a fixed non-red color attached to one side of the anchor
      - marker length encodes clockwise rotation: 1 -> 90 deg, 2 -> 180 deg, 3 -> 270 deg
      - quadrant position no longer determines the transform; reflection is removed
    marker_geometry:
      - the ray starts from the middle cell of one anchor side and extends outward
      - side choice is random but semantically irrelevant; only the pixel count matters
      - TL source quadrant has no instruction marker

  behavior_trace:
    original:
      - if the target anchor was in the right half, the patch was mirrored left/right
      - if the target anchor was in the bottom half, the patch was mirrored up/down
    variant_example:
      - suppose TR has a 3x3 red anchor and a 2-pixel marker attached to its left side
      - old rule would mirror the TL motif horizontally because TR is on the right
      - new rule scales by 3/2 as before, then rotates the motif 180 degrees because the marker count is 2
      - the quadrant itself is no longer a transform cue; the counted symbol is

  complexity_calibration:
    - every generated sample uses all three counts {1,2,3} exactly once across TR/BL/BR
    - source motifs are forced to be asymmetric under flips and 90/180/270 degree rotations
    - this prevents the new rotation rule from visually collapsing back into the old reflection rule
    - marker color is fixed and reserved, so humans can read it as a symbol instead of as motif content

  implicit_assumption_review:
    - original main() assumed qb/qr directly mapped to flip axes; this is invalid now and replaced by marker decoding
    - original stamping skipped only existing red pixels; that would overwrite the new instruction markers
    - the variant therefore treats any pre-existing non-bg target pixel as protected input structure
    - marker decoding assumes each non-TL quadrant contains only the red anchor plus one attached marker component
    - generator guarantees this by clearing those quadrants before re-placing anchor + marker

  placement_legality:
    - target anchors are placed only where the full scaled rotated motif fits inside the quadrant
    - marker rays are also required to fit entirely inside the quadrant
    - marker side is chosen from sides that do not intersect the transformed motif, so the symbol stays readable

  visual_observability:
    - the three target quadrants visibly display marker lengths 1, 2, 3 in the same puzzle
    - the source motif uses multiple colors and an asymmetric layout, making 90/180/270 degree outputs distinguishable
    - no color-id ordering or hidden numeric color logic is used; only visible count and geometry matter

  visual_cleanliness:
    - motif colors exclude the divider and marker colors to reduce clutter
    - generator rejects placements where the marker would sit on top of the stamped motif

  new_main_assumptions_checked_against_generator:
    - main() expects one TL component containing the source red anchor; generator builds exactly one
    - main() expects each target quadrant to expose a marker count in {1,2,3}; generator always places one
    - main() expects scaling roundtrip to remain lossless; generator only emits block-based motifs that satisfy it
"""

import random
from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider."""
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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _rotate_cw(img: np.ndarray, turns: int) -> np.ndarray:
    """Rotate clockwise by turns * 90 degrees."""
    turns = int(turns) % 4
    if turns == 0:
        return img.copy()
    return np.rot90(img, -turns)


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, qb: int, qr: int) -> tuple[int, int, int, int]:
    """Inclusive bounds for one quadrant."""
    h, w = shape
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r - 1 if qb == 0 else h - 1
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c - 1 if qr == 0 else w - 1
    return int(r0), int(r1), int(c0), int(c1)


def _marker_rel_cells(red_r0: int, red_c0: int, size: int, side: str, length: int) -> list[tuple[int, int]]:
    """Marker cells in patch coordinates, using the anchor-side midpoint as the attachment cell."""
    mid_r = red_r0 + (size - 1) // 2
    mid_c = red_c0 + (size - 1) // 2
    if side == "top":
        return [(red_r0 - 1 - k, mid_c) for k in range(length)]
    if side == "bottom":
        return [(red_r0 + size + k, mid_c) for k in range(length)]
    if side == "left":
        return [(mid_r, red_c0 - 1 - k) for k in range(length)]
    if side == "right":
        return [(mid_r, red_c0 + size + k) for k in range(length)]
    raise ValueError(f"unknown marker side: {side}")


def _target_marker_cells(anchor_r: int, anchor_c: int, size: int, side: str, length: int) -> list[tuple[int, int]]:
    """Marker cells in global coordinates."""
    mid_r = anchor_r + (size - 1) // 2
    mid_c = anchor_c + (size - 1) // 2
    if side == "top":
        return [(anchor_r - 1 - k, mid_c) for k in range(length)]
    if side == "bottom":
        return [(anchor_r + size + k, mid_c) for k in range(length)]
    if side == "left":
        return [(mid_r, anchor_c - 1 - k) for k in range(length)]
    if side == "right":
        return [(mid_r, anchor_c + size + k) for k in range(length)]
    raise ValueError(f"unknown marker side: {side}")


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Return quadrant -> (anchor_top_r, anchor_top_c, size)."""
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
        assert (r1 - r0 + 1) == size and (c1 - c0 + 1) == size, "red anchors must be squares"
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert len(anchors) == 4, "grid must contain one red anchor per quadrant"
    return anchors


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    """Extract the TL non-bg component containing the source red anchor."""
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red anchor"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _decode_turns(g: np.ndarray, bg: int, mid_r: int, mid_c: int,
                  key: tuple[int, int]) -> int:
    """Read marker length from the non-TL quadrant component that contains the red anchor."""
    qb, qr = key
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(g.shape, mid_r, mid_c, qb, qr)
    quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    comps = components_mask(quad != bg, connectivity=4)
    match = None
    for pts in comps:
        if any(quad[r, c] == RED for r, c in pts):
            match = pts
            break
    assert match is not None, "target quadrant must contain an anchor-plus-marker component"
    turns = sum(1 for r, c in match if quad[r, c] != RED)
    assert turns in (1, 2, 3), "marker length must encode one of 90/180/270 degrees"
    return int(turns)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread the top-left motif to the other quadrants, resizing by anchor size and rotating by edge markers.

    Input structure:
    - The grid is split into four quadrants by one solid full row and one solid full column of the same non-bg color.
    - Every quadrant has a red square anchor, and the TL quadrant also contains the source motif attached to that anchor.
    - Each non-TL anchor has a short non-red marker ray attached to one side; its length is 1, 2, or 3 pixels.

    Rule:
    - Extract the TL motif component and keep using the target/source red-square size ratio as the scale factor.
    - For each non-TL quadrant, decode the marker length and rotate the source motif clockwise by 90/180/270 degrees accordingly before scaling.
    - Align the transformed motif so its red region lands on the target anchor, then stamp only onto background cells.

    Key insight:
    - Red square size still carries the scaling parameter, but the old implicit quadrant-based reflection cue is replaced by an explicit counted symbol next to each target anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)
    assert (0, 0) in anchors, "top-left quadrant must contain a source anchor"

    patch = _extract_source_patch(g, bg, mid_r, mid_c)
    _, _, source_size = anchors[(0, 0)]

    for key, (tr0, tc0, target_size) in anchors.items():
        if key == (0, 0):
            continue

        turns = _decode_turns(g, bg, mid_r, mid_c, key)
        patch_t = _rotate_cw(patch, turns)

        ratio = float(target_size) / float(source_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve motif information"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled patch must still contain the red anchor"
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] != bg:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


# --------------- generator (o5x) ---------------

BG = 4
DIV = 1
RED = 2
MARK = 0
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
MARKER_SIDES = ("top", "bottom", "left", "right")
TURN_ASSIGNMENTS = [perm for perm in permutations((1, 2, 3)) if perm[2] != 2]
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


def _assign_colors(filled: set[tuple[int, int]], red_cell: tuple[int, int], h: int, w: int) -> np.ndarray | None:
    """Assign colors to a logical motif. Every row and col must participate to keep the bbox tight."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p != red_cell]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = COLOR_POOL[:]
    random.shuffle(extras)
    extras = extras[: n_colors - 1]
    grid = np.full((h, w), BG, dtype=int)
    grid[red_cell] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _is_asymmetric(patch: np.ndarray) -> bool:
    """Reject motifs that visually collapse under the new rotations or the old reflections."""
    transforms = [
        np.fliplr(patch),
        np.flipud(patch),
        _rotate_cw(patch, 1),
        _rotate_cw(patch, 2),
        _rotate_cw(patch, 3),
    ]
    for other in transforms:
        if other.shape == patch.shape and np.array_equal(other, patch):
            return False
    return True


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build an asymmetric block-based source patch whose 2x2 red area scales losslessly."""
    s = SOURCE_ANCHOR_SIZE
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(4, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, (ar, ac), lh, lw)
    if logical is None:
        return None
    patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
    if not _is_asymmetric(patch):
        return None
    return patch, ar * s, ac * s


def _scaled_rotated_patch(patch: np.ndarray, turns: int, target_size: int) -> np.ndarray:
    """Rotate first, then scale according to the anchor-size ratio."""
    p = _rotate_cw(patch, turns)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _legacy_reflected_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Old transform used only for degeneracy checks."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _valid_marker_sides(patch_s: np.ndarray, size: int, length: int) -> list[str]:
    """Marker sides that keep the symbol disjoint from the stamped motif."""
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())
    sides = []
    for side in MARKER_SIDES:
        ok = True
        for mr, mc in _marker_rel_cells(red_r0, red_c0, size, side, length):
            if 0 <= mr < patch_s.shape[0] and 0 <= mc < patch_s.shape[1] and patch_s[mr, mc] != BG:
                ok = False
                break
        if ok:
            sides.append(side)
    return sides


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray,
                   side: str, length: int) -> tuple[int, int] | None:
    """Pick an anchor position where both the full transformed motif and its marker fit cleanly."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    if side == "top":
        anch_r0 = max(anch_r0, q_r0 + length)
    elif side == "bottom":
        anch_r1 = min(anch_r1, q_r1 - size - length + 1)
    elif side == "left":
        anch_c0 = max(anch_c0, q_c0 + length)
    elif side == "right":
        anch_c1 = min(anch_c1, q_c1 - size - length + 1)
    else:
        raise ValueError(f"unknown marker side: {side}")

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _paint_marker(out: np.ndarray, anchor_r: int, anchor_c: int, size: int, side: str, length: int) -> None:
    """Paint the instruction ray."""
    for rr, cc in _target_marker_cells(anchor_r, anchor_c, size, side, length):
        out[rr, cc] = MARK


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp transformed motif pixels onto background cells only."""
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    valid = ((tgt_rr >= 0) & (tgt_rr < GRID_SIZE) &
             (tgt_cc >= 0) & (tgt_cc < GRID_SIZE) &
             (tgt_rr != MID) & (tgt_cc != MID))
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    empty = out[tgt_rr, tgt_cc] == BG
    out[tgt_rr[empty], tgt_cc[empty]] = patch_s[src_rr[empty], src_cc[empty]]


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }
    turn_perm = random.choice(TURN_ASSIGNMENTS)
    turn_map = {
        (0, 1): turn_perm[0],
        (1, 0): turn_perm[1],
        (1, 1): turn_perm[2],
    }

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    transformed = {}
    marker_sides = {}
    for key, turns in turn_map.items():
        patch_s = _scaled_rotated_patch(patch, turns, target_sizes[key])
        roundtrip = _resize_nn(patch_s, _rotate_cw(patch, turns).shape[0], _rotate_cw(patch, turns).shape[1])
        if not np.array_equal(roundtrip, _rotate_cw(patch, turns)):
            return None
        legacy = _legacy_reflected_patch(patch, key[0], key[1], target_sizes[key])
        if legacy.shape == patch_s.shape and np.array_equal(legacy, patch_s):
            return None
        sides = _valid_marker_sides(patch_s, target_sizes[key], turns)
        if not sides:
            return None
        transformed[key] = patch_s
        marker_sides[key] = random.choice(sides)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in transformed.items():
        anchor = _choose_anchor(key[0], key[1], target_sizes[key], patch_s, marker_sides[key], turn_map[key])
        if anchor is None:
            return None
        anchors[key] = anchor

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
        _paint_marker(out, ar, ac, size, marker_sides[key], turn_map[key])
    for key, patch_s in transformed.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
        _paint_marker(inp, ar, ac, size, marker_sides[key], turn_map[key])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D12 input")
