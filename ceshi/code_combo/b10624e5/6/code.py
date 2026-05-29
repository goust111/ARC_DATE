# Combo variant of b10624e5 (index 6)
# combo_spec: D5/0+D12/3
# Name: 旋转替代反射 + 锚块边缘像素编码旋转角度

"""
Design analysis for the fused b10624e5 combo variant.

spec:
  preserved_backbone:
    - 27x27 grid with bg=4 and a full-row/full-column divider of color 1 at index 13
    - exactly one red square anchor per quadrant, with the TL anchor fixed at 2x2
    - the TL quadrant contains one 4-connected colored source motif attached to the red anchor
    - non-TL outputs still align a transformed+scaled copy of that TL motif to each target anchor
    - scale is still target_anchor_size / source_anchor_size via nearest-neighbor resize

  fused_change:
    - D5/0 removes reflection and says the copied motif should be rotated instead
    - D12/3 removes the fixed quadrant->angle mapping and makes the angle explicit:
      marker length 1/2/3 means rotate clockwise by 90/180/270 degrees
    - therefore quadrant location no longer determines the transform at all; only the counted marker does

behavior_trace:
  concrete_example:
    - imagine the TL patch is a 4x6 asymmetric motif with the red 2x2 block near its upper-left area
    - original task:
      TR uses fliplr, BL uses flipud, BR uses both flips
    - D5-only task:
      TR uses 90 cw, BL uses 270 cw, BR uses 180
    - fused task:
      if TR shows a 3-cell marker, TR must use 270 cw even though D5-only would have used 90 cw there
      if BL shows a 1-cell marker, BL must use 90 cw even though D5-only would have used 270 cw there
    - visible difference:
      the same quadrant can now hold different rotations across examples, so the solver must read the marker count
      instead of inferring the angle from position

complexity_calibration:
  - every sample uses the three marker lengths {1,2,3} exactly once across TR/BL/BR
  - the source motif is block-based and rejected if symmetric under flips or quarter-turn rotations
  - the generator also rejects any sample whose combo output matches either:
    - the legacy reflection behavior
    - the fixed quadrant-rotation behavior from D5/0
  - this keeps the fusion visually observable without adding extra symbol vocabularies

implicit_assumption_review:
  affected_main_assumptions:
    - original code assumed quadrant side encoded the transform; no longer valid
    - original stamping allowed overwriting any non-red pixel; unsafe once markers exist
    - original generator could use freeform motifs; here block motifs are safer because all chosen scales must roundtrip
  repaired_in_combo:
    - main() decodes turns by counting the non-red cells attached to each target anchor
    - stamping now writes only onto background, preserving both markers and anchors
    - generator guarantees the source patch remains lossless under all chosen scales after rotation

placement_legality:
  - each target anchor is chosen where the full transformed motif footprint fits inside its quadrant when possible
  - the marker ray must also fit inside the same quadrant
  - valid marker sides are filtered so the marker never sits on top of the stamped motif
  - divider cells are never painted over

visual_observability:
  - marker color is fixed and reserved, so humans can read "count of pixels" directly
  - motif colors exclude both divider color and marker color to avoid accidental legend confusion
  - the sample set spans all three turn counts, making the count->rotation rule inferable from few examples

visual_cleanliness:
  - non-TL inputs contain only the red anchor plus one short marker ray
  - output stamping respects existing non-bg structure, so anchors and markers remain legible
  - side choice for the marker is random but only among collision-free options

new_main_assumptions_checked_against_generator:
  - each target quadrant has exactly one anchor-plus-marker component: enforced by clearing quadrants before repainting
  - marker length is always in {1,2,3}: enforced by using a permutation of those lengths
  - the decoded transform must differ from the fixed D5 mapping in every accepted sample:
    enforced by rejecting the default permutation and measuring output differences vs the fixed-rotation baseline
"""

import random
from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
FIXED_TURNS = {
    (0, 1): 1,
    (1, 0): 3,
    (1, 1): 2,
}
TURN_ASSIGNMENTS = [perm for perm in permutations((1, 2, 3)) if perm != (1, 3, 2)]
MAX_RETRIES = 600


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
    """Marker cells in patch coordinates, attached to the midpoint cell of one anchor side."""
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


def _decode_turns(g: np.ndarray, bg: int, mid_r: int, mid_c: int, key: tuple[int, int]) -> int:
    """Read marker length from the non-TL quadrant component containing the red anchor."""
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


def _transform_patch(patch: np.ndarray, key: tuple[int, int], mode: str, turns: int | None = None) -> np.ndarray:
    """Apply the transform mode used by the combo solver or its baselines."""
    if mode == "decoded":
        assert turns in (1, 2, 3), "decoded mode requires 1/2/3 turns"
        return _rotate_cw(patch, int(turns))
    if mode == "fixed_rotation":
        return _rotate_cw(patch, FIXED_TURNS[key])
    if mode == "reflection":
        out = patch
        if key[1] == 1:
            out = np.fliplr(out)
        if key[0] == 1:
            out = np.flipud(out)
        return out
    raise ValueError(f"unknown solve mode: {mode}")


def _solve(grid: np.ndarray, mode: str) -> np.ndarray:
    """Shared solver body for the combo rule and comparison baselines."""
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

        turns = _decode_turns(g, bg, mid_r, mid_c, key) if mode == "decoded" else None
        patch_t = _transform_patch(patch, key, mode, turns)

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


def _solve_fixed_rotation(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="fixed_rotation")


def _solve_reflection(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="reflection")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants by scaling from anchor size and rotating from edge markers.

    Input structure:
    - One solid non-background row and one solid non-background column split the board into four quadrants.
    - Each quadrant contains one red square anchor, and the top-left quadrant also contains the full colored source motif.
    - Every non-top-left anchor has a short attached marker ray; the ray length is 1, 2, or 3 cells.

    Rule:
    - Extract the top-left motif component that contains the source red anchor.
    - For each target quadrant, keep using the anchor-size ratio to scale the motif, but replace the old mirror logic with a clockwise rotation decoded from the marker length: 1->90, 2->180, 3->270.
    - Align the transformed red area to the target anchor and stamp only onto background cells so the input anchors and markers stay intact.

    Key insight:
    - The red block still carries scale, but the transformation is no longer implied by quadrant position; the solver must count the marker pixels next to each anchor to recover the rotation angle.
    """
    return _solve(grid, mode="decoded")


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
    """Assign colors to a logical motif; every row and col must be used so the bbox is tight."""
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
    """Reject motifs that collapse under flips or quarter-turn rotations."""
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
    rotated = _rotate_cw(patch, turns)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(rotated.shape[0] * ratio)))
    w = max(1, int(round(rotated.shape[1] * ratio)))
    return _resize_nn(rotated, h, w)


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


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray, side: str, length: int) -> tuple[int, int] | None:
    """Pick an anchor position where both the transformed motif and the marker fit cleanly."""
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
    """Paint the marker ray."""
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

    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    empty = out[tgt_rr, tgt_cc] == BG
    out[tgt_rr[empty], tgt_cc[empty]] = patch_s[src_rr[empty], src_cc[empty]]


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count non-anchor, non-marker differences per non-TL quadrant."""
    stats: dict[tuple[int, int], int] = {}
    for key, (r0, r1, c0, c1) in QUADS.items():
        if key == (0, 0):
            continue
        qa = a[r0 : r1 + 1, c0 : c1 + 1]
        qb = b[r0 : r1 + 1, c0 : c1 + 1]
        protected = (qa == RED) | (qb == RED) | (qa == MARK) | (qb == MARK)
        stats[key] = int(((qa != qb) & (~protected)).sum())
    return stats


def _observable_combo(combo_out: np.ndarray, fixed_out: np.ndarray, reflected_out: np.ndarray) -> bool:
    """Reject samples that visually collapse back to either baseline."""
    fixed_stats = _diff_stats(combo_out, fixed_out)
    refl_stats = _diff_stats(combo_out, reflected_out)
    if sum(fixed_stats.values()) == 0:
        return False
    if sum(refl_stats.values()) == 0:
        return False
    if refl_stats[(0, 1)] == 0 and refl_stats[(1, 0)] == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check, and reject degeneracy."""
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

    transformed: dict[tuple[int, int], np.ndarray] = {}
    marker_sides: dict[tuple[int, int], str] = {}
    for key, turns in turn_map.items():
        rotated = _rotate_cw(patch, turns)
        patch_s = _scaled_rotated_patch(patch, turns, target_sizes[key])
        roundtrip = _resize_nn(patch_s, rotated.shape[0], rotated.shape[1])
        if not np.array_equal(roundtrip, rotated):
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

    combo_out = main(inp)
    if not np.array_equal(combo_out, out):
        return None
    if not _observable_combo(combo_out, _solve_fixed_rotation(inp), _solve_reflection(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
