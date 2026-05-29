# Combo variant of b10624e5 (index 33)
# combo_spec: D3/2+D16/1
# Name: 目标锚块居中对齐 + 锚块到边界距离比决定缩放

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant.
# a) Concrete behavior:
#    Original behavior used the target red block's top-left corner and scaled by
#    target-anchor-size / source-anchor-size. Here the transformed motif is
#    reflected as before, but its discrete red-center cell is aligned to the
#    target anchor's discrete center cell, and the resize ratio becomes
#    nearest_boundary_distance(target_anchor) / nearest_boundary_distance(source_anchor).
#    Example with TL distance 2: a target anchor at distance 1 gets a 0.5x motif,
#    while another anchor with the same size but distance 2 gets a 1.0x motif.
# b) Complexity calibration:
#    Distances are restricted to a few visible bins (1, 2, 3) so humans can
#    compare gaps to the quadrant borders by eye. The generator also forces at
#    least one "same size, different distance" pair and one "same distance,
#    different size" pair, preventing a silent collapse back to size-based scaling.
# c) Implicit assumptions replaced:
#    - Target anchor size no longer predicts scale, so solver cannot use size ratio.
#    - Target motif placement can no longer align the red bboxes by top-left corner.
#    - Because scale and anchor size are decoupled, reproducing scaled red pixels
#      would create misleading extra-red blobs; the red anchor is now treated as a
#      fixed destination reference, so only non-red motif pixels are stamped.
# d) Placement legality:
#    The generator searches anchor positions that satisfy the requested nearest
#    boundary distance exactly, keep the anchor inside its quadrant, and prefer
#    candidates where the full centered scaled motif fits inside the quadrant.
# e) Visual observability:
#    The TL motif is a 6x6 blocky sprite with a central 2x2 red anchor and an
#    asymmetric colored surround, so reflections remain easy to spot and the
#    centered placement visibly spreads around the target anchor.
# f) Visual cleanliness:
#    Only one red square remains in each non-TL quadrant, divider cells are never
#    overwritten, and target motifs are usually fully contained in their quadrants.
# g) New solver-side assumptions guaranteed by the generator:
#    exactly one red square per quadrant; TL has one connected non-bg component
#    containing that square; the TL source anchor has positive nearest-boundary
#    distance; target anchors use the requested distance bins and yield at least
#    one visible non-red motif pixel after centered stamping.


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


def _quadrant_bounds(mid_r: int, mid_c: int, qb: int, qr: int) -> tuple[int, int, int, int]:
    """Inclusive row/col bounds for one quadrant."""
    if qb == 0:
        r0, r1 = 0, mid_r - 1
    else:
        r0, r1 = mid_r + 1, 2 * mid_r
    if qr == 0:
        c0, c1 = 0, mid_c - 1
    else:
        c0, c1 = mid_c + 1, 2 * mid_c
    return int(r0), int(r1), int(c0), int(c1)


def _nearest_boundary_distance(
    r0: int, c0: int, r1: int, c1: int, bounds: tuple[int, int, int, int]
) -> int:
    """Gap from an anchor bbox to its closest quadrant border."""
    q_r0, q_r1, q_c0, q_c1 = bounds
    return int(min(r0 - q_r0, q_r1 - r1, c0 - q_c0, q_c1 - c1))


def _center_cell(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    """Discrete midpoint cell for an inclusive bbox."""
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize with 1-cell minimum extent."""
    src_h, src_w = img.shape
    new_h = max(1, int(new_h))
    new_w = max(1, int(new_w))
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Mirror the source motif according to quadrant position."""
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _scale_for_distance(patch: np.ndarray, qb: int, qr: int, ratio: float) -> np.ndarray:
    """Reflect and resize a patch by the requested ratio."""
    p = _reflect_patch(patch, qb, qr)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _patch_red_bbox(patch: np.ndarray) -> tuple[int, int, int, int]:
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0, "transformed source patch must still contain red pixels"
    return int(rr_red.min()), int(cc_red.min()), int(rr_red.max()), int(cc_red.max())


def _centered_origin(patch: np.ndarray, anchor_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Top-left placement that aligns the patch's red center cell to anchor center cell."""
    ar0, ac0, ar1, ac1 = anchor_bbox
    pr0, pc0, pr1, pc1 = _patch_red_bbox(patch)
    tgt_cr, tgt_cc = _center_cell(ar0, ac0, ar1, ac1)
    src_cr, src_cc = _center_cell(pr0, pc0, pr1, pc1)
    return int(tgt_cr - src_cr), int(tgt_cc - src_cc)


def _stamp_centered(
    out: np.ndarray,
    patch: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    *,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    """Stamp non-bg, non-red motif pixels using center alignment to the target anchor."""
    place_r0, place_c0 = _centered_origin(patch, anchor_bbox)
    rr_p, cc_p = np.where((patch != bg) & (patch != RED))
    h, w = out.shape
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < h and 0 <= cc_t < w):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch[rr0, cc0])


# --------------- solver ---------------

# --- Grid constants reused by the generator ---
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants using centered anchor alignment and border-distance scaling.

    Input structure:
    - A 27x27 grid is split into four 13x13 quadrants by one full non-background row and one full non-background column.
    - Each quadrant contains exactly one red square anchor block.
    - Only the top-left quadrant contains an additional colored 4-connected motif joined to its red anchor.

    Rule:
    - Extract the top-left connected motif containing the red anchor and use it as the source sprite.
    - For each non-TL quadrant, reflect the sprite by quadrant relation as in the original task.
    - Resize it by (target anchor's nearest quadrant-border distance) / (source anchor's nearest quadrant-border distance).
    - Place the transformed sprite so the center cell of its red region aligns with the center cell of the target anchor, then stamp only the non-red motif pixels.

    Key insight:
    - The target red square is no longer a size template; it is a positioning pivot.
    - Scale is encoded by how close that anchor sits to its quadrant border, while quadrant side still determines the reflection axes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    sr0, sc0, sr1, sc1 = anchors[(0, 0)]
    src_bounds = _quadrant_bounds(mid_r, mid_c, 0, 0)
    src_dist = _nearest_boundary_distance(sr0, sc0, sr1, sc1, src_bounds)
    assert src_dist > 0, "top-left anchor must not touch its quadrant border in this variant"

    for (qb, qr), anchor_bbox in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        bounds = _quadrant_bounds(mid_r, mid_c, qb, qr)
        dst_dist = _nearest_boundary_distance(*anchor_bbox, bounds)
        ratio = float(dst_dist) / float(src_dist)
        patch_s = _scale_for_distance(patch, qb, qr, ratio)
        _stamp_centered(out, patch_s, anchor_bbox, bg=bg, mid_r=mid_r, mid_c=mid_c)

    return out


# --------------- generator ---------------

SOURCE_ANCHOR_SIZE = 2
LOGICAL_SIDE = 3
CELL = 2
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400
FILL_RANGE = (6, 8)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
TARGET_TEMPLATES = [
    [(1, 3), (2, 1), (2, 3)],
    [(1, 4), (3, 2), (3, 4)],
]


def _grow_4conn(side: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected logical mask."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.remove(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _make_source_patch() -> np.ndarray | None:
    """Create a 6x6 asymmetric blocky motif with a central 2x2 red anchor."""
    anchor = (1, 1)
    n_fill = random.randint(*FILL_RANGE)
    filled = _grow_4conn(LOGICAL_SIDE, {anchor}, n_fill)
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != {0, 1, 2} or cols != {0, 1, 2}:
        return None

    logical = np.full((LOGICAL_SIDE, LOGICAL_SIDE), BG, dtype=int)
    non_red = [cell for cell in filled if cell != anchor]
    random.shuffle(non_red)

    logical[anchor] = RED
    n_colors = random.randint(3, min(5, len(non_red) + 1))
    extras = COLOR_POOL[:]
    random.shuffle(extras)
    extras = extras[: n_colors - 1]
    for idx, (r, c) in enumerate(non_red):
        logical[r, c] = extras[idx] if idx < len(extras) else random.choice(extras)

    if np.array_equal(logical, np.fliplr(logical)) or np.array_equal(logical, np.flipud(logical)):
        return None

    return np.repeat(np.repeat(logical, CELL, axis=0), CELL, axis=1)


def _source_positions_for_distance(patch: np.ndarray, desired_distance: int) -> list[tuple[int, int]]:
    """Feasible TL patch origins whose source anchor has the requested border distance."""
    positions = []
    rel_r0, rel_c0 = 2, 2
    rel_r1, rel_c1 = 3, 3
    q_r0, q_r1, q_c0, q_c1 = QUADS[(0, 0)]
    for pr0 in range(q_r0, q_r1 - patch.shape[0] + 2):
        for pc0 in range(q_c0, q_c1 - patch.shape[1] + 2):
            ar0 = pr0 + rel_r0
            ac0 = pc0 + rel_c0
            ar1 = pr0 + rel_r1
            ac1 = pc0 + rel_c1
            dist = _nearest_boundary_distance(ar0, ac0, ar1, ac1, QUADS[(0, 0)])
            if dist == desired_distance:
                positions.append((pr0, pc0))
    return positions


def _visible_non_red_pixels(
    patch: np.ndarray, anchor_bbox: tuple[int, int, int, int], bounds: tuple[int, int, int, int]
) -> tuple[int, bool]:
    """Return visible non-red count and whether the patch fits fully in the quadrant."""
    place_r0, place_c0 = _centered_origin(patch, anchor_bbox)
    q_r0, q_r1, q_c0, q_c1 = bounds
    rr_p, cc_p = np.where((patch != BG) & (patch != RED))
    visible = 0
    full_fit = True
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        inside = q_r0 <= rr_t <= q_r1 and q_c0 <= cc_t <= q_c1
        visible += int(inside)
        if not inside:
            full_fit = False
    return visible, full_fit


def _choose_target_anchor(
    qb: int, qr: int, size: int, desired_distance: int, patch: np.ndarray
) -> tuple[int, int] | None:
    """Search anchor placements matching the desired border distance."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    bounds = QUADS[(qb, qr)]
    full_fit: list[tuple[int, int]] = []
    partial_fit: list[tuple[int, int]] = []
    for ar0 in range(q_r0, q_r1 - size + 2):
        for ac0 in range(q_c0, q_c1 - size + 2):
            anchor_bbox = (ar0, ac0, ar0 + size - 1, ac0 + size - 1)
            dist = _nearest_boundary_distance(*anchor_bbox, bounds)
            if dist != desired_distance:
                continue
            visible, fits = _visible_non_red_pixels(patch, anchor_bbox, bounds)
            if visible <= 0:
                continue
            if fits:
                full_fit.append((ar0, ac0))
            else:
                partial_fit.append((ar0, ac0))
    if full_fit:
        return random.choice(full_fit)
    if partial_fit:
        return random.choice(partial_fit)
    return None


def _stamp_with_bbox(out: np.ndarray, patch: np.ndarray, anchor_bbox: tuple[int, int, int, int]) -> None:
    """Thin wrapper to share the same centered stamping logic."""
    _stamp_centered(out, patch, anchor_bbox, bg=BG, mid_r=MID, mid_c=MID)


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    patch = _make_source_patch()
    if patch is None:
        return None

    src_positions = _source_positions_for_distance(patch, desired_distance=2)
    if not src_positions:
        return None
    tl_r0, tl_c0 = random.choice(src_positions)

    target_specs = random.choice(TARGET_TEMPLATES)[:]
    random.shuffle(target_specs)
    keys = [(0, 1), (1, 0), (1, 1)]
    spec_map = {key: target_specs[idx] for idx, key in enumerate(keys)}

    transformed: dict[tuple[int, int], np.ndarray] = {}
    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    anchors[(0, 0)] = (tl_r0 + 2, tl_c0 + 2, tl_r0 + 3, tl_c0 + 3)

    for key in keys:
        dist, size = spec_map[key]
        if not (TARGET_SIZE_RANGE[0] <= size <= TARGET_SIZE_RANGE[1]):
            return None
        ratio = dist / 2.0
        patch_s = _scale_for_distance(patch, key[0], key[1], ratio)
        choice = _choose_target_anchor(key[0], key[1], size, dist, patch_s)
        if choice is None:
            return None
        ar0, ac0 = choice
        transformed[key] = patch_s
        anchors[key] = (ar0, ac0, ar0 + size - 1, ac0 + size - 1)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)

    for key in keys:
        ar0, ac0, ar1, ac1 = anchors[key]
        paste(out, np.full((ar1 - ar0 + 1, ac1 - ac0 + 1), RED, dtype=int), ar0, ac0)
        _stamp_with_bbox(out, transformed[key], anchors[key])

    inp = out.copy()
    for key in keys:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar0, ac0, ar1, ac1 = anchors[key]
        paste(inp, np.full((ar1 - ar0 + 1, ac1 - ac0 + 1), RED, dtype=int), ar0, ac0)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo-variant input")
