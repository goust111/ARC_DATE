# Combo variant of b10624e5 (index 40)
# combo_spec: D9/1+D13/3
# Name: 图案位置由锚块朝向决定 + L形锚块

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D9/1 + D13/3
# ---------------------------------------
# a) Concrete behavior:
#    - Original task: every quadrant had a red square anchor, the TL quadrant
#      contained the full multicolor patch, and the other quadrants received
#      scaled copies reflected by quadrant location. Placement was effectively
#      "align the red square's top-left corner".
#    - Fused task: the red anchor is now an oriented L, and the motif sits on
#      the same corner side as the L elbow. In the TL source patch the elbow is
#      northwest, so the non-red motif grows to the northwest of the red L.
#      After left-right / up-down reflection, the other quadrants show the same
#      patch but with elbow directions NE / SW / SE, and the motif follows that
#      direction as well.
#    - Small trace:
#      old: square size 3 in TR => reflect patch LR, resize by 3/2, align the
#      red square bbox, paste.
#      new: TR anchor is an NE-facing L. The solver reflects the TL patch LR,
#      tries the allowed size ratios, and keeps the unique resized version whose
#      red mask exactly matches that NE L. Translation is computed only after
#      this mask match, so the motif lands on the NE side of the anchor.
# b) Complexity calibration:
#    - The rule still has one anchor per quadrant, one source patch, quadrant
#      reflections, and 4 size candidates. The new freedom is limited to four
#      visible L orientations, which is enough to make the change obvious but
#      still learnable from 2-3 examples.
#    - The generator forces at least two distinct non-TL scales and keeps the
#      four quadrant reflections, so both "scale by anchor size" and "place by
#      L direction" are repeatedly visible. It also avoids the nominal size-1
#      case during generation because that half-scale collapse would erase the
#      L orientation and make the combo visually degenerate.
# c) Implicit assumptions broken from the original main():
#    - `size = max(bbox_h, bbox_w)` no longer recovers the intended scale,
#      because L anchors are not filled squares.
#    - Aligning by `min(red_rows), min(red_cols)` is only valid after the solver
#      has matched the full transformed red L mask, not before.
#    - The original exact roundtrip assertion was tuned for blocky square
#      anchors. Here the generator instead guarantees that every target anchor
#      is literally produced by resizing the source red L, so main() can recover
#      the correct scale by mask matching.
# d) Placement legality:
#    - Every transformed patch is placed fully inside its quadrant. This avoids
#      clipping ambiguity and keeps the directional corner relation legible.
#    - Non-TL quadrants keep only the red L in the input, so red components stay
#      isolated and quadrant assignment is unambiguous.
# e) Visual observability:
#    - The elbow corner of the red L is a human-visible orientation cue. The
#      copied motif consistently appears on that same corner side in the output.
#      This is directly observable from shape, not from hidden color ordering.
# f) Visual cleanliness:
#    - The patch is compact, quadrant-local, and never overlaps the divider. The
#      motif uses only a few colors and stays attached to one corner of the L,
#      so anchor and decoration remain easy to distinguish.
# g) Color inference:
#    - Colors do not encode control logic. They only decorate the motif. The
#      directional rule is conveyed by geometry alone, which is necessary for a
#      2-4 example ARC task.
# h) Post-implementation combo metric:
#    - For generated samples, detect each non-TL anchor's elbow direction and
#      verify that the output motif extends into the same corner direction
#      relative to the anchor bbox. Also compare against the original square-
#      anchor solver; equality should be rare or impossible because the inputs no
#      longer satisfy the original square-anchor premise.
# i) New main() assumptions audited below:
#    - Exactly one red connected component exists per quadrant.
#    - The TL non-background component containing red is the full source patch.
#    - Every non-TL red anchor is generated from the reflected source patch with
#      one of the discrete size ratios 1/2, 1, 3/2, 2.


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique matching full-row/full-col divider."""
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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _norm_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return {(int(r - r0), int(c - c0)) for r, c in coords.tolist()}


def _transform_patch(patch: np.ndarray, qb: int, qr: int, nominal_size: int, source_nominal_size: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    ratio = float(nominal_size) / float(source_nominal_size)
    new_h = max(1, int(round(out.shape[0] * ratio)))
    new_w = max(1, int(round(out.shape[1] * ratio)))
    return _resize_nn(out, new_h, new_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif by matching directional L anchors rather than square blocks.

    Input structure:
    - A full-row/full-column divider splits the board into four 13x13 quadrants.
    - Each quadrant contains exactly one red connected anchor, and each anchor is
      an oriented L shape rather than a solid square.
    - Only the top-left quadrant contains the complete multicolor patch; the
      other quadrants contain only their red L anchors.

    Rule:
    - Extract the top-left connected non-background component containing red as
      the source patch.
    - Reflect that patch horizontally for right quadrants and vertically for
      bottom quadrants, preserving the original quadrant-reflection logic.
    - For each non-TL quadrant, try the allowed size ratios until the reflected
      patch's red pixels match the target red L exactly; then paste that whole
      transformed patch with the same translation.
    - Leave divider cells and pre-existing red anchor pixels unchanged.

    Key insight: the red object is no longer just a size token. Its L geometry
    carries both scale and direction, so the solver must match the transformed
    red mask itself; once that mask matches, the motif automatically lands on
    the side indicated by the L elbow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red_comps = components_mask(g == 2, connectivity=4)
    assert red_comps, "grid must contain red anchors"

    anchors: dict[tuple[int, int], np.ndarray] = {}
    for pts in red_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = pts

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "expected one red anchor per quadrant"

    tl = g[:mid_r, :mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(tl[r, c] == 2 for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with red pixels"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), target_pts in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        target_norm = _norm_coords(target_pts)
        target_r0, target_c0, _tr1, _tc1 = bbox_coords(target_pts)
        matched_patch = None

        for nominal_size in ALLOWED_SIZES:
            patch_s = _transform_patch(patch, qb, qr, nominal_size, SOURCE_NOMINAL_SIZE)
            red_pts = np.argwhere(patch_s == RED)
            if red_pts.size == 0:
                continue
            if _norm_coords(red_pts) == target_norm:
                matched_patch = patch_s
                break

        assert matched_patch is not None, "target anchor does not match any transformed source L"

        red_pts = np.argwhere(matched_patch == RED)
        src_r0, src_c0, _sr1, _sc1 = bbox_coords(red_pts)
        place_r0 = target_r0 - src_r0
        place_c0 = target_c0 - src_c0

        rr_p, cc_p = np.where(matched_patch != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(matched_patch[rr0, cc0])
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


# --------------- generator ---------------

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

SOURCE_NOMINAL_SIZE = 2
ALLOWED_SIZES = (1, 2, 3, 4)
GEN_TARGET_SIZES = (2, 3, 4)
MOTIF_H_RANGE = (2, 4)
MOTIF_W_RANGE = (2, 4)
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.9
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400

SOURCE_ANCHOR = np.array(
    [
        [RED, RED, RED],
        [RED, BG, BG],
        [RED, BG, BG],
    ],
    dtype=int,
)


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
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_red = [p for p in filled if p not in red_cells]
    if len(non_red) < MOTIF_COLORS_MIN - 1:
        return None

    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _build_source_patch() -> np.ndarray | None:
    mh = random.randint(*MOTIF_H_RANGE)
    mw = random.randint(*MOTIF_W_RANGE)
    red_cells = {(mh - 1, mw - 1)}
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * mh * mw)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * mh * mw)))
    filled = _grow_4conn(mh, mw, red_cells, random.randint(min_fill, max_fill))
    motif = _assign_colors(filled, red_cells, mh, mw)
    if motif is None:
        return None

    patch = np.full((mh + 2, mw + 2), BG, dtype=int)
    patch[:mh, :mw] = motif
    anchor_mask = SOURCE_ANCHOR != BG
    anchor_zone = patch[mh - 1 : mh + 2, mw - 1 : mw + 2]
    anchor_zone[anchor_mask] = SOURCE_ANCHOR[anchor_mask]
    return patch


def _anchor_only(patch: np.ndarray) -> np.ndarray:
    return np.where(patch == RED, RED, BG)


def _corner_relation_ok(patch: np.ndarray, key: tuple[int, int]) -> bool:
    motif = np.argwhere((patch != BG) & (patch != RED))
    if motif.size == 0:
        return False
    red = np.argwhere(patch == RED)
    ar0, ac0, ar1, ac1 = bbox_coords(red)
    mr0, mc0, mr1, mc1 = bbox_coords(motif)
    if key[0] == 0:
        if mr0 >= ar0:
            return False
    else:
        if mr1 <= ar1:
            return False
    if key[1] == 0:
        if mc0 >= ac0:
            return False
    else:
        if mc1 <= ac1:
            return False
    return True


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    while True:
        non_tl = [3, random.choice(GEN_TARGET_SIZES), random.choice(GEN_TARGET_SIZES)]
        random.shuffle(non_tl)
        sizes = {
            (0, 0): SOURCE_NOMINAL_SIZE,
            (0, 1): non_tl[0],
            (1, 0): non_tl[1],
            (1, 1): non_tl[2],
        }
        if len(set(non_tl)) >= 2 and any(s != SOURCE_NOMINAL_SIZE for s in non_tl):
            return sizes


def _choose_patch_pos(patch: np.ndarray, qb: int, qr: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert max_r0 >= q_r0 and max_c0 >= q_c0, "patch must fit inside the quadrant"
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _gen_attempt() -> np.ndarray | None:
    patch = _build_source_patch()
    if patch is None or not _corner_relation_ok(patch, (0, 0)):
        return None

    sizes = _sample_target_sizes()
    placed: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    for key, size in sizes.items():
        patch_t = _transform_patch(patch, key[0], key[1], size, SOURCE_NOMINAL_SIZE)
        if not _corner_relation_ok(patch_t, key):
            return None
        r0, c0 = _choose_patch_pos(patch_t, key[0], key[1])
        placed[key] = (patch_t, r0, c0)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    for patch_t, r0, c0 in placed.values():
        paste(out, patch_t, r0, c0, transparent=BG)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    tl_patch, tl_r0, tl_c0 = placed[(0, 0)]
    paste(inp, tl_patch, tl_r0, tl_c0, transparent=BG)
    for key, (patch_t, r0, c0) in placed.items():
        if key == (0, 0):
            continue
        paste(inp, _anchor_only(patch_t), r0, c0, transparent=BG)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
