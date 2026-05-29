# Variant of b10624e5
# Dimension: D13 — 图案与锚块的空间关系 (meta)
# Change: 图案紧邻锚块放置（与锚块共享边或角），图案的锚定点是锚块的特定角/边... → 图案位置由锚块朝向决定

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D13 variant
# -----------------------------------
# a) Concrete behavior:
#    - The old task used one red NxN square per quadrant. The colored motif was
#      attached to the TL square and other quadrants received reflected/scaled
#      copies aligned by the square's top-left corner.
#    - Here the red anchor is asymmetric: a 2x2 red body with a corner-hook
#      protrusion. The TL source anchor points to the patch's bottom-right
#      corner, and the motif is grown in that pointed corner. After LR/UD
#      reflection, the protrusion points to a different corner in each target
#      quadrant, so the motif must appear in that corner as well.
#    - Small trace:
#      old: source patch red bbox top-left == target red square top-left.
#      new: source patch is first reflected, then candidate-scaled until its red
#      hook shape exactly matches the target red hook; only then is the whole
#      patch translated. The red hook orientation, not a fixed square corner,
#      determines the motif side.
# b) Complexity calibration:
#    - Each sample still has four anchors, a divider, scaling, and quadrant-wise
#      reflections. The new anchor shape adds one visible control variable
#      without introducing many arbitrary cases.
#    - Source orientation is fixed to a southeast hook so the quadrant
#      reflections produce all four corner directions in every sample. This
#      keeps the rule easy to notice while still giving repeated evidence.
# c) Implicit assumptions from the original main():
#    - `size = max(bbox_h, bbox_w)` depended on anchors being squares. That is
#      now false, so scaling is recovered by explicit hook-shape matching.
#    - Aligning with `min(red_rows), min(red_cols)` used to be harmless because
#      every red anchor was a solid square. With asymmetric hooks, translation
#      is only safe after the full scaled red mask has been matched exactly.
#    - The old roundtrip-resize assertion relied on blocky square anchors. The
#      new hook uses asymmetric micro-geometry, so the generator guarantees
#      recognizable NN scaling instead of exact back-resize identity.
# d) Placement legality:
#    - Every full patch is placed wholly inside its quadrant; no clipping is
#      needed. Inputs keep only the red hook in non-TL quadrants, so components
#      stay disjoint and the solver can identify one anchor per quadrant.
# e) Visual observability:
#    - The four output quadrants display the motif in four different corner
#      directions because the hook points SE / SW / NE / NW after reflection.
#      The change is visible from shape alone; no hidden ordering by color id.
# f) Visual cleanliness:
#    - The hook is compact, the motif grows from one pointed corner, and the
#      generator keeps whole patches inside quadrants. There is no overlap with
#      divider cells or with other quadrant content.
# g) Color inference:
#    - Colors are decorative only. The rule does not depend on color ranking or
#      any color remapping; colors just make the reflected motif easier to see.
# h) Variant metric used after implementation:
#    - For generated samples, detect each quadrant's red-hook orientation and
#      verify the set is {SE, SW, NE, NW}. Also verify that the output motif
#      centroid lies in the same corner direction as the hook for every non-TL
#      quadrant.
# i) New main() assumptions to audit:
#    - Exactly one red component exists per quadrant.
#    - The TL non-bg component containing red is the full source patch.
#    - Non-TL red anchors are generated from the same reflected/scaled source
#      hook, so one of the allowed scale candidates matches exactly.


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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _norm_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return {(int(r - r0), int(c - c0)) for r, c in coords.tolist()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif by matching directional red anchors instead of square blocks.

    Input structure:
    - A full-row/full-column divider splits the grid into four quadrants.
    - Each quadrant contains exactly one red connected anchor; the anchor is a hooked shape with a visible corner-facing protrusion.
    - Only the top-left quadrant contains the full multicolor component; the other quadrants contain just the red hooked anchor.

    Rule:
    - Extract the full connected component from the top-left quadrant as the source patch.
    - Reflect that patch horizontally for right quadrants and vertically for bottom quadrants, just as in the original task.
    - For each target quadrant, try the allowed scale factors until the scaled red hook matches the target red anchor exactly, then paste the whole transformed patch with that same translation.
    - Keep divider cells and already-present red anchor pixels unchanged.

    Key insight: the red anchor is no longer a symmetric size marker; its asymmetric hook tells you which corner the motif belongs to, so the solver must match anchor shape and orientation before placing the motif.
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

    allowed_sizes = (1, 2, 3, 4)
    source_nominal_size = 2

    for (qb, qr), target_pts in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        target_norm = _norm_coords(target_pts)
        target_r0, target_c0, _tr1, _tc1 = bbox_coords(target_pts)
        matched_patch = None

        for size in allowed_sizes:
            ratio = float(size) / float(source_nominal_size)
            new_h = max(1, int(round(patch_t.shape[0] * ratio)))
            new_w = max(1, int(round(patch_t.shape[1] * ratio)))
            patch_s = _resize_nn(patch_t, new_h, new_w)
            red_pts = np.argwhere(patch_s == 2)
            if red_pts.size == 0:
                continue
            if _norm_coords(red_pts) == target_norm:
                matched_patch = patch_s
                break

        assert matched_patch is not None, "target anchor does not match any scaled reflected source anchor"

        red_pts = np.argwhere(matched_patch == 2)
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
            if out[rr_t, cc_t] == 2:
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
TARGET_SIZE_RANGE = (1, 4)
MOTIF_H_RANGE = (2, 4)
MOTIF_W_RANGE = (2, 4)
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.9
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 300

SOURCE_ANCHOR = np.array(
    [
        [RED, RED, BG],
        [RED, RED, RED],
        [BG, RED, RED],
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


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
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
    red_cells = {(0, 0)}
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * mh * mw)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * mh * mw)))
    filled = _grow_4conn(mh, mw, red_cells, random.randint(min_fill, max_fill))
    motif = _assign_colors(filled, red_cells, mh, mw)
    if motif is None:
        return None

    patch = np.full((mh + 2, mw + 2), BG, dtype=int)
    patch[:3, :3] = SOURCE_ANCHOR
    zone = patch[2:, 2:]
    mask = motif != BG
    zone[mask] = motif[mask]
    return patch


def _transform_patch(patch: np.ndarray, qb: int, qr: int, size: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    ratio = float(size) / float(SOURCE_NOMINAL_SIZE)
    new_h = max(1, int(round(out.shape[0] * ratio)))
    new_w = max(1, int(round(out.shape[1] * ratio)))
    return _resize_nn(out, new_h, new_w)


def _choose_patch_pos(patch: np.ndarray, qb: int, qr: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    max_r0 = q_r1 - patch.shape[0] + 1
    max_c0 = q_c1 - patch.shape[1] + 1
    assert max_r0 >= q_r0 and max_c0 >= q_c0, "patch must fit inside the quadrant"
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


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
        if mr1 <= ar1:
            return False
    else:
        if mr0 >= ar0:
            return False
    if key[1] == 0:
        if mc1 <= ac1:
            return False
    else:
        if mc0 >= ac0:
            return False
    return True


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    while True:
        sizes = {
            (0, 0): SOURCE_NOMINAL_SIZE,
            (0, 1): random.randint(*TARGET_SIZE_RANGE),
            (1, 0): random.randint(*TARGET_SIZE_RANGE),
            (1, 1): random.randint(*TARGET_SIZE_RANGE),
        }
        if len({sizes[(0, 1)], sizes[(1, 0)], sizes[(1, 1)]}) >= 2:
            return sizes


def _gen_attempt() -> np.ndarray | None:
    patch = _build_source_patch()
    if patch is None:
        return None
    if not _corner_relation_ok(patch, (0, 0)):
        return None

    sizes = _sample_target_sizes()
    placed: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    for key, size in sizes.items():
        patch_t = _transform_patch(patch, key[0], key[1], size)
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
    raise RuntimeError("failed to generate a valid b10624e5 D13 input")
