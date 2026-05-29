# Variant of 6ffbe589
# Dimension: D17 — 层间重叠处理 (meta)
# Change: 三个颜色层在板内互不重叠，旋转前后也不重叠... → 旋转后重叠区域被擦除为背景

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the D17 variant
#
# a) Concrete behavior:
#    The board is still a square crop with three cue-controlled color layers.
#    Input layers stay pairwise disjoint, so the board can still be read as
#    "three separate masks on one board". The variant change happens only after
#    rotation: two rotated masks are allowed to land on the same cell, and such
#    cells must be cleared to background instead of keeping either color.
#    Small traced example:
#    - old logic: rotate mid -> write mid cells; rotate inner -> overwrite or
#      assume no conflict; output keeps every rotated cell.
#    - new logic: rotate mid and inner independently from the original crop,
#      count per-cell hits, and erase cells with hit_count >= 2.
#    - visible difference: the final cropped board contains deliberate holes in
#      the central region where the rotated middle and inner features collide.
#
# b) Complexity calibration:
#    Only the overlap rule changes, so the cue mechanism, square crop, and
#    color-matched rotations are preserved. To make the new rule observable, the
#    generator always creates a 2-4 cell collision between the middle and inner
#    layers after rotation. This keeps the branching small and avoids ambiguous
#    color-mapping logic.
#
# c) Old hidden assumptions that break:
#    - `rotate_color_layer(out, col, ...)` used the evolving grid as the source
#      mask. That was harmless only because rotated layers never touched.
#    - Sequential writes also assumed "later layers cannot destroy the source
#      shape of earlier layers". With overlap erasure this is false.
#    The solver therefore now extracts all original color masks first, rotates
#    them independently, and merges them in one collision-aware pass.
#
# d) Placement legality:
#    Input masks remain non-overlapping and inside the same NxN board. Cue
#    sprites still stay outside the crop with the original distance rules, so
#    cropping and cue detection remain valid. The board is still forced to be
#    the unique largest connected component.
#
# e) Visual observability:
#    The collision is not encoded by color ids or hidden ordering. It is visible
#    as central black holes that appear exactly where two cue-rotated masks
#    would otherwise meet. The generator requires repeated mid/inner collisions
#    so this behavior shows up across samples instead of being a rare edge case.
#
# f) Visual cleanliness:
#    Collisions are restricted to the central interaction zone. The outer frame
#    keeps its border-heavy role, and the erased cells stay in the middle of the
#    board instead of scattering noise across the entire crop.
#
# g) Color inference:
#    No new color remapping is introduced. Colors still identify layers, and the
#    only new effect is background erasure on multi-layer hits after rotation.
# -----------------------------------------------------------------------------


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper: crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def compose_rotated_layers(
    color_masks: dict[int, np.ndarray],
    rot_ks: dict[int, int],
    *,
    bg: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Rotate each color mask independently, then erase cells hit by 2+ layers."""
    shape = next(iter(color_masks.values())).shape
    hit_count = np.zeros(shape, dtype=int)
    rotated_masks: dict[int, np.ndarray] = {}

    for col, mask in color_masks.items():
        k = int(rot_ks.get(int(col), 0)) % 4
        rmask = np.rot90(mask.astype(np.uint8), k).astype(bool)
        rotated_masks[int(col)] = rmask
        hit_count += rmask.astype(int)

    out = np.full(shape, int(bg), dtype=int)
    safe = hit_count == 1
    for col, rmask in rotated_masks.items():
        out[safe & rmask] = int(col)
    return out, rotated_masks, hit_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and rotate its color layers from matching outside cues.

    Input structure:
    - The background is the dominant color on the full canvas.
    - The largest foreground component is a square three-layer board.
    - Small monochrome cue sprites of matching colors sit away from the board.

    Rule:
    - Crop the board region away from the distant cue sprites.
    - For each board color, use its same-color cue area modulo 4 to choose a quarter-turn rotation.
    - Rotate every color mask independently inside the fixed square crop.
    - If multiple rotated masks land on the same output cell, erase that cell to background.

    Key insight:
    Cue size still tells you how far each layer rotates, but the final picture is
    the collision-resolved merge of those rotated layers rather than a simple overwrite.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    color_masks = {int(col): crop == int(col) for col in colors}
    rot_ks: dict[int, int] = {}
    for col in colors:
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(outside_comps) <= 1, f"expected at most one outside cue for color {col}"
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        rot_ks[int(col)] = (-cue_size) % 4

    out, _rotated_masks, _hit_count = compose_rotated_layers(color_masks, rot_ks, bg=int(bg))
    return out


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8
INNER_SPAN_MIN = 3

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

INTERACTION_OVERLAP_MIN = 2
INTERACTION_OVERLAP_MAX = 4
INTERACTION_WINDOW = 2.25

GEN_RETRIES = 600


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4-position orbit of (r, c) under 90-degree rotations."""
    pts = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _rotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    """Rotate one coordinate like np.rot90(mask, k) in an n*n grid."""
    rr, cc = int(r), int(c)
    for _ in range(int(k) % 4):
        rr, cc = n - 1 - cc, rr
    return int(rr), int(cc)


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True if mask changes under every non-zero 90-degree rotation."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + int(phase)) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < int(width)
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    random.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= int(target):
            return True
        mask[r, c] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    extra_lo_terms: list[int] | tuple[int, ...] = (),
) -> np.ndarray | None:
    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    lo = max(int(np.ceil(float(density_lo) * n * n)), int(mask.sum()), *extra_lo_terms)
    hi = int(np.floor(float(density_hi) * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {ASYM_MARKER}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int) -> np.ndarray | None:
    """Build the middle layer inside an inset square; collision cells are added later."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[int(min_area)])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(INNER_SPAN_MIN, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n, require_asym))
    top = (n - span) // 2

    if span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = random.choice(variants)
        local = np.rot90(local, random.randrange(4))
    else:
        local = np.zeros((span, span), dtype=bool)
        if span == 4:
            pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
        else:
            pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
        for r, c in pts:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))
    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _pick_rotations() -> list[int]:
    """Pick rotations with guaranteed middle/inner disagreement so collisions can be induced."""
    while True:
        rot_ks = [random.randrange(4) for _ in range(3)]
        if not any(rot_ks):
            continue
        if rot_ks[1] != rot_ks[2]:
            return rot_ks


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < int(area):
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return int(max(dr, dc))


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _interaction_candidates(n: int) -> list[tuple[int, int]]:
    """Central cells whose 4-step orbit stays distinct, so collisions are visible but local."""
    center = (n - 1) / 2.0
    out: list[tuple[int, int]] = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if max(abs(r - center), abs(c - center)) > INTERACTION_WINDOW:
                continue
            if len(_rot_orbit(r, c, n)) < 4:
                continue
            out.append((int(r), int(c)))
    return out


def _touches_cluster(cell: tuple[int, int], cluster: list[tuple[int, int]]) -> bool:
    r, c = cell
    return any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in cluster)


def _inject_mid_inner_collision(
    outer_mask: np.ndarray,
    mid_mask: np.ndarray,
    inner_mask: np.ndarray,
    rot_mid: int,
    rot_inner: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    """Add disjoint input cells whose rotated images collide in the center."""
    n = int(outer_mask.shape[0])
    candidates = _interaction_candidates(n)
    if not candidates:
        return None

    inv_mid = (-int(rot_mid)) % 4
    inv_inner = (-int(rot_inner)) % 4

    for _ in range(240):
        target_count = random.randint(INTERACTION_OVERLAP_MIN, INTERACTION_OVERLAP_MAX)
        pool = random.sample(candidates, len(candidates))
        targets: list[tuple[int, int]] = []
        while pool and len(targets) < target_count:
            cell = pool.pop()
            if not targets or _touches_cluster(cell, targets):
                targets.append(cell)
        if len(targets) < target_count:
            continue

        new_mid = mid_mask.copy()
        new_inner = inner_mask.copy()
        ok = True
        for tr, tc in targets:
            sm = _rotate_coord(tr, tc, n, inv_mid)
            si = _rotate_coord(tr, tc, n, inv_inner)

            if sm == si:
                ok = False
                break
            if outer_mask[sm] or outer_mask[si]:
                ok = False
                break
            if new_inner[sm] or new_mid[si]:
                ok = False
                break

            new_mid[sm] = True
            new_inner[si] = True

        if not ok:
            continue
        if np.any(new_mid & new_inner):
            continue

        r_mid = np.rot90(new_mid.astype(np.uint8), int(rot_mid)).astype(bool)
        r_inner = np.rot90(new_inner.astype(np.uint8), int(rot_inner)).astype(bool)
        pair_overlap = r_mid & r_inner
        if int(pair_overlap.sum()) < target_count:
            continue
        return new_mid, new_inner, targets
    return None


def _variant_profile(grid: np.ndarray) -> dict[str, int]:
    """Quantify the D17 behavior on a generated input."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg

    cols = colors_present(crop, exclude={bg})
    masks = {int(col): crop == int(col) for col in cols}
    rot_ks: dict[int, int] = {}
    for col in cols:
        comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, colors=[int(col)], return_format="dict")
        cue_size = int(comps[0]["area"]) if comps else 0
        rot_ks[int(col)] = (-cue_size) % 4

    _out, rotated, hit_count = compose_rotated_layers(masks, rot_ks, bg=bg)
    cols = sorted(rotated, key=lambda col: int(np.sum(masks[int(col)])))
    pair_mid_inner = 0
    if len(cols) == 3:
        pair_mid_inner = int(np.sum(rotated[cols[1]] & rotated[cols[2]]))
    return {
        "overlap_cells": int(np.sum(hit_count >= 2)),
        "mid_inner_overlap": int(pair_mid_inner),
        "surviving_colors": len(colors_present(_out, exclude={bg})),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = colors
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, rot_ks[2] > 0)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=rot_ks[2] > 0)
    if rot_ks[2] > 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    injected = _inject_mid_inner_collision(outer_mask, mid_mask, inner_mask, rot_ks[1], rot_ks[2])
    if injected is None:
        return None
    mid_mask, inner_mask, targets = injected

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    color_masks = {
        int(c_frame): outer_mask,
        int(c_mid): mid_mask,
        int(c_inner): inner_mask,
    }
    rot_map = {
        int(c_frame): int(rot_ks[0]),
        int(c_mid): int(rot_ks[1]),
        int(c_inner): int(rot_ks[2]),
    }
    out_board, rotated_masks, hit_count = compose_rotated_layers(color_masks, rot_map, bg=0)

    pair_mid_inner = int(np.sum(rotated_masks[int(c_mid)] & rotated_masks[int(c_inner)]))
    if pair_mid_inner < len(targets):
        return None
    if int(np.sum(hit_count >= 2)) < INTERACTION_OVERLAP_MIN:
        return None
    if np.any(rotated_masks[int(c_frame)] & rotated_masks[int(c_mid)]):
        return None
    if np.any(rotated_masks[int(c_frame)] & rotated_masks[int(c_inner)]):
        return None
    if len(colors_present(out_board, exclude={0})) != NUM_COLORS:
        return None

    in_board = np.zeros((n, n), dtype=int)
    in_board[outer_mask] = int(c_frame)
    in_board[mid_mask] = int(c_mid)
    in_board[inner_mask] = int(c_inner)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (col, random.choice([1, 5]) if (-k) % 4 == 1 else (-k) % 4)
        for col, k in zip(colors, rot_ks)
        if k != 0
    ]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(col, area), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D17 input")
