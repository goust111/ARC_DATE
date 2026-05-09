# Combo variant of 6ffbe589 (index 19)
# combo_spec: D12/2+D17/2
# Name: 标记像素+重叠擦除

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior:
#    The original task rotated three same-color masks inside a square board by
#    cue-controlled quarter turns. Here the three large layer bodies are made
#    deliberately 90-degree symmetric, so the only orientation witness is a
#    small same-color marker pixel near the border. The outer layer has its own
#    marker slot A, while the middle and inner layers both target slot B in the
#    output frame. A traced example is:
#    - input outer marker sits at rot^-1(A), input middle marker already sits at
#      B because its cue is absent, input inner marker sits at rot^-1(B).
#    - solver rotates all three full masks independently from the cropped input.
#    - after rotation, the outer marker lands on A and survives; the middle and
#      inner markers both land on B, so B is erased to background.
#    Relative to the original task, visible motion comes from markers instead of
#    from the body shapes, and one moved marker disappears because of overlap.
#
# b) Complexity calibration:
#    The generator keeps only two active rotation-bearing colors per sample:
#    outer always rotates and keeps a surviving marker; inner always rotates and
#    collides with the static middle marker. This avoids an underconstrained
#    "all markers vanish" case while still making the overlap rule unavoidable.
#    The control space stays small: one stable collision slot, one surviving
#    marker slot, and cue sizes still encode {90, 180, 270}.
#
# c) Hidden assumptions that break:
#    The original sequential `rotate_color_layer` logic assumed no post-rotation
#    collisions, so in-place writes were safe. D17/2 breaks that directly.
#    D12/2 also removes the old assumption that a rotating layer's large body is
#    itself visibly asymmetric. After the combo change:
#    - masks must be extracted from the original crop before any writes;
#    - rotated masks must be merged in one collision-aware pass;
#    - the solver must not assume a surviving marker exists for every rotated
#      layer, because one marker is intentionally erased at the shared slot.
#
# d) Placement legality:
#    Input masks are kept pairwise disjoint by construction. The border marker
#    orbits are chosen outside the middle/inner body regions, so inverse-rotated
#    markers never fall onto another body's cells. The full board stays 8-
#    connected because the static middle marker bridges the outer ring to the
#    middle annulus in the input, while the middle annulus touches the inner
#    sprite around the centered hole.
#
# e) Visual observability:
#    Across samples, the outer marker visibly moves and survives, so the cue ->
#    rotation rule remains easy to see. The inner marker also moves, but it
#    always disappears at the same border slot already occupied by the middle
#    marker. That gives a human-solvable "moved there, then got erased" pattern
#    instead of a hidden numeric trick.
#
# f) Visual cleanliness:
#    Only one output cell is forced to collide, and it is a marker cell near the
#    border rather than clutter in the dense center. The three main bodies stay
#    as clean concentric structures, so the output does not devolve into noisy
#    multi-layer interference.
#
# g) Inference stability:
#    No color remapping is introduced. Colors still identify layers exactly as
#    in the base task; the new reasoning burden is purely spatial: move each
#    layer by its same-color cue, then blank any cell hit by multiple layers.
# -----------------------------------------------------------------------------


CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

MIDDLE_INSET = 2
OUTER_MARKER = (1, 2)
SHARED_MARKER = (2, 1)

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4

GEN_RETRIES = 300


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper: crop the foreground cluster near the unique largest component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox uses inclusive coordinates.
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
    assert keep_coords.size > 0, "expected the crop to retain the board cluster"

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
    """Rotate each color mask independently, then erase cells hit by 2+ masks."""
    shape = next(iter(color_masks.values())).shape
    rotated_masks: dict[int, np.ndarray] = {}
    hit_count = np.zeros(shape, dtype=int)

    for col, mask in color_masks.items():
        rmask = np.rot90(mask.astype(np.uint8), int(rot_ks.get(int(col), 0)) % 4).astype(bool)
        rotated_masks[int(col)] = rmask
        hit_count += rmask.astype(int)

    out = np.full(shape, int(bg), dtype=int)
    safe = hit_count == 1
    for col, rmask in rotated_masks.items():
        out[safe & rmask] = int(col)
    return out, rotated_masks, hit_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster, rotate cue-matched color layers, and erase collisions.

    Input structure:
    - The dominant color is background on a 20x20 canvas.
    - The unique largest foreground component is a square three-layer board.
    - Each board color has a mostly rotation-symmetric body plus a small border
      marker pixel, and distant same-color cue sprites may appear outside it.

    Rule:
    - Crop the board away from the distant cue sprites.
    - For each board color, read the area of its same-color outside cue modulo 4
      to determine the quarter-turn rotation; missing cue means no rotation.
    - Rotate every color mask independently inside the fixed square crop.
    - Merge the rotated masks in one pass, clearing any cell claimed by 2 or
      more colors back to background.

    Key insight:
    The cues tell you where each color's marker should move, but a marker can
    vanish if another rotated layer lands on the same destination cell.
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


def _rot_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    """Return the orbit of one coordinate under repeated 90-degree rotation."""
    pts: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        pt = (int(rr), int(cc))
        if pt not in pts:
            pts.append(pt)
        rr, cc = n - 1 - cc, rr
    return tuple(pts)


def _marker_input_pos(marker: tuple[int, int], rot_k: int, n: int) -> tuple[int, int]:
    """Where a canonical output marker must appear in the input before rotation."""
    return tuple(map(int, _rot_orbit(int(marker[0]), int(marker[1]), int(n))[(-int(rot_k)) % 4]))


def _with_marker(mask: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    out = mask.copy()
    out[int(marker[0]), int(marker[1])] = True
    return out


def _outer_base(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    mask[n - 1, :] = True
    mask[:, 0] = True
    mask[:, n - 1] = True
    return mask


def _middle_base(n: int) -> np.ndarray:
    """Centered annulus: full inset square minus a tiny centered hole."""
    mask = np.zeros((n, n), dtype=bool)
    lo = int(MIDDLE_INSET)
    hi = int(n - MIDDLE_INSET)
    mask[lo:hi, lo:hi] = True

    hole_span = 2 if (n % 2) == 0 else 3
    hole_top = (n - hole_span) // 2
    mask[hole_top : hole_top + hole_span, hole_top : hole_top + hole_span] = False
    return mask


def _inner_base(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if (n % 2) == 0:
        c0 = n // 2 - 1
        mask[c0 : c0 + 2, c0 : c0 + 2] = True
    else:
        cr = cc = n // 2
        mask[cr, cc] = True
        mask[cr - 1, cc] = True
        mask[cr + 1, cc] = True
        mask[cr, cc - 1] = True
        mask[cr, cc + 1] = True
    return mask


def _pick_rotations() -> tuple[int, int, int]:
    """Outer rotates visibly, middle stays put, inner rotates into the shared slot."""
    return random.randint(1, 3), 0, random.randint(1, 3)


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
        arr[int(r), int(c)] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _cue_area_for_rotation(rot_k: int) -> int:
    mod = (-int(rot_k)) % 4
    if mod == 0:
        raise ValueError("rotation 0 should not request a cue")
    return 5 if mod == 1 else mod


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


def _plain_rotated_merge(
    color_masks: dict[int, np.ndarray],
    rot_ks: dict[int, int],
    *,
    bg: int,
) -> np.ndarray:
    """Reference merge without overlap erasure, used only for variant diagnostics."""
    shape = next(iter(color_masks.values())).shape
    out = np.full(shape, int(bg), dtype=int)
    for col in sorted(color_masks):
        rmask = np.rot90(color_masks[int(col)].astype(np.uint8), int(rot_ks.get(int(col), 0)) % 4).astype(bool)
        out[rmask] = int(col)
    return out


def _variant_profile(grid: np.ndarray) -> dict[str, int]:
    """Quantify that the combo behavior is present on a generated sample."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg

    cols = colors_present(crop, exclude={bg})
    color_masks = {int(col): crop == int(col) for col in cols}
    rot_ks: dict[int, int] = {}
    for col in cols:
        comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, colors=[int(col)], return_format="dict")
        cue_size = int(comps[0]["area"]) if comps else 0
        rot_ks[int(col)] = (-cue_size) % 4

    out, rotated, hit_count = compose_rotated_layers(color_masks, rot_ks, bg=bg)
    plain = _plain_rotated_merge(color_masks, rot_ks, bg=bg)
    sr, sc = SHARED_MARKER
    orr, occ = OUTER_MARKER
    return {
        "collision_cells": int(np.sum(hit_count >= 2)),
        "shared_slot_hits": int(hit_count[sr, sc]),
        "shared_slot_erased": int(out[sr, sc] == bg),
        "outer_marker_survives": int(hit_count[orr, occ] == 1 and out[orr, occ] != bg),
        "diff_vs_plain_merge": int(np.sum(out != plain)),
        "rotating_colors": int(sum(1 for k in rot_ks.values() if int(k) != 0)),
        "surviving_colors": int(len(colors_present(out, exclude={bg}))),
        "rotated_masks_with_shared_hit": int(sum(1 for mask in rotated.values() if mask[sr, sc])),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_outer, c_middle, c_inner = colors
    rot_outer, rot_middle, rot_inner = _pick_rotations()

    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

    out_masks = {
        int(c_outer): _with_marker(_outer_base(n), OUTER_MARKER),
        int(c_middle): _with_marker(_middle_base(n), SHARED_MARKER),
        int(c_inner): _with_marker(_inner_base(n), SHARED_MARKER),
    }
    out_board, _rotated_masks, hit_count = compose_rotated_layers(
        out_masks,
        {int(c_outer): 0, int(c_middle): 0, int(c_inner): 0},
        bg=0,
    )
    if int(hit_count[SHARED_MARKER[0], SHARED_MARKER[1]]) != 2:
        return None
    if out_board[SHARED_MARKER] != 0:
        return None
    if out_board[OUTER_MARKER] == 0:
        return None

    in_masks = {
        int(c_outer): np.rot90(out_masks[int(c_outer)].astype(np.uint8), (-rot_outer) % 4).astype(bool),
        int(c_middle): out_masks[int(c_middle)].copy(),
        int(c_inner): np.rot90(out_masks[int(c_inner)].astype(np.uint8), (-rot_inner) % 4).astype(bool),
    }

    occupied = np.zeros((n, n), dtype=int)
    for col in colors:
        mask = in_masks[int(col)]
        if np.any((occupied != 0) & mask):
            return None
        occupied[mask] = int(col)

    if not np.all(occupied == np.where(occupied == 0, 0, occupied)):
        return None

    # Sanity-check the intended marker behavior directly on the input board.
    input_outer_marker = _marker_input_pos(OUTER_MARKER, rot_outer, n)
    input_inner_marker = _marker_input_pos(SHARED_MARKER, rot_inner, n)
    if occupied[input_outer_marker] != int(c_outer):
        return None
    if occupied[SHARED_MARKER] != int(c_middle):
        return None
    if occupied[input_inner_marker] != int(c_inner):
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, occupied, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (int(c_outer), _cue_area_for_rotation(rot_outer)),
        (int(c_inner), _cue_area_for_rotation(rot_inner)),
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

    solved = main(canvas)
    if not np.array_equal(solved, out_board):
        return None

    profile = _variant_profile(canvas)
    if profile["shared_slot_hits"] != 2:
        return None
    if profile["shared_slot_erased"] != 1:
        return None
    if profile["outer_marker_survives"] != 1:
        return None
    if profile["diff_vs_plain_merge"] <= 0:
        return None
    if profile["rotating_colors"] != 2:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")
