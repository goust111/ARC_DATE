# Combo variant of 898e7135 (index 9)
# combo_spec: D1/2+D2/2
# Name: 轮廓方向匹配

from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np
from scipy import ndimage

from arc_utils import (
    bbox,
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: each outside sprite is a solid k-scaled polyomino; the solver can
#      rotate/flip it before comparing its block mask with a frame hole.
#    - Combo task: each outside sprite keeps the slot's original orientation, but is shown
#      only as a 1-pixel outline of the k-scaled solid. The solver must first recover the
#      solid region from that outline, then compare the recovered block mask with no D4 move.
#    - Small trace: consider two L-shaped slots that are the same under rotation but face
#      different directions. The input now shows two hollow L outlines with those exact two
#      directions. A relaxed D4 solver could still swap them; the combo solver cannot, because
#      it matches the recovered block masks exactly as drawn.
#
# b) Complexity calibration
#    - Hollow outlines alone are too weak if every non-unit slot is D4-unique; the task would
#      only test "fill the outline". Strict orientation alone is too weak if sprites stay solid.
#    - To make both dimensions visible, the generator always includes one non-unit pair that
#      shares a D4 family but differs in exact orientation, while keeping all remaining non-unit
#      slots D4-unique. It also rejects samples where every outline is visually identical to its
#      filled solid, so the recovery step is not cosmetic.
#
# c) Implicit-assumption audit of main()
#    - The original `k` inference from a smallest solid square piece is broken by hollow inputs.
#      The combo solver instead takes the gcd of all piece bbox dimensions; the generator keeps
#      every piece as a k-scaled slot and preserves 1-2 unit slots, so the gcd is exactly k.
#    - The original direct block downsampling is broken by outline masks because many k-by-k
#      blocks become partial. The combo solver must call binary hole-fill before downsampling.
#    - The original D4 candidate expansion is no longer valid for the real solve path; strict
#      matching uses only the given orientation. A relaxed D4 solve is kept only for anti-
#      degeneracy validation inside the generator.
#
# d) Placement legality
#    - Hollow pieces contain background inside their bbox, so using the visible patch for overlap
#      checks would allow other objects to fall into the cavity. The generator therefore keeps a
#      separate `reserve` canvas occupied by each piece's recovered solid mask and by the whole
#      frame bbox. All placements and distractors avoid that reserve.
#
# e) Visual observability
#    - Across examples, humans can see two stable signals: outside pieces are hollow while output
#      insertions are solid, and at least one pair of same-family shapes keeps orientation rather
#      than being freely rotatable. This is inferable from a few examples without hidden state.
#
# f) Visual cleanliness
#    - Object classes remain only frame, hollow pieces, and isolated singleton distractors.
#      Reserve-based placement prevents clutter inside outlines, and 8-neighborhood spacing keeps
#      all objects visually separated.
#
# g) Color inferability
#    - Color still only marks identity: frame color is the output background, non-unit pieces use
#      distinct colors, and unit pieces may share one color. No rule depends on numeric color ids.


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8
PAIR_PLACE_RETRIES = 24

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400
FAMILY_ID = 0

# All bases here are asymmetric so they provide at least two exact orientations.
ORIENTATION_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


# --- shared helpers ----------------------------------------------------------

def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_shape_key(v) for v in _unique_d4_variants(mask))


def _unique_d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for var in (rot, np.fliplr(rot)):
            key = _shape_key(var)
            if key in seen:
                continue
            seen.add(key)
            out.append(var.copy())
    return out


def _outline_mask(filled: np.ndarray) -> np.ndarray:
    """Keep only the 1-pixel outer contour of a solid orthogonal region."""
    boundary = filled.copy()
    if filled.shape[0] >= 3 and filled.shape[1] >= 3:
        interior = (
            filled[1:-1, 1:-1]
            & filled[:-2, 1:-1]
            & filled[2:, 1:-1]
            & filled[1:-1, :-2]
            & filled[1:-1, 2:]
        )
        boundary[1:-1, 1:-1] &= ~interior
    return boundary


def _recover_filled(mask: np.ndarray) -> np.ndarray:
    """Recover the solid sprite represented by a hollow contour."""
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(bool)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    out_h, out_w = hh // int(k), ww // int(k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[r * int(k) : (r + 1) * int(k), c * int(k) : (c + 1) * int(k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _roundtrips_under_outline(mask: np.ndarray, k: int) -> bool:
    """Check that outline -> fill -> block-downsample returns the original block mask."""
    solid = np.kron(mask.astype(bool), np.ones((k, k), dtype=bool))
    recovered = _recover_filled(_outline_mask(solid))
    down = _downsample_blocks(recovered, k)
    return down is not None and np.array_equal(down, mask)


def _mask_to_slot(coords: np.ndarray, *, family_id: int | None = None) -> dict:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    sm = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "r0": int(r0),
        "c0": int(c0),
        "mask": sm,
        "shape_key": _shape_key(sm),
        "canon_key": _canonical_shape_key(sm),
        "family_id": family_id,
    }


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = list(map(tuple, np.argwhere(~forbidden).tolist()))
    if not free_cells:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < int(target_size) and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((int(nr), int(nc)))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == int(target_size):
            return np.array(sorted(chosen), dtype=int)
    return None


def _place_mask_slot(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    ones = np.argwhere(mask)
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            coords = ones + np.array([[int(r0), int(c0)]], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    return ones + np.array([[int(r0), int(c0)]], dtype=int)


def _random_reserved_location(reserve: np.ndarray, solid_mask: np.ndarray) -> tuple[int, int]:
    return random_free_location(
        reserve,
        solid_mask.astype(int),
        transparent=0,
        border=0,
        padding=1,
        padding_connectivity=8,
    )


def _legal_positions(reserve: np.ndarray, solid_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, solid_mask.shape)
    ch, cw = map(int, reserve.shape)
    if h > ch or w > cw:
        return []
    occ = _dilate8(reserve != 0)
    out: list[tuple[int, int]] = []
    for r0 in range(ch - h + 1):
        for c0 in range(cw - w + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & solid_mask).any()):
                out.append((int(r0), int(c0)))
    return out


def _reserve_place(reserve: np.ndarray, solid_mask: np.ndarray, top: int, left: int) -> np.ndarray:
    out = reserve.copy()
    paste(out, np.ones_like(solid_mask, dtype=int), int(top), int(left), mask=solid_mask, overwrite=True)
    return out


def _find_swapped_pair_positions(
    reserve: np.ndarray, early_piece: dict, late_piece: dict
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Force the later slot's piece to appear earlier in scan order.

    Strict orientation still solves correctly, but a relaxed D4 solver that greedily assigns
    the first matching component to the first slot will swap this pair.
    """
    late_candidates = _legal_positions(reserve, late_piece["reserve_mask"])
    for late_pos in late_candidates:
        reserve_after_late = _reserve_place(reserve, late_piece["reserve_mask"], late_pos[0], late_pos[1])
        early_candidates = [
            pos for pos in _legal_positions(reserve_after_late, early_piece["reserve_mask"]) if pos > late_pos
        ]
        if early_candidates:
            return late_pos, random.choice(early_candidates)
    return None


# --- solver ------------------------------------------------------------------

def _solve(grid: np.ndarray, *, allow_dihedral: bool, recover_outline: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one background hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(tuple(map(int, patch.shape)), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        colors = patch[pm]
        color = int(Counter(map(int, colors.tolist())).most_common(1)[0][0])
        pieces.append({"mask": pm, "color": color})
    assert pieces, "at least one non-frame piece must exist"

    dims = [int(v) for p in pieces for v in p["mask"].shape]
    k = dims[0]
    for v in dims[1:]:
        k = gcd(int(k), int(v))
    assert k >= 2, "combo variant expects a block scale of at least 2"

    candidates = []
    for pi, p in enumerate(pieces):
        variants = _unique_d4_variants(p["mask"]) if allow_dihedral else [p["mask"]]
        for vm in variants:
            solid = _recover_filled(vm) if recover_outline else vm.astype(bool)
            um = _downsample_blocks(solid, k)
            if um is None:
                continue
            vp = np.where(solid, int(p["color"]), 0).astype(int)
            candidates.append(
                {
                    "pi": int(pi),
                    "solid_area": int(solid.sum()),
                    "um": um,
                    "vp": vp,
                    "vm": solid,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["solid_area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the frame layout and refill each slot with the correctly oriented recovered sprite.

    Input structure:
    - The largest foreground component is a monochrome frame whose bbox encloses background holes.
    - Outside the frame are monochrome pieces drawn as hollow 1-pixel contours of k-scaled slot
      shapes, plus isolated singleton distractors in another color.

    Rule:
    - Extract each hole region inside the frame as a slot mask.
    - For every outside piece, fill its contour back into the solid k-scaled shape, convert that
      solid to a block-level mask, and match it to a slot in the piece's current orientation only.
      No rotation or flip is allowed during matching.
    - Output the tightly cropped frame layout enlarged by k, with each matched piece restored as a
      solid colored insertion and the frame color used as the background.

    Key insight:
    - The input pieces are not directly block-downsampleable because their interiors are missing.
      First recover the filled region from the outline, then compare the recovered block masks
      exactly as seen; the decisive ambiguity is orientation, not color or position.
    """
    return _solve(grid, allow_dihedral=False, recover_outline=True)


# --- layout generation -------------------------------------------------------

def _gen_layout(h: int, w: int, n_slots: int, n_units: int, k: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_exact: set[tuple[int, int, bytes]] = set()
    used_canon: set[tuple[int, int, bytes]] = set()

    for _ in range(int(n_units)):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            slots.append(slot)
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    pair_slots = None
    for _ in range(PAIR_PLACE_RETRIES):
        base = random.choice(ORIENTATION_BASE_SHAPES)
        variants = _unique_d4_variants(base)
        if len(variants) < 2:
            continue
        m1, m2 = random.sample(variants, 2)
        if _shape_key(m1) == _shape_key(m2):
            continue
        if not (_roundtrips_under_outline(m1, k) and _roundtrips_under_outline(m2, k)):
            continue
        trial_forbidden = forbidden.copy()
        coords1 = _place_mask_slot(trial_forbidden, m1)
        if coords1 is None:
            continue
        _forbid_coords(trial_forbidden, coords1)
        coords2 = _place_mask_slot(trial_forbidden, m2)
        if coords2 is None:
            continue
        _forbid_coords(trial_forbidden, coords2)
        s1 = _mask_to_slot(coords1, family_id=FAMILY_ID)
        s2 = _mask_to_slot(coords2, family_id=FAMILY_ID)
        pair_slots = [s1, s2]
        forbidden = trial_forbidden
        used_exact.update({s1["shape_key"], s2["shape_key"]})
        used_canon.add(s1["canon_key"])
        break
    if pair_slots is None:
        return None
    slots.extend(pair_slots)

    for _ in range(n_slots - n_units - 2):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            if not _roundtrips_under_outline(slot["mask"], k):
                continue
            if slot["shape_key"] in used_exact:
                continue
            if slot["canon_key"] in used_canon:
                continue
            slots.append(slot)
            used_exact.add(slot["shape_key"])
            used_canon.add(slot["canon_key"])
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None
    return slots


# --- piece construction ------------------------------------------------------

def _make_piece(slot: dict, color: int, k: int) -> dict:
    solid_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    outline_mask = _outline_mask(solid_mask)
    out_patch = np.where(solid_mask, int(color), 0).astype(int)
    in_patch = np.where(outline_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "family_id": slot.get("family_id"),
        "out_patch": out_patch,
        "out_mask": solid_mask,
        "in_patch": in_patch,
        "in_mask": outline_mask,
        "reserve_mask": solid_mask,
        "hollow_gap": int(solid_mask.sum()) - int(outline_mask.sum()),
    }


# --- output assembly ---------------------------------------------------------

def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], int(top), int(left), mask=p["out_mask"], overwrite=True)
    return out


# --- input assembly ----------------------------------------------------------

def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(p["reserve_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (
        canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]
    ):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    reserve = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    fr = random.randint(1, fr_max)
    fc = random.randint(1, fc_max)
    paste(canvas, frame_img, fr, fc)
    paste(reserve, np.ones((frame_h, frame_w), dtype=int), fr, fc, overwrite=True)

    family_pieces = [p for p in pieces if p.get("family_id") == FAMILY_ID]
    assert len(family_pieces) == 2
    family_pieces = sorted(family_pieces, key=lambda p: _slot_sort_key(p["slot"]))
    early_piece, late_piece = family_pieces[0], family_pieces[1]
    swapped = _find_swapped_pair_positions(reserve, early_piece, late_piece)
    if swapped is None:
        return None
    late_pos, early_pos = swapped
    paste(canvas, late_piece["in_patch"], late_pos[0], late_pos[1], mask=late_piece["in_mask"], overwrite=True)
    reserve = _reserve_place(reserve, late_piece["reserve_mask"], late_pos[0], late_pos[1])
    paste(canvas, early_piece["in_patch"], early_pos[0], early_pos[1], mask=early_piece["in_mask"], overwrite=True)
    reserve = _reserve_place(reserve, early_piece["reserve_mask"], early_pos[0], early_pos[1])

    handled = {id(early_piece), id(late_piece)}
    others = [p for p in pieces if id(p) not in handled]
    for p in sorted(others, key=lambda item: -int(item["reserve_mask"].sum())):
        try:
            pr, pc = _random_reserved_location(reserve, p["reserve_mask"])
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)
        reserve = _reserve_place(reserve, p["reserve_mask"], pr, pc)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0 or reserve[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        if np.any(reserve[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


# --- generation --------------------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - 2)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units, k)
    if slots is None:
        return None
    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
        return None

    non_unit_sorted = sorted(non_unit_slots, key=_slot_sort_key)
    pieces = [_make_piece(s, non_unit_colors[i], k) for i, s in enumerate(non_unit_sorted)]
    pieces += [_make_piece(s, unit_color, k) for s in unit_slots]

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    if frame_area <= max(int(p["in_mask"].sum()) for p in pieces):
        return None
    if not any(int(p["hollow_gap"]) > 0 for p in pieces):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        relaxed = _solve(canvas, allow_dihedral=True, recover_outline=True)
    except Exception:
        return None
    try:
        no_fill = _solve(canvas, allow_dihedral=False, recover_outline=False)
        no_fill_same = np.array_equal(no_fill, expected_out)
    except Exception:
        no_fill_same = False
    if not np.array_equal(produced, expected_out):
        return None
    if np.array_equal(relaxed, expected_out):
        return None
    if no_fill_same:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
