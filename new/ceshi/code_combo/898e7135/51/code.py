# Combo variant of 898e7135 (index 51)
# combo_spec: D1/2+D2/2+D6/3
# Name: 轮廓辨别严格匹配

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: every outside piece is a solid k-scaled sprite, every non-singleton
#      component is assumed real, shapes may be matched after rotation/flip, and the
#      smallest solid square is used to infer k.
#    - Combo task:
#      1. true pieces are hollow 1-pixel outlines of the k-scaled slot shapes;
#      2. 1-2 extra solid pseudo-pieces are also scattered outside the frame;
#      3. recovered block masks must match slots in the observed orientation only.
#    - Small trace:
#      slot A and slot B are two different orientations of the same L triomino family.
#      Their true outside pieces are hollow outlines in those same two orientations.
#      A solid pseudo-piece is also present nearby.
#      Old logic either fails on the hollow masks or rotates one L to fit the other.
#      New logic first keeps only outline-compatible components, fills each outline back
#      to a solid k-scaled polyomino, and then compares the recovered block mask directly.
#      The solid decoy is ignored, and the two L pieces can no longer swap by rotation.
#
# b) Complexity calibration
#    - The generator always includes one non-unit pair from the same D4 family but with
#      different exact orientations. This makes strict orientation observable.
#    - All non-unit true pieces are chosen so that their hollow outline visibly differs
#      from the solid fill (`outline_area < solid_area`), so the hollow-vs-solid cue is
#      not limited to an invisible internal convention.
#    - Pseudo-pieces are few (1-2), solid, and D4-distinct from every true slot family.
#      This keeps the scene readable while still forcing a real filtering step.
#
# c) Implicit-assumption audit of the original main()
#    - "Every multi-pixel component is a true piece": no longer valid; solid decoys exist.
#    - "Input masks can be block-downsampled directly": no longer valid; hollow outlines
#      must be filled first.
#    - "Rotations/flips are allowed while matching": no longer valid; exact orientation
#      must be preserved.
#    - "Smallest solid square gives k": no longer valid in the presence of solid decoys.
#      The combo solver instead infers k from outline-compatible pieces only.
#    - New solver assumption: every true piece is exactly the outer contour of a scaled
#      solid polyomino, so `outline_mask(fill_holes(piece)) == piece`. The generator
#      guarantees this for true pieces and generates solid decoys with visible interior so
#      they fail that test.
#
# d) Placement legality
#    - Hollow true pieces reserve the footprint of their recovered solid shape, not just
#      their visible contour. This prevents other objects from being placed inside a hollow.
#    - The frame reserves its full bounding rectangle, so no outside object can land inside
#      the frame interior even if the visible frame is only a border.
#    - All objects are placed with 8-neighborhood padding of at least one background cell.
#
# e) Visual observability
#    - Across examples, humans can see that the useful outside pieces are hollow outlines
#      while the irrelevant extras are solid blobs.
#    - The output restores those hollow outlines as solid fills inside the frame, making the
#      contour-recovery step visible.
#    - Because one true pair shares a D4 family but not an exact orientation, the samples
#      also reveal that "same shape up to rotation" is not enough.
#
# f) Visual cleanliness
#    - Only one frame, a small number of separated hollow true pieces, and 1-2 separated
#      solid decoys are present.
#    - Reserve-based placement avoids contour-overlap artifacts and keeps every object easy
#      to read by eye.
#
# g) Color inferability and post-change generator guarantees
#    - Colors still identify objects only; they do not encode hidden ordering.
#    - Non-unit true pieces keep distinct colors, unit pieces may share one color, and each
#      decoy gets its own fresh color.
#    - The generator guarantees:
#      1. at least one visibly hollow true piece;
#      2. at least one same-family-different-orientation true pair;
#      3. every solid decoy fails the outline-compatibility test used by main().

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


# --- shared helpers ----------------------------------------------------------

FAMILY_ID = 0

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
PAIR_PLACE_RETRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

N_DECOYS_CHOICES = [1, 2]
DECOY_MASK_TRIES = 80
GENERATE_MAX_RETRIES = 400

ORIENTATION_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _shape_key(mask: np.ndarray) -> bytes:
    return bytes([int(mask.shape[0]), int(mask.shape[1])]) + mask.astype(np.uint8).tobytes()


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(_shape_key(r))
        variants.append(_shape_key(np.fliplr(r)))
    return min(variants)


def _unique_d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for kk in range(4):
        r = np.rot90(mask, kk)
        for variant in (r, np.fliplr(r)):
            key = _shape_key(variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(variant.copy())
    return out


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_to_slot(coords: np.ndarray, *, family_id: int | None = None) -> dict:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "r0": int(r0),
        "c0": int(c0),
        "mask": mask,
        "shape_key": _shape_key(mask),
        "canon_key": _canonical_shape_key(mask),
        "family_id": family_id,
    }


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _outline_mask(filled: np.ndarray) -> np.ndarray:
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
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(bool)


def _solid_outline_gap(base_mask: np.ndarray, k: int) -> int:
    solid = np.kron(base_mask, np.ones((k, k), dtype=bool))
    outline = _outline_mask(solid)
    return int(solid.sum()) - int(outline.sum())


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if k <= 0 or hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


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


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(mask.sum()), "r0": rr0, "c0": cc0, "mask": mask})
    slots.sort(key=lambda slot: (-int(slot["area"]), int(slot["r0"]), int(slot["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        filled = _recover_filled(mask)
        outline = _outline_mask(filled)
        color = int(Counter(map(int, patch[mask].tolist())).most_common(1)[0][0])
        pieces.append(
            {
                "mask": mask,
                "filled": filled,
                "outline_like": bool(np.array_equal(mask, outline)),
                "color": color,
                "visible_area": int(mask.sum()),
                "filled_area": int(filled.sum()),
                "hollow_gap": int(filled.sum()) - int(mask.sum()),
            }
        )
    assert pieces, "at least one non-frame multi-pixel piece must exist"

    return {
        "grid": g,
        "frame_color": frame_color,
        "layout_shape": tuple(map(int, layout.shape)),
        "slots": slots,
        "pieces": pieces,
    }


def _infer_k(pieces: list[dict]) -> int:
    outline_pieces = [piece for piece in pieces if piece["outline_like"]]
    assert outline_pieces, "need at least one outline-compatible piece"
    dims = [int(v) for piece in outline_pieces for v in piece["filled"].shape]
    k = dims[0]
    for v in dims[1:]:
        k = gcd(int(k), int(v))
    assert k >= 2, "outline-compatible pieces must imply k >= 2"
    return int(k)


def _match_slots(slots: list[dict], pieces: list[dict], k: int, *, allow_dihedral: bool) -> list[tuple[dict, dict]]:
    candidates_by_slot: list[list[dict]] = [[] for _ in slots]
    for pi, piece in enumerate(pieces):
        if not piece["outline_like"]:
            continue
        masks = _unique_d4_variants(piece["filled"]) if allow_dihedral else [piece["filled"]]
        for vm in masks:
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            area = int(vm.sum())
            vp = np.where(vm, int(piece["color"]), 0).astype(int)
            for si, slot in enumerate(slots):
                if area != int(slot["area"]) * int(k) * int(k):
                    continue
                if um.shape != slot["mask"].shape:
                    continue
                if np.array_equal(um, slot["mask"]):
                    candidates_by_slot[si].append({"pi": int(pi), "vp": vp, "vm": vm})

    if any(not cands for cands in candidates_by_slot):
        raise AssertionError("every slot must have at least one outline-compatible candidate")

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(candidates_by_slot[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
    )
    chosen: list[dict | None] = [None] * len(slots)
    used: set[int] = set()

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        si = order[idx]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            chosen[si] = cand
            if dfs(idx + 1):
                return True
            chosen[si] = None
            used.remove(pi)
        return False

    assert dfs(0), "outline-compatible pieces must admit a full one-to-one assignment"
    return [(slots[i], chosen[i]) for i in range(len(slots)) if chosen[i] is not None]


def _solve(grid: np.ndarray, *, allow_dihedral: bool) -> np.ndarray:
    scene = _extract_scene(grid)
    slots = scene["slots"]
    pieces = scene["pieces"]
    k = _infer_k(pieces)
    placements = _match_slots(slots, pieces, k, allow_dihedral=allow_dihedral)

    out_h = (int(scene["layout_shape"][0]) + 2) * int(k)
    out_w = (int(scene["layout_shape"][1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(scene["frame_color"]), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the framed hole layout and refill it with the recovered strictly oriented sprites.

    Input structure:
    - The most common color is background. The largest foreground component is a monochrome
      rectangular frame whose interior background cells define the slot layout.
    - Outside the frame, true sprites are monochrome hollow 1-pixel contours of k-scaled slot
      shapes, while 1-2 extra solid monochrome pseudo-sprites may also appear.
    - True sprites keep the same orientation as their destination slots; no corrective rotation
      or reflection is implied by the input.

    Rule:
    - Read the connected hole regions inside the frame as block-level slots.
    - Among the outside components, keep the ones that are exact outer contours of a filled
      shape, recover each contour to its solid form, infer k from those recovered shapes, and
      match them to slots without rotation or flip.
    - Ignore leftover solid pseudo-sprites and output the tight scaled frame filled with the
      recovered solid colors.

    Key insight:
    - A real sprite is not the visible contour itself but the solid k-scaled polyomino enclosed
      by that contour. Filling the contour reconstructs the slot shape, and only the orientation
      already shown in the input is valid.
    """
    return _solve(grid, allow_dihedral=False)


# --- generator ---------------------------------------------------------------

def _gen_layout(h: int, w: int, n_slots: int, n_units: int, k: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_exact: set[bytes] = set()
    used_canon: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append(_mask_to_slot(coords))
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    pair_slots = None
    valid_bases = [mask for mask in ORIENTATION_BASE_SHAPES if _solid_outline_gap(mask, k) > 0]
    if not valid_bases:
        return None
    for _ in range(PAIR_PLACE_RETRIES):
        base = random.choice(valid_bases)
        variants = [variant for variant in _unique_d4_variants(base) if _solid_outline_gap(variant, k) > 0]
        if len(variants) < 2:
            continue
        m1, m2 = random.sample(variants, 2)
        trial_forbidden = forbidden.copy()
        coords1 = _place_mask_slot(trial_forbidden, m1)
        if coords1 is None:
            continue
        _forbid_coords(trial_forbidden, coords1)
        coords2 = _place_mask_slot(trial_forbidden, m2)
        if coords2 is None:
            continue
        s1 = _mask_to_slot(coords1, family_id=FAMILY_ID)
        s2 = _mask_to_slot(coords2, family_id=FAMILY_ID)
        if s1["shape_key"] == s2["shape_key"]:
            continue
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
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            if _solid_outline_gap(slot["mask"], k) <= 0:
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

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None
    return slots


def _make_true_piece(slot: dict, color: int, k: int) -> dict:
    solid_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    outline_mask = _outline_mask(solid_mask)
    out_patch = np.where(solid_mask, int(color), 0).astype(int)
    in_patch = np.where(outline_mask, int(color), 0).astype(int)
    return {
        "kind": "true",
        "color": int(color),
        "slot": slot,
        "family_id": slot.get("family_id"),
        "out_patch": out_patch,
        "out_mask": solid_mask,
        "in_patch": in_patch,
        "in_mask": outline_mask,
        "reserve_mask": solid_mask,
        "visible_area": int(outline_mask.sum()),
        "hollow_gap": int(solid_mask.sum()) - int(outline_mask.sum()),
    }


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _make_solid_decoy(color: int, k: int, forbidden_canon: set[bytes]) -> dict | None:
    for _ in range(DECOY_MASK_TRIES):
        size = random.choice([3, 4, 5, 6])
        coords = _grow_slot(np.zeros((5, 5), dtype=bool), size)
        if coords is None:
            continue
        base_mask = _tight_mask_from_coords(coords)
        canon = _canonical_shape_key(base_mask)
        if canon in forbidden_canon:
            continue
        if _solid_outline_gap(base_mask, k) <= 0:
            continue
        solid_mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
        solid_mask = _random_dihedral(solid_mask)
        patch = np.where(solid_mask, int(color), 0).astype(int)
        return {
            "kind": "decoy",
            "color": int(color),
            "in_patch": patch,
            "in_mask": solid_mask,
            "reserve_mask": solid_mask,
            "visible_area": int(solid_mask.sum()),
        }
    return None


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


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
    occupied = _dilate8(reserve != 0)
    out: list[tuple[int, int]] = []
    for r0 in range(ch - h + 1):
        for c0 in range(cw - w + 1):
            if not bool((occupied[r0 : r0 + h, c0 : c0 + w] & solid_mask).any()):
                out.append((int(r0), int(c0)))
    return out


def _find_swapped_pair_positions(
    reserve: np.ndarray, early_piece: dict, late_piece: dict
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    late_candidates = sorted(_legal_positions(reserve, late_piece["reserve_mask"]))
    for late_pos in late_candidates:
        trial = reserve.copy()
        paste(
            trial,
            np.ones_like(late_piece["reserve_mask"], dtype=int),
            late_pos[0],
            late_pos[1],
            mask=late_piece["reserve_mask"],
            overwrite=True,
        )
        early_candidates = [
            pos for pos in _legal_positions(trial, early_piece["reserve_mask"]) if pos > late_pos
        ]
        if early_candidates:
            return late_pos, random.choice(early_candidates)
    return None


def _build_input_canvas(
    pieces: list[dict],
    decoys: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    outside_objects = pieces + decoys
    total_area = sum(int(obj["reserve_mask"].sum()) for obj in outside_objects)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_area + AREA_BUFFER_CONST

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

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    frame_visible_area = int((frame_img != 0).sum())
    max_outside_visible = max(int(obj["visible_area"]) for obj in outside_objects)
    if max_outside_visible >= frame_visible_area:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    reserve = np.zeros((canvas_h, canvas_w), dtype=int)

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    fr = random.randint(1, fr_max)
    fc = random.randint(1, fc_max)
    paste(canvas, frame_img, fr, fc)
    paste(reserve, np.ones((frame_h, frame_w), dtype=int), fr, fc)

    family_pieces = [piece for piece in pieces if piece.get("family_id") == FAMILY_ID]
    assert len(family_pieces) == 2
    family_pieces = sorted(family_pieces, key=lambda piece: _slot_sort_key(piece["slot"]))
    early_piece, late_piece = family_pieces[0], family_pieces[1]
    swapped = _find_swapped_pair_positions(reserve, early_piece, late_piece)
    if swapped is None:
        return None
    late_pos, early_pos = swapped
    paste(canvas, late_piece["in_patch"], late_pos[0], late_pos[1], mask=late_piece["in_mask"], overwrite=True)
    paste(
        reserve,
        np.ones_like(late_piece["reserve_mask"], dtype=int),
        late_pos[0],
        late_pos[1],
        mask=late_piece["reserve_mask"],
        overwrite=True,
    )
    paste(canvas, early_piece["in_patch"], early_pos[0], early_pos[1], mask=early_piece["in_mask"], overwrite=True)
    paste(
        reserve,
        np.ones_like(early_piece["reserve_mask"], dtype=int),
        early_pos[0],
        early_pos[1],
        mask=early_piece["reserve_mask"],
        overwrite=True,
    )

    handled = {id(early_piece), id(late_piece)}
    others = [piece for piece in pieces if id(piece) not in handled] + decoys
    for obj in sorted(others, key=lambda item: -int(item["reserve_mask"].sum())):
        try:
            pr, pc = _random_reserved_location(reserve, obj["reserve_mask"])
        except ValueError:
            return None
        paste(canvas, obj["in_patch"], pr, pc, mask=obj["in_mask"], overwrite=True)
        paste(
            reserve,
            np.ones_like(obj["reserve_mask"], dtype=int),
            pr,
            pc,
            mask=obj["reserve_mask"],
            overwrite=True,
        )

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _sample_metrics(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    strict = _solve(grid, allow_dihedral=False)
    relaxed = _solve(grid, allow_dihedral=True)
    true_outline = [piece for piece in scene["pieces"] if piece["outline_like"]]
    solid_decoys = [piece for piece in scene["pieces"] if not piece["outline_like"]]
    return {
        "n_outline_like": len(true_outline),
        "n_solid_decoys": len(solid_decoys),
        "n_visibly_hollow_true": sum(int(piece["hollow_gap"]) > 0 for piece in true_outline),
        "strict_differs_from_relaxed": not np.array_equal(strict, relaxed),
    }


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - 2)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)
    n_decoys = random.choice(N_DECOYS_CHOICES)

    palette = list(range(1, 10))
    random.shuffle(palette)
    needed_piece_colors = (n_slots - n_units) + 1
    needed_colors = 1 + needed_piece_colors + n_decoys
    if len(palette) < needed_colors:
        return None

    frame_color = palette.pop()
    piece_palette = [palette.pop() for _ in range(needed_piece_colors)]
    decoy_colors = [palette.pop() for _ in range(n_decoys)]
    non_unit_colors = piece_palette[: n_slots - n_units]
    unit_color = piece_palette[-1]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units, k)
    if slots is None:
        return None
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    non_unit_sorted = sorted(non_unit_slots, key=_slot_sort_key)
    pieces = [_make_true_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_sorted)]
    pieces += [_make_true_piece(slot, unit_color, k) for slot in unit_slots]

    if not any(int(piece["hollow_gap"]) > 0 for piece in pieces):
        return None

    forbidden_canon = {_canonical_shape_key(slot["mask"]) for slot in non_unit_slots}
    decoys: list[dict] = []
    for color in decoy_colors:
        decoy = _make_solid_decoy(color, k, forbidden_canon)
        if decoy is None:
            return None
        decoys.append(decoy)

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, decoys, layout_mask, layout_h, layout_w, frame_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        relaxed = _solve(canvas, allow_dihedral=True)
        metrics = _sample_metrics(canvas)
    except Exception:
        return None

    if not np.array_equal(produced, expected_out):
        return None
    if np.array_equal(relaxed, expected_out):
        return None
    if metrics["n_solid_decoys"] != len(decoys):
        return None
    if metrics["n_outline_like"] != len(pieces):
        return None
    if metrics["n_visibly_hollow_true"] < 1:
        return None
    if not metrics["strict_differs_from_relaxed"]:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
