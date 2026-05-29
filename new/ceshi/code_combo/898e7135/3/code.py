# Combo variant of 898e7135 (index 3)
# combo_spec: D1/2+D9/3
# Name: 等面积严格形状匹配

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The preserved scaffold is still: one monochrome frame enclosing background holes, one
#      scattered piece per hole outside the frame, and isolated singleton distractors.
#    - The fused change hits the matching stage from two directions at once:
#      1. D1/2 removes all rotation/flip freedom. A piece only matches a slot in its current
#         orientation.
#      2. D9/3 makes every non-unit slot have the same area, so area is no longer informative.
#    - Small trace:
#      suppose the non-unit slots are three area-4 polyominoes: an L facing down-right, the same
#      L family facing up-right, and a T tetromino. All three outside pieces also have area 4*k*k.
#      Original logic would first filter by area, keeping all three candidates, then it could rotate
#      either L to fit either L-shaped hole. The fused logic is stricter: the two L pieces are only
#      distinguishable by exact orientation, and the T by exact shape. Matching is therefore done by
#      downsampling each piece to a block mask and comparing that mask directly to each hole mask.
#
# b) Complexity calibration
#    - Equal area alone is too weak if there is only one non-unit hole, so every generated sample
#      contains at least two non-unit slots.
#    - Strict orientation alone can visually collapse back to the original task if every non-unit
#      shape belongs to a different D4 family. To prevent that, each sample contains one forced pair
#      of non-unit slots from the same D4 family but with different exact orientations.
#    - The common non-unit area is restricted to {3, 4, 5}. Those areas provide enough distinct
#      polyomino families to support 2-5 non-unit slots without making the search space explode.
#
# c) Implicit-assumption review of main()
#    - Frame detection still assumes the largest foreground component is the frame. The generator
#      keeps the frame largest and all pieces disconnected from it.
#    - Tight hole cropping still assumes the inner layout is reconstructed from the bbox of all hole
#      cells. The generator therefore crops the designed slot set to its own hole bbox before drawing
#      the frame, so solver and generator agree on the layout coordinates.
#    - k inference still assumes at least one solid k-by-k square piece exists. The generator keeps
#      1-2 unit slots, which become solid squares outside the frame.
#    - The original `need_area = slot_area * k * k` pre-filter is invalid here because all non-unit
#      slots intentionally share one area. The fused solver removes that filter and matches only by
#      exact downsampled mask.
#    - Once rotation/flip are forbidden, the solver also assumes no two distinct non-unit slots share
#      the same exact oriented mask. The generator guarantees exact-shape uniqueness while still
#      allowing one repeated canonical D4 family with different orientations.
#
# d) Placement legality
#    - Layout generation places each slot on a staging grid and marks its 8-neighborhood forbidden,
#      preserving the original 4-connected / 8-separated hole constraint.
#    - Outside pieces are placed with at least one background cell of 8-connected clearance from the
#      frame and from one another. The special same-family pair is placed first so its scan order on
#      the canvas can be controlled without overlap.
#
# e) Visual observability
#    - A human can see that multiple non-unit holes have equal size, because their cell counts match
#      while their silhouettes differ.
#    - A human can also see the strict-orientation twist, because at least one pair of holes belongs
#      to the same D4 family yet has visibly different orientations, and the outside sprites appear in
#      those already-correct orientations rather than needing mental rotation.
#    - No hidden ordering uses raw color numbers. Colors only identify pieces; matching evidence comes
#      from visible shape and orientation.
#
# f) Visual cleanliness
#    - The frame, pieces, and singleton distractors remain separated objects with no overlays.
#    - The layout stays sparse enough to read because hole density is bounded and non-overlap spacing
#      is enforced both inside and outside the frame.
#
# g) Inferability of the change
#    - The fused rule is structurally supported across samples: all non-unit holes repeat the same
#      area, and at least one canonical family repeats with a different orientation.
#    - This avoids the degenerate case where "strict orientation" never matters or where "equal area"
#      is true but irrelevant because only one non-unit piece exists.

from collections import Counter
import random

import numpy as np

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

PAIR_FAMILY_ID = 0


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = np.array(mask, dtype=bool, copy=False)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool, copy=False)


def _shape_key(mask: np.ndarray) -> bytes:
    tight = _tight_mask(mask)
    return bytes([int(tight.shape[0]), int(tight.shape[1])]) + tight.astype(np.uint8).tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best = None
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            sig = _mask_signature(_tight_mask(var))
            if best is None or sig < best:
                best = sig
    assert best is not None
    return best


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            tight = _tight_mask(var)
            sig = _mask_signature(tight)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(tight.copy())
    return out


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, bytes]] = set()
    for kk in range(4):
        rot_p = np.rot90(patch, int(kk))
        rot_m = np.rot90(mask, int(kk))
        for vp, vm in ((rot_p, rot_m), (np.fliplr(rot_p), np.fliplr(rot_m))):
            sig = _mask_signature(_tight_mask(vm))
            if sig in seen:
                continue
            seen.add(sig)
            out.append((vp, vm))
    return out


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = True
    return out


def _enumerate_free_polyominoes(max_area: int) -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {1: [np.ones((1, 1), dtype=bool)]}
    for area in range(2, max_area + 1):
        next_shapes: dict[tuple[int, int, bytes], np.ndarray] = {}
        for shape in by_area[area - 1]:
            coords = {tuple(map(int, rc)) for rc in np.argwhere(shape)}
            for r, c in list(coords):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in coords:
                        continue
                    grown = set(coords)
                    grown.add((int(nr), int(nc)))
                    mask = _coords_to_mask(grown)
                    key = _canonical_shape_key(mask)
                    if key not in next_shapes:
                        next_shapes[key] = _tight_mask(mask).copy()
        by_area[area] = list(next_shapes.values())
    return by_area


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


def _slot_sort_key(slot: dict) -> tuple[int, int]:
    return int(slot["r0"]), int(slot["c0"])


# --- solver ------------------------------------------------------------------

def _solve(grid: np.ndarray, *, allow_dihedral: bool, use_area_filter: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "shape_sig": _mask_signature(sm),
            }
        )
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
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    candidates = []
    for pi, piece in enumerate(pieces):
        variants = _dihedral_variants(piece["patch"], piece["mask"]) if allow_dihedral else [(piece["patch"], piece["mask"])]
        for vp, vm in variants:
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "shape_sig": _mask_signature(um),
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        found = None
        need_area = int(slot["area"]) * int(k) * int(k)
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if use_area_filter and int(cand["area"]) != int(need_area):
                continue
            if cand["shape_sig"] != slot["shape_sig"]:
                continue
            found = cand
            break
        assert found is not None, "every slot must match one piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole layout and fill it using exact-orientation shape matches.

    Input structure:
    - The most common color is background. The largest foreground component is a monochrome frame
      that encloses several 4-connected background holes.
    - Outside the frame sit monochrome pieces made from solid k-by-k blocks, plus isolated
      single-pixel distractors; at least one unit hole yields a solid k-by-k square piece.

    Rule:
    - Recover the hole layout, split it into slot masks, and downsample each outside piece to block
      level using k.
    - Match every slot to the one unused piece whose downsampled mask is exactly identical in the
      given orientation. No rotation and no flip are allowed, and equal non-unit areas give no
      shortcut.
    - Output the tight hole layout enlarged by k with the matched pieces pasted into their slots on
      the frame color background.

    Key insight:
    - All non-unit candidates can have the same area, so the reliable signal is the exact oriented
      block mask. Downsampling removes the k-by-k scale and exposes the strict shape equality test.
    """
    return _solve(grid, allow_dihedral=False, use_area_filter=False)


# --- generator ---------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_AREA_CHOICES = [3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
LAYOUT_PLACE_RETRIES = 60

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 600

SHAPES_BY_AREA = _enumerate_free_polyominoes(max(NON_UNIT_AREA_CHOICES))
FAMILIES_WITH_ORIENTATION = {
    area: [shape for shape in SHAPES_BY_AREA[area] if len(_dihedral_masks(shape)) >= 2]
    for area in NON_UNIT_AREA_CHOICES
}


def _place_exact_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    mh, mw = map(int, mask.shape)
    if mh > h or mw > w:
        return None
    rel = np.argwhere(mask)
    locs = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
    random.shuffle(locs)
    for r0, c0 in locs:
        coords = rel + np.array([[int(r0), int(c0)]], dtype=int)
        if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
            return coords
    return None


def _slot_from_coords(coords: np.ndarray, *, family_id: int | None = None) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
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


def _crop_slots(slots: list[dict]) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not slots:
        return None, None
    all_coords = np.concatenate([slot["coords"] for slot in slots], axis=0)
    r0, c0, r1, c1 = bbox_coords(all_coords)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    layout_h = int(r1 - r0 + 1)
    layout_w = int(c1 - c0 + 1)
    if not (LAYOUT_H_RANGE[0] <= layout_h <= LAYOUT_H_RANGE[1]):
        return None, None
    if not (LAYOUT_W_RANGE[0] <= layout_w <= LAYOUT_W_RANGE[1]):
        return None, None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    out_slots: list[dict] = []
    for slot in slots:
        coords = slot["coords"] - np.array([[int(r0), int(c0)]], dtype=int)
        cropped = _slot_from_coords(coords, family_id=slot.get("family_id"))
        out_slots.append(cropped)
        layout_mask[coords[:, 0], coords[:, 1]] = True

    density = float(layout_mask.sum()) / float(layout_h * layout_w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None, None
    return out_slots, layout_mask


def _choose_non_unit_masks(n_non_unit: int) -> tuple[int, list[dict]] | tuple[None, None]:
    area_choices = [
        area
        for area in NON_UNIT_AREA_CHOICES
        if len(SHAPES_BY_AREA[area]) >= n_non_unit - 1 and FAMILIES_WITH_ORIENTATION[area]
    ]
    if not area_choices:
        return None, None
    area = random.choice(area_choices)

    pair_family = random.choice(FAMILIES_WITH_ORIENTATION[area])
    pair_variants = _dihedral_masks(pair_family)
    if len(pair_variants) < 2:
        return None, None
    v1, v2 = random.sample(pair_variants, 2)

    others_needed = int(n_non_unit - 2)
    remaining_families = [shape for shape in SHAPES_BY_AREA[area] if _canonical_shape_key(shape) != _canonical_shape_key(pair_family)]
    if len(remaining_families) < others_needed:
        return None, None

    chosen: list[dict] = [
        {"mask": v1, "family_id": PAIR_FAMILY_ID},
        {"mask": v2, "family_id": PAIR_FAMILY_ID},
    ]
    for base in random.sample(remaining_families, others_needed):
        chosen.append({"mask": random.choice(_dihedral_masks(base)), "family_id": None})

    exact_keys = [_shape_key(item["mask"]) for item in chosen]
    if len(exact_keys) != len(set(exact_keys)):
        return None, None
    return int(area), chosen


def _gen_layout(stage_h: int, stage_w: int, n_non_unit: int, n_units: int) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    common_area, chosen_masks = _choose_non_unit_masks(n_non_unit)
    if common_area is None or chosen_masks is None:
        return None, None

    order = sorted(chosen_masks, key=lambda item: (-int(item["mask"].shape[0] * item["mask"].shape[1]), -int(item["mask"].sum())))
    for _ in range(LAYOUT_PLACE_RETRIES):
        forbidden = np.zeros((stage_h, stage_w), dtype=bool)
        placed_slots: list[dict] = []
        ok = True

        for item in order:
            coords = _place_exact_mask(forbidden, item["mask"])
            if coords is None:
                ok = False
                break
            placed_slots.append(_slot_from_coords(coords, family_id=item["family_id"]))
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        for _unit_idx in range(n_units):
            coords = _place_exact_mask(forbidden, np.ones((1, 1), dtype=bool))
            if coords is None:
                ok = False
                break
            placed_slots.append(_slot_from_coords(coords))
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        cropped_slots, layout_mask = _crop_slots(placed_slots)
        if cropped_slots is None or layout_mask is None:
            continue

        non_unit = [slot for slot in cropped_slots if int(slot["area"]) > 1]
        if len(non_unit) != n_non_unit:
            continue
        if {int(slot["area"]) for slot in non_unit} != {int(common_area)}:
            continue
        exact_keys = [slot["shape_key"] for slot in non_unit]
        if len(exact_keys) != len(set(exact_keys)):
            continue
        pair_count = sum(1 for slot in non_unit if slot.get("family_id") == PAIR_FAMILY_ID)
        if pair_count != 2:
            continue
        canon_counts = Counter(slot["canon_key"] for slot in non_unit)
        if max(canon_counts.values()) < 2:
            continue
        return cropped_slots, layout_mask

    return None, None


def _make_piece(slot: dict, color: int, k: int) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "family_id": slot.get("family_id"),
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch.copy(),
            "in_mask": mask.copy(),
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "family_id": slot.get("family_id"),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _legal_positions(canvas: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, patch.shape)
    ch, cw = map(int, canvas.shape)
    if h > ch or w > cw:
        return []
    occ = _dilate8(canvas != 0)
    mask = patch != 0
    out = []
    for r0 in range(ch - h + 1):
        for c0 in range(cw - w + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                out.append((int(r0), int(c0)))
    return out


def _find_swapped_pair_positions(canvas: np.ndarray, early_piece: dict, late_piece: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    late_candidates = _legal_positions(canvas, late_piece["in_patch"])
    if not late_candidates:
        return None
    random.shuffle(late_candidates)
    late_candidates.sort()
    for late_pos in late_candidates:
        trial = canvas.copy()
        paste(trial, late_piece["in_patch"], late_pos[0], late_pos[1], mask=late_piece["in_mask"], overwrite=True)
        early_candidates = [pos for pos in _legal_positions(trial, early_piece["in_patch"]) if pos > late_pos]
        if early_candidates:
            return late_pos, random.choice(early_candidates)
    return None


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    family_pieces = [piece for piece in pieces if piece.get("family_id") == PAIR_FAMILY_ID]
    assert len(family_pieces) == 2
    family_pieces = sorted(family_pieces, key=lambda p: _slot_sort_key(p["slot"]))
    early_piece, late_piece = family_pieces[0], family_pieces[1]
    swapped = _find_swapped_pair_positions(canvas, early_piece, late_piece)
    if swapped is None:
        return None
    late_pos, early_pos = swapped
    paste(canvas, late_piece["in_patch"], late_pos[0], late_pos[1], mask=late_piece["in_mask"], overwrite=True)
    paste(canvas, early_piece["in_patch"], early_pos[0], early_pos[1], mask=early_piece["in_mask"], overwrite=True)

    handled = {id(early_piece), id(late_piece)}
    others = [piece for piece in pieces if id(piece) not in handled]
    for piece in sorted(others, key=lambda p: -int(p["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - 2)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)
    n_non_unit = int(n_slots - n_units)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = n_non_unit + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[:n_non_unit]
    unit_color = palette[n_non_unit]

    stage_h = random.randint(*LAYOUT_H_RANGE)
    stage_w = random.randint(*LAYOUT_W_RANGE)
    slots, layout_mask = _gen_layout(stage_h, stage_w, n_non_unit, n_units)
    if slots is None or layout_mask is None:
        return None

    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None
    non_unit_areas = {int(slot["area"]) for slot in non_unit_slots}
    if len(non_unit_areas) != 1:
        return None
    exact_non_unit = [slot["shape_key"] for slot in non_unit_slots]
    if len(exact_non_unit) != len(set(exact_non_unit)):
        return None
    canon_counts = Counter(slot["canon_key"] for slot in non_unit_slots)
    if max(canon_counts.values()) < 2:
        return None

    non_unit_sorted = sorted(non_unit_slots, key=_slot_sort_key)
    pieces = [_make_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_sorted)]
    pieces += [_make_piece(slot, unit_color, k) for slot in unit_slots]

    layout_h, layout_w = map(int, layout_mask.shape)
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        relaxed = _solve(canvas, allow_dihedral=True, use_area_filter=True)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    if np.array_equal(relaxed, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
