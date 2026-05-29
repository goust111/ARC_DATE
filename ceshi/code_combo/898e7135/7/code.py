# Combo variant of 898e7135 (index 7)
# combo_spec: D1/1+D4/3
# Name: 旋转-缩放复合变换

from __future__ import annotations

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


"""
Design analysis for combo D1/1 + D4/3
=====================================

spec_variant:
  preserved_structure:
    grid: still 15-30 by 15-30 with bg=0 as the most common color
    frame: still the largest monochrome foreground component, enclosing a hole layout
    slots: still 3-6 hole components, 8-separated, with 1-2 unit slots and 4-12 x 4-8 layout bounds
    k: still inferred from the scattered pieces and still limited to {2, 3}
    distractors: still isolated singleton noise in a dedicated color

  fused_change:
    D1/1:
      matching is restricted to rotations only; mirror images are no longer interchangeable
    D4/3:
      the scattered piece is not a rotated pixel sprite anymore
      instead, its block-level mask is rotated first, then each occupied block is expanded with a fixed
      asymmetric kxk micro-tile
      the output is rebuilt from the matched slot mask by applying the same asymmetric micro-tile after
      rotating the abstract mask back into slot orientation

  concrete_behavior_trace:
    small_example:
      let k=3 and let one occupied abstract cell expand to
        0 c c
        c c c
        c c c
      where 0 is background and c is the piece color
      suppose a slot is an L-shape:
        1 0
        1 1
      and its input piece uses the abstract rotation
        1 1
        1 0
      original_task_style:
        scale first, then rotate the full pixel sprite to align with the slot
        the missing corner inside every 3x3 block rotates together with the piece
      fused_variant:
        rotate only the 2x2 abstract mask back to slot orientation, then scale with the same fixed micro-tile
        the global L silhouette matches the slot, but every occupied 3x3 block keeps the missing corner at the
        micro-tile's canonical top-left position
      visible_difference:
        on the same input, old-order rendering and new-order rendering disagree pixelwise whenever a piece uses
        a non-zero abstract rotation

  complexity_calibration:
    why generator changes are necessary:
      with solid monochrome kxk blocks, rotate-then-scale and scale-then-rotate commute, so D4/3 would be invisible
    chosen fix:
      every occupied block uses one fixed asymmetric micro-tile with a single bg notch
      this keeps the task readable, but makes operation order observable
    ambiguity control:
      only one micro-tile type is used across the whole puzzle, so humans see a single structural rule
      non-unit rotation classes remain unique, while one forced mirror-pair makes the "no flips" restriction visible

  implicit_assumption_audit_for_main:
    - largest connected foreground component is the frame:
      generator enforces frame foreground area > every piece foreground area and keeps 8-neighbor spacing
    - hole bbox can be tight-cropped:
      generator rejects layouts whose holes do not touch all four borders of the layout crop
    - k can be inferred globally:
      all scattered pieces have bbox sizes divisible by k, and unit slots guarantee only one of {2, 3} fits all
    - every piece decomposes into canonical kxk micro-tiles:
      generator only creates pieces from that single template, so downsampling is deterministic
    - slot matching can be greedy after sorting by area:
      non-unit rotation classes are unique; only unit slots are duplicated, and those are interchangeable
    - flips are not needed:
      generator injects at least one chiral mirror-pair, so forbidding flips changes actual pairings

  placement_legality:
    all slots are placed with 8-neighbor exclusion inside the layout
    scattered pieces are packed with random_free_location(..., padding=1, padding_connectivity=8)
    distractors are isolated singletons and never touch meaningful objects
    every attempt is rejected if bounds, density, background-majority, or frame-dominance constraints fail

  visual_observability:
    the D1/1 signal is a mirror pair of hole silhouettes that are not rotation-equivalent
    the D4/3 signal is the repeated asymmetric notch inside every expanded block
    because the same notch motif appears on every piece, a human can compare input and output and infer that
    the piece is matched at block scale and then re-expanded, rather than simply pixel-rotated as a whole

  cleanliness:
    there is no overlaying or color remapping beyond the original task's piece placement
    the frame, holes, pieces, and distractors remain visually separated and easy to parse
"""


RotKey = tuple[tuple[int, int], bytes]

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
CHIRAL_PAIR_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 70
LAYOUT_RETRIES = 140

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    """Canonical key for a tight boolean mask under rotations only."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, r.shape)), r.tobytes()))
    return min(variants)


def _rotation_orbit_size(mask: np.ndarray) -> int:
    seen: set[RotKey] = set()
    for kk in range(4):
        r = np.rot90(mask, int(kk))
        seen.add((tuple(map(int, r.shape)), r.tobytes()))
    return int(len(seen))


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


def _block_mask(k: int) -> np.ndarray:
    """Canonical asymmetric kxk tile used for every occupied abstract cell."""
    out = np.ones((int(k), int(k)), dtype=bool)
    out[0, 0] = False
    return out


def _render_scaled(mask: np.ndarray, color: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Scale an abstract mask with the fixed asymmetric micro-tile."""
    h, w = map(int, mask.shape)
    tile = _block_mask(int(k))
    out_mask = np.zeros((h * int(k), w * int(k)), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            rr = r * int(k)
            cc = c * int(k)
            out_mask[rr : rr + int(k), cc : cc + int(k)] = tile
    out = np.zeros_like(out_mask, dtype=int)
    out[out_mask] = int(color)
    return out, out_mask


def _downsample_piece_mask(mask: np.ndarray, k: int) -> np.ndarray | None:
    """Recover the abstract block mask from a scattered piece."""
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    tile = _block_mask(int(k))
    out = np.zeros((hh // int(k), ww // int(k)), dtype=bool)
    for r in range(hh // int(k)):
        for c in range(ww // int(k)):
            block = mask[r * int(k) : (r + 1) * int(k), c * int(k) : (c + 1) * int(k)]
            if np.array_equal(block, tile):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out if bool(out.any()) else None


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _slot_from_coords(coords: np.ndarray) -> dict:
    mask = _tight_mask_from_coords(coords)
    r0, c0, _r1, _c1 = map(int, bbox_coords(coords))
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "rot_key": _rotation_shape_key(mask),
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


def _random_shape_mask(
    target_size: int,
    *,
    require_chiral: bool,
    require_nontrivial_rotation: bool,
    used_rot_keys: set[RotKey] | None = None,
) -> np.ndarray | None:
    """Grow a connected tight mask of target_size that satisfies the requested symmetry constraints."""
    used_rot_keys = used_rot_keys or set()
    max_side = min(5, max(2, int(target_size)))

    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(2, max_side)
        box_w = random.randint(2, max_side)
        if box_h * box_w < int(target_size):
            continue

        start = (random.randint(0, box_h - 1), random.randint(0, box_w - 1))
        chosen = {start}
        frontier = {
            (start[0] - 1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1),
        }

        while len(chosen) < int(target_size):
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < box_h and 0 <= p[1] < box_w and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

        if len(chosen) != int(target_size):
            continue

        coords = np.array(sorted(chosen), dtype=int)
        mask = _tight_mask_from_coords(coords)
        key = _rotation_shape_key(mask)
        if key in used_rot_keys:
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        if require_nontrivial_rotation and _rotation_orbit_size(mask) == 1:
            continue
        return mask

    return None


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Place a tight mask into the layout without 8-neighbor contact."""
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None

    base_coords = np.argwhere(mask)
    candidates: list[tuple[int, int]] = []
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            coords = base_coords + np.array([int(top), int(left)], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(top), int(left)))

    if not candidates:
        return None
    top, left = random.choice(candidates)
    coords = base_coords + np.array([int(top), int(left)], dtype=int)
    _forbid_coords(forbidden, coords)
    return coords


def _layout_touches_all_borders(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _count_mirror_pairs(non_unit_slots: list[dict]) -> int:
    count = 0
    for i in range(len(non_unit_slots)):
        key_i = non_unit_slots[i]["rot_key"]
        mirror_i = _rotation_shape_key(np.fliplr(non_unit_slots[i]["mask"]))
        if mirror_i == key_i:
            continue
        for j in range(i + 1, len(non_unit_slots)):
            if non_unit_slots[j]["rot_key"] == mirror_i:
                count += 1
    return int(count)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> tuple[list[dict], list[dict], np.ndarray] | None:
    """
    Generate layout slots with:
    - at least one chiral mirror-pair among non-unit slots
    - at least one non-unit slot with a non-trivial rotation orbit
    """
    n_non_unit = int(n_slots - n_units)
    if n_non_unit < 2:
        return None

    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        used_rot_keys: set[RotKey] = set()
        non_unit_slots: list[dict] = []
        unit_slots: list[dict] = []

        pair_size = random.choice(CHIRAL_PAIR_SIZE_CHOICES)
        base_mask = _random_shape_mask(
            pair_size,
            require_chiral=True,
            require_nontrivial_rotation=True,
            used_rot_keys=used_rot_keys,
        )
        if base_mask is None:
            continue
        mirror_mask = np.fliplr(base_mask)
        if _rotation_shape_key(base_mask) == _rotation_shape_key(mirror_mask):
            continue

        pair_masks = [base_mask, mirror_mask]
        random.shuffle(pair_masks)
        ok = True
        for mask in pair_masks:
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            used_rot_keys.add(slot["rot_key"])
            non_unit_slots.append(slot)
        if not ok:
            continue

        rem_sizes = [random.choice(NON_UNIT_SIZE_CHOICES) for _ in range(n_non_unit - 2)]
        rem_sizes.sort(reverse=True)
        for size in rem_sizes:
            mask = _random_shape_mask(
                size,
                require_chiral=False,
                require_nontrivial_rotation=False,
                used_rot_keys=used_rot_keys,
            )
            if mask is None:
                ok = False
                break
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            used_rot_keys.add(slot["rot_key"])
            non_unit_slots.append(slot)
        if not ok:
            continue

        for _ in range(n_units):
            coords = _place_mask_randomly(forbidden, np.ones((1, 1), dtype=bool))
            if coords is None:
                ok = False
                break
            unit_slots.append(_slot_from_coords(coords))
        if not ok:
            continue

        layout_mask = np.zeros((h, w), dtype=bool)
        for slot in unit_slots + non_unit_slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

        density = float(layout_mask.sum()) / float(h * w)
        lo, hi = HOLE_DENSITY_RANGE
        if not (lo <= density <= hi):
            continue
        if not _layout_touches_all_borders(layout_mask):
            continue
        if _count_mirror_pairs(non_unit_slots) < 1:
            continue
        if not any(_rotation_orbit_size(slot["mask"]) > 1 for slot in non_unit_slots):
            continue

        return unit_slots, non_unit_slots, layout_mask

    return None


def _make_piece(slot: dict, color: int, k: int, rot_steps: int) -> dict:
    """Build input/output sprites from the slot mask using abstract rotation before scaling."""
    in_mask_abstract = np.rot90(slot["mask"], int(rot_steps))
    in_patch, in_mask = _render_scaled(in_mask_abstract, int(color), int(k))
    out_patch, out_mask = _render_scaled(slot["mask"], int(color), int(k))
    return {
        "color": int(color),
        "slot": slot,
        "rot_steps": int(rot_steps),
        "in_abstract": in_mask_abstract,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "out_patch": out_patch,
        "out_mask": out_mask,
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((int(layout_h) + 2) * int(k), (int(layout_w) + 2) * int(k)), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], int(top), int(left), mask=p["out_mask"], overwrite=True)
    return out


def _build_old_order_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    """Reference rendering for degeneration checks: rotate the scaled input patch as a whole."""
    out = np.full(((int(layout_h) + 2) * int(k), (int(layout_w) + 2) * int(k)), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        back_rot = (-int(p["rot_steps"])) % 4
        old_patch = np.rot90(p["in_patch"], int(back_rot))
        old_mask = np.rot90(p["in_mask"], int(back_rot))
        paste(out, old_patch, int(top), int(left), mask=old_mask, overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
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

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
        try:
            loc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        if loc is None:
            return None
        pr, pc = map(int, loc)
        paste(canvas, p["in_patch"], int(pr), int(pc), mask=p["in_mask"], overwrite=True)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and refill it with the scattered pieces.

    Input structure:
    - The most common color is background. The largest foreground component is a monochrome frame around bg holes.
    - Each outside piece is built from a block-level shape expanded by a fixed asymmetric kxk micro-tile; k is 2 or 3.

    Rule:
    - Extract the hole components inside the frame as slots.
    - Downsample each scattered piece back to its block mask, then match that mask to a slot using rotations only.
    - Rebuild the matched slot by scaling the slot-oriented block mask with the same asymmetric micro-tile and place it into the enlarged frame.

    Key insight:
    - The scattered piece does not need a pixel-level rotation in the output. The correct operation is: infer the abstract block mask, rotate that abstract mask to the slot, then scale it again. The asymmetric micro-tile makes this order visible.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
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
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        colors = sorted({int(v) for v in patch[pm]})
        assert len(colors) == 1, "each piece must be monochrome aside from the shared bg notches"
        pieces.append({"mask": pm, "color": int(colors[0])})
    assert pieces, "at least one non-frame piece must exist"

    k_candidates = [
        cand
        for cand in K_CHOICES
        if all(int(p["mask"].shape[0]) % cand == 0 and int(p["mask"].shape[1]) % cand == 0 for p in pieces)
    ]
    assert len(k_candidates) == 1, "piece dimensions must reveal a unique k"
    k = int(k_candidates[0])

    for p in pieces:
        abstract = _downsample_piece_mask(p["mask"], int(k))
        assert abstract is not None, "every piece must decode into canonical micro-tiles"
        p["abstract_mask"] = abstract
        p["area"] = int(abstract.sum())

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        found = None
        for pi, piece in enumerate(pieces):
            if int(pi) in used_piece or int(piece["area"]) != int(slot["area"]):
                continue
            for kk in range(4):
                rot = np.rot90(piece["abstract_mask"], int(kk))
                if rot.shape == slot["mask"].shape and np.array_equal(rot, slot["mask"]):
                    found = {"pi": int(pi), "piece": piece}
                    break
            if found is not None:
                break
        assert found is not None, "every slot must match one piece under rotations only"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found["piece"]))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, piece in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        patch, mask = _render_scaled(slot["mask"], int(piece["color"]), int(k))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    layout_res = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if layout_res is None:
        return None
    unit_slots, non_unit_slots, layout_mask = layout_res

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(max([slot["area"] * (k * k - 1) for slot in non_unit_slots + unit_slots], default=0))
    if frame_area <= max_piece_area:
        return None

    forced_visible_idx = None
    candidates = [i for i, slot in enumerate(non_unit_slots) if _rotation_orbit_size(slot["mask"]) > 1]
    if candidates:
        forced_visible_idx = int(random.choice(candidates))

    pieces: list[dict] = []
    visible_rotation_count = 0
    for i, slot in enumerate(non_unit_slots):
        if forced_visible_idx is not None and int(i) == int(forced_visible_idx):
            valid_rots = [r for r in (1, 2, 3) if not np.array_equal(np.rot90(slot["mask"], int(r)), slot["mask"])]
            if not valid_rots:
                return None
            rot_steps = int(random.choice(valid_rots))
        else:
            rot_steps = random.randint(0, 3)
        if rot_steps != 0 and not np.array_equal(np.rot90(slot["mask"], int(rot_steps)), slot["mask"]):
            visible_rotation_count += 1
        pieces.append(_make_piece(slot, non_unit_colors[i], k, rot_steps))

    for slot in unit_slots:
        pieces.append(_make_piece(slot, unit_color, k, 0))

    if visible_rotation_count < 1:
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    old_order_out = _build_old_order_output(pieces, layout_h, layout_w, k, frame_color)
    if np.array_equal(expected_out, old_order_out):
        return None

    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
