from __future__ import annotations

# Combo variant of 898e7135 (index 15)
# combo_spec: D6/3+D7/2
# Name: 真假精灵双重筛选
#
# Design analysis for the combo variant
# -------------------------------------
# concretization_and_behavior_trace:
#   input_visual:
#     - The frame remains the largest foreground component: a 1-cell-thick
#       monochrome rectangle around a hole/wall layout.
#     - The correct pieces are unchanged in kind: k_true-scaled monochrome block
#       sprites whose downsampled masks equal the hole shapes up to D4 symmetry.
#     - New outside inventory now mixes three object families:
#         1. true solution pieces at k_true=3;
#         2. spare normal sprites at k_extra=2, so there are more regular-looking
#            sprites than holes even after obvious garbage is ignored;
#         3. 1-2 pseudo-pieces in unique colors, built as multi-pixel objects
#            whose block-level shapes do not match any hole.
#   small_trace:
#     - Suppose the frame contains slots {L area 3, domino area 2, unit area 1}.
#     - True pieces outside are {3x L, 3x domino, 3x3 unit square}.
#     - Add spare normal sprites {2x2 unit square, 2x domino}; these are still
#       clean block sprites, so they look like plausible pieces but belong to the
#       wrong common scale.
#     - Add one pseudo-piece {3x zig-zag} in a color not used by any real/spare
#       sprite; it is block-aligned but matches no hole shape.
#     - Old solver path:
#         * takes every non-singleton component as a piece;
#         * infers k from the smallest solid square, so it picks 2;
#         * cannot complete all slot matches and fails.
#     - New solver path:
#         * still collects every multi-pixel component;
#         * tries each square-derived k candidate and searches for an injective
#           slot assignment using only a subset of components;
#         * at k=2 the spare sprites are too few to cover all slots;
#         * at k=3 all slots are covered by the true pieces, while the pseudo
#           shapes remain unmatched and are discarded.
#   visible_difference:
#     - A human sees both kinds of extras: some outside sprites are simply too
#       many but still well-formed, and some are unique-color shapes with no hole
#       counterpart at all. The output keeps only the consistent k_true family.
#
# complexity_calibration:
#   - Fix k_true=3 and k_extra=2. This keeps the second filter legible and avoids
#     uncontrolled multi-scale ambiguity.
#   - Keep pseudo-piece count at 1-2 and spare normal sprite count at 1-2. The
#     scene changes are visible without turning the canvas into clutter.
#   - Non-unit hole shapes remain D4-distinct, so matching is structural rather
#     than relying on arbitrary ordering.
#   - Spare normal sprites reuse real piece colors, while pseudo-pieces get fresh
#     colors. This avoids introducing a separate color-mapping puzzle while still
#     making the fake family visually inspectable across examples.
#
# implicit_assumptions_audit_of_original_main:
#   - "all non-singleton foreground components are true pieces": broken by both
#     spare sprites and pseudo-pieces.
#   - "smallest solid square reveals k": broken because a 2x2 spare unit square
#     is present while the real scale is 3.
#   - "there are exactly as many usable pieces as slots": broken because some
#     components are pseudo-pieces and some are leftover regular sprites.
#   - "greedy first-fit matching is enough": broken because we now need a subset
#     assignment under each k candidate, not a linear consume-once scan.
#   - "mask bytes alone identify D4 variants": unsafe once 1x2 and 2x1 or other
#     rectangular masks coexist; shape dimensions must be part of the key.
#
# placement_legality:
#   - The generator still uses `random_free_location(..., padding=1,
#     padding_connectivity=8)`, so the frame, true pieces, spare sprites, and
#     pseudo-pieces never touch or overlap.
#   - The frame remains the largest component because all outside objects are kept
#     small relative to the border rectangle.
#   - The hole layout is forced to touch all four inner borders, so tight-cropping
#     inside the frame still reconstructs the intended layout.
#
# visual_observability_and_cleanliness:
#   - The fusion is observable from 2-4 examples because the same two effects
#     repeat: unique-color fake shapes vanish, and among the remaining regular
#     sprites only one common block scale consistently explains the filled frame.
#   - No object is layered, no touching components create parsing ambiguity, and
#     pseudo-pieces live in the same outside-sprite region instead of adding a
#     separate annotation system.
#
# post-change_solver_assumptions_and_generator_guarantees:
#   - main() assumes there is at least one true unit slot, so a real 3x3 square
#     exists and contributes k=3 as a candidate. The generator enforces 1-2 unit
#     slots.
#   - main() assumes exactly one square-derived k yields a full assignment. The
#     generator guarantees this by providing only 1-2 spare k=2 sprites, fewer
#     than the number of slots, while every true slot still has its k=3 piece.
#   - main() assumes pseudo-pieces never match a real hole at k=3. The generator
#     builds pseudo base masks D4-distinct from all slot masks.

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


TRUE_K = 3
EXTRA_K = 2

LAYOUT_H_RANGE = (4, 8)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
N_SPARE_RANGE = (1, 2)
N_PSEUDO_RANGE = (1, 2)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
EXTRA_SHAPE_RETRIES = 80
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 5
AREA_BUFFER_CONST = 65

GENERATE_MAX_RETRIES = 500

ShapeKey = tuple[tuple[int, int], bytes]


def _canonical_shape_key(mask: np.ndarray) -> ShapeKey:
    variants: list[ShapeKey] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((tuple(map(int, flip.shape)), flip.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
    out = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    out[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
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
    height, width = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    starts = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(starts)
    for r0, c0 in starts[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < height and 0 <= p[1] < width and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        base.append((rot_patch, rot_mask))
        base.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[ShapeKey] = set()
    for patch_variant, mask_variant in base:
        key = (tuple(map(int, mask_variant.shape)), mask_variant.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((patch_variant, mask_variant))
    return uniq


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
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


def _match_slots_with_k(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = [[] for _ in slots]
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            scaled_area = int(vm.sum())
            for si, slot in enumerate(slots):
                if scaled_area != int(slot["area"]) * int(k) * int(k):
                    continue
                if um.shape != slot["mask"].shape:
                    continue
                if np.array_equal(um, slot["mask"]):
                    candidates_by_slot[si].append(
                        {
                            "pi": int(pi),
                            "vp": vp,
                            "vm": vm,
                        }
                    )

    if any(not slot_cands for slot_cands in candidates_by_slot):
        return None

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(candidates_by_slot[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
    )
    chosen: list[dict | None] = [None] * len(slots)
    used_pieces: set[int] = set()

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        slot_index = order[depth]
        for cand in candidates_by_slot[slot_index]:
            pi = int(cand["pi"])
            if pi in used_pieces:
                continue
            used_pieces.add(pi)
            chosen[slot_index] = cand
            if dfs(depth + 1):
                return True
            chosen[slot_index] = None
            used_pieces.remove(pi)
        return False

    if not dfs(0):
        return None
    return [(slots[i], chosen[i]) for i in range(len(slots)) if chosen[i] is not None]


def _infer_k_and_match(slots: list[dict], pieces: list[dict]) -> tuple[int, list[tuple[dict, dict]]]:
    square_sizes = sorted(
        {
            int(piece["mask"].shape[0])
            for piece in pieces
            if piece["mask"].shape[0] == piece["mask"].shape[1]
            and int(piece["mask"].sum()) == int(piece["mask"].shape[0] * piece["mask"].shape[1])
            and int(piece["mask"].shape[0]) > 1
        }
    )
    assert square_sizes, "need at least one solid square to infer k candidates"

    best: tuple[int, list[tuple[dict, dict]]] | None = None
    for k in square_sizes:
        placements = _match_slots_with_k(slots, pieces, int(k))
        if placements is None:
            continue
        if best is not None:
            raise AssertionError("ambiguous scale: more than one k explains all slots")
        best = (int(k), placements)
    assert best is not None, "no square-derived k yields a full assignment"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the frame layout and paste only the subset of sprites that truly solves it.

    Input structure:
    - Background is the most common color. The largest foreground component is a
      monochrome frame enclosing bg holes and same-color walls.
    - Outside the frame are three kinds of multi-pixel components: true pieces at
      one common scale, spare regular sprites at another scale, and 1-2 unique-color
      pseudo-pieces whose shapes fit no hole.

    Rule:
    - Extract the hole components inside the frame as slot masks.
    - Consider every outside multi-pixel component as a candidate, try each scale
      suggested by solid-square sprites, and find a one-to-one assignment from some
      subset of components to all slots under rotation/flip.
    - Use the only scale that covers every slot; unmatched regular sprites and
      pseudo-pieces are discarded.

    Key insight:
    - The correct answer is the unique global explanation in which all slots are
      covered by one common block scale. Fake sprites fail by shape, while spare
      normal sprites fail because they belong to the wrong scale family or remain
      unused after the full assignment is found.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_mask = sub == bg
    hole_box = bbox(hole_mask)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

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
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "patch": patch, "mask": mask})
    assert pieces, "at least one non-frame multi-pixel piece must exist"

    k, placements = _infer_k_and_match(slots, pieces)

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[ShapeKey] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            mask = np.ones((1, 1), dtype=bool)
            slots.append({"coords": coords, "area": 1, "mask": mask, "shape_key": _canonical_shape_key(mask)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for _ in range(n_slots - n_units):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            mask = _tight_mask_from_coords(coords)
            shape_key = _canonical_shape_key(mask)
            if shape_key in used_shape_keys:
                continue
            used_shape_keys.add(shape_key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": shape_key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece_from_mask(cell_mask: np.ndarray, color: int, scale: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(cell_mask, np.ones((scale, scale), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    if apply_dihedral and int(cell_mask.sum()) > 1:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": int(color),
        "area": int(out_mask.sum()),
        "cell_mask": cell_mask,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _random_nonmatching_mask(target_sizes: list[int], forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(EXTRA_SHAPE_RETRIES):
        size = random.choice(target_sizes)
        hh = random.randint(2, min(5, size + 2))
        ww = random.randint(2, min(5, size + 2))
        if hh * ww < size:
            continue
        coords = _grow_slot(np.zeros((hh, ww), dtype=bool), size)
        if coords is None:
            continue
        mask = _tight_mask_from_coords(coords)
        shape_key = _canonical_shape_key(mask)
        if shape_key in forbidden_keys:
            continue
        return mask
    return None


def _build_expected_output(pieces: list[dict], slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * TRUE_K, (layout_w + 2) * TRUE_K), int(frame_color), dtype=int)
    for piece, slot in zip(pieces, slots):
        coords = slot["coords"]
        top = (int(coords[:, 0].min()) + 1) * TRUE_K
        left = (int(coords[:, 1].min()) + 1) * TRUE_K
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    all_objects: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_area = sum(int(obj["in_mask"].sum()) for obj in all_objects)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_area + AREA_BUFFER_CONST

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
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for obj in sorted(all_objects, key=lambda item: -int(item["in_mask"].sum())):
        try:
            rr, cc = random_free_location(
                canvas,
                obj["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, obj["in_patch"], rr, cc, mask=obj["in_mask"], overwrite=True)

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt(return_meta: bool = False):
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    n_spare = random.randint(*N_SPARE_RANGE)
    n_pseudo = random.randint(*N_PSEUDO_RANGE)

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    needed_piece_colors = (n_slots - n_units) + 1
    needed_colors = 1 + needed_piece_colors + n_pseudo
    palette = list(range(1, 10))
    random.shuffle(palette)
    if len(palette) < needed_colors:
        return None

    frame_color = palette.pop()
    piece_palette = [palette.pop() for _ in range(needed_piece_colors)]
    pseudo_colors = [palette.pop() for _ in range(n_pseudo)]
    non_unit_colors = piece_palette[: n_slots - n_units]
    unit_color = piece_palette[-1]

    true_pieces = [
        _make_piece_from_mask(slot["mask"], non_unit_colors[i], TRUE_K, apply_dihedral=True)
        for i, slot in enumerate(non_unit_slots)
    ]
    true_pieces += [_make_piece_from_mask(slot["mask"], unit_color, TRUE_K, apply_dihedral=False) for slot in unit_slots]

    real_piece_colors = [piece["color"] for piece in true_pieces]
    slot_keys = {slot["shape_key"] for slot in slots}
    non_unit_slot_keys = {slot["shape_key"] for slot in non_unit_slots}
    non_unit_areas = [int(slot["area"]) for slot in non_unit_slots] or [2]

    spare_masks: list[np.ndarray] = [np.ones((1, 1), dtype=bool)]
    while len(spare_masks) < n_spare:
        if non_unit_slots and random.random() < 0.7:
            spare_masks.append(random.choice(non_unit_slots)["mask"].copy())
        else:
            extra_mask = _random_nonmatching_mask(non_unit_areas, slot_keys)
            if extra_mask is None:
                return None
            spare_masks.append(extra_mask)
    spare_pieces = [
        _make_piece_from_mask(mask, random.choice(real_piece_colors), EXTRA_K, apply_dihedral=(int(mask.sum()) > 1))
        for mask in spare_masks
    ]

    forbidden_pseudo_keys = set(slot_keys) | { _canonical_shape_key(mask) for mask in spare_masks if int(mask.sum()) > 1 or mask.shape != (1, 1) }
    pseudo_pieces: list[dict] = []
    for idx in range(n_pseudo):
        pseudo_mask = _random_nonmatching_mask(non_unit_areas, forbidden_pseudo_keys | non_unit_slot_keys)
        if pseudo_mask is None:
            return None
        forbidden_pseudo_keys.add(_canonical_shape_key(pseudo_mask))
        pseudo_pieces.append(
            _make_piece_from_mask(pseudo_mask, pseudo_colors[idx], TRUE_K, apply_dihedral=True)
        )

    expected_out = _build_expected_output(true_pieces, non_unit_slots + unit_slots, layout_h, layout_w, frame_color)
    all_objects = true_pieces + spare_pieces + pseudo_pieces
    canvas = _build_input_canvas(all_objects, layout_mask, layout_h, layout_w, frame_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    if not return_meta:
        return canvas
    return {
        "input": canvas,
        "output": expected_out,
        "slot_count": int(n_slots),
        "true_count": int(len(true_pieces)),
        "spare_count": int(len(spare_pieces)),
        "pseudo_count": int(len(pseudo_pieces)),
        "normal_count": int(len(true_pieces) + len(spare_pieces)),
        "has_double_filter": bool(len(spare_pieces) > 0 and len(pseudo_pieces) > 0),
        "spare_minus_slots": int(len(spare_pieces)),
        "normal_minus_slots": int(len(true_pieces) + len(spare_pieces) - n_slots),
    }


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt(return_meta=False)
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")


def generate_sample_with_meta() -> dict:
    for _ in range(GENERATE_MAX_RETRIES):
        sample = _gen_attempt(return_meta=True)
        if sample is not None:
            return sample
    raise RuntimeError(f"generate_sample_with_meta: failed after {GENERATE_MAX_RETRIES} attempts")


def sample_combo_metrics(trials: int = 100) -> dict:
    stats = {
        "accepted": 0,
        "double_filter_cases": 0,
        "pseudo_count_hist": {},
        "spare_count_hist": {},
        "normal_minus_slots_hist": {},
    }
    for _ in range(trials):
        sample = generate_sample_with_meta()
        stats["accepted"] += 1
        if sample["has_double_filter"]:
            stats["double_filter_cases"] += 1
        for key_name, hist_name in [
            ("pseudo_count", "pseudo_count_hist"),
            ("spare_count", "spare_count_hist"),
            ("normal_minus_slots", "normal_minus_slots_hist"),
        ]:
            val = int(sample[key_name])
            stats[hist_name][val] = int(stats[hist_name].get(val, 0)) + 1
    return stats
