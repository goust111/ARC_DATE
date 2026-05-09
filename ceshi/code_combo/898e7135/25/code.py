# Combo variant of 898e7135 (index 25)
# combo_spec: D2/2+D6/3
# Name: 轮廓真伪辨别

from __future__ import annotations

from collections import Counter
from math import gcd
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


# Design analysis for the combo variant
# -------------------------------------
# combo_analysis:
#   fused_rule:
#     - The frame and inner hole-layout keep the source task's role.
#     - Every true outside piece is the 1-pixel hollow contour of a k-scaled slot shape.
#     - There are also 1-2 extra solid pseudo-pieces outside the frame; none matches any slot.
#     - The solver must first separate hollow true pieces from solid fake ones, then recover each
#       true piece's filled area and match it to a hole under D4.
#
#   concrete_pixel_form:
#     frame:
#       - Same 1-cell monochrome rectangle as the source task.
#       - Inside the frame, bg holes form 3-6 slots, 4-connected internally and 8-separated.
#     true_pieces:
#       - Built from the same block-level slot masks as before, but first scaled by k and then
#         replaced by their 1-pixel contour only.
#       - They are monochrome, disconnected, and placed outside the frame.
#       - Output still restores them as solid filled regions.
#     pseudo_pieces:
#       - Extra multi-pixel solid components, also outside the frame and disconnected.
#       - They use fresh colors and are never needed in the output.
#
#   behavior_trace:
#     small_example:
#       - Suppose the frame contains three slots: a unit cell, a domino, and an L-shape.
#       - In the source task with k=3, the outside pieces are solid 3x3 / 3x6 / 6x6-style sprites.
#       - In this combo variant, those three true sprites become hollow outlines instead, and we add
#         one solid fake polyomino and maybe one extra solid 2x2 square.
#     original_solver_failure_modes:
#       - If it treats every outside component as real, pseudo-pieces pollute the candidate pool.
#       - If it directly downsamples the hollow masks, many 3x3 blocks are only partially filled and
#         matching fails immediately.
#     combo_solver_path:
#       1. Detect the frame and extract the hole layout exactly as in the source task.
#       2. Split outside components into hollow candidates and solid leftovers by checking whether
#          filling enclosed background increases the area.
#       3. Infer k from the hollow candidates only.
#       4. Recover each hollow candidate's solid mask, downsample blockwise, and solve a one-to-one
#          slot assignment under rotation/flip.
#       5. Ignore the solid leftovers entirely when building the output.
#
#   complexity_calibration:
#     - To keep the "true = visibly hollow, fake = visibly solid" cue stable, the generator fixes
#       k=3. With k=2, unit or skinny true pieces can visually collapse back into solid shapes and
#       blur the first-stage classification rule.
#     - The original layout freedom is otherwise preserved: 3-6 slots, 1-2 unit slots, distinct
#       non-unit D4 shapes, random placements, and optional extra fake count 1-2.
#     - At least one solid fake is always a plausible multi-cell pseudo-piece; some samples also add
#       a small solid square so the outside area visibly contains multiple solid intruders.
#
#   implicit_assumption_audit_of_original_main:
#     broken_assumptions:
#       - "all non-singleton components are true pieces": false once solid pseudo-pieces are added.
#       - "input masks are already solid k-scaled pieces": false because true pieces are outlines.
#       - "smallest solid square reveals k": unsafe because fake solid squares may exist.
#       - "greedy first match over all components is enough": unsafe after the classification step.
#     repaired_logic:
#       - Only hollow components are eligible for matching.
#       - k is inferred from hollow candidates, not from all components.
#       - Matching is solved as a subset-free one-to-one assignment among hollow candidates only.
#     new_assumptions:
#       - Every true piece has strictly positive hollow gap (`filled_area > outline_area`).
#       - At least one true unit slot exists, so one hollow 3x3 square is present and the gcd of
#         hollow bbox dimensions is exactly k=3.
#       - Pseudo-pieces are solid, so filling them does not enlarge their mask.
#     generator_guarantees:
#       - It preserves 1-2 unit slots and k=3.
#       - It rejects any attempt where a true outline is not visibly hollow.
#       - It builds pseudo-pieces only as solid components and keeps their D4 shapes distinct from
#         the real slot masks whenever they are block-style decoys.
#
#   placement_legality_and_cleanliness:
#     - Hollow outlines contain background cavities, so a plain canvas occupancy test is not enough:
#       another object could otherwise land inside the cavity.
#     - The generator therefore maintains a reserve canvas that marks each object's full solid
#       occupancy, not just its visible pixels.
#     - The frame reserves its whole bounding rectangle; true outlines reserve their recovered solid
#       region; solid pseudo-pieces reserve their own solid region.
#     - All placements use 8-neighbor padding=1, keeping components cleanly separated and readable.
#
#   visual_observability_and_inference:
#     - Across examples, humans can see two outside-object classes: hollow contours that reappear as
#       filled shapes in the output, and solid extras that vanish.
#     - The classification cue is visual, not numeric: no reliance on color ids or hidden ordering.
#     - Because the output only uses filled versions of the hollow objects, the two-stage reasoning
#       "keep hollow outlines, discard solid fakes, then fill and place" is inferable from a few
#       examples.
#
#   post_change_main_assumption_check:
#     - Frame extraction assumes the frame remains the largest foreground component.
#       generate_input() keeps pseudo-pieces small and disjoint, so this remains true.
#     - Hollow detection assumes true outlines enclose interior zeros while fakes do not.
#       generate_input() enforces positive hollow gap for every true piece and uses only solid fakes.
#     - k inference assumes all hollow bbox dimensions are multiples of k and their gcd equals k.
#       generate_input() fixes k=3 and includes at least one unit slot.
#     - Downsampling assumes recovered solid masks are perfectly k-block-aligned.
#       generate_input() constructs every true piece by scaling a slot mask and only then outlining it.
#     - DFS matching assumes slot shapes are jointly coverable by the hollow candidate set.
#       generate_input() is output-first and validates `main(input) == expected_output`.


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Return unique D4 variants of a boolean mask."""
    base: list[np.ndarray] = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        base.append(r)
        base.append(np.fliplr(r))
    uniq: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for m in base:
        key = (tuple(map(int, m.shape)), m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq


def _outline_mask(filled: np.ndarray) -> np.ndarray:
    """Keep only the 1-pixel contour of a solid orthogonal shape."""
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
    """Fill enclosed cavities of a hollow contour to recover the solid shape."""
    h, w = map(int, mask.shape)
    empty = ~mask.astype(bool)
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return ~(empty & outside)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    """Convert a solid k-scaled pixel mask back to block level."""
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


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    return slots


def _match_slots(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = [[] for _ in slots]
    for pi, piece in enumerate(pieces):
        for vm in _dihedral_masks(piece["mask"]):
            filled = _recover_filled(vm)
            if int(filled.sum()) == int(vm.sum()):
                continue
            um = _downsample_blocks(filled, k)
            if um is None:
                continue
            area = int(filled.sum())
            vp = np.where(filled, int(piece["color"]), 0).astype(int)
            for si, slot in enumerate(slots):
                if area != int(slot["area"]) * int(k) * int(k):
                    continue
                if um.shape != slot["mask"].shape:
                    continue
                if np.array_equal(um, slot["mask"]):
                    candidates_by_slot[si].append(
                        {
                            "pi": int(pi),
                            "vp": vp,
                            "vm": filled,
                        }
                    )

    if any(not cands for cands in candidates_by_slot):
        return None

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

    if not dfs(0):
        return None
    return [(slots[i], chosen[i]) for i in range(len(slots)) if chosen[i] is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a framed hole-layout and refill it with the genuine outside sprites.

    Input structure:
    - Background is the most common color. The largest foreground component is a monochrome frame
      enclosing a bg hole-layout.
    - Outside the frame are two visible classes of components: true sprites are hollow 1-pixel
      contours of k-scaled slot shapes, while 1-2 extra pseudo-sprites are solid multi-pixel decoys.

    Rule:
    - Extract the frame's hole regions as block-level slots.
    - Keep only hollow outside components, recover each one to its filled solid form, and compare the
      recovered block-level shape to the slots under rotation/flip.
    - Ignore solid pseudo-sprites and output the cropped scaled frame filled with the matched solid
      pieces, using the frame color as background.

    Key insight:
    - The decisive cue is not color but topology: genuine pieces gain area when their hollow contour
      is filled, whereas pseudo-pieces do not. After filtering by that cue, hole filling plus
      k-block downsampling restores the original matching problem.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected foreground components"
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

    slots = _extract_slots(layout)
    assert slots, "frame must contain at least one slot"

    hollow_pieces: list[dict] = []
    solid_pieces: list[dict] = []
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
        color = int(Counter(map(int, patch[pm].tolist())).most_common(1)[0][0])
        filled = _recover_filled(pm)
        piece = {"mask": pm, "color": color, "filled_area": int(filled.sum()), "area": int(pm.sum())}
        if int(filled.sum()) > int(pm.sum()):
            hollow_pieces.append(piece)
        else:
            solid_pieces.append(piece)

    assert hollow_pieces, "expected hollow true pieces"
    assert solid_pieces, "expected at least one solid pseudo-piece"

    dims = [int(v) for p in hollow_pieces for v in p["mask"].shape]
    k = dims[0]
    for v in dims[1:]:
        k = gcd(int(k), int(v))
    assert k >= 2, "invalid inferred scale"

    placements = _match_slots(slots, hollow_pieces, int(k))
    assert placements is not None and len(placements) == len(slots), "every slot must match a hollow piece"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

K_CHOICES = [3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DECOY_COUNT_RANGE = (1, 2)
GENERATE_MAX_RETRIES = 1200


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
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


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(int(n_units)):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for _ in range(int(n_slots - n_units)):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            sm = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_true_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    solid_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    in_solid = solid_mask.copy()
    if apply_dihedral:
        rot = random.randint(0, 3)
        in_solid = np.rot90(in_solid, rot)
        if random.random() < 0.5:
            in_solid = np.fliplr(in_solid)
    outline = _outline_mask(in_solid)
    hollow_gap = int(in_solid.sum()) - int(outline.sum())
    out_patch = np.where(solid_mask, int(color), 0).astype(int)
    in_patch = np.where(outline, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": solid_mask,
        "in_patch": in_patch,
        "in_mask": outline,
        "reserve_mask": in_solid,
        "hollow_gap": int(hollow_gap),
    }


def _scaled_solid_piece(base_mask: np.ndarray, color: int, k: int) -> dict:
    solid_mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
    rot = random.randint(0, 3)
    solid_mask = np.rot90(solid_mask, rot)
    if random.random() < 0.5:
        solid_mask = np.fliplr(solid_mask)
    patch = np.where(solid_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "in_patch": patch,
        "in_mask": solid_mask,
        "reserve_mask": solid_mask,
    }


def _make_small_square_decoy(color: int) -> dict:
    mask = np.ones((2, 2), dtype=bool)
    patch = np.full((2, 2), int(color), dtype=int)
    return {
        "color": int(color),
        "in_patch": patch,
        "in_mask": mask,
        "reserve_mask": mask,
    }


def _random_decoy_base_mask(forbidden_shape_keys: set[bytes]) -> np.ndarray | None:
    for _ in range(80):
        size = random.randint(2, 6)
        coords = _grow_slot(np.zeros((5, 5), dtype=bool), size)
        if coords is None:
            continue
        mask = _tight_mask_from_coords(coords)
        key = _canonical_shape_key(mask)
        if key in forbidden_shape_keys:
            continue
        return mask
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
    reserve = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    fr = random.randint(1, fr_max)
    fc = random.randint(1, fc_max)
    paste(canvas, frame_img, fr, fc)
    paste(reserve, np.ones((frame_h, frame_w), dtype=int), fr, fc)

    for obj in sorted(outside_objects, key=lambda item: -int(item["reserve_mask"].sum())):
        try:
            pr, pc = _random_reserved_location(reserve, obj["reserve_mask"])
        except ValueError:
            return None
        paste(canvas, obj["in_patch"], pr, pc, mask=obj["in_mask"], overwrite=True)
        paste(reserve, np.ones_like(obj["reserve_mask"], dtype=int), pr, pc, mask=obj["reserve_mask"], overwrite=True)

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    n_decoys = random.randint(*DECOY_COUNT_RANGE)

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

    pieces = [_make_true_piece(slot, non_unit_colors[i], k, apply_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_true_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]
    if not pieces or not all(int(piece["hollow_gap"]) > 0 for piece in pieces):
        return None

    forbidden_shape_keys = {_canonical_shape_key(slot["mask"]) for slot in slots}
    decoys: list[dict] = []
    block_decoy_mask = _random_decoy_base_mask(forbidden_shape_keys)
    if block_decoy_mask is None:
        return None
    decoys.append(_scaled_solid_piece(block_decoy_mask, decoy_colors[0], k))
    if n_decoys == 2:
        decoys.append(_make_small_square_decoy(decoy_colors[1]))

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, decoys, layout_mask, layout_h, layout_w, frame_color)
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
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
