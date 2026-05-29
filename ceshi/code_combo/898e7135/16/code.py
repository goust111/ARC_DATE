# Combo variant of 898e7135 (index 16)
# combo_spec: D6/2+D10/0
# Name: 单色混淆干扰

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


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   - Start from the original frame-and-slots task.
#   - Apply D10/0: every true piece uses one shared piece color.
#   - Apply D6/2 on top of that: singleton distractors also use that same shared
#     piece color instead of any dedicated distractor color.
#   - Visible result: outside the frame there are only two semantic foreground
#     roles left for the solver to separate:
#       1. multi-pixel k-by-k-block pieces
#       2. isolated 1-pixel noise
#     but both roles now look identical by color.
#
# concrete_trace:
#   - Suppose k = 2 and the frame contains three slots: an L triomino, a T
#     tetromino, and one unit slot.
#   - Original task:
#       * L piece might be color 3, T piece color 5, unit piece color 7.
#       * Distractors might be color 8.
#       * A solver could partly reason by color exclusivity.
#   - After D10/0 only:
#       * all three pieces become color 3, but distractors still color 8.
#       * shape still matters, yet singleton noise is color-filterable.
#   - After D6/2 + D10/0:
#       * all three pieces are color 3 and every distractor singleton is also
#         color 3.
#       * The only stable distinction is connectivity and size:
#           - isolated area-1 components are removable distractors
#           - area>1 components are candidate pieces
#       * Matching from pieces to slots remains purely geometric via downsampled
#         block masks under dihedral transforms.
#
# complexity_calibration:
#   - We keep non-unit slots D4-distinct, otherwise same-colored pieces could
#     introduce ambiguous many-to-one matching.
#   - We require at least two non-unit slots per sample, so "all complex pieces
#     share one color" is visually obvious instead of only affecting unit squares.
#   - Distractors stay singleton-only. This keeps the new difficulty focused on
#     preprocessing, not on inventing a new matching rule.
#
# implicit_assumption_audit_main:
#   - get_bg / find_components(..., monochromatic=False):
#       assumes background 0 remains the most common color. Generator preserves it.
#   - frame_comp = largest foreground component:
#       assumes the frame beats every scattered piece in area. Generator checks this.
#   - hole extraction inside frame bbox:
#       assumes distractors and pieces never overlap the frame interior. Generator
#       places frame first and only places other objects on empty exterior cells.
#   - piece extraction by "area > 1":
#       this is now the intended discriminator. Color no longer carries meaning.
#   - k inference from smallest solid square:
#       assumes at least one unit slot exists, so at least one real k-by-k solid
#       piece exists. Generator still guarantees 1-2 unit slots.
#   - slot matching by area + downsampled mask equality:
#       assumes every non-unit slot is unique up to D4, and unit slots are safe
#       duplicates because multiple k-by-k squares can fill multiple unit holes.
#
# placement_legality:
#   - The frame is pasted first.
#   - Pieces are placed with padding=1 and 8-neighborhood clearance, so same-color
#     pieces never merge with each other or with the frame.
#   - Distractors are placed only on empty cells whose full 3x3 neighborhood is
#     empty, so they remain isolated singleton components even though they reuse
#     the exact piece color.
#
# visual_observability_and_cleanliness:
#   - Each sample shows multiple large same-colored sprites plus many same-colored
#     isolated dots. Humans can see that color is useless and size/connectivity
#     must be the cue.
#   - Only three visible colors remain: background 0, frame color, and one shared
#     piece/distractor color. This is minimal but still readable because spacing
#     prevents overlaps and role confusion inside a component.
#
# new_solver_assumptions_after_fusion:
#   - All non-frame area-1 foreground components are distractors.
#   - Every real piece stays disconnected from every other foreground object.
#   - At least one real piece is a solid square whose side equals k.
#   - Generator explicitly enforces all three.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame's hole layout and refill each hole with its matching sprite.

    Input structure:
    - The most common color is background. The largest foreground component is a
      monochrome rectangular frame enclosing background holes.
    - Every scattered piece outside the frame is built from solid k-by-k blocks,
      but all real pieces share one color and singleton distractors reuse that
      same color.
    - The smallest solid square piece still reveals k.

    Rule:
    - Read the connected hole regions inside the frame as slot masks.
    - Ignore isolated area-1 foreground components as noise; treat all remaining
      non-frame components as true pieces.
    - Match each true piece to a slot by comparing block-level shapes under
      rotation and reflection, then place the matched piece into the scaled frame.

    Key insight:
    - Color carries no identity at all in this fusion. Both pieces and distractors
      can look identical chromatically, so the invariant is purely geometric:
      true pieces are multi-cell k-by-k upscales of the hole shapes, whereas
      distractors are only isolated singleton components.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
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
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    distractor_count = 0
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area <= 1:
            distractor_count += 1
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": area, "mask": pm, "patch": patch})

    assert pieces, "at least one non-frame piece must exist"
    _ = distractor_count

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((int(hh // k), int(ww // k)), dtype=bool)
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            rp = np.rot90(patch, kk)
            rm = np.rot90(mask, kk)
            base.append((rp, rm))
            base.append((np.fliplr(rp), np.fliplr(rm)))
        uniq = []
        seen = set()
        for vp, vm in base:
            key = (vm.shape, vm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out = np.full(((int(layout.shape[0]) + 2) * k, (int(layout.shape[1]) + 2) * k), int(bgc), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * k
        left = (int(s["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


"""Generator for ARC-AGI-2 task 898e7135, combo D6/2 + D10/0.

Output-first strategy:
  1. Build a frame layout with 4-connected hole slots, including unit slots.
  2. Pick one shared piece_color for every slot and apply random D4 transforms
     to non-unit pieces.
  3. Assemble the expected enlarged output.
  4. Scatter the pieces outside the frame on a bg=0 canvas.
  5. Add 8-13 isolated singleton distractors in that same piece_color, so color
     is completely uninformative.

The generator validates main(input) == expected_output before returning.
"""


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
MIN_NON_UNIT_SLOTS = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a tight-cropped bool mask."""
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow one 4-connected slot on free cells."""
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen))
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a bool mask by one step in 8-connectivity."""
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
    """Mark cells and their 8-neighbors as forbidden."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """Generate 4-connected, pairwise 8-separated slots with unit slots first."""
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1})
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
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
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


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Create input/output patches for one slot using one shared color."""
    if int(slot["area"]) == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {"color": color, "slot": slot, "out_patch": patch, "out_mask": mask, "in_patch": patch, "in_mask": mask}

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": color,
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    """Assemble the enlarged filled frame."""
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _place_camouflage_distractors(canvas: np.ndarray, piece_color: int, n_distractors: int) -> bool:
    """Place isolated singleton distractors in the exact same color as all pieces."""
    coords: list[tuple[int, int]] = []
    h, w = canvas.shape
    mark = -1
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if len(coords) >= n_distractors:
            break
        rr = random.randint(0, h - 1)
        cc = random.randint(0, w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = mark
        coords.append((rr, cc))

    if len(coords) < MIN_DISTRACTORS:
        for rr, cc in coords:
            canvas[rr, cc] = 0
        return False

    for rr, cc in coords:
        canvas[rr, cc] = int(piece_color)
    return True


def _build_input_canvas(
    pieces: list[dict], layout_mask: np.ndarray, layout_h: int, layout_w: int, k: int, frame_color: int, piece_color: int
) -> np.ndarray | None:
    """Build the input grid with frame, same-colored pieces, and same-colored singleton noise."""
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
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

    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
    frame_area = int(np.count_nonzero(frame_img))
    largest_piece_area = max(int(p["in_mask"].sum()) for p in pieces)
    if frame_area <= largest_piece_area:
        return None

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    if not _place_camouflage_distractors(canvas, piece_color, n_distractors):
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - MIN_NON_UNIT_SLOTS)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    if not palette:
        return None
    piece_color = palette.pop()

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]
    if len(non_unit_slots) < MIN_NON_UNIT_SLOTS:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(slot, piece_color, k, apply_dihedral=True) for slot in non_unit_slots]
    pieces += [_make_piece(slot, piece_color, k, apply_dihedral=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, piece_color)
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
