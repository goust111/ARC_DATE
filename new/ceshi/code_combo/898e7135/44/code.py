# Combo variant of 898e7135 (index 44)
# combo_spec: D6/2+D10/0+D11/0
# Name: 极限对象分离
#
# Design analysis for the fused variant
# -------------------------------------
# fusion_summary:
#   - Start from the original "match scattered k-by-k block pieces to frame holes" task.
#   - Apply D11 first: pieces are no longer guaranteed to stay outside the frame bbox;
#     a piece may hug or intrude into the frame bbox as long as it never overwrites a
#     frame-colored pixel. This breaks the old "largest multicolor foreground component
#     is the frame" shortcut and also breaks `sub == bg` as the hole detector, because a
#     true hole may be temporarily covered by a piece.
#   - Apply D10 and D6 together: every real piece and every singleton distractor now
#     shares one common object color. Color contributes zero information: humans and the
#     solver must separate objects only by monochrome connectivity and size.
#
# a_concretized_trace:
#   - Small mental example:
#     - Original task: frame is blue, one L piece is red, one bar is green, distractors
#       are yellow singletons, and all pieces sit outside the frame bbox.
#     - Original solver path:
#       1. `find_components(..., monochromatic=False)` finds one largest foreground blob:
#          the frame.
#       2. Inside the frame bbox, `sub == bg` recovers the hole layout because holes are
#          still visible black cells.
#       3. Each non-frame non-singleton component is cropped from the raw grid and matched
#          to one slot by downsampling.
#     - Fused variant:
#       1. Every non-background non-frame pixel is, say, color 3. A large piece can touch
#          the frame, some other same-colored pieces are elsewhere, and same-colored
#          singleton noise pixels are scattered around.
#       2. The old largest-component heuristic fails because frame-adjacent object pixels
#          merge into the same multicolor foreground region.
#       3. The old `sub == bg` hole recovery fails because a true hole covered by a piece
#          would no longer look like background.
#       4. The correct fused interpretation is:
#            - find the frame as the unique monochrome rectangular-border component;
#            - inside its bbox, every non-frame-color cell still belongs to the hole
#              layout, whether it is visible background or temporarily occupied by a piece;
#            - among same-colored non-frame monochrome components, area-1 means distractor
#              and area>1 means real piece.
#
# b_complexity_calibration:
#   - To keep the "all objects same color" change visible, the generator requires at least
#     two non-unit slots, so multiple complex shapes share the same color.
#   - To keep the D11 change visible and to ensure the fused inputs differ materially from
#     the base task, at least one large piece is forced into a frame-bbox-overlapping,
#     frame-touching placement.
#   - To avoid ambiguous matching despite color removal, non-unit slot shapes remain
#     unique up to D4, and 1-2 unit slots still provide the solid k-by-k square needed to
#     infer k.
#   - Distractors remain singleton components only. This keeps the geometry hard enough to
#     require connectivity reasoning without making the board unreadable.
#
# c_implicit_assumption_audit_main:
#   - Base line `frame_comp = max(multicolor_comps, area)` is invalid after D11 because a
#     touching piece or touching distractor can merge with the frame in multicolor space.
#   - Base line `hole = sub == bg` is invalid after D11 because covered holes are no longer
#     visible background cells.
#   - Base line `patch = crop_bbox(g, piece_bbox)` is unsafe when piece bboxes overlap the
#     frame bbox; the crop may include frame pixels. The fused solver rebuilds each piece
#     patch from its monochrome coords only.
#   - New solver assumptions that the generator must satisfy:
#       1. exactly one monochrome component forms a rectangular border with interior
#          non-frame cells, so the frame is identifiable;
#       2. every real piece is one monochrome area>1 component in the shared object color;
#       3. every distractor is a singleton monochrome component in that same color;
#       4. at least one unit slot exists, so k is recoverable from a solid square piece;
#       5. non-unit slot shapes are unique under D4, so geometric matching is injective.
#
# d_placement_legality:
#   - Pieces are never allowed to overwrite nonzero canvas cells, so they cannot cover
#     frame pixels or each other.
#   - Piece-piece spacing still uses an 8-neighborhood exclusion, so same-colored pieces
#     never merge.
#   - Distractors are placed only on empty cells outside the frame bbox and only when no
#     same-colored object already exists in their 3x3 neighborhood, so they stay singleton
#     object-color components even though frame adjacency is allowed.
#
# e_visual_observability:
#   - Across examples, a human can see that:
#       1. multiple non-singleton shapes and multiple singleton dots are the exact same
#          color;
#       2. at least one larger shape visibly hugs or overlaps the frame bbox region;
#       3. output removes all singleton dots and reuses the same shared object color to
#          fill matched holes.
#   - No hidden ordering by numeric color ids is used; only visible geometry, adjacency,
#     and the rectangular frame structure matter.
#
# f_cleanliness:
#   - The palette is intentionally compressed to three semantic colors: background, frame,
#     shared object color.
#   - Only one or occasionally a few pieces are near the frame; the rest remain separated,
#     which keeps the input readable even though color no longer separates roles.
#
# g_inferability_of_change:
#   - There is no arbitrary color remapping to infer. The only color rule is structural:
#     every non-frame foreground object shares one color.
#   - Because slot identity is still determined by D4 shape matching and not by an unseen
#     lookup table, 2-4 examples are enough for a human to infer the transformation.

from __future__ import annotations

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
)


def _solid_square_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _dihedral_variants(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rp = np.rot90(patch, int(kk))
        rm = np.rot90(mask, int(kk))
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


def _recover_layout_from_frame_bbox(sub: np.ndarray, frame_color: int) -> np.ndarray:
    non_frame = sub != int(frame_color)
    hole_box = bbox(non_frame)
    assert hole_box is not None, "frame must enclose at least one non-frame cell"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = non_frame[lr0 : lr1 + 1, lc0 : lc1 + 1]
    assert layout[0].any() and layout[-1].any()
    assert layout[:, 0].any() and layout[:, -1].any()
    return layout


def _find_frame_component(g: np.ndarray, bg: int) -> dict:
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    candidates: list[tuple[tuple[int, int, int, int], dict]] = []
    for comp in mono:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if not (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        interior = sub[1:-1, 1:-1]
        if interior.size == 0 or not np.any(interior != color):
            continue
        try:
            layout = _recover_layout_from_frame_bbox(sub, color)
        except AssertionError:
            continue
        if not components_mask(layout, connectivity=4):
            continue
        rank = (h * w, int(comp["area"]), h + w, color)
        candidates.append((rank, comp))
    assert candidates, "no valid frame component found"
    return max(candidates, key=lambda item: item[0])[1]


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"
    return slots


def _extract_pieces(g: np.ndarray, bg: int, frame_comp: dict) -> list[dict]:
    frame_color = int(frame_comp["color"])
    frame_bbox = tuple(map(int, frame_comp["bbox"]))
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces = []
    for comp in mono:
        if int(comp["color"]) == frame_color and tuple(map(int, comp["bbox"])) == frame_bbox:
            continue
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        pm = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        patch = np.zeros(pm.shape, dtype=int)
        patch[pm] = int(comp["color"])
        pieces.append({"area": area, "mask": pm, "patch": patch, "color": int(comp["color"])})
    assert pieces, "at least one non-frame piece must exist"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a framed hole-layout and fill each hole with its matching piece.

    Input structure:
    - Background is the most common color. The frame is the unique monochrome rectangular-border
      component whose interior contains non-frame cells.
    - Every non-frame foreground object, both real pieces and singleton distractors, shares one
      common color.
    - A real piece is a non-singleton k-by-k-block sprite, possibly rotated/flipped, and it may
      touch or overlap the frame bbox as long as it never replaces frame pixels.

    Rule:
    - Recover the slot layout from the frame bbox by treating every non-frame-color cell inside
      that bbox as part of the hole pattern, even if a piece currently covers it.
    - Separate shared-color objects by monochrome connectivity: singleton components are
      distractors, larger components are candidate pieces.
    - Match each piece to one slot by downsampling its k-block structure under all dihedral
      variants, then output the scaled frame filled with the matched pieces.

    Key insight:
    - Once color stops carrying any role information, the invariant is purely structural: the
      frame is detectable by its rectangular monochrome border, singleton shared-color components
      are removable noise, and every larger shared-color component is a k-scaled slot shape whose
      downsampled mask matches exactly one hole under rotation/flip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame_comp = _find_frame_component(g, bg)
    frame_color = int(frame_comp["color"])
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    layout = _recover_layout_from_frame_bbox(sub, frame_color)
    slots = _extract_slots(layout)
    pieces = _extract_pieces(g, bg, frame_comp)
    k = _solid_square_k(pieces)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
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

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

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
DISTRACTOR_RETRY_MULT = 16
MIN_DISTRACTORS = 8

FORCED_OVERLAPPERS_RANGE = (1, 1)
OPTIONAL_OVERLAP_PROB = 0.20
GENERATE_MAX_RETRIES = 500


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
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


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
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
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
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
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _bbox_overlap_area(top: int, left: int, shape: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> int:
    pr0, pc0 = int(top), int(left)
    pr1 = int(top) + int(shape[0]) - 1
    pc1 = int(left) + int(shape[1]) - 1
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    oh = min(pr1, fr1) - max(pr0, fr0) + 1
    ow = min(pc1, fc1) - max(pc0, fc0) + 1
    if oh <= 0 or ow <= 0:
        return 0
    return int(oh * ow)


def _candidate_positions(
    canvas: np.ndarray,
    piece_mask: np.ndarray,
    piece_occ: np.ndarray,
    frame_occ: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    require_overlap: bool,
) -> list[tuple[int, int, int, bool]]:
    canvas_h, canvas_w = canvas.shape
    ph, pw = piece_mask.shape
    padded_piece_occ = _dilate8(piece_occ)
    touch_frame_occ = _dilate4(frame_occ)
    candidates = []
    for top in range(canvas_h - ph + 1):
        for left in range(canvas_w - pw + 1):
            view_nonzero = canvas[top : top + ph, left : left + pw] != 0
            if bool(np.logical_and(view_nonzero, piece_mask).any()):
                continue
            if bool(np.logical_and(padded_piece_occ[top : top + ph, left : left + pw], piece_mask).any()):
                continue
            overlap = _bbox_overlap_area(top, left, piece_mask.shape, frame_bbox)
            touches_frame = bool(
                np.logical_and(touch_frame_occ[top : top + ph, left : left + pw], piece_mask).any()
            )
            if require_overlap and not (overlap > 0 and touches_frame):
                continue
            candidates.append((int(top), int(left), int(overlap), bool(touches_frame)))
    return candidates


def _choose_position(candidates: list[tuple[int, int, int, bool]], require_overlap: bool) -> tuple[int, int, int] | None:
    if not candidates:
        return None
    if require_overlap:
        best = max(int(c[2]) for c in candidates)
        strong = [c for c in candidates if int(c[2]) >= max(1, best - 2)]
        top, left, overlap, _ = random.choice(strong)
        return int(top), int(left), int(overlap)
    overlap_cands = [c for c in candidates if int(c[2]) > 0 and bool(c[3])]
    if overlap_cands and random.random() < OPTIONAL_OVERLAP_PROB:
        top, left, overlap, _ = random.choice(overlap_cands)
        return int(top), int(left), int(overlap)
    top, left, overlap, _ = random.choice(candidates)
    return int(top), int(left), int(overlap)


def _place_singleton_distractors(
    canvas: np.ndarray,
    object_color: int,
    frame_bbox: tuple[int, int, int, int],
    n_distractors: int,
) -> bool:
    h, w = canvas.shape
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    coords: list[tuple[int, int]] = []
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if len(coords) >= n_distractors:
            break
        rr = random.randint(0, h - 1)
        cc = random.randint(0, w - 1)
        if fr0 <= rr <= fr1 and fc0 <= cc <= fc1:
            continue
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] == int(object_color)):
            continue
        canvas[rr, cc] = int(object_color)
        coords.append((rr, cc))
    if len(coords) < MIN_DISTRACTORS:
        for rr, cc in coords:
            canvas[rr, cc] = 0
        return False
    return True


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    object_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
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
    frame_top = random.randint(1, fr_max)
    frame_left = random.randint(1, fc_max)
    paste(canvas, frame_img, frame_top, frame_left)
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    frame_occ = canvas != 0

    ordered = sorted(pieces, key=lambda p: -int(p["in_mask"].sum()))
    piece_occ = np.zeros_like(canvas, dtype=bool)
    n_forced = min(len(ordered), random.randint(*FORCED_OVERLAPPERS_RANGE))
    overlaps: list[int] = []

    for idx, p in enumerate(ordered):
        require_overlap = idx < n_forced
        candidates = _candidate_positions(
            canvas,
            p["in_mask"],
            piece_occ,
            frame_occ,
            frame_bbox,
            require_overlap,
        )
        pos = _choose_position(candidates, require_overlap)
        if pos is None:
            return None
        top, left, overlap = pos
        overlaps.append(int(overlap))
        paste(canvas, p["in_patch"], top, left, mask=p["in_mask"], overwrite=True)
        piece_occ[top : top + p["in_mask"].shape[0], left : left + p["in_mask"].shape[1]] |= p["in_mask"]

    if not any(v > 0 for v in overlaps[:n_forced]):
        return None

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    if not _place_singleton_distractors(canvas, object_color, frame_bbox, n_distractors):
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
    frame_color = int(palette.pop())
    if not palette:
        return None
    object_color = int(palette.pop())
    if object_color == frame_color:
        return None

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

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

    pieces = [_make_piece(s, object_color, k, apply_dihedral=True) for s in non_unit_slots]
    pieces += [_make_piece(s, object_color, k, apply_dihedral=False) for s in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, object_color)
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
