# Variant of 898e7135
# Dimension: D1 — 匹配条件：块级形状匹配（含旋转/翻转） (match_cond)
# Change: shape_match + rotation_flip — 精灵下采样为块级掩码后，与洞的形状在二面体群（4次旋转 × ... → 禁止旋转和翻转（严格方向匹配）

from __future__ import annotations

# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and flow impact
#    - The frame, slots, block scale k, singleton distractors, and output format stay the same.
#    - The changed behavior is only in piece-slot matching:
#      original task: a scattered piece may be rotated or mirrored before matching a hole;
#      this variant: the scattered piece must already have the same orientation as its hole.
#    - In pixels, each non-unit piece is still a solid k-by-k block expansion of a slot mask.
#      The visual difference is that the input now contains direction-sensitive shapes that must
#      be matched "as seen", not after mental rotation.
#    - Small trace:
#      slot A = horizontal domino, slot B = vertical domino. The input contains one horizontal
#      piece and one vertical piece. Original logic can rotate either piece, so both pieces are
#      valid candidates for both slots. Variant logic compares the downsampled mask directly, so
#      the horizontal piece only matches slot A and the vertical piece only matches slot B.
#
# b) Complexity calibration
#    - If every slot had a unique D4 shape class, removing rotation/flip would be visually weak:
#      the same samples would still be solvable by the old rule.
#    - To make the change observable without adding unrelated mechanics, the generator always
#      includes one pair of non-unit slots that belong to the same D4 class but have different
#      exact orientations. All other non-unit slots remain unique under D4, so the puzzle stays
#      unambiguous and the only added burden is orientation checking.
#
# c) Implicit-assumption review of main()
#    - Frame detection: unchanged. It still assumes the largest multicolor component is the frame.
#      The generator keeps the frame largest.
#    - Hole extraction and slot cropping: unchanged. The generator still makes holes touch all
#      four borders of the tight crop so the cropped layout is exact.
#    - Piece extraction: unchanged. Distractors are still singletons and are ignored by area<=1.
#    - k inference: unchanged. The generator still includes at least one unit slot, hence at least
#      one solid k-by-k square piece.
#    - Candidate generation: this is the critical line-level change. The original code expanded
#      each piece to all D4 variants before shape comparison. That assumption no longer holds.
#      The variant solver keeps only the original orientation. Therefore the generator must ensure
#      no two distinct non-unit slots share the same exact oriented mask, otherwise strict matching
#      would become ambiguous. The generator enforces exact-shape uniqueness.
#
# d) Placement legality
#    - Slots are still 8-separated in the layout.
#    - Pieces are still placed outside the frame with at least one background cell of separation.
#    - The special orientation-sensitive pair is placed first on the canvas so their scan order is
#      controlled without overlap, then the remaining pieces use random_free_location.
#
# e) Visual observability
#    - Each sample contains a same-D4-but-different-orientation pair such as two orientations of
#      an L triomino or domino. Humans can see that the outside sprite orientation is already the
#      one that appears in the filled output; no rotation is being "corrected" during solving.
#    - The generator also checks that the old relaxed solver would produce a different output on
#      the same input, so the samples do not silently collapse back to the original behavior.
#
# f) Visual cleanliness
#    - Only one frame, a few separated pieces, and isolated singleton distractors remain.
#    - No extra markings or overlays are introduced. The variant relies on shape orientation, which
#      is already visually salient in ARC grids.
#
# g) Color inferability
#    - Colors are unchanged structurally: frame color, one distractor color, distinct colors for
#      non-unit pieces, and a shared color for unit pieces.
#    - Color is not used to encode hidden state. It only preserves object identity once strict
#      shape matching has chosen the destination slot.

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


# --- solver ------------------------------------------------------------------

def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
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


def _solve(grid: np.ndarray, *, allow_dihedral: bool) -> np.ndarray:
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
    assert hole_box is not None, "frame must enclose at least one bg hole"
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
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    candidates = []
    for pi, p in enumerate(pieces):
        variants = _dihedral_variants(p["patch"], p["mask"]) if allow_dihedral else [(p["patch"], p["mask"])]
        for vp, vm in variants:
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
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole-layout and fill each slot with the strictly oriented matching piece.

    Variant difference from the original task:
    - pieces are matched only in their given orientation;
    - no rotation and no flip are allowed during comparison.
    """
    return _solve(grid, allow_dihedral=False)


# --- generator ---------------------------------------------------------------

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
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300
FAMILY_ID = 0

ORIENTATION_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
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
    out = []
    seen = set()
    for kk in range(4):
        r = np.rot90(mask, kk)
        for v in (r, np.fliplr(r)):
            key = _shape_key(v)
            if key in seen:
                continue
            seen.add(key)
            out.append(v.copy())
    return out


def _mask_to_slot(coords: np.ndarray, *, family_id: int | None = None) -> dict:
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


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = list(map(tuple, free_cells.tolist()))
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
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
            chosen.add((int(nr), int(nc)))
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


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _place_mask_slot(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    ones = np.argwhere(mask)
    candidates = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            coords = ones + np.array([[int(r0), int(c0)]], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    return ones + np.array([[int(r0), int(c0)]], dtype=int)


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_exact: set[bytes] = set()
    used_canon: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            slots.append(slot)
            used_exact.add(slot["shape_key"])
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


def _make_piece(slot: dict, color: int, k: int) -> dict:
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
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], int(top), int(left), mask=p["out_mask"], overwrite=True)
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
    late_candidates = sorted(_legal_positions(canvas, late_piece["in_patch"]))
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
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
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
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    family_pieces = [p for p in pieces if p.get("family_id") == FAMILY_ID]
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
    others = [p for p in pieces if id(p) not in handled]
    for p in sorted(others, key=lambda q: -int(q["in_mask"].sum())):
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
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
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

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    non_unit_sorted = sorted(non_unit_slots, key=_slot_sort_key)
    pieces = [_make_piece(s, non_unit_colors[i], k) for i, s in enumerate(non_unit_sorted)]
    pieces += [_make_piece(s, unit_color, k) for s in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        relaxed = _solve(canvas, allow_dihedral=True)
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
