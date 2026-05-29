# Combo variant of 898e7135 (index 32)
# combo_spec: D1/2+D2/0
# Name: 多色严格方向匹配

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and flow impact
#    - The original task detects one rectangular frame, extracts the hole layout inside it,
#      matches outside sprites to hole shapes at block scale k, then pastes the matched sprites
#      into the enlarged frame.
#    - This combo changes two linked parts at once:
#      1. shape matching is strict-orientation only: no rotation, no mirror;
#      2. every non-frame sprite is multicolored at block level: each occupied layout cell expands
#         to a solid k-by-k block, but different cells of the same sprite use 2-3 colors.
#    - Small trace:
#      two L-shaped holes belong to the same D4 family but face different directions. In the
#      original task, either outside sprite could be rotated/flipped to fit either hole. Here each
#      sprite already appears in its final direction, and its internal color arrangement is also
#      direction-sensitive. The solver must therefore compare the downsampled mask exactly as seen
#      and paste the piece without any reorientation.
#
# b) Complexity calibration
#    - Using multicolor pieces alone would not force a new inference rule if all shapes were unique.
#      Using strict orientation alone would be visually weak if every D4 family appeared only once.
#    - The generator therefore always includes one pair of non-unit slots that share a D4 family but
#      differ in exact orientation. All remaining non-unit slots are unique under D4. This keeps the
#      puzzle unambiguous while making the "no rotate / no flip" rule observable in every sample.
#    - Colors are reused from one small global palette instead of assigning a fresh palette per piece.
#      That keeps the scene readable and makes the multicolor block structure a visible regularity
#      rather than arbitrary decoration.
#
# c) Implicit-assumption review of main()
#    - Largest multicolor component = frame: still required. The generator keeps the bordered frame
#      larger than any outside piece.
#    - Tight crop of bg holes inside the frame bbox: unchanged. The generator still ensures the hole
#      layout touches all four sides of its tight crop.
#    - Non-singleton foreground components are pieces: still valid because distractors remain isolated
#      single pixels and slots of area 1 are removed from this combo.
#    - Old k inference from a solid square piece no longer applies because every piece must use 2-3
#      colors. New assumption: each occupied layout cell becomes one monochrome k-by-k block and
#      same-colored blocks never touch orthogonally inside a piece. Then the smallest non-singleton
#      monochrome component area is exactly k^2, which the generator enforces.
#    - Candidate generation may not enumerate dihedral variants in the strict solver. Exact oriented
#      block masks must therefore be unique across different slots; the generator enforces that, while
#      allowing one duplicate only at the D4-family level.
#
# d) Placement legality
#    - Slots stay 8-separated in the layout, so enlarged pasted pieces never overlap in the output.
#    - The frame and all pieces are placed on a bg canvas with at least one background cell of 8-gap.
#      The special orientation-sensitive pair is placed first with controlled scan order so that a
#      relaxed dihedral matcher would consume them in the wrong order.
#    - Distractors remain singleton pixels in one dedicated color and are placed away from all other
#      objects, so they never merge into the frame or pieces.
#
# e) Visual observability
#    - A human can see two simultaneous cues:
#      1. two holes have the same coarse family but different facing directions;
#      2. each sprite contains a small repeating 2-3-color block pattern that would visibly change if
#         the sprite were rotated or mirrored.
#    - The generator validates that a relaxed dihedral solver produces a different output from the
#      strict solver, preventing silent collapse back to the original rule.
#
# f) Visual cleanliness
#    - Only one frame, a handful of separated multicolor sprites, and isolated singleton distractors
#      appear. There are no overlaid markers or hidden encodings.
#    - Each sprite is colored per block, not per pixel noise, so the enlarged pieces remain legible
#      and the output still looks like a clean inflated frame-filling task.
#
# g) Change inferability
#    - Colors do not encode a lookup table. They stay attached to block positions within each sprite,
#      and the output simply preserves that local arrangement.
#    - Because every sprite uses only 2-3 colors from a small shared palette, the viewer can infer
#      that color layout is part of the object's orientation rather than an unrelated recoloring rule.

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

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8
PAIR_PLACE_RETRIES = 20
PAIR_COLOR_RETRIES = 50

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 800
FAMILY_ID = 0

ORIENTATION_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
]

COLOR_FORMULAS = {
    2: [(1, 1), (1, 0), (0, 1)],
    3: [(1, 1), (1, 2), (2, 1), (2, 2)],
}


def _shape_key(mask: np.ndarray) -> bytes:
    return f"{int(mask.shape[0])}x{int(mask.shape[1])}|".encode() + mask.astype(np.uint8).tobytes()


def _colored_key(block_colors: np.ndarray, mask: np.ndarray) -> bytes:
    arr = np.where(mask, block_colors, -1).astype(np.int16)
    return (
        f"{int(arr.shape[0])}x{int(arr.shape[1])}|".encode()
        + arr.tobytes()
    )


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(_shape_key(rot))
        variants.append(_shape_key(np.fliplr(rot)))
    return min(variants)


def _unique_d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for var in (rot, np.fliplr(rot)):
            key = _shape_key(var)
            if key in seen:
                continue
            seen.add(key)
            out.append(var.copy())
    return out


def _dihedral_variants(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for kk in range(4):
        rp = np.rot90(patch, kk)
        rm = np.rot90(mask, kk)
        for vp, vm in ((rp, rm), (np.fliplr(rp), np.fliplr(rm))):
            key = _colored_key(vp, vm)
            if key in seen:
                continue
            seen.add(key)
            out.append((vp.copy(), vm.copy()))
    return out


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _mask_to_slot(coords: np.ndarray, *, family_id: int | None = None) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
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


def _downsample_piece(
    patch: np.ndarray, mask: np.ndarray, k: int, bg: int
) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    out_colors = np.full((out_h, out_w), -1, dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            rs = int(r * k)
            re = int((r + 1) * k)
            cs = int(c * k)
            ce = int((c + 1) * k)
            block_mask = mask[rs:re, cs:ce]
            block_patch = patch[rs:re, cs:ce]
            if bool(block_mask.all()):
                vals = np.unique(block_patch)
                if len(vals) != 1 or int(vals[0]) == int(bg):
                    return None
                out_mask[r, c] = True
                out_colors[r, c] = int(vals[0])
            elif bool(block_mask.any()):
                return None
            elif np.any(block_patch != int(bg)):
                return None
    return out_mask, out_colors


def _infer_k_from_mono_blocks(grid: np.ndarray, bg: int) -> int:
    mono = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    areas = [int(comp["area"]) for comp in mono if int(comp["area"]) > 1]
    assert areas, "need non-singleton monochrome components"
    area = min(areas)
    k = int(round(area ** 0.5))
    assert k * k == area, "smallest monochrome component must be a square block"
    return k


def _solve(grid: np.ndarray, *, allow_dihedral: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one background hole"
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
        mask = np.zeros(tuple(map(int, patch.shape)), dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = _infer_k_from_mono_blocks(g, bg)

    candidates = []
    for pi, piece in enumerate(pieces):
        variants = _dihedral_variants(piece["patch"], piece["mask"]) if allow_dihedral else [(piece["patch"], piece["mask"])]
        for vp, vm in variants:
            down = _downsample_piece(vp, vm, k, bg)
            if down is None:
                continue
            um, uc = down
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": um,
                    "uc": uc,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a strictly oriented piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame's hole map and insert each multicolor sprite into its exactly aligned slot.

    Input structure:
    - Background is the most frequent color. The largest foreground component is a single-color frame
      whose interior contains background holes; each outside sprite is a connected multicolor object
      made from solid k-by-k color blocks.
    - Distractors are isolated one-pixel foreground cells. Piece colors may repeat across sprites, but
      within a sprite the occupied blocks use 2-3 colors and keep their original orientation.

    Rule:
    - Crop the hole layout inside the frame, split it into connected slots, and infer k from the
      smallest non-singleton monochrome block component.
    - Downsample each outside sprite to block scale, match it to the unique slot with the same mask in
      the same orientation only, then paste the sprite unchanged into the k-scaled output frame.

    Key insight:
    - The outside sprites are already the final filled-in tiles of the enlarged frame. Because color
      layout is attached to block positions, rotating or mirroring a sprite would alter the intended
      pattern, so exact block-mask orientation is the decisive match cue.
    """
    return _solve(grid, allow_dihedral=False)


# --- generator ---------------------------------------------------------------

def _layout_touches_all_borders(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_exact: set[bytes] = set()
    used_canon: set[bytes] = set()

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

    for _ in range(n_slots - 2):
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


def _dihedral_colored_variants(block_colors: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for kk in range(4):
        rot_c = np.rot90(block_colors, kk)
        rot_m = np.rot90(mask, kk)
        for vc, vm in ((rot_c, rot_m), (np.fliplr(rot_c), np.fliplr(rot_m))):
            key = _colored_key(vc, vm)
            if key in seen:
                continue
            seen.add(key)
            out.append(np.where(vm, vc, -1).astype(int))
    return out


def _choose_block_colors(mask: np.ndarray, palette: list[int]) -> np.ndarray | None:
    coords = np.argwhere(mask)
    area = int(coords.shape[0])
    target_options = [2] if area == 2 else random.sample([2, 3], k=2)
    for target in target_options:
        if target > area:
            continue
        color_choices = palette[:target]
        formulas = COLOR_FORMULAS[target][:]
        random.shuffle(formulas)
        offsets = list(range(target))
        random.shuffle(offsets)
        for a, b in formulas:
            for offset in offsets:
                block_colors = np.full(mask.shape, -1, dtype=int)
                for r, c in coords:
                    idx = (int(a) * int(r) + int(b) * int(c) + int(offset)) % int(target)
                    block_colors[int(r), int(c)] = int(color_choices[idx])
                used = {int(v) for v in block_colors[mask]}
                if len(used) != target:
                    continue
                if any(np.array_equal(var, np.where(mask, block_colors, -1)) for var in _dihedral_colored_variants(block_colors, mask)[1:]):
                    continue
                return block_colors
    return None


def _transform_equivalent_coloring(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray) -> bool:
    target = np.where(b_mask, b, -1)
    for var in _dihedral_colored_variants(a, a_mask):
        if var.shape == target.shape and np.array_equal(var, target):
            return True
    return False


def _scale_block_colors(block_colors: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, mask.shape)
    patch = np.zeros((h * k, w * k), dtype=int)
    out_mask = np.zeros((h * k, w * k), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            rs = int(r * k)
            re = int((r + 1) * k)
            cs = int(c * k)
            ce = int((c + 1) * k)
            patch[rs:re, cs:ce] = int(block_colors[r, c])
            out_mask[rs:re, cs:ce] = True
    return patch, out_mask


def _make_piece(slot: dict, k: int, palette: list[int]) -> dict | None:
    block_colors = _choose_block_colors(slot["mask"], palette)
    if block_colors is None:
        return None
    patch, mask = _scale_block_colors(block_colors, slot["mask"], k)
    return {
        "slot": slot,
        "family_id": slot.get("family_id"),
        "block_colors": block_colors,
        "out_patch": patch,
        "out_mask": mask,
        "in_patch": patch.copy(),
        "in_mask": mask.copy(),
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
    out: list[tuple[int, int]] = []
    for r0 in range(ch - h + 1):
        for c0 in range(cw - w + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                out.append((int(r0), int(c0)))
    return out


def _find_swapped_pair_positions(
    canvas: np.ndarray, early_piece: dict, late_piece: dict
) -> tuple[tuple[int, int], tuple[int, int]] | None:
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
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
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

    family_pieces = [piece for piece in pieces if piece.get("family_id") == FAMILY_ID]
    assert len(family_pieces) == 2
    family_pieces = sorted(family_pieces, key=lambda piece: _slot_sort_key(piece["slot"]))
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
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

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

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_palette = palette[:3]
    if len(piece_palette) != 3:
        return None

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not _layout_touches_all_borders(layout_mask):
        return None

    pieces: list[dict] = []
    for slot in sorted(slots, key=_slot_sort_key):
        piece = _make_piece(slot, k, piece_palette)
        if piece is None:
            return None
        pieces.append(piece)

    for _ in range(PAIR_COLOR_RETRIES):
        family_pieces = [piece for piece in pieces if piece.get("family_id") == FAMILY_ID]
        if len(family_pieces) != 2:
            return None
        if not _transform_equivalent_coloring(
            family_pieces[0]["block_colors"],
            family_pieces[0]["slot"]["mask"],
            family_pieces[1]["block_colors"],
            family_pieces[1]["slot"]["mask"],
        ):
            break
        for idx, piece in enumerate(pieces):
            if piece.get("family_id") != FAMILY_ID:
                continue
            new_piece = _make_piece(piece["slot"], k, piece_palette)
            if new_piece is None:
                return None
            pieces[idx] = new_piece
    else:
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        strict = main(canvas)
        relaxed = _solve(canvas, allow_dihedral=True)
    except Exception:
        return None
    if not np.array_equal(strict, expected_out):
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
