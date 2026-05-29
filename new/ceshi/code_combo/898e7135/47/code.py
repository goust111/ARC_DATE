# Combo variant of 898e7135 (index 47)
# combo_spec: D1/1+D4/4+D13/2
# Name: 旋转-重着色-填充三步流水线

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
Design analysis for combo 47
============================

a) Concrete behavior and process trace
--------------------------------------
- Original task:
  1. Read the frame and its hole-components.
  2. Read outside pieces as k-by-k block sprites.
  3. Match a piece to a slot under rotation or flip.
  4. Paste the matched piece into the scaled frame, keeping the piece color.
- Combo task:
  1. Keep the same frame extraction and the same block-level slot geometry.
  2. Restrict matching to rotations only. A left-handed L and a right-handed L
     are now different.
  3. After a piece is matched and reoriented, recolor its occupied cells to the
     frame-wall color.
  4. Paste that recolored patch into the scaled slot.
- Small trace:
  - Suppose two slots are mirror partners: one is an L with its foot on the
    lower-right, the other is the mirrored L with its foot on the lower-left.
  - Input contains two scattered pieces that are rotated copies of those two
    chiral shapes.
  - The original solver could match either slot using a flip if needed.
  - The combo solver must first recover the correct rotation-only match, then
    discard the piece's original color, then fill the slot.
  - Final output loses the piece colors, but the input still genuinely contains
    a chirality-sensitive matching problem.

b) Complexity calibration
-------------------------
- The risky part is D4/4: if every filled piece is recolored to the same wall
  color, the final image becomes a solid scaled frame rectangle.
- To avoid the generator collapsing into samples where D1/1 is irrelevant, the
  layout is forced to contain at least one non-unit mirror pair that is not
  rotation-equivalent.
- Non-unit rotation classes remain unique, so the rotation-only matching is
  still unambiguous rather than combinatorially messy.
- This keeps the latent reasoning chain meaningful without introducing extra
  arbitrary color rules beyond the requested recolor.

c) Implicit assumption audit for `main()`
-----------------------------------------
- `frame_comp = max(comps, key=area)` assumes the frame is the largest
  foreground component.
  Generator guarantee: reject samples where frame wall area is not larger than
  every piece area, and keep pieces separated from the frame.
- Tight-cropping the hole layout assumes at least one hole exists and that the
  union of holes reaches all four sides of the cropped layout.
  Generator guarantee: reject layouts that do not touch all four borders.
- Inferring `k` from the smallest solid square assumes there is at least one
  unit slot.
  Generator guarantee: every sample includes 1 or 2 unit slots.
- Enumerating piece variants assumes only rotations are legal now.
  Generator guarantee: non-unit pieces are rotated but never flipped.
- The matching loop assumes each slot has exactly one unused piece with the same
  rotation-class block mask.
  Generator guarantee: all non-unit rotation keys are unique, while one forced
  mirror pair ensures flips would matter but rotations still separate them.
- The recolor stage assumes the final paint color is the frame-wall color.
  Generator guarantee: the frame is monochrome, so that target color is unique.

d) Placement legality
---------------------
- Slots are placed with 8-neighbor exclusion, so hole components never touch.
- The frame is placed first on the background canvas.
- Pieces are scattered by `random_free_location(..., padding=1,
  padding_connectivity=8)`, which keeps them disjoint from the frame and from
  one another.
- Distractors remain isolated single pixels in one dedicated color.

e) Visual observability
-----------------------
- A human can still see the D1 change in the input because at least one chiral
  mirror pair appears among the holes and among the rotated pieces.
- A human can also see the D4 change from input/output examples because the
  multi-colored input pieces vanish into the wall color in the output.
- The multi-step story is inferable as:
  1. match by rotated shape,
  2. recolor matched content to the wall color,
  3. fill the frame.

f) Visual cleanliness
---------------------
- No overlays, markers, or extra substructure are added to the pieces.
- The input remains close to the original task: one frame, several scattered
  monochrome block sprites, and singleton noise.
- The output is visually simple rather than cluttered: a clean solid rectangle
  at the scaled frame size.

g) Change inferability
----------------------
- The recolor rule is structurally anchored: every matched piece becomes the
  same color as the surrounding wall it fills into.
- Only one recolor target exists per sample, so the user does not need to infer
  arbitrary per-piece color correspondences.
- Because color is erased after matching, this combo would be degenerate if the
  input distribution never instantiated chirality-sensitive cases. The generator
  therefore forces such cases and they are audited after implementation.

9) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumption: at least one non-unit mirror pair exists, so rotation-only
  vs rotation-plus-flip is a real distinction in the input distribution.
  Generator guarantee: `_gen_layout` explicitly constructs one such pair.
- Solver assumption: piece shapes are exact k-by-k block upscales of the slot
  masks.
  Generator guarantee: every piece is built by `np.kron(slot_mask, ones(k,k))`.
- Solver assumption: recoloring after matching uses only occupancy, not the
  original piece color.
  Generator guarantee: pieces are monochrome, so erasing their color is always
  well-defined.
"""


RotKey = tuple[tuple[int, int], bytes]


# --- shared helpers ----------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
CHIRAL_PAIR_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 60
LAYOUT_RETRIES = 120

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


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


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


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(hh // k):
        for c in range(ww // k):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _rotation_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = []
    seen: set[RotKey] = set()
    for kk in range(4):
        vp = np.rot90(patch, int(kk))
        vm = np.rot90(mask, int(kk))
        key = (tuple(map(int, vm.shape)), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _extract_frame_and_layout(g: np.ndarray) -> tuple[int, dict, int, np.ndarray, list[dict]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain frame and pieces"

    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "coords": coords,
                "rot_key": _rotation_shape_key(sm),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    return bg, frame_comp, frame_color, layout, slots


def _extract_pieces(g: np.ndarray, bg: int, frame_comp: dict) -> list[dict]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
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
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"
    return pieces


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None, "a unit slot is required to infer k"
    return int(k)


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a framed hole-layout and run a three-stage piece pipeline on it.

    Input structure:
    - The most common color is the background. The largest foreground component
      is a monochrome rectangular frame enclosing background holes and same-color walls.
    - Outside the frame are separated piece sprites built from solid k×k blocks,
      with k recoverable from the smallest solid square piece.

    Rule:
    - Stage 1: extract hole components as slots and match each outside piece to a
      slot by block-level shape using rotations only; flips are illegal.
    - Stage 2: once a piece is matched and reoriented, recolor all occupied cells
      of that piece to the frame-wall color.
    - Stage 3: paste the recolored piece into the corresponding slot of the k-scaled
      frame layout and return the cropped scaled frame.

    Key insight: The input still hides a rotation-only shape-bijection problem,
    but after the correct orientation is recovered every matched piece loses its
    own color and merges into the wall color of the enlarged frame.
    """
    g = np.array(grid, dtype=int)
    bg, frame_comp, frame_color, layout, slots = _extract_frame_and_layout(g)
    pieces = _extract_pieces(g, bg, frame_comp)
    k = _infer_k(pieces)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _rotation_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vm": vm, "vp": vp})

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
        assert found is not None, "every slot must match a piece under rotations only"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        recolored = np.full(cand["vp"].shape, int(frame_color), dtype=int)
        paste(out, recolored, int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- generator ---------------------------------------------------------------

def _random_shape_mask(
    target_size: int,
    *,
    require_chiral: bool,
    used_rot_keys: set[RotKey] | None = None,
) -> np.ndarray | None:
    used_rot_keys = used_rot_keys or set()
    max_side = min(5, max(2, int(target_size)))
    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(2, max_side)
        box_w = random.randint(2, max_side)
        if box_h * box_w < int(target_size):
            continue

        r0 = random.randint(0, box_h - 1)
        c0 = random.randint(0, box_w - 1)
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
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
        return mask
    return None


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None

    candidates: list[tuple[int, int]] = []
    base_coords = np.argwhere(mask)
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            coords = base_coords + np.array([top, left], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(top), int(left)))

    if not candidates:
        return None
    top, left = random.choice(candidates)
    coords = base_coords + np.array([top, left], dtype=int)
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
    return count


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> tuple[list[dict], list[dict], np.ndarray] | None:
    n_non_unit = int(n_slots - n_units)
    if n_non_unit < 2:
        return None

    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        unit_slots: list[dict] = []
        non_unit_slots: list[dict] = []
        used_rot_keys: set[RotKey] = set()

        pair_size = random.choice(CHIRAL_PAIR_SIZE_CHOICES)
        base_mask = _random_shape_mask(pair_size, require_chiral=True, used_rot_keys=used_rot_keys)
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
            mask = _random_shape_mask(size, require_chiral=False, used_rot_keys=used_rot_keys)
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

        return unit_slots, non_unit_slots, layout_mask

    return None


def _make_piece(slot: dict, color: int, k: int, *, apply_rotation: bool) -> dict:
    if slot["area"] == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "aligned_patch": patch,
            "aligned_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    aligned_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    aligned_patch = np.where(aligned_mask, int(color), 0).astype(int)
    if apply_rotation:
        rot = random.randint(0, 3)
        in_mask = np.rot90(aligned_mask, int(rot))
        in_patch = np.rot90(aligned_patch, int(rot))
    else:
        in_mask, in_patch = aligned_mask, aligned_patch

    return {
        "color": int(color),
        "slot": slot,
        "aligned_patch": aligned_patch,
        "aligned_mask": aligned_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        recolored = np.full(p["aligned_patch"].shape, int(frame_color), dtype=int)
        paste(out, recolored, int(top), int(left), mask=p["aligned_mask"], overwrite=True)
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


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = list(map(int, palette[: n_slots - n_units]))
    unit_color = int(palette[n_slots - n_units])

    layout_res = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if layout_res is None:
        return None
    unit_slots, non_unit_slots, layout_mask = layout_res

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(max([slot["area"] * k * k for slot in non_unit_slots + unit_slots], default=0))
    if frame_area <= max_piece_area:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_rotation=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_rotation=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
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
