# Combo variant of 898e7135 (index 8)
# combo_spec: D1/1+D2/0
# Name: 多色旋转匹配

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
    first_free_location,
    get_bg,
    paste,
    random_free_location,
)


"""
Design analysis for combo D1/1 + D2/0
=====================================

spec_variant:
  preserved_structure:
    grid: still 15-30 by 15-30 with background 0 as the most common color
    frame: still the largest foreground component, a monochrome rectangular border
      around an inner hole/wall layout
    layout: still a 4-12 by 4-8 inner grid whose hole components are the slots
    placement: frame and scattered pieces remain 8-separated; distractors remain
      isolated single pixels in a dedicated color
    output: still the tight-cropped frame layout, scaled by k, with the matched
      pieces pasted into the corresponding slots

  fused_change:
    old_rule:
      pieces were monochrome block-scaled sprites and matching allowed any D4
      transform, so rotation and flip were both legal
    new_rule:
      every real piece is multicolor, using 2-3 colors across its occupied block
      cells, and matching is rotation-only
    practical consequence:
      the solver must recover k from monochrome k-by-k color atoms instead of
      from a smallest solid-square unit piece, and it must preserve the handed
      color arrangement because mirror flips are no longer legal

  concrete_behavior_trace:
    miniature example:
      slot_A: a 3-cell L shape
      slot_B: the left-right mirror of slot_A
      piece_A colors: red at the elbow, blue on one arm, yellow on the other
      piece_B colors: green at the elbow, orange on one arm, teal on the other
    original task behavior:
      a monochrome solver with flips could treat the two shapes as equivalent up
      to reflection; with no internal color structure, that ambiguity is often invisible
    fused behavior:
      only rotations are legal, so piece_A can fill only slot_A and piece_B can
      fill only slot_B; after placement, their distinct multicolor layouts are retained
    visible difference:
      if a flip-enabled solver swapped the pair, the output colors in those two
      slots would visibly change, so the fusion is observable from pixels alone

  complexity_calibration:
    why not keep the old unit slots:
      the original k inference relied on a 1-cell slot becoming a solid k-by-k
      square, but a genuine 2-3-color piece cannot be represented cleanly by one
      block cell without introducing sub-block striping that obscures the block model
    adaptation:
      all slots are non-unit and every same-color region inside every piece is
      exactly one solid k-by-k square; k is inferred as the gcd of those square sizes
    ambiguity control:
      non-unit slot masks are unique up to rotation, and at least one chiral
      mirror pair is forced into every example
    degeneracy control:
      the generator places that mirror pair in a canvas order that makes a
      flip-enabled greedy matcher disagree, and examples are rejected unless the
      flip-enabled reference output differs from the intended rotation-only output

  implicit_assumption_audit_for_main:
    - find_components(..., monochromatic=False):
      assumption: each multicolor piece is still a single 4-connected component
      generator guarantee: adjacent occupied blocks always touch by edges inside
      the piece, even when their colors differ
    - largest component is the frame:
      assumption: no piece exceeds the frame area
      generator guarantee: examples are rejected unless frame_area > max_piece_area
    - tight crop of holes:
      assumption: the union of slot holes touches all four sides of the hole bbox
      generator guarantee: layouts are rejected unless holes hit all four borders
    - k inference:
      old assumption: a smallest solid square piece exists
      fused status: invalid; replaced by the assumption that each monochrome color
      atom is a solid k-by-k square
      generator guarantee: adjacent occupied slot cells never share a color, so
      each monochrome component is exactly one block atom
    - candidate enumeration:
      old assumption: flips are legal
      fused status: invalid for main(); only the generator's reference check uses flips
    - first exact match wins:
      assumption: for each slot there is exactly one unused piece whose block mask
      matches under allowed transforms
      generator guarantee: rotation classes are unique, except the forced mirror pair
      which only becomes ambiguous when flips are allowed
    - output determinacy:
      assumption: no slot shape has non-trivial rotational self-symmetry that would
      allow multiple color-distinct rotations to fit the same slot
      generator guarantee: every slot mask is strict-orientation, so non-zero
      rotations never reproduce the same tight mask

  placement_legality:
    layout slots are placed with 8-neighbor exclusion, so hole components never
    touch each other, even diagonally
    pieces are scattered outside the frame with padding 1 under 8-connectivity,
    so they never merge with each other or with the frame
    distractors are singleton pixels placed only in empty 8-isolated cells

  visual_observability:
    the change is not hidden in numeric color ids:
    at least one sample feature is a left-handed/right-handed shape pair and every
    real piece uses 2-3 visible colors
    because those colors are kept in the output, humans can see that the task is
    "match by rotated shape, then preserve the multicolor sprite"

  cleanliness_and_inferability:
    there is no overlap layering beyond the normal paste into output slots
    color count stays controlled: one frame color, one distractor color, and a
    small reusable piece palette
    the generator rejects examples where the flip-enabled reference would produce
    the same output, so the fusion does not silently collapse back to the original rule
"""


RotKey = tuple[tuple[int, int], bytes]


# --- shared solver helpers ---------------------------------------------------


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _extract_slots(layout: np.ndarray) -> list[dict]:
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
                "rot_key": _rotation_shape_key(sm),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    return slots


def _extract_pieces(g: np.ndarray, comps: list[dict], frame_comp: dict) -> list[dict]:
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
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch, "bbox": comp["bbox"]})
    return pieces


def _infer_k_from_pieces(pieces: list[dict]) -> int:
    k = 0
    found = False
    for piece in pieces:
        patch = piece["patch"]
        mask = piece["mask"]
        inside_colors = sorted(set(map(int, patch[mask])))
        for col in inside_colors:
            mono = (patch == int(col)) & mask
            for coords in components_mask(mono, connectivity=4):
                rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
                h = int(rr1 - rr0 + 1)
                w = int(cc1 - cc0 + 1)
                area = int(coords.shape[0])
                assert h == w and area == h * w, "monochrome atoms must be solid squares"
                k = int(h) if k == 0 else int(gcd(k, int(h)))
                found = True
    assert found and k >= 2, "failed to infer multicolor block scale"
    return int(k)


def _downsample_piece(patch: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    block_colors = np.zeros((out_h, out_w), dtype=int)
    block_mask = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            rs = int(r * k)
            cs = int(c * k)
            sub_mask = mask[rs : rs + k, cs : cs + k]
            if bool(sub_mask.all()):
                vals = np.unique(patch[rs : rs + k, cs : cs + k][sub_mask])
                if vals.size != 1:
                    return None
                block_mask[r, c] = True
                block_colors[r, c] = int(vals[0])
            elif bool(sub_mask.any()):
                return None
    return block_colors, block_mask


def _piece_variants(
    patch: np.ndarray, mask: np.ndarray, allow_flip: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    variants = []
    seen: set[tuple[tuple[int, int], bytes, bytes]] = set()
    for kk in range(4):
        vp = np.rot90(patch, int(kk))
        vm = np.rot90(mask, int(kk))
        key = (tuple(map(int, vm.shape)), vm.tobytes(), vp.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append((vp, vm))
        if allow_flip:
            fp = np.fliplr(vp)
            fm = np.fliplr(vm)
            fkey = (tuple(map(int, fm.shape)), fm.tobytes(), fp.tobytes())
            if fkey not in seen:
                seen.add(fkey)
                variants.append((fp, fm))
    return variants


def _solve(grid: np.ndarray, *, allow_flip: bool) -> np.ndarray:
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
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = _extract_slots(layout)
    assert slots, "frame must contain at least one slot"

    pieces = _extract_pieces(g, comps, frame_comp)
    assert pieces, "input must contain non-frame pieces"
    k = _infer_k_from_pieces(pieces)

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _piece_variants(piece["patch"], piece["mask"], allow_flip=allow_flip):
            down = _downsample_piece(vp, vm, k)
            if down is None:
                continue
            _block_colors, block_mask = down
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": block_mask,
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
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and refill its slots with matching multicolor pieces.

    Input structure:
    - Background 0 is the most common color. The largest foreground component is a
      monochrome frame around a hole/wall layout.
    - Every outside piece is a connected multicolor sprite made of solid k-by-k
      color blocks; singleton pixels are distractors, not pieces.

    Rule:
    - Read the hole components inside the frame as block-level slots.
    - Match each scattered piece to a slot by block-mask shape using rotations
      only; mirror flips are forbidden.
    - Paste the correctly rotated multicolor piece into the scaled slot position
      and return the cropped enlarged frame.

    Key insight:
    - The occupied mask still reveals the slot shape after block downsampling, but
      k must now be inferred from the monochrome block atoms, and forbidding flips
      preserves the handed color layout of each sprite.
    """
    return _solve(grid, allow_flip=False)


# --- generator ---------------------------------------------------------------


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 5)
NON_UNIT_SIZE_CHOICES = [3, 4, 5, 6]
CHIRAL_PAIR_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 80
LAYOUT_RETRIES = 140
COLOR_RETRIES = 60

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _slot_from_coords(coords: np.ndarray, *, slot_id: int, pair_tag: int = -1) -> dict:
    mask = _tight_mask_from_coords(coords)
    r0, c0, _r1, _c1 = map(int, bbox_coords(coords))
    return {
        "id": int(slot_id),
        "pair_tag": int(pair_tag),
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


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


def _is_strict_orientation(mask: np.ndarray) -> bool:
    for kk in (1, 2, 3):
        rot = np.rot90(mask, int(kk))
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return False
    return True


def _random_shape_mask(target_size: int, *, used_rot_keys: set[RotKey]) -> np.ndarray | None:
    max_side = min(5, max(3, int(target_size)))
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
        if not _is_chiral(mask):
            continue
        if not _is_strict_orientation(mask):
            continue
        if _rotation_shape_key(mask) in used_rot_keys:
            continue
        return mask
    return None


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None
    base_coords = np.argwhere(mask)
    candidates: list[tuple[int, int]] = []
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


def _gen_layout(h: int, w: int, n_slots: int) -> tuple[list[dict], np.ndarray, tuple[int, int]] | None:
    if n_slots < 3:
        return None
    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        slots: list[dict] = []
        used_rot_keys: set[RotKey] = set()
        next_slot_id = 0

        pair_size = random.choice(CHIRAL_PAIR_SIZE_CHOICES)
        base_mask = _random_shape_mask(pair_size, used_rot_keys=used_rot_keys)
        if base_mask is None:
            continue
        mirror_mask = np.fliplr(base_mask)
        if not _is_strict_orientation(mirror_mask):
            continue
        if _rotation_shape_key(base_mask) == _rotation_shape_key(mirror_mask):
            continue

        pair_ids: list[int] = []
        for pair_tag, mask in enumerate(random.sample([base_mask, mirror_mask], 2)):
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                break
            slot = _slot_from_coords(coords, slot_id=next_slot_id, pair_tag=pair_tag)
            used_rot_keys.add(slot["rot_key"])
            slots.append(slot)
            pair_ids.append(int(next_slot_id))
            next_slot_id += 1
        if len(pair_ids) != 2:
            continue

        ok = True
        rem_sizes = [random.choice(NON_UNIT_SIZE_CHOICES) for _ in range(n_slots - 2)]
        rem_sizes.sort(reverse=True)
        for size in rem_sizes:
            mask = _random_shape_mask(size, used_rot_keys=used_rot_keys)
            if mask is None:
                ok = False
                break
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords, slot_id=next_slot_id)
            used_rot_keys.add(slot["rot_key"])
            slots.append(slot)
            next_slot_id += 1
        if not ok:
            continue

        layout_mask = np.zeros((h, w), dtype=bool)
        for slot in slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
        density = float(layout_mask.sum()) / float(h * w)
        lo, hi = HOLE_DENSITY_RANGE
        if not (lo <= density <= hi):
            continue
        if not _layout_touches_all_borders(layout_mask):
            continue
        return slots, layout_mask, (int(pair_ids[0]), int(pair_ids[1]))
    return None


def _block_adjacency(mask: np.ndarray, coords: np.ndarray) -> dict[int, set[int]]:
    pos_to_idx = {tuple(map(int, coord)): i for i, coord in enumerate(coords)}
    graph = {i: set() for i in range(len(coords))}
    for i, (r, c) in enumerate(coords):
        for nr, nc in ((int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))):
            j = pos_to_idx.get((nr, nc))
            if j is not None:
                graph[i].add(int(j))
    return graph


def _assign_block_colors(mask: np.ndarray, colors: list[int]) -> np.ndarray | None:
    coords = np.argwhere(mask)
    order = list(range(len(coords)))
    graph = _block_adjacency(mask, coords)
    order.sort(key=lambda i: (-len(graph[i]), int(coords[i][0]), int(coords[i][1])))
    colors = [int(c) for c in colors]
    assign = [-1] * len(coords)
    used = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return used == set(colors)
        idx = order[pos]
        blocked = {assign[j] for j in graph[idx] if assign[j] != -1}
        options = [c for c in colors if c not in blocked]
        random.shuffle(options)
        remain = len(order) - pos - 1
        for color in options:
            newly_used = color not in used
            if newly_used and remain < len(set(colors) - (used | {color})):
                continue
            assign[idx] = int(color)
            if newly_used:
                used.add(int(color))
            if backtrack(pos + 1):
                return True
            if newly_used:
                used.remove(int(color))
            assign[idx] = -1
        return False

    if not backtrack(0):
        return None

    out = np.zeros(mask.shape, dtype=int)
    for idx, (r, c) in enumerate(coords):
        out[int(r), int(c)] = int(assign[idx])
    return out


def _build_piece(
    slot: dict,
    k: int,
    block_colors: np.ndarray,
    *,
    allow_rotation: bool,
    piece_name: str,
) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.kron(block_colors, np.ones((k, k), dtype=int))
    if allow_rotation:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, int(rot))
        in_patch = np.rot90(out_patch, int(rot))
    else:
        in_mask, in_patch = out_mask, out_patch
    return {
        "name": piece_name,
        "slot": slot,
        "block_colors": block_colors,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    distractor_color: int,
    *,
    forced_first_piece_names: tuple[str, str],
) -> np.ndarray | None:
    frame_h = int(layout_h + 2)
    frame_w = int(layout_w + 2)
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
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
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    piece_by_name = {piece["name"]: piece for piece in pieces}
    ordered_names = list(forced_first_piece_names) + [
        piece["name"]
        for piece in sorted(pieces, key=lambda item: (-int(item["in_mask"].sum()), item["name"]))
        if piece["name"] not in set(forced_first_piece_names)
    ]

    for name in ordered_names:
        piece = piece_by_name[name]
        loc = first_free_location(
            canvas,
            piece["in_patch"],
            transparent=0,
            border=0,
            padding=1,
            padding_connectivity=8,
        )
        if loc is None:
            try:
                loc = random_free_location(
                    canvas,
                    piece["in_patch"],
                    transparent=0,
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                return None
        paste(canvas, piece["in_patch"], int(loc[0]), int(loc[1]), mask=piece["in_mask"], overwrite=True)

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


def _gen_pair_colorings(slot_a: dict, slot_b: dict, piece_palette: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(piece_palette) < 4:
        return None
    pair_colors_a = random.sample(piece_palette, 2)
    remaining = [c for c in piece_palette if c not in pair_colors_a]
    pair_colors_b = random.sample(remaining, 2)
    for _ in range(COLOR_RETRIES):
        grid_a = _assign_block_colors(slot_a["mask"], pair_colors_a)
        grid_b = _assign_block_colors(slot_b["mask"], pair_colors_b)
        if grid_a is None or grid_b is None:
            continue
        return grid_a, grid_b
    return None


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    piece_palette_size = random.choice([5, 6, min(7, len(palette))])
    piece_palette = [int(c) for c in palette[:piece_palette_size]]
    if len(piece_palette) < 5:
        return None

    layout_res = _gen_layout(layout_h, layout_w, n_slots)
    if layout_res is None:
        return None
    slots, layout_mask, pair_ids = layout_res

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(max(slot["area"] * k * k for slot in slots))
    if frame_area <= max_piece_area:
        return None

    slot_by_id = {slot["id"]: slot for slot in slots}
    pair_slot_a = slot_by_id[int(pair_ids[0])]
    pair_slot_b = slot_by_id[int(pair_ids[1])]
    pair_colors = _gen_pair_colorings(pair_slot_a, pair_slot_b, piece_palette)
    if pair_colors is None:
        return None
    pair_grid_a, pair_grid_b = pair_colors

    pieces: list[dict] = []
    pieces.append(_build_piece(pair_slot_a, k, pair_grid_a, allow_rotation=True, piece_name="pair_a"))
    pieces.append(_build_piece(pair_slot_b, k, pair_grid_b, allow_rotation=True, piece_name="pair_b"))

    used_slot_ids = {int(pair_slot_a["id"]), int(pair_slot_b["id"])}
    for slot in sorted(slots, key=lambda s: int(s["id"])):
        if int(slot["id"]) in used_slot_ids:
            continue
        ncols = random.randint(2, min(3, int(slot["area"])))
        block_colors = None
        for _ in range(COLOR_RETRIES):
            chosen = random.sample(piece_palette, ncols)
            block_colors = _assign_block_colors(slot["mask"], chosen)
            if block_colors is not None:
                break
        if block_colors is None:
            return None
        pieces.append(
            _build_piece(
                slot,
                k,
                block_colors,
                allow_rotation=True,
                piece_name=f"piece_{int(slot['id'])}",
            )
        )

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    slot_order = sorted(slots, key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    pair_names_by_slot = {
        int(pair_slot_a["id"]): "pair_a",
        int(pair_slot_b["id"]): "pair_b",
    }
    ordered_pair_slot_ids = [int(slot["id"]) for slot in slot_order if int(slot["id"]) in pair_names_by_slot]
    if len(ordered_pair_slot_ids) != 2:
        return None
    forced_first_piece_names = (
        pair_names_by_slot[int(ordered_pair_slot_ids[1])],
        pair_names_by_slot[int(ordered_pair_slot_ids[0])],
    )

    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        layout_h,
        layout_w,
        k,
        frame_color,
        distractor_color,
        forced_first_piece_names=forced_first_piece_names,
    )
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        flip_enabled = _solve(canvas, allow_flip=True)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    if np.array_equal(flip_enabled, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
