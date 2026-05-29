# Variant of 898e7135
# Dimension: D3 — 目标对象：框架与洞的结构 (dst)
# Change: hole_filling — 单色矩形框架（1 像素厚边框），内部布局由洞（背景色单元格）和墙壁（框架色单元格）组成，3... → 不规则外形框架（非矩形）

from __future__ import annotations

"""
Design analysis for the D3 variant
==================================

spec_variant:
  core_change:
    from: rectangular 1-cell border around a rectangular hole/wall layout
    to: irregular monochrome frame whose outer silhouette is non-rectangular
  chosen_realization:
    support_mask:
      meaning: coarse "interior layout domain" where holes may live
      families: L, T, plus/cross
      property: tight-cropped, connected, touches all 4 sides of its own bbox
    frame_footprint:
      meaning: the full visible frame region in input/output
      construction: 4-neighbor dilation of pad(support_mask, 1)
      consequence: the outside border is still 1 cell thick but now follows an irregular shape
    holes:
      meaning: bg cells inside support_mask, still 4-connected per slot and pairwise 8-separated
      output_role: each hole is filled by its matching k-scaled piece
    output_crop:
      meaning: crop to the bbox of the irregular frame footprint
      visibility: cells inside that bbox but outside the footprint stay bg=0, making the non-rectangular crop visible

  a_concretization_and_trace:
    small_example:
      support: a 5x5 T shape
      old_logic:
        - take frame bbox
        - interpret every bg cell in the bbox as a slot cell
        - output a solid rectangle of frame color
      new_logic:
        - take frame bbox
        - split bg in the bbox into exterior bg (touches bbox edge) vs enclosed bg (holes)
        - use enclosed bg only as slots
        - scale only the irregular footprint; keep bbox cutouts as bg
      difference:
        - old solver would wrongly treat exterior cutouts as holes
        - old solver would also erase the irregular silhouette by filling the whole bbox

  b_complexity_calibration:
    keep_same:
      - pieces are still kxk-block upscales of hole shapes
      - same D4 matching rule
      - same distractor idea
    constrain_new_freedom:
      - only 3 silhouette families: L, T, plus
      - frame color still monochrome
      - holes still 3-6 regions, areas 1-7, with 1-2 unit holes
    reason:
      - enough variety to make the new silhouette obvious
      - not so much freedom that matching becomes ambiguous or visually noisy

  c_implicit_assumption_audit_of_original_main:
    - "largest non-bg component bbox contains only true layout bg":
        broken_now: irregular bboxes contain exterior cutouts that are bg but are not holes
    - "tight-crop hole bbox and then add a rectangular +1 border in output":
        broken_now: that reconstructs a rectangle, not the irregular frame
    - "frame background color can fill the whole output":
        broken_now: bbox cutouts must remain bg to show the non-rectangular crop
    - "pieces can be placed at (slot_r+1, slot_c+1) inside a rectangle":
        adjusted_now: slots are measured directly inside the full irregular footprint bbox

  d_placement_legality:
    - generator reserves the entire frame bbox when scattering pieces and distractors
    - this prevents pieces from landing inside bbox cutouts, which would visually clash with the frame
    - generator also checks wall connectivity so the frame remains one component and the solver can still pick the largest component reliably

  e_visual_observability:
    - every generated frame has bbox cutouts (footprint area < bbox area)
    - output keeps those cutouts as bg=0, so the D3 change is visible in every example
    - the piece-matching logic stays the same, so the new difficulty is specifically "identify the irregular frame region before filling"

  f_cleanliness:
    - silhouette families are bold and simple
    - pieces stay outside the frame bbox
    - distractors remain singleton pixels in a dedicated color

  g_color_inference:
    - no new color rule is introduced
    - frame keeps one color, pieces keep their own colors, outside cutouts use visible bg=0

  h_new_solver_assumptions_introduced_here:
    - within the frame bbox, all non-edge-reachable bg cells are true holes
    - the frame wall component is connected and larger than any piece component
    - no piece/distractor enters the frame bbox
    generator_guarantees:
      - hole construction + bbox reservation + connectivity checks enforce these assumptions
"""

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


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
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


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(bool), b)


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + f":{r.shape[0]}x{r.shape[1]}".encode())
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + f":{fr.shape[0]}x{fr.shape[1]}".encode())
    return min(variants)


def _edge_reachable_bg(bg_mask: np.ndarray) -> np.ndarray:
    reach = np.zeros_like(bg_mask, dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == bg_mask.shape[0] - 1):
            reach[coords[:, 0], coords[:, 1]] = True
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == bg_mask.shape[1] - 1):
            reach[coords[:, 0], coords[:, 1]] = True
    return reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of an irregular monochrome frame with their matching pieces.

    The input still contains:
    - bg = most common color
    - one large connected frame component
    - scattered k-scaled pieces outside the frame

    The D3 change is that the frame bbox is no longer a full rectangle:
    bg cells in the bbox must be split into
    - exterior cutouts (touch the bbox edge): not slots
    - enclosed holes: true fill targets
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground objects"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    bg_mask = sub == bg
    exterior_bg = _edge_reachable_bg(bg_mask)
    hole = bg_mask & ~exterior_bg
    assert hole.any(), "irregular frame must enclose at least one hole"

    # Full coarse footprint of the target frame: walls plus hole cells, excluding bbox cutouts.
    footprint = ~exterior_bg

    slots = []
    for coords in components_mask(hole, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
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
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
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

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            rp = np.rot90(patch, kk)
            rm = np.rot90(mask, kk)
            base.append((rp, rm))
            base.append((np.fliplr(rp), np.fliplr(rm)))
        uniq = []
        seen = set()
        for vp, vm in base:
            key = vm.tobytes() + f":{vm.shape[0]}x{vm.shape[1]}".encode()
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

    scaled_footprint = np.kron(footprint, np.ones((k, k), dtype=bool))
    out = np.where(scaled_footprint, frame_color, bg).astype(int)
    for s, cand in placements:
        paste(out, cand["vp"], int(s["r0"]) * k, int(s["c0"]) * k, mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

K_CHOICES = [2, 3]
SUPPORT_H_RANGE = (5, 8)
SUPPORT_W_RANGE = (5, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
FRAME_FAMILIES = ["L", "T", "PLUS"]

HOLE_DENSITY_RANGE = (0.20, 0.48)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 35
GROW_SEED_TRIES = 10
MIN_VISIBLE_CUTOUTS = 3

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 2
AREA_BUFFER_CONST = 24

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

OUTPUT_MAX_DIM = 30
GENERATE_MAX_RETRIES = 500


def _make_support_shape() -> tuple[str, np.ndarray] | None:
    family = random.choice(FRAME_FAMILIES)
    h = random.randint(*SUPPORT_H_RANGE)
    w = random.randint(*SUPPORT_W_RANGE)
    max_t = min(3, h - 2, w - 2)
    if max_t < 2:
        return None
    t = random.randint(2, max_t)

    mask = np.zeros((h, w), dtype=bool)
    if family == "L":
        mask[:, :t] = True
        mask[-t:, :] = True
    elif family == "T":
        if w - t < 2:
            return None
        stem_c = random.randint(1, w - t - 1)
        mask[:t, :] = True
        mask[:, stem_c : stem_c + t] = True
    else:
        if h - t < 2 or w - t < 2:
            return None
        stem_r = random.randint(1, h - t - 1)
        stem_c = random.randint(1, w - t - 1)
        mask[:, stem_c : stem_c + t] = True
        mask[stem_r : stem_r + t, :] = True

    mask = np.rot90(mask, random.randint(0, 3))
    mask = _trim_mask(mask)
    return family, mask


def _grow_slot(support: np.ndarray, forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    free_cells = np.argwhere(support & ~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_list)
    for start in free_list[:GROW_SEED_TRIES]:
        chosen = {start}
        frontier = {
            (start[0] - 1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1),
        }
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < support.shape[0]
                and 0 <= p[1] < support.shape[1]
                and bool(support[p[0], p[1]])
                and not bool(forbidden[p[0], p[1]])
                and p not in chosen
            ]
            if not cands:
                break
            pick = random.choice(cands)
            chosen.add(pick)
            frontier.discard(pick)
            frontier.update(
                {
                    (pick[0] - 1, pick[1]),
                    (pick[0] + 1, pick[1]),
                    (pick[0], pick[1] - 1),
                    (pick[0], pick[1] + 1),
                }
            )
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden, dtype=bool)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_slots(support: np.ndarray, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = ~support.copy()
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(support, forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for _ in range(n_slots - n_units):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(support, forbidden, size)
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

    density = sum(int(s["area"]) for s in slots) / float(int(support.sum()))
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), color, dtype=int)
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
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict],
    footprint: np.ndarray,
    frame_color: int,
    bg: int,
    k: int,
) -> np.ndarray:
    scaled_footprint = np.kron(footprint, np.ones((k, k), dtype=bool))
    out = np.where(scaled_footprint, frame_color, bg).astype(int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    frame_img: np.ndarray,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
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
    occupancy = np.zeros((canvas_h, canvas_w), dtype=int)

    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    fr = random.randint(1, fr_max)
    fc = random.randint(1, fc_max)
    paste(canvas, frame_img, fr, fc)
    occupancy[fr : fr + frame_h, fc : fc + frame_w] = frame_color

    for p in sorted(pieces, key=lambda q: -int(q["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                occupancy,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)
        paste(occupancy, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if occupancy[rr, cc] != 0:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(occupancy[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = distractor_color
        occupancy[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt(return_meta: bool = False):
    bg = 0
    k = random.choice(K_CHOICES)

    made = _make_support_shape()
    if made is None:
        return None
    family, support = made

    support_pad = np.pad(support, 1, constant_values=False)
    footprint = _dilate4(support_pad)
    if footprint.size - int(footprint.sum()) < MIN_VISIBLE_CUTOUTS:
        return None
    if footprint.shape[0] * k > OUTPUT_MAX_DIM or footprint.shape[1] * k > OUTPUT_MAX_DIM:
        return None

    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_slots(support, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    hole_support = np.zeros_like(support, dtype=bool)
    for s in slots:
        hole_support[s["coords"][:, 0], s["coords"][:, 1]] = True
    hole_pad = np.pad(hole_support, 1, constant_values=False)
    wall_mask = footprint & ~hole_pad

    if len(components_mask(wall_mask, connectivity=4)) != 1:
        return None

    max_piece_area = max(int(s["area"]) * k * k for s in slots)
    if int(wall_mask.sum()) <= max_piece_area:
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    expected_out = _build_expected_output(pieces, footprint, frame_color, bg, k)

    frame_img = np.where(wall_mask, frame_color, bg).astype(int)
    canvas = _build_input_canvas(pieces, frame_img, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    meta = {
        "family": family,
        "k": int(k),
        "slots": int(n_slots),
        "units": int(n_units),
        "cutout_cells": int(footprint.size - int(footprint.sum())),
        "output_shape": tuple(map(int, expected_out.shape)),
    }
    return (canvas, meta) if return_meta else canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")


def _generate_input_with_meta() -> tuple[np.ndarray, dict]:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt(return_meta=True)
        if res is not None:
            return res
    raise RuntimeError(f"_generate_input_with_meta: failed to generate after {GENERATE_MAX_RETRIES} attempts")
