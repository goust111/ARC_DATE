# Combo variant of 898e7135 (index 30)
# combo_spec: D1/1+D13/2
# Name: 旋转匹配后变换

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


"""
Design analysis for combo D1/1 + D13/2
======================================

a) Concrete behavior and process trace
--------------------------------------
- Preserved scaffold from the base task:
  - A monochrome rectangular frame encloses several background holes.
  - Each outside piece is still a k-by-k block upscaling of one hole shape.
  - Output is still the cropped frame layout scaled by k and filled.
- Fusion rule in pixels:
  1. Every occupied k-by-k block inside a piece contains one shared accent pixel
     at its top-left corner in the canonical orientation.
  2. Input pieces may be globally rotated, but never mirrored.
  3. Matching is done only by block-shape under the 4 rotations. This makes
     chiral pairs such as L and mirrored-L genuinely different.
  4. After a piece is matched and reoriented to the slot, a second transform is
     applied: slots whose centers lie in the left half of the layout have every
     occupied block mirrored horizontally, so the accent moves from top-left to
     top-right. Right-half slots keep the canonical block pattern.
- Small trace:
  - Suppose slot A is a left-handed 3-cell L on the left half, and slot B is
    its mirrored partner on the right half.
  - Input piece A appears rotated 90 degrees, input piece B rotated 270 degrees.
  - Stage 1 recovers their canonical orientations using rotation-only matching;
    A cannot match B because their chiral silhouettes differ.
  - Stage 2 inspects matched slot positions: A is in the left half so each
    occupied block is locally mirrored, B is in the right half so it is not.
  - The final outputs differ from the original task in two visible ways:
    chirality matters during matching, and block-corner accents depend on slot
    position after matching.

b) Complexity calibration
-------------------------
- If the generator only removed flips but never produced mirror-only shape pairs,
  many examples would collapse back to the old task. Therefore each sample
  forces at least one non-unit chiral mirror pair.
- If the second step only recolored monochrome pieces, the change would be
  invisible. The shared accent pixel gives a low-entropy but observable cue.
- The post-match transform has only two branches, left-half vs right-half, and
  each sample is forced to contain both. This keeps the rule learnable from
  2-4 examples without exploding the case count.
- Remaining non-unit slots keep unique rotation classes so the first-stage
  matching stays unambiguous even after adding the mirror pair.

c) Implicit assumption audit for main()
---------------------------------------
- Component detection:
  - The original "largest foreground component is the frame" heuristic becomes
    weaker because pieces are now multicolor. The solver therefore identifies a
    frame-like monochrome rectangle explicitly.
- Tight hole crop:
  - The solver still assumes the hole layout touches all four borders of its
    tight crop. The generator rejects layouts that fail this.
- k inference:
  - The solver assumes at least one piece is a full square mask so k can be
    inferred directly, with a gcd fallback for robustness. The generator always
    includes 1-2 unit slots, whose pieces are full k-by-k masks.
- Candidate enumeration:
  - The original task allowed dihedral matching. That assumption is invalid
    here, so the solver enumerates rotations only.
  - The original task could deduplicate variants by mask alone. That is invalid
    once block-internal markers exist, so the solver preserves patch-distinct
    rotations and explicitly tests for the canonical accent orientation.
- Direct placement:
  - The original matched orientation was already final. In the combo it is only
    an intermediate state; the solver must apply a second slot-driven transform
    before pasting.
- Match uniqueness:
  - The solver assumes each slot has exactly one unused piece whose canonical
    rotated block mask matches. The generator guarantees distinct rotation-class
    masks except for the forced chiral pair, whose two members remain distinct
    because flips are forbidden.

d) Placement legality
---------------------
- Layout slots are placed with 8-neighbor exclusion, so holes never touch.
- The forced chiral pair is placed as separate 4-connected regions and is
  rejected unless the two members land on opposite halves of the layout.
- The frame is placed first on the background canvas, then pieces are packed
  outside it using `random_free_location(..., padding=1, padding_connectivity=8)`.
- Distractors remain isolated singleton pixels in one dedicated color.
- Samples are rejected if they exceed bounds, violate density, fail border-touch,
  fail background-majority, or make the frame no longer the largest meaningful
  component.

e) Visual observability
-----------------------
- Humans can see the chirality constraint because every sample contains a mirror
  pair that is not rotation-equivalent.
- Humans can see the second stage because the same shared accent color appears in
  the top-left corner of occupied blocks on the right side, but top-right on the
  left side after placement.
- No branch depends on hidden numeric properties such as color id ordering or
  generation order; all control signals are visible in geometry or position.

f) Visual cleanliness
---------------------
- Pieces use only two visible colors: a piece-specific base color and one shared
  accent color. This is enough to show the local transform without clutter.
- The local mirror acts inside each occupied block, so silhouettes stay crisp and
  the slot shapes remain easy to compare.
- Distractors stay isolated and never overlap meaningful structures.

g) Change inferability
----------------------
- There is no arbitrary color remapping table to memorize.
- The accent color is shared across pieces, so the learner sees repeated
  evidence that the meaningful change is geometric.
- Because the forced mirror pair and the forced left/right split co-occur in
  every sample, the dataset does not silently degenerate into "rotation-only"
  or "post-transform-only" behavior.

9) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each piece has exactly one rotation where every occupied k-by-k
  block has one accent pixel at the block's top-left corner.
  Guarantee: generator builds that canonical form first and then applies only a
  global rotation.
- Assumption: samples contain both left-half and right-half slots.
  Guarantee: the forced mirror-pair members must land on opposite halves.
- Assumption: the rotation-only stage is behaviorally necessary.
  Guarantee: every sample includes at least one chiral mirror pair.
- Assumption: the post-match transform is behaviorally necessary.
  Guarantee: every sample includes both half-branches, so some placed pieces
  keep top-left accents and some switch to top-right accents.
"""


RotKey = tuple[tuple[int, int], bytes]


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
CHIRAL_PAIR_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 60
LAYOUT_RETRIES = 140

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


# --- shared helpers -----------------------------------------------------------

def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
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
    variants = []
    seen = set()
    for kk in range(4):
        vp = np.rot90(patch, int(kk))
        vm = np.rot90(mask, int(kk))
        key = (tuple(map(int, vp.shape)), vp.tobytes(), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append((vp, vm))
    return variants


def _slot_is_left(coords: np.ndarray, layout_w: int) -> bool:
    center_c = float(coords[:, 1].mean())
    return center_c < (float(layout_w) - 1.0) / 2.0


def _slot_is_centered(coords: np.ndarray, layout_w: int) -> bool:
    center_c = float(coords[:, 1].mean())
    return abs(center_c - (float(layout_w) - 1.0) / 2.0) < 1e-9


def _decorate_scaled_piece(block_mask: np.ndarray, color: int, accent_color: int, k: int) -> np.ndarray:
    h, w = map(int, block_mask.shape)
    patch = np.zeros((h * k, w * k), dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(block_mask[r, c]):
                continue
            rr = r * k
            cc = c * k
            patch[rr : rr + k, cc : cc + k] = int(color)
            patch[rr, cc] = int(accent_color)
    return patch


def _is_canonical_marker_orientation(patch: np.ndarray, mask: np.ndarray, k: int) -> bool:
    vals, counts = np.unique(patch[mask], return_counts=True)
    if len(vals) != 2:
        return False
    accent = int(vals[np.argmin(counts)])
    block_mask = _downsample_blocks(mask, int(k))
    if block_mask is None:
        return False
    for r in range(block_mask.shape[0]):
        for c in range(block_mask.shape[1]):
            if not bool(block_mask[r, c]):
                continue
            block = patch[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if int(block[0, 0]) != accent:
                return False
            if int(np.count_nonzero(block == accent)) != 1:
                return False
    return True


def _apply_left_slot_transform(patch: np.ndarray, mask: np.ndarray, k: int, *, is_left: bool) -> np.ndarray:
    if not is_left:
        return patch.copy()
    out = patch.copy()
    block_mask = _downsample_blocks(mask, int(k))
    assert block_mask is not None
    for r in range(block_mask.shape[0]):
        for c in range(block_mask.shape[1]):
            if not bool(block_mask[r, c]):
                continue
            rr = r * k
            cc = c * k
            out[rr : rr + k, cc : cc + k] = np.fliplr(out[rr : rr + k, cc : cc + k])
    return out


def _find_frame_component(g: np.ndarray, comps: list[dict], bg: int) -> dict:
    frame_like = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        cols = {int(g[int(r), int(c)]) for r, c in coords}
        if len(cols) != 1:
            continue
        color = next(iter(cols))
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if not np.any(sub == int(bg)):
            continue
        if not bool(np.all(sub[0, :] == color) and np.all(sub[-1, :] == color)):
            continue
        if not bool(np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
            continue
        frame_like.append(comp)
    assert frame_like, "could not identify frame"
    return max(frame_like, key=lambda d: int(d["area"]))


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    if k is not None:
        return int(k)
    dims = []
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        dims.extend([int(h), int(w)])
    gg = 0
    for d in dims:
        gg = gcd(int(gg), int(d))
    return int(max(gg, 1))


# --- layout generation -------------------------------------------------------

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

    base_coords = np.argwhere(mask)
    candidates = []
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


def _annotate_slots(slots: list[dict], layout_w: int) -> bool:
    left_count = 0
    right_count = 0
    for slot in slots:
        coords = slot["coords"]
        if _slot_is_centered(coords, int(layout_w)):
            return False
        is_left = _slot_is_left(coords, int(layout_w))
        slot["is_left"] = bool(is_left)
        if is_left:
            left_count += 1
        else:
            right_count += 1
    return left_count > 0 and right_count > 0


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> tuple[list[dict], list[dict], np.ndarray] | None:
    n_non_unit = int(n_slots - n_units)
    if n_non_unit < 2:
        return None

    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        used_rot_keys: set[RotKey] = set()
        unit_slots: list[dict] = []
        non_unit_slots: list[dict] = []

        pair_size = random.choice(CHIRAL_PAIR_SIZE_CHOICES)
        base_mask = _random_shape_mask(pair_size, require_chiral=True, used_rot_keys=used_rot_keys)
        if base_mask is None:
            continue
        mirror_mask = np.fliplr(base_mask)
        if _rotation_shape_key(base_mask) == _rotation_shape_key(mirror_mask):
            continue

        ok = True
        for mask in random.sample([base_mask, mirror_mask], 2):
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

        slots = unit_slots + non_unit_slots
        layout_mask = np.zeros((h, w), dtype=bool)
        for slot in slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

        density = float(layout_mask.sum()) / float(h * w)
        lo, hi = HOLE_DENSITY_RANGE
        if not (lo <= density <= hi):
            continue
        if not _layout_touches_all_borders(layout_mask):
            continue
        if not _annotate_slots(slots, int(w)):
            continue
        if bool(non_unit_slots[0]["is_left"]) == bool(non_unit_slots[1]["is_left"]):
            continue

        return unit_slots, non_unit_slots, layout_mask

    return None


# --- piece construction ------------------------------------------------------

def _make_piece(slot: dict, color: int, accent_color: int, k: int) -> dict:
    aligned_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    aligned_patch = _decorate_scaled_piece(slot["mask"], int(color), int(accent_color), int(k))

    rot = random.randint(0, 3)
    in_mask = np.rot90(aligned_mask, int(rot))
    in_patch = np.rot90(aligned_patch, int(rot))

    return {
        "color": int(color),
        "slot": slot,
        "aligned_patch": aligned_patch,
        "aligned_mask": aligned_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


# --- output assembly ---------------------------------------------------------

def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        placed = _apply_left_slot_transform(
            piece["aligned_patch"],
            piece["aligned_mask"],
            int(k),
            is_left=bool(piece["slot"]["is_left"]),
        )
        paste(out, placed, int(top), int(left), mask=piece["aligned_mask"], overwrite=True)
    return out


# --- input canvas assembly ---------------------------------------------------

def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
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
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
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
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

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


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame layout, identify each rotated sprite, then apply a second slot-driven transform before filling.

    Input structure:
    - Color 0 is the global background. One monochrome rectangular frame encloses several 4-connected hole regions.
    - Outside the frame sit separated multicolor sprites built from k×k solid blocks; every occupied block carries one accent pixel, and k is recoverable from the smallest square-mask piece.

    Rule:
    - Read the hole regions as slots and the outside sprites as candidate pieces.
    - Match each piece to a slot using only block-shape equality under the four rotations; mirrored silhouettes are different and do not match.
    - After a piece is matched and reoriented, locally mirror every occupied k×k block iff the destination slot lies in the left half of the frame layout, then paste the result into the k-scaled frame.

    Key insight:
    - The puzzle splits into two transformation stages: first undo the piece's global rotation to recover the canonical block layout, then use the matched slot position to decide whether the block-internal accent corner must flip left-to-right.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame and pieces"
    frame_comp = _find_frame_component(g, comps, bg)
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
                "is_left": _slot_is_left(coords, int(layout.shape[1])),
            }
        )
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

    k = _infer_k(pieces)

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _rotation_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, int(k))
            if um is None:
                continue
            if not _is_canonical_marker_orientation(vp, vm, int(k)):
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": um,
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
        assert found is not None, "every slot must match a piece under rotations only"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        placed = _apply_left_slot_transform(cand["vp"], cand["vm"], int(k), is_left=bool(slot["is_left"]))
        paste(out, placed, int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- generator ----------------------------------------------------------------

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
    accent_color = int(palette.pop())
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
    max_piece_area = int(
        max([slot["area"] * k * k for slot in non_unit_slots + unit_slots], default=0)
    )
    if frame_area <= max_piece_area:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], accent_color, k) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, accent_color, k) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        layout_h,
        layout_w,
        k,
        frame_color,
        distractor_color,
    )
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
