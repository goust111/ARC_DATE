# Variant of 898e7135
# Dimension: D15 — 洞形状的唯一性约束 (meta)
# Change: 非单元洞的形状在二面体群下互不相同（canonical shape key 去重），单元洞（面积=1）可重复... → 洞形状互为旋转变体（需要精确旋转角度匹配）

from __future__ import annotations

"""
Design analysis for the D15 rotation-family variant.

a) Concrete behavior and flow impact
   - The frame and output format stay the same: a monochrome frame encloses a hole
     layout, and the output is the cropped layout scaled by k with frame-color
     background.
   - The changed part is the hole-set itself. In the original task, each non-unit
     hole had a D4-unique shape, so any matching piece could be rotated/flipped to
     fit. Here, all non-unit holes come from one base polyomino and appear as 2-4
     different rotations of that same shape.
   - The outside pieces keep those exact rotations. Therefore the solver must match
     by exact block-level orientation, not by "any dihedral variant".
   - Small example: suppose the base hole is an L-tetromino. Original logic would
     treat a red L-piece as usable for every rotated L-hole by rotating it during
     matching. New logic forbids that collapse: if the frame contains red/blue/green
     holes that are 0/90/180 degree versions of the same L, each scattered piece is
     already in one exact orientation and may only fill the hole with the same exact
     orientation.

b) Complexity calibration
   - Using one shared rotation family makes the D15 change visible: several holes are
     clearly "the same shape, different angle".
   - To avoid ambiguity, the base polyomino is required to have 4 distinct rotations.
     This removes symmetric shapes whose orientation would be visually invisible.
   - We keep 1-2 unit holes, but require 2-4 non-unit holes so the rotation-family
     pattern appears repeatedly within each example.

c) Hidden assumption audit against original main()
   - Frame extraction, hole extraction, slot bboxing, piece extraction, and k
     inference remain valid because the frame/piece geometry is unchanged.
   - The old `dihedral_variants()` step depended on the old invariant that non-unit
     holes were D4-unique, so rotating/flipping a piece could not steal another
     piece's destination. That assumption is false now: rotating a piece would erase
     the exact-angle signal that distinguishes holes in the same rotation family.
   - The new solver therefore downscales each piece once, in its observed
     orientation, and matches slots by exact mask equality.
   - This introduces a new assumption: non-unit pieces are generated already in the
     orientation they must occupy in the frame. The generator guarantees that.

d) Placement legality
   - Slot placement still enforces 8-separation, so holes do not merge visually or
     topologically.
   - Pieces are still placed outside the frame with one background cell of padding,
     and distractors remain isolated singleton pixels in a separate color.
   - The generator uses a staging layout, then tight-crops the combined holes before
     building the final frame. This keeps the solver's `bbox(hole)` crop aligned with
     the generator's expected output.

e) Visual observability
   - Human solvers can see multiple identical silhouettes in different rotations both
     inside the frame and among the scattered pieces.
   - Because the base shape has no rotational symmetry, the "correct angle" is a
     visible property of the silhouette itself; no hidden color ordering is needed.

f) Visual cleanliness
   - Only three object types remain: frame, rotated solid pieces, and isolated
     singleton distractors. No overlays or in-piece decorations are added, so the
     grids stay legible.

g) Color reasoning
   - Non-unit pieces keep distinct colors as in the original task. Colors merely help
     track which exact oriented piece goes into which hole; they do not encode any
     hidden ordering or numeric rule.

h) Variant-specific validation target
   - A generated example is only accepted if the non-unit slots share one canonical
     rotation key, use at least two distinct exact rotations, and the new `main()`
     reproduces the expected output.

i) New solver assumptions to audit after implementation
   - Every non-unit slot has an exact-orientation mate among the scattered pieces.
   - No two non-unit slots share the same exact rotation.
   - Each piece is a clean k-by-k block scaling of its slot shape.
   The generator enforces all three.
"""

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# --- shared helpers ----------------------------------------------------------

K_CHOICES = [2, 3]
STAGE_H_RANGE = (6, 12)
STAGE_W_RANGE = (5, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
NON_UNIT_COUNT_CHOICES = [2, 3, 4]
N_UNITS_RANGE = (1, 2)
BASE_SHAPE_SIZE_CHOICES = [4, 5, 6]
HOLE_DENSITY_RANGE = (0.20, 0.50)

GROW_SEED_TRIES = 8
BASE_SHAPE_RETRIES = 40
PLACE_SLOT_RETRIES = 60
UNIT_SLOT_RETRIES = 40

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[tuple[int, np.ndarray]] = []
    for rot in range(4):
        rm = np.rot90(mask, rot)
        key = _mask_key(rm)
        if key in seen:
            continue
        seen.add(key)
        out.append((int(rot), rm))
    return out


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [_mask_key(np.rot90(mask, rot)) for rot in range(4)]
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    H, W = forbidden.shape
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
                p for p in frontier
                if 0 <= p[0] < H and 0 <= p[1] < W and not forbidden[p[0], p[1]] and p not in chosen
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


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _place_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    H, W = forbidden.shape
    h, w = map(int, mask.shape)
    if h > H or w > W:
        return None
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        region = forbidden[top : top + h, left : left + w]
        if np.any(region & mask):
            continue
        coords = np.argwhere(mask) + np.array([[top, left]])
        return coords.astype(int)
    return None


def _generate_base_shape(target_size: int, stage_h: int, stage_w: int) -> np.ndarray | None:
    side = max(4, target_size + 1)
    canvas = np.zeros((side, side), dtype=bool)
    for _ in range(BASE_SHAPE_RETRIES):
        coords = _grow_slot(canvas, target_size)
        if coords is None:
            continue
        mask = _coords_to_mask(coords)
        if min(mask.shape) <= 1:
            continue
        rots = _rotation_variants(mask)
        if len(rots) != 4:
            continue
        if any(rm.shape[0] > stage_h or rm.shape[1] > stage_w for _, rm in rots):
            continue
        return mask
    return None


def _crop_layout(slots: list[dict], stage_h: int, stage_w: int) -> tuple[list[dict], np.ndarray] | None:
    layout_mask = np.zeros((stage_h, stage_w), dtype=bool)
    for slot in slots:
        coords = slot["coords"]
        layout_mask[coords[:, 0], coords[:, 1]] = True
    box = bbox(layout_mask)
    if box is None:
        return None
    r0, c0, r1, c1 = map(int, box)
    cropped = layout_mask[r0 : r1 + 1, c0 : c1 + 1]
    ch, cw = map(int, cropped.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= ch <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= cw <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(cropped.sum()) / float(ch * cw)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    cropped_slots = []
    for slot in slots:
        coords = slot["coords"] - np.array([[r0, c0]])
        mask = _coords_to_mask(coords)
        rr0, cc0, _, _ = bbox_coords(coords)
        cropped_slots.append(
            {
                "coords": coords.astype(int),
                "area": int(slot["area"]),
                "mask": mask,
                "r0": int(rr0),
                "c0": int(cc0),
                "rotation": slot.get("rotation"),
                "canonical_key": _canonical_rotation_key(mask),
                "exact_key": _mask_key(mask),
            }
        )
    return cropped_slots, cropped


def _gen_layout() -> tuple[list[dict], np.ndarray] | None:
    stage_h = random.randint(*STAGE_H_RANGE)
    stage_w = random.randint(*STAGE_W_RANGE)
    n_non_units = random.choice(NON_UNIT_COUNT_CHOICES)
    max_units = min(N_UNITS_RANGE[1], 6 - n_non_units)
    min_units = N_UNITS_RANGE[0]
    if max_units < min_units:
        return None
    n_units = random.randint(min_units, max_units)
    base_size = random.choice(BASE_SHAPE_SIZE_CHOICES)

    base_mask = _generate_base_shape(base_size, stage_h, stage_w)
    if base_mask is None:
        return None
    rot_variants = _rotation_variants(base_mask)
    chosen_rots = random.sample(rot_variants, n_non_units)

    forbidden = np.zeros((stage_h, stage_w), dtype=bool)
    slots: list[dict] = []

    for rot, mask in sorted(chosen_rots, key=lambda item: -(item[1].shape[0] * item[1].shape[1])):
        coords = None
        for _ in range(PLACE_SLOT_RETRIES):
            coords = _place_mask(forbidden, mask)
            if coords is not None:
                break
        if coords is None:
            return None
        slots.append({"coords": coords, "area": int(mask.sum()), "mask": mask, "rotation": int(rot)})
        _forbid_coords(forbidden, coords)

    for _ in range(n_units):
        coords = None
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                break
        if coords is None:
            return None
        slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool), "rotation": None})
        _forbid_coords(forbidden, coords)

    cropped = _crop_layout(slots, stage_h, stage_w)
    if cropped is None:
        return None
    cropped_slots, layout_mask = cropped

    non_units = [slot for slot in cropped_slots if int(slot["area"]) > 1]
    if len(non_units) < 2:
        return None
    canonical_keys = {slot["canonical_key"] for slot in non_units}
    exact_keys = {slot["exact_key"] for slot in non_units}
    if len(canonical_keys) != 1:
        return None
    if len(exact_keys) != len(non_units):
        return None
    return cropped_slots, layout_mask


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        top = (int(piece["slot"]["r0"]) + 1) * int(k)
        left = (int(piece["slot"]["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict], layout_mask: np.ndarray, layout_h: int, layout_w: int, k: int, frame_color: int, distractor_color: int
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


# --- solver -----------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and fill each slot with the exactly-oriented matching piece.

    Changed rule versus the original task:
    - Non-unit holes belong to one rotation family, so orientation is informative.
    - Pieces are matched in their observed orientation; rotating/flipping them would
      destroy the intended one-to-one mapping between rotated holes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
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
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    slots.sort(key=lambda slot: (-int(slot["area"]), int(slot["r0"]), int(slot["c0"])))
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
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = map(int, mask.shape)
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

    # Original code tried all dihedral variants here; that is the old D15 assumption.
    # In this variant, exact orientation is part of the signal, so each piece only
    # contributes one candidate: its observed block-level mask.
    candidates_by_key: dict[tuple[int, tuple[tuple[int, int], bytes]], list[dict]] = {}
    for piece in pieces:
        um = downsample_blocks(piece["mask"])
        if um is None:
            continue
        key = (int(piece["area"]), _mask_key(um))
        candidates_by_key.setdefault(key, []).append(
            {
                "area": int(piece["area"]),
                "um": um,
                "vp": piece["patch"],
                "vm": piece["mask"],
            }
        )

    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        key = (need_area, slot["key"])
        bucket = candidates_by_key.get(key, [])
        assert bucket, "every slot must have an exactly-oriented matching piece"
        found = bucket.pop()
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --- generator ---------------------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout = _gen_layout()
    if layout is None:
        return None
    slots, layout_mask = layout
    layout_h, layout_w = map(int, layout_mask.shape)

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    if not (2 <= len(non_unit_slots) <= 4):
        return None
    if len(non_unit_slots) + len(unit_slots) > 6:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = len(non_unit_slots) + (1 if unit_slots else 0)
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: len(non_unit_slots)]
    unit_color = palette[len(non_unit_slots)] if unit_slots else None

    pieces = [_make_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_slots)]
    if unit_slots:
        pieces.extend(_make_piece(slot, unit_color, k) for slot in unit_slots)

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
