# Combo variant of 898e7135 (index 19)
# combo_spec: D9/3+D15/2
# Name: 等面积旋转变体

from __future__ import annotations

"""
Design analysis for the fused equal-area rotation variant.

a) Concrete behavior and step-by-step trace
   - The outer scene still has three visible object types: one monochrome frame,
     several scattered monochrome sprites outside it, and isolated singleton
     distractors in one extra color.
   - Original task: after extracting frame holes, the solver could compare a slot
     against all rotated/flipped versions of each piece, and area was a strong
     pre-filter because non-unit holes often had different sizes.
   - Fused variant: every non-unit hole has the same area and all of them are
     rotations of one base polyomino. Example: three size-5 holes might be the
     0/90/180 degree rotations of the same pentomino. Their scattered sprites keep
     exactly those observed rotations. The solver must therefore match exact
     downsampled orientation, not "shape up to rotation", and area no longer helps.
   - Difference: original logic could rotate a red sprite into any compatible hole.
     Here that would destroy the intended signal because one rotated sibling hole
     would steal another's piece.

b) Complexity calibration
   - Each sample uses 2-4 non-unit holes, so the shared-area and shared-rotation
     family are both visually repeated.
   - The common non-unit area is chosen from a small set (4, 5, 6), which keeps
     examples readable and avoids overfitting to one exact size.
   - The base shape must expose all 4 rotations distinctly; symmetric shapes are
     rejected so "correct angle" remains visible to humans.

c) Hidden assumption review against the original solver
   - Frame extraction, hole cropping, connected-component parsing, and k inference
     remain valid.
   - The old `need_area = slot_area * k * k` filter becomes non-informative for
     non-unit holes because all of them share the same area.
   - The old `dihedral_variants()` candidate expansion is actively wrong for this
     combo: rotating a piece would erase the exact-angle constraint introduced by
     the rotation family.
   - New solver assumption: each non-unit piece is already shown in the one
     orientation it must occupy in the frame. The generator guarantees this.

d) Placement legality
   - Hole placement still uses 8-neighborhood forbidding, so distinct holes remain
     separated by at least one wall cell, including diagonals.
   - Pieces are placed by `random_free_location(..., padding=1,
     padding_connectivity=8)`, which keeps them isolated from the frame and from
     one another.
   - The frame is built from a tight crop of the staged layout so the solver's
     `bbox(hole)` crop aligns exactly with the generated output target.

e) Visual observability
   - A human can see that the non-unit holes all occupy the same number of cells,
     yet their silhouettes differ only by angle. The same is true for the outside
     sprites after block downscaling.
   - No hidden ordering by color id, area value, or scan order is needed; the only
     informative cues are visible shape and orientation.

f) Visual cleanliness
   - No overlays or interior marks are added. The scene stays readable because the
     frame, sprites, and singleton distractors are spatially separated.
   - Using one dedicated distractor color prevents noise from being mistaken for
     real pieces.

g) Change inferability
   - Color is only an object identity aid. The rule does not depend on numeric
     color values, and non-unit colors are distinct only to help track pieces.
   - The repeated presence of same-area rotated siblings ensures that the fused rule
     can be inferred from a few examples instead of looking like arbitrary
     per-piece color placement.

h) Quantifiable fusion target
   - Accepted samples must satisfy all of the following at once:
     1. every non-unit slot has the same area;
     2. all non-unit slots share one canonical rotation key;
     3. non-unit slots use at least two distinct exact rotations;
     4. the fused solver reproduces the output.

i) Audit of new solver assumptions and generator guarantees
   - Assumption: at least one unit slot exists so `k` can be recovered from the
     smallest solid square piece. Generator guarantee: 1-2 unit slots per sample.
   - Assumption: each piece is an exact k-by-k block scaling with no partial blocks.
     Generator guarantee: all sprites are built with `np.kron`.
   - Assumption: no two non-unit holes share the same exact orientation, otherwise
     exact-shape greedy matching would be ambiguous. Generator guarantee: non-unit
     rotations are sampled without replacement.
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
COMMON_NON_UNIT_AREAS = [4, 5, 6]
HOLE_DENSITY_RANGE = (0.20, 0.50)

GROW_SEED_TRIES = 8
BASE_SHAPE_RETRIES = 60
PLACE_SLOT_RETRIES = 60
UNIT_SLOT_RETRIES = 40

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = np.array(mask, dtype=bool, copy=False)
    return tuple(map(int, tight.shape)), tight.tobytes()


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
        out.append((int(rot), np.array(rm, dtype=bool, copy=True)))
    return out


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(np.rot90(mask, rot)) for rot in range(4))


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
        chosen = {(int(r0), int(c0))}
        frontier = {(int(r0 - 1), int(c0)), (int(r0 + 1), int(c0)), (int(r0), int(c0 - 1)), (int(r0), int(c0 + 1))}
        while len(chosen) < int(target_size) and frontier:
            cands = [
                (int(r), int(c))
                for r, c in frontier
                if 0 <= int(r) < h and 0 <= int(c) < w and not bool(forbidden[int(r), int(c)]) and (int(r), int(c)) not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((int(nr), int(nc)))
            frontier.discard((int(nr), int(nc)))
            frontier.update(
                {
                    (int(nr - 1), int(nc)),
                    (int(nr + 1), int(nc)),
                    (int(nr), int(nc - 1)),
                    (int(nr), int(nc + 1)),
                }
            )
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


def _place_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    mh, mw = map(int, mask.shape)
    if mh > h or mw > w:
        return None
    positions = [(int(r0), int(c0)) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
    random.shuffle(positions)
    rel = np.argwhere(mask)
    for top, left in positions:
        coords = rel + np.array([[int(top), int(left)]], dtype=int)
        if np.any(forbidden[coords[:, 0], coords[:, 1]]):
            continue
        return coords.astype(int)
    return None


def _generate_base_shape(target_size: int, stage_h: int, stage_w: int) -> np.ndarray | None:
    side = max(4, int(target_size + 1))
    seed_canvas = np.zeros((side, side), dtype=bool)
    for _ in range(BASE_SHAPE_RETRIES):
        coords = _grow_slot(seed_canvas, int(target_size))
        if coords is None:
            continue
        mask = _coords_to_mask(coords)
        if min(mask.shape) <= 1:
            continue
        rotations = _rotation_variants(mask)
        if len(rotations) != 4:
            continue
        if any(int(rm.shape[0]) > int(stage_h) or int(rm.shape[1]) > int(stage_w) for _, rm in rotations):
            continue
        return mask
    return None


def _crop_layout(slots: list[dict], stage_h: int, stage_w: int) -> tuple[list[dict], np.ndarray] | None:
    layout_mask = np.zeros((int(stage_h), int(stage_w)), dtype=bool)
    for slot in slots:
        coords = slot["coords"]
        layout_mask[coords[:, 0], coords[:, 1]] = True
    box = bbox(layout_mask)
    if box is None:
        return None
    r0, c0, r1, c1 = map(int, box)
    cropped_mask = layout_mask[r0 : r1 + 1, c0 : c1 + 1]
    layout_h, layout_w = map(int, cropped_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= layout_h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= layout_w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(cropped_mask.sum()) / float(layout_h * layout_w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots = []
    for slot in slots:
        coords = slot["coords"] - np.array([[int(r0), int(c0)]], dtype=int)
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
    return cropped_slots, cropped_mask


def _gen_layout() -> tuple[list[dict], np.ndarray] | None:
    stage_h = random.randint(*STAGE_H_RANGE)
    stage_w = random.randint(*STAGE_W_RANGE)
    n_non_units = random.choice(NON_UNIT_COUNT_CHOICES)
    max_units = min(N_UNITS_RANGE[1], 6 - n_non_units)
    if max_units < N_UNITS_RANGE[0]:
        return None
    n_units = random.randint(N_UNITS_RANGE[0], max_units)
    common_area = random.choice(COMMON_NON_UNIT_AREAS)

    base_mask = _generate_base_shape(common_area, stage_h, stage_w)
    if base_mask is None:
        return None
    rot_variants = _rotation_variants(base_mask)
    if len(rot_variants) < n_non_units:
        return None
    chosen_rots = random.sample(rot_variants, n_non_units)

    forbidden = np.zeros((stage_h, stage_w), dtype=bool)
    slots: list[dict] = []

    for rot, mask in sorted(chosen_rots, key=lambda item: -(int(item[1].shape[0]) * int(item[1].shape[1]))):
        coords = None
        for _ in range(PLACE_SLOT_RETRIES):
            coords = _place_mask(forbidden, mask)
            if coords is not None:
                break
        if coords is None:
            return None
        slots.append({"coords": coords, "area": int(mask.sum()), "mask": mask, "rotation": int(rot)})
        _forbid_coords(forbidden, coords)

    unit_mask = np.ones((1, 1), dtype=bool)
    for _ in range(n_units):
        coords = None
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                break
        if coords is None:
            return None
        slots.append({"coords": coords, "area": 1, "mask": unit_mask, "rotation": None})
        _forbid_coords(forbidden, coords)

    cropped = _crop_layout(slots, stage_h, stage_w)
    if cropped is None:
        return None
    cropped_slots, layout_mask = cropped

    non_units = [slot for slot in cropped_slots if int(slot["area"]) > 1]
    if len(non_units) < 2:
        return None
    if len({int(slot["area"]) for slot in non_units}) != 1:
        return None
    if len({slot["canonical_key"] for slot in non_units}) != 1:
        return None
    if len({slot["exact_key"] for slot in non_units}) != len(non_units):
        return None
    return cropped_slots, layout_mask


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((int(k), int(k)), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((int(layout_h) + 2) * int(k), (int(layout_w) + 2) * int(k)), int(frame_color), dtype=int)
    for piece in pieces:
        top = (int(piece["slot"]["r0"]) + 1) * int(k)
        left = (int(piece["slot"]["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


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
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the cropped frame layout and fill each hole with the single piece whose observed angle already matches it.

    Input structure:
    - The most common color is background. The largest non-background component is a monochrome frame surrounding background holes.
    - Outside the frame are k-by-k block sprites plus isolated singleton distractors. Non-unit holes all have the same area and belong to one rotation family.

    Rule:
    - Extract the hole components from inside the frame, tight-crop them, and infer k from the smallest solid square piece.
    - Downsample each outside piece in its current orientation only, then place it into the hole whose exact block mask matches. Do not rotate or flip pieces during matching.

    Key insight:
    - Equal non-unit areas remove area-based elimination, and rotated sibling holes make orientation itself meaningful. The correct match is the piece whose observed block-level silhouette exactly equals the slot mask.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

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
        slots.append({"r0": int(rr0), "c0": int(cc0), "mask": mask, "key": _mask_key(mask)})
    slots.sort(key=lambda slot: (-int(slot["mask"].sum()), int(slot["r0"]), int(slot["c0"])))
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
        pieces.append({"patch": patch, "mask": mask})
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

    candidates_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for piece in pieces:
        um = downsample_blocks(piece["mask"])
        if um is None:
            continue
        candidates_by_key.setdefault(_mask_key(um), []).append({"vp": piece["patch"], "vm": piece["mask"]})

    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        bucket = candidates_by_key.get(slot["key"], [])
        assert bucket, "every slot must have an exactly oriented matching piece"
        placements.append((slot, bucket.pop()))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


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
    if len({int(slot["area"]) for slot in non_unit_slots}) != 1:
        return None
    if len({slot["canonical_key"] for slot in non_unit_slots}) != 1:
        return None
    if len({slot["exact_key"] for slot in non_unit_slots}) != len(non_unit_slots):
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
