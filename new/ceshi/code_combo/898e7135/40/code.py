# Combo variant of 898e7135 (index 40)
# combo_spec: D1/1+D9/2+D15/2
# Name: 大形状精确旋转鉴别

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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
Design analysis for combo D1/1 + D9/2 + D15/2
==============================================

a) Concrete behavior and flow impact
   - Original task: each non-unit hole had a D4-unique shape, so the solver could
     rotate or flip every scattered piece independently until it matched one slot.
   - New combo: all non-unit holes come from one large chiral base polyomino with
     area 8-12, and the holes shown inside the frame are different rotations of that
     same base shape. Flips are illegal.
   - If we stopped there, the task would be ambiguous: several same-family holes and
     several same-family colored pieces would let the old "rotate each piece any way"
     logic permute colors arbitrarily across the holes.
   - To make the family visible but still uniquely solvable, the generator uses one
     shared non-zero rotation offset for all non-unit pieces. Small example:
       slot rotations inside frame: {0, 90, 180}
       scattered piece rotations:    {90, 180, 270}
     Every large piece must be rotated by the same angle (here 270 degrees CCW /
     90 degrees CW) to recover its destination slot. Unit pieces remain solid k-by-k
     squares and only stabilize k inference.

b) Complexity calibration
   - D9/2 is made visible by forcing the shared base shape to be large: area is
     always 8-12, never a tiny 2-4 toy shape. The shape must also be irregular
     (bbox contains at least one missing cell) so orientation is readable.
   - D1/1 is made real by forcing chirality: the mirror image is not rotation-
     equivalent, so "rotation only" is a meaningful restriction instead of a no-op.
   - D15/2 is made real by using 2-3 distinct exact rotations of the same base mask
     inside the frame. Four rotations are avoided because the full set would be
     invariant under global shifts and the common offset would stop being unique.

c) Hidden assumption audit against the original main()
   - `find_components(..., monochromatic=False)` assumes the frame is the largest
     foreground component. The generator keeps every piece and distractor separated
     from the frame by background and rejects layouts where any piece could exceed
     the frame area.
   - Tight-cropping the hole layout still works because the frame structure is
     unchanged; the generator builds a staged hole mask and then crops it exactly the
     same way the solver does.
   - Inferring `k` from the smallest solid square piece still requires unit slots.
     The generator always includes 1-2 area-1 holes.
   - The old `dihedral_variants()` assumption is broken in two ways: flips are no
     longer legal, and allowing per-piece arbitrary rotations would destroy the exact
     rotation-family signal. The new solver therefore infers one common rotation
     offset across all non-unit pieces, while unit pieces stay orientation-free.
   - The solver now assumes:
       1. all non-unit slots share one rotation canonical key,
       2. their exact orientations are distinct,
       3. the non-unit pieces share one unique common rotation offset.
     The generator explicitly validates all three before accepting an instance.

d) Placement legality
   - Slot masks are placed on a staged 12x8 layout with 8-neighbor exclusion, so
     holes never touch or merge.
   - Pieces are placed with `random_free_location(..., padding=1, padding_connectivity=8)`,
     so they stay detached from each other and from the frame.
   - Distractors remain isolated singleton pixels in a fresh color.

e) Visual observability
   - Human solvers can see one big handed silhouette repeated at several angles in
     the frame and again among the scattered pieces.
   - Because the shape is chiral and irregular, the correct angle is readable from
     geometry alone; no hidden color-value ordering is involved.
   - The shared offset creates a visible family-level pattern across 2-3 colored
     pieces, which makes the "exact rotation" requirement learnable from a few ARC
     examples.

f) Visual cleanliness
   - There are still only three object types: the frame, monochrome pieces, and
     singleton distractors. No overlays, internal markers, or mixed-color pieces are
     added.
   - Large pieces are few in number, which avoids overcrowding the 15-30 canvas.

g) Color reasoning
   - Colors only track identity of distinct non-unit pieces. They do not encode the
     rotation amount or any hidden ordinal relation.
   - Unit pieces may share a color, as in the source task, because swapping equal
     squares does not change the output.

h) Variant-specific validation target
   - Accepted samples must satisfy all of the following:
       1. every non-unit slot has area 8-12,
       2. all non-unit slots share one rotation canonical key,
       3. the exact orientations used inside the frame are all distinct,
       4. a unique common non-zero rotation offset maps observed non-unit pieces back
          to the slot orientations,
       5. the new `main()` reproduces the expected output.

i) Why this is not a degenerate copy of the original task
   - On these inputs, the original solver's per-piece D4 matching loses the shared
     orientation constraint and can permute colored large pieces across same-family
     holes. The combo solver preserves the orientation family by solving for one
     global rotation offset under rotations only.
"""


ExactKey = tuple[tuple[int, int], bytes]


K_CHOICES = [2, 2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
HOLE_DENSITY_RANGE = (0.20, 0.50)

NON_UNIT_COUNT_BY_K = {2: [3], 3: [2]}
UNIT_COUNT_RANGE = (1, 2)
BASE_SHAPE_SIZE_BY_K = {2: [8, 9, 10, 11, 12], 3: [8, 9, 10]}
TOTAL_SLOT_AREA_BUDGET = {2: 32, 3: 20}

BASE_SHAPE_RETRIES = 80
PLACE_SLOT_RETRIES = 80
UNIT_SLOT_RETRIES = 50

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 600


def _mask_key(mask: np.ndarray) -> ExactKey:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen: set[ExactKey] = set()
    out: list[tuple[int, np.ndarray]] = []
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        key = _mask_key(rm)
        if key in seen:
            continue
        seen.add(key)
        out.append((int(rot), rm))
    return out


def _canonical_rotation_key(mask: np.ndarray) -> ExactKey:
    return min(_mask_key(np.rot90(mask, int(rot))) for rot in range(4))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w


def _is_chiral(mask: np.ndarray) -> bool:
    return _canonical_rotation_key(mask) != _canonical_rotation_key(np.fliplr(mask))


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
    mask = np.zeros_like(forbidden)
    mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(mask)


def _grow_connected_mask(box_h: int, box_w: int, target_size: int) -> np.ndarray | None:
    start = (random.randint(0, box_h - 1), random.randint(0, box_w - 1))
    chosen = {start}
    frontier = {
        (start[0] - 1, start[1]),
        (start[0] + 1, start[1]),
        (start[0], start[1] - 1),
        (start[0], start[1] + 1),
    }

    while len(chosen) < int(target_size) and frontier:
        cands = []
        for nr, nc in frontier:
            if not (0 <= nr < box_h and 0 <= nc < box_w):
                continue
            if (nr, nc) in chosen:
                continue
            freedom = 0
            for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                if 0 <= ar < box_h and 0 <= ac < box_w and (ar, ac) not in chosen:
                    freedom += 1
            cands.append((freedom, nr, nc))
        if not cands:
            break
        cands.sort(reverse=True)
        top = cands[: min(4, len(cands))]
        _freedom, nr, nc = random.choice(top)
        chosen.add((nr, nc))
        frontier.discard((nr, nc))
        frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

    if len(chosen) != int(target_size):
        return None
    coords = np.array(sorted(chosen), dtype=int)
    return _coords_to_mask(coords)


def _generate_base_shape(target_size: int, stage_h: int, stage_w: int) -> np.ndarray | None:
    max_side = min(6, max(4, int(target_size)))
    for _ in range(BASE_SHAPE_RETRIES):
        box_h = random.randint(3, max_side)
        box_w = random.randint(3, max_side)
        if box_h * box_w < int(target_size):
            continue
        mask = _grow_connected_mask(box_h, box_w, target_size)
        if mask is None:
            continue
        rots = _rotation_variants(mask)
        if len(rots) != 4:
            continue
        if not _is_irregular_large_shape(mask):
            continue
        if not _is_chiral(mask):
            continue
        if any(int(rm.shape[0]) > int(stage_h) or int(rm.shape[1]) > int(stage_w) for _rot, rm in rots):
            continue
        return mask
    return None


def _place_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    H, W = map(int, forbidden.shape)
    h, w = map(int, mask.shape)
    if h > H or w > W:
        return None
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    fg = np.argwhere(mask)
    for top, left in positions[:PLACE_SLOT_RETRIES]:
        coords = fg + np.array([top, left], dtype=int)
        if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
            continue
        return coords.astype(int)
    return None


def _crop_layout(slots: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    full = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    for slot in slots:
        coords = slot["coords"]
        full[coords[:, 0], coords[:, 1]] = True
    box = bbox(full)
    if box is None:
        return None
    r0, c0, r1, c1 = map(int, box)
    layout_mask = full[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, layout_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(layout_mask.sum()) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots = []
    for slot in slots:
        coords = slot["coords"].copy()
        coords[:, 0] -= int(r0)
        coords[:, 1] -= int(c0)
        rr0, cc0, _rr1, _cc1 = map(int, bbox_coords(coords))
        mask = _coords_to_mask(coords)
        entry = {
            "coords": coords,
            "area": int(slot["area"]),
            "r0": int(rr0),
            "c0": int(cc0),
            "mask": mask,
            "exact_key": _mask_key(mask),
            "canonical_key": _canonical_rotation_key(mask),
            "rotation": slot.get("rotation"),
        }
        cropped_slots.append(entry)
    return cropped_slots, layout_mask


def _rotation_set_is_shift_unique(rotations: tuple[int, ...]) -> bool:
    base = set(int(r) for r in rotations)
    for delta in (1, 2, 3):
        if {(int(r) + int(delta)) % 4 for r in base} == base:
            return False
    return True


def _infer_common_delta(piece_masks: list[np.ndarray], slot_masks: list[np.ndarray]) -> list[int]:
    slot_counter = Counter(_mask_key(mask) for mask in slot_masks)
    deltas = []
    for delta in range(4):
        piece_counter = Counter(_mask_key(np.rot90(mask, int(delta))) for mask in piece_masks)
        if piece_counter == slot_counter:
            deltas.append(int(delta))
    return deltas


def _gen_layout(k: int) -> tuple[list[dict], np.ndarray, int] | None:
    stage_h, stage_w = map(int, WORK_LAYOUT_SHAPE)
    n_non_units = random.choice(NON_UNIT_COUNT_BY_K[int(k)])
    n_units = random.randint(UNIT_COUNT_RANGE[0], min(UNIT_COUNT_RANGE[1], 6 - n_non_units))

    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])
    size_choices = [
        size
        for size in BASE_SHAPE_SIZE_BY_K[int(k)]
        if int(size) * int(n_non_units) + int(n_units) <= int(budget)
    ]
    if not size_choices:
        return None

    base_size = random.choice(size_choices)
    base_mask = _generate_base_shape(base_size, stage_h, stage_w)
    if base_mask is None:
        return None

    rot_variants = _rotation_variants(base_mask)
    if len(rot_variants) != 4:
        return None
    rot_map = {int(rot): mask for rot, mask in rot_variants}

    valid_rotation_sets = []
    for comb in itertools.combinations([0, 1, 2, 3], n_non_units):
        if _rotation_set_is_shift_unique(tuple(int(v) for v in comb)):
            valid_rotation_sets.append(tuple(int(v) for v in comb))
    if not valid_rotation_sets:
        return None

    chosen_rotations = random.choice(valid_rotation_sets)
    piece_input_rot = random.choice([1, 2, 3])

    forbidden = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    slots: list[dict] = []

    order = sorted(
        chosen_rotations,
        key=lambda rot: -(int(rot_map[int(rot)].shape[0]) * int(rot_map[int(rot)].shape[1])),
    )
    for rot in order:
        coords = _place_mask(forbidden, rot_map[int(rot)])
        if coords is None:
            return None
        slot = {"coords": coords, "area": int(base_size), "rotation": int(rot), "mask": rot_map[int(rot)]}
        slots.append(slot)
        _forbid_coords(forbidden, coords)

    for _ in range(n_units):
        coords = _place_mask(forbidden, np.ones((1, 1), dtype=bool))
        if coords is None:
            return None
        slots.append({"coords": coords, "area": 1, "rotation": None, "mask": np.ones((1, 1), dtype=bool)})
        _forbid_coords(forbidden, coords)

    cropped = _crop_layout(slots)
    if cropped is None:
        return None
    cropped_slots, layout_mask = cropped

    non_units = [slot for slot in cropped_slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in cropped_slots if int(slot["area"]) == 1]
    if len(non_units) != int(n_non_units) or not unit_slots:
        return None
    if any(not (8 <= int(slot["area"]) <= 12) for slot in non_units):
        return None
    if len({slot["canonical_key"] for slot in non_units}) != 1:
        return None
    if len({slot["exact_key"] for slot in non_units}) != len(non_units):
        return None

    observed_masks = [np.rot90(slot["mask"], int(piece_input_rot)) for slot in non_units]
    delta_candidates = _infer_common_delta(observed_masks, [slot["mask"] for slot in non_units])
    expected_delta = (4 - int(piece_input_rot)) % 4
    if delta_candidates != [int(expected_delta)]:
        return None

    return cropped_slots, layout_mask, int(piece_input_rot)


def _make_piece(slot: dict, color: int, k: int, piece_input_rot: int) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((int(k), int(k)), int(color), dtype=int)
        mask = np.ones((int(k), int(k)), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch.copy(),
            "in_mask": mask.copy(),
        }

    out_mask = np.kron(slot["mask"], np.ones((int(k), int(k)), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    in_mask = np.rot90(out_mask, int(piece_input_rot))
    in_patch = np.rot90(out_patch, int(piece_input_rot))
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
    out = np.full(((int(layout_h) + 2) * int(k), (int(layout_w) + 2) * int(k)), int(frame_color), dtype=int)
    for piece in pieces:
        top = (int(piece["slot"]["r0"]) + 1) * int(k)
        left = (int(piece["slot"]["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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

    canvas = np.zeros((int(canvas_h), int(canvas_w)), dtype=int)
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))

    fr_max, fc_max = int(canvas_h - frame_h - 1), int(canvas_w - frame_w - 1)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the framed hole layout and refill it with the matching scattered sprites.

    Input structure:
    - The largest foreground component is a monochrome rectangular frame around a
      tight-cropped hole layout. Outside the frame sit isolated piece components
      built from solid k-by-k blocks, plus singleton noise pixels.
    - The non-unit holes are large chiral shapes from one rotation family; unit holes
      remain k-by-k solid squares used to reveal k.

    Rule:
    - Extract the hole components inside the frame and the block-scaled pieces
      outside it, then downsample each piece back to a block mask.
    - For non-unit pieces, allow rotations only, forbid flips, and infer the one
      common rotation offset that maps the observed piece orientations to the hole
      orientations. Paste each rotated piece into its matching hole position.
    - Output the cropped framed layout scaled by k, using the frame color as the
      background.

    Key insight:
    The large holes are not arbitrary unique silhouettes: they are different
    rotations of one handed base shape. Matching therefore depends on preserving
    chirality and recovering a shared exact rotation, not on generic dihedral
    equivalence.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
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
                "exact_key": _mask_key(mask),
                "canonical_key": _canonical_rotation_key(mask),
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

    piece_infos = []
    for piece in pieces:
        um = downsample_blocks(piece["mask"])
        if um is None:
            continue
        piece_infos.append(
            {
                "area": int(piece["area"]),
                "patch": piece["patch"],
                "mask": piece["mask"],
                "um": um,
                "uarea": int(um.sum()),
                "exact_key": _mask_key(um),
                "canonical_key": _canonical_rotation_key(um),
            }
        )
    assert piece_infos, "every meaningful piece must be block-scaled"

    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_pieces = [piece for piece in piece_infos if int(piece["uarea"]) == 1]
    non_unit_pieces = [piece for piece in piece_infos if int(piece["uarea"]) > 1]

    assert unit_slots and unit_pieces, "unit slots and pieces are required to infer k"
    assert len(unit_slots) == len(unit_pieces), "unit piece count must match unit slot count"
    assert len(non_unit_slots) == len(non_unit_pieces), "non-unit piece count must match slot count"
    assert len({slot["canonical_key"] for slot in non_unit_slots}) == 1, "large slots must share one rotation family"
    assert len({piece["canonical_key"] for piece in non_unit_pieces}) == 1, "large pieces must share one rotation family"
    assert len({slot["exact_key"] for slot in non_unit_slots}) == len(non_unit_slots), "large slot orientations must be distinct"

    delta_candidates = _infer_common_delta(
        [piece["um"] for piece in non_unit_pieces],
        [slot["mask"] for slot in non_unit_slots],
    )
    assert len(delta_candidates) == 1, "non-unit pieces must admit one shared rotation offset"
    delta = int(delta_candidates[0])

    placements: list[tuple[dict, np.ndarray, np.ndarray]] = []
    slot_by_key = {slot["exact_key"]: slot for slot in non_unit_slots}
    for piece in non_unit_pieces:
        rotated_patch = np.rot90(piece["patch"], int(delta))
        rotated_mask = np.rot90(piece["mask"], int(delta))
        rotated_um = np.rot90(piece["um"], int(delta))
        target = slot_by_key.get(_mask_key(rotated_um))
        assert target is not None, "every large piece must rotate to one slot"
        placements.append((target, rotated_patch, rotated_mask))

    unit_slots_sorted = sorted(unit_slots, key=lambda slot: (int(slot["r0"]), int(slot["c0"])))
    unit_pieces_sorted = sorted(unit_pieces, key=lambda piece: int(piece["area"]))
    for slot, piece in zip(unit_slots_sorted, unit_pieces_sorted):
        placements.append((slot, piece["patch"], piece["mask"]))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for slot, patch, mask in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, patch, top, left, mask=mask, overwrite=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout = _gen_layout(int(k))
    if layout is None:
        return None
    slots, layout_mask, piece_input_rot = layout
    layout_h, layout_w = map(int, layout_mask.shape)

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    if len(non_unit_slots) not in (2, 3):
        return None
    if not unit_slots:
        return None

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = max(int(slot["area"]) * int(k) * int(k) for slot in non_unit_slots + unit_slots)
    if frame_area <= max_piece_area:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    needed_piece_colors = len(non_unit_slots) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: len(non_unit_slots)]
    unit_color = int(palette[len(non_unit_slots)])

    pieces = [
        _make_piece(slot, int(non_unit_colors[i]), int(k), int(piece_input_rot))
        for i, slot in enumerate(non_unit_slots)
    ]
    pieces.extend(_make_piece(slot, int(unit_color), int(k), int(piece_input_rot)) for slot in unit_slots)

    observed_non_unit_masks = [np.rot90(slot["mask"], int(piece_input_rot)) for slot in non_unit_slots]
    delta_candidates = _infer_common_delta(observed_non_unit_masks, [slot["mask"] for slot in non_unit_slots])
    if delta_candidates != [int((4 - int(piece_input_rot)) % 4)]:
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(k), int(frame_color))
    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        int(layout_h),
        int(layout_w),
        int(frame_color),
        int(distractor_color),
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
