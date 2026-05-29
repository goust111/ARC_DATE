# Combo variant of 898e7135 (index 1)
# combo_spec: D1/1+D15/2
# Name: 精确旋转角度匹配

from __future__ import annotations

"""
Design analysis for combo D1/1 + D15/2
======================================

a) Concrete behavior and flow trace
   - Preserved scaffold:
     the input still contains one largest monochrome frame, background 0, 1-2 unit
     holes, several non-unit holes, scattered block-scaled pieces, and singleton
     distractors. The output is still the cropped inner layout enlarged by k with
     the frame color as background.
   - Combo-specific change:
     every non-unit hole is a rotation of one shared chiral base polyomino. The
     outside non-unit pieces keep those exact observed rotations, and flips are not
     legal matches.
   - Small concrete example:
     suppose the shared base shape is a 4-cell L with one longer arm. Inside the
     frame we may see three holes: 0 degree, 90 degree, and 270 degree. Outside the
     frame we see three solid k-by-k-block pieces in those same three silhouettes,
     each with a distinct color.
     original task:
       the solver canonicalizes by the full dihedral group, so any one of the three
       pieces can be rotated or flipped to fit any of the three holes. Exact angle is
       discarded, so color placement among those holes becomes arbitrary.
     combo task:
       the solver must keep the exact block-level angle. Because the family is chiral,
       a flip would produce the wrong handedness, and because several holes share the
       same rotation class, a rotation-invariant key would also lose which color goes
       to which hole. Matching is therefore exact-orientation matching inside the
       rotation-only family.

b) Complexity calibration
   - If we only removed flips but kept D4-unique holes, many samples would look
     unchanged from the original task.
   - If we only made holes share a rotation family but used an achiral base shape,
     forbidding flips would be visually irrelevant.
   - The generator therefore uses one chiral base shape with four distinct rotations.
     Non-unit holes are sampled from those rotations, and at least two different exact
     rotations must appear in each example.
   - Non-unit piece colors stay distinct so wrong assignments are visible in the
     output. Unit holes remain as repeated k-by-k squares to preserve the original k
     inference cue without dominating the puzzle.

c) Hidden assumption audit for main()
   - find_components(..., monochromatic=False):
     still assumes the frame is the largest foreground component.
     generator guarantee: frame area must exceed every piece area, and all pieces are
     placed with 1-cell separation from the frame and each other.
   - bbox(hole) tight crop:
     still assumes the combined holes define the intended inner layout bbox.
     generator guarantee: a staging layout is tight-cropped before frame assembly, so
     the solver sees the same cropped layout.
   - infer k from the smallest solid square piece:
     still assumes at least one unit slot exists.
     generator guarantee: every sample contains 1-2 unit slots.
   - old dihedral candidate enumeration:
     this assumption is invalid under the combo. Full D4 search would erase both the
     chirality signal and the exact-angle signal.
     replacement: downsample each observed piece once and match by exact tight mask.
   - bucket match by (area, exact mask):
     new assumption: each non-unit exact orientation appears at most once.
     generator guarantee: chosen non-unit rotations are distinct; unit duplicates are
     allowed and harmless because their patches are identical.

d) Placement legality
   - Non-overlap inside the layout is enforced by 8-neighbor forbidding, so holes stay
     disconnected even diagonally.
   - Rotated masks are only placed where every occupied cell is free and in bounds.
   - Pieces are scattered with random_free_location(..., padding=1,
     padding_connectivity=8), so they do not touch the frame, each other, or
     distractors.
   - The final canvas is rejected unless background 0 remains the most common color.

e) Visual observability
   - A human can see that several holes are the same silhouette at different angles.
   - Because the base shape is chiral, mirror images are visibly different, so the
     "no flip" restriction has a readable consequence.
   - Because non-unit colors are distinct, a wrong angle assignment would move colors
     to different holes in the output, making the orientation rule observable across
     a few training pairs.

f) Visual cleanliness
   - Only three object families remain: the frame, solid block pieces, and singleton
     distractors.
   - No overlays, per-cell markings, or hidden numeric encodings are introduced.
   - Tight cropping of the staged layout avoids excessive empty space inside the frame.

g) Change inferability and anti-degeneration
   - The critical control variable is "exact non-unit rotation identity within one
     shared chiral family".
   - Generator-side validation rejects any sample that the legacy dihedral solver would
     solve to the same output. This prevents silent fallback to the original behavior.
   - A useful quantitative metric after implementation is:
       distinct_non_unit_exact_rotations >= 2
       and legacy_output != combo_output
     The validation script later measures both.

h) Main-loop assumptions after modification
   - Every non-unit slot has one piece with the same exact block mask after
     downsampling.
   - No non-unit slot shares its exact mask with another non-unit slot.
   - Unit slots may repeat, but all unit pieces are identical k-by-k squares so any
     bucket pop is safe.
   The generator explicitly enforces these conditions.
"""

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


MaskKey = tuple[tuple[int, int], bytes]


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
BASE_SHAPE_RETRIES = 60
PLACE_SLOT_RETRIES = 80
UNIT_SLOT_RETRIES = 40

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _mask_key(mask: np.ndarray) -> MaskKey:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _rotation_canonical_key(mask: np.ndarray) -> MaskKey:
    return min(_mask_key(np.rot90(mask, rot)) for rot in range(4))


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen: set[MaskKey] = set()
    out: list[tuple[int, np.ndarray]] = []
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        key = _mask_key(rm)
        if key in seen:
            continue
        seen.add(key)
        out.append((int(rot), rm))
    return out


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_canonical_key(mask) != _rotation_canonical_key(np.fliplr(mask))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
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


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < int(target_size) and frontier:
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
    h, w = map(int, mask.shape)
    big_h, big_w = map(int, forbidden.shape)
    if h > big_h or w > big_w:
        return None
    positions = [(r, c) for r in range(big_h - h + 1) for c in range(big_w - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        region = forbidden[top : top + h, left : left + w]
        if np.any(region & mask):
            continue
        return (np.argwhere(mask) + np.array([[top, left]], dtype=int)).astype(int)
    return None


def _generate_base_shape(target_size: int, stage_h: int, stage_w: int) -> np.ndarray | None:
    side = max(4, int(target_size) + 1)
    free = np.zeros((side, side), dtype=bool)
    for _ in range(BASE_SHAPE_RETRIES):
        coords = _grow_slot(free, int(target_size))
        if coords is None:
            continue
        mask = _coords_to_mask(coords)
        if min(mask.shape) <= 1:
            continue
        rots = _rotation_variants(mask)
        if len(rots) != 4:
            continue
        if not _is_chiral(mask):
            continue
        if any(rm.shape[0] > stage_h or rm.shape[1] > stage_w for _, rm in rots):
            continue
        return mask
    return None


def _crop_slots(slots: list[dict], stage_h: int, stage_w: int) -> tuple[list[dict], np.ndarray] | None:
    layout_mask = np.zeros((stage_h, stage_w), dtype=bool)
    for slot in slots:
        coords = slot["coords"]
        layout_mask[coords[:, 0], coords[:, 1]] = True

    box = bbox(layout_mask)
    if box is None:
        return None
    r0, c0, r1, c1 = map(int, box)
    cropped_layout = layout_mask[r0 : r1 + 1, c0 : c1 + 1]
    ch, cw = map(int, cropped_layout.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= ch <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= cw <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(cropped_layout.sum()) / float(ch * cw)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    cropped_slots: list[dict] = []
    for slot in slots:
        coords = slot["coords"] - np.array([[r0, c0]], dtype=int)
        mask = _coords_to_mask(coords)
        rr0, cc0, _rr1, _cc1 = map(int, bbox_coords(coords))
        cropped_slots.append(
            {
                "coords": coords.astype(int),
                "area": int(slot["area"]),
                "mask": mask,
                "r0": int(rr0),
                "c0": int(cc0),
                "rotation": slot.get("rotation"),
                "canonical_key": _rotation_canonical_key(mask),
                "exact_key": _mask_key(mask),
            }
        )
    return cropped_slots, cropped_layout


def _gen_layout() -> tuple[list[dict], np.ndarray] | None:
    stage_h = random.randint(*STAGE_H_RANGE)
    stage_w = random.randint(*STAGE_W_RANGE)
    n_non_units = random.choice(NON_UNIT_COUNT_CHOICES)
    max_units = min(N_UNITS_RANGE[1], 6 - n_non_units)
    if max_units < N_UNITS_RANGE[0]:
        return None
    n_units = random.randint(N_UNITS_RANGE[0], max_units)
    base_size = random.choice(BASE_SHAPE_SIZE_CHOICES)

    base_mask = _generate_base_shape(base_size, stage_h, stage_w)
    if base_mask is None:
        return None
    rot_variants = _rotation_variants(base_mask)
    if len(rot_variants) != 4:
        return None
    chosen_rotations = random.sample(rot_variants, n_non_units)

    forbidden = np.zeros((stage_h, stage_w), dtype=bool)
    slots: list[dict] = []

    for rot, mask in sorted(chosen_rotations, key=lambda item: -(item[1].shape[0] * item[1].shape[1])):
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

    cropped = _crop_slots(slots, stage_h, stage_w)
    if cropped is None:
        return None
    cropped_slots, layout_mask = cropped

    non_units = [slot for slot in cropped_slots if int(slot["area"]) > 1]
    if len(non_units) < 2:
        return None
    if len({slot["canonical_key"] for slot in non_units}) != 1:
        return None
    if len({slot["exact_key"] for slot in non_units}) != len(non_units):
        return None
    if len({slot["rotation"] for slot in non_units}) != len(non_units):
        return None
    if len({slot["exact_key"] for slot in non_units}) < 2:
        return None
    return cropped_slots, layout_mask


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots: list[dict] = []
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
    return slots


def _extract_pieces(g: np.ndarray, comps: list[dict], frame_comp: dict) -> list[dict]:
    pieces: list[dict] = []
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
    return pieces


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original D4-based solver, kept only for anti-degeneration rejection."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "legacy solver requires a frame"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = _extract_slots(layout)
    pieces = _extract_pieces(g, comps, frame_comp)
    assert pieces
    k = _infer_k(pieces)

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base: list[tuple[np.ndarray, np.ndarray]] = []
        for rot in range(4):
            rp = np.rot90(patch, int(rot))
            rm = np.rot90(mask, int(rot))
            base.append((rp, rm))
            base.append((np.fliplr(rp), np.fliplr(rm)))
        uniq: list[tuple[np.ndarray, np.ndarray]] = []
        seen: set[MaskKey] = set()
        for vp, vm in base:
            key = _mask_key(vm)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(piece["area"]), "um": um, "vp": vp, "vm": vm})

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
        assert found is not None
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --- solver -----------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole layout and fill each slot with the correctly oriented piece.

    Input structure:
    - Background 0 is the most common color. The largest foreground component is a
      monochrome rectangular frame enclosing background holes.
    - Outside the frame sit monochrome pieces made of solid k-by-k blocks, plus
      isolated singleton distractors in another color.

    Rule:
    - Extract the connected hole regions inside the frame as slots and infer k from
      the smallest solid square piece.
    - Downsample each non-singleton outside piece to its block mask in the observed
      orientation, then match slots by exact block-level mask equality. Flips are not
      allowed, and collapsing to a rotation-invariant key is also wrong here because
      several slots are rotations of the same chiral base shape.
    - Output the cropped frame layout scaled by k, using the frame color as the
      background and pasting each matched piece into its slot location.

    Key insight:
    The non-unit holes intentionally form one rotation family, so "same shape up to
    orientation" is insufficient. The exact observed rotation of each block-scaled
    piece is the signal that determines which same-family hole it fills.
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

    slots = _extract_slots(layout)
    assert slots, "frame must contain at least one slot"

    pieces = _extract_pieces(g, comps, frame_comp)
    assert pieces, "at least one non-frame piece must exist"
    k = _infer_k(pieces)

    candidates_by_key: dict[tuple[int, MaskKey], list[dict]] = {}
    for piece in pieces:
        um = _downsample_blocks(piece["mask"], k)
        if um is None:
            continue
        key = (int(piece["area"]), _mask_key(um))
        candidates_by_key.setdefault(key, []).append(
            {
                "area": int(piece["area"]),
                "vp": piece["patch"],
                "vm": piece["mask"],
            }
        )

    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        key = (need_area, slot["key"])
        bucket = candidates_by_key.get(key, [])
        assert bucket, "every slot must have an exactly oriented matching piece"
        placements.append((slot, bucket.pop()))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --- generator ---------------------------------------------------------------

def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
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
    layout = _gen_layout()
    if layout is None:
        return None
    slots, layout_mask = layout
    layout_h, layout_w = map(int, layout_mask.shape)

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    if len(non_unit_slots) + len(unit_slots) > 6:
        return None
    if len({slot["exact_key"] for slot in non_unit_slots}) < 2:
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

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(max(slot["area"] * k * k for slot in slots))
    if frame_area <= max_piece_area:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_slots)]
    if unit_slots:
        pieces.extend(_make_piece(slot, unit_color, k) for slot in unit_slots)

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
        legacy = _legacy_main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    if np.array_equal(legacy, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
