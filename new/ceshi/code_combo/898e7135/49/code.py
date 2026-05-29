# Combo variant of 898e7135 (index 49)
# combo_spec: D3/1+D5/0+D7/2
# Name: 多框架全局选择填充

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

source_tasks:
  - base: 898e7135
  - D3 inspiration: 8698868d (multiple independent targets on one canvas)
  - D7 reference: single-variant 898e7135/D7/2

concretization_and_behavior_trace:
  input_visual:
    - There are 2-3 independent monochrome frames on the same bg=0 canvas.
      Each frame is already drawn at the final block scale k_true, so its outer
      border and internal walls are made of k_true x k_true blocks rather than
      1-pixel lines.
    - The holes inside each frame are bg-colored k_true x k_true cell blocks.
      Outside the frames, a larger global pool of candidate sprites is
      scattered; some belong to frame A, some to frame B, some are spare.
    - Output keeps the full canvas size and the original frame positions. The
      only transformation is: remove external sprites/noise and fill the holes
      in place.
  small_trace:
    - Original puzzle: one unscaled frame with slots {domino, unit}. Outside
      sprites are {2x-scaled domino, 2x2 square}. Output crops to the enlarged
      frame only.
    - Combo puzzle: suppose frame A has {domino, unit}, frame B has {L-tromino,
      line-tromino}. Outside sprites are those four true pieces plus a spare
      3x3 square and maybe another nonmatching spare. Now the solver must:
      1. find both frames instead of only the largest one,
      2. interpret each frame as already scaled, so the answer is painted back
         onto the original canvas rather than extracted/cropped,
      3. choose a subset of external sprites whose downsampled shapes globally
         cover all holes injectively, discarding leftovers.
  visible_difference:
    - A human can directly see that there are several separated frames, that
      the output is not a cropped enlargement but the same canvas cleaned up,
      and that the outside sprite inventory is larger than the number of holes.

complexity_calibration:
  - The true frame/piece scale is fixed at k_true=2 so 2-3 frames fit cleanly
    inside a 30x30 ARC canvas. Spare pieces use k_extra=3 often enough to make
    the inventory visibly mixed.
  - Total slots are capped at 4-5, with 1-2 unit slots overall. This keeps the
    global assignment nontrivial but avoids combinatorial clutter.
  - Non-unit slot shapes are D4-distinct globally, so matching is decided by
    shape/scale/frame structure rather than arbitrary color memorization.
  - Spare pieces reuse real piece colors. This prevents the task from
    degenerating into "ignore a dedicated distractor color".

implicit_assumptions_review:
  - Original `frame_comp = max(area)` is invalid because there are multiple
    frames. The new solver scans every monochrome component and tests whether
    it is a block-scaled frame.
  - Original hole extraction by tight-cropping bg inside the frame bbox is no
    longer appropriate. Here the frame is already scaled, so the solver must
    downsample the whole frame mask by candidate k and read holes from the
    interior cell grid.
  - Original output shape `(layout+2)*k` is invalid under D5/0. The new solver
    returns a canvas with the same shape as the input and fills holes in place.
  - Original greedy 1:1 slot loop is invalid under D7/2 because unused pieces
    are allowed. The new solver performs a global injective slot->piece search.
  - D4 deduplication must include mask shape as well as bytes; otherwise full
    1x2 and 2x1 masks collide and harm subset search.
  - New solver-side assumption: at least one true unit slot exists so the true
    k contributes a solid-square candidate. The generator enforces 1-2 unit
    slots globally.

placement_legality:
  - Frames are placed by reserving their full bounding rectangles, so no other
    object can overlap even with a frame hole.
  - External sprites are placed with >=1 cell of 8-neighborhood padding, so
    connected components stay separate.
  - Distractors remain isolated single pixels in one dedicated color.

visual_cleanliness:
  - Frame colors are distinct from piece colors and from the distractor color.
  - Spare sprites are few (1-2) and spatially separated, so the canvas remains
    readable.
  - The output is visually simple: same canvas, same frames, no outside clutter.

color_inference:
  - There is no hidden color remapping. Chosen sprites keep their own colors
    when pasted into holes; frame colors only define the wall structure.
"""

import random
import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


TRUE_K = 2
EXTRA_K = 3

N_FRAMES_RANGE = (2, 3)
TOTAL_SLOTS_RANGE = (4, 5)
TOTAL_UNITS_RANGE = (1, 2)
FRAME_SLOT_MAX = 3

FRAME_LAYOUT_CHOICES = [(3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]
NON_UNIT_SIZE_CHOICES = [2, 3, 4]
HOLE_DENSITY_RANGE = (0.18, 0.55)

UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 40
FRAME_LAYOUT_RETRIES = 50
EXTRA_SHAPE_RETRIES = 60
GROW_SEED_TRIES = 10

N_EXTRA_RANGE = (1, 2)
CANVAS_DIM_RANGE = (22, 30)
AREA_BUFFER_MULT = 3
AREA_BUFFER_CONST = 30

DISTRACTOR_RANGE = (8, 12)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500

ShapeKey = tuple[tuple[int, int], bytes]


def _canonical_shape_key(mask: np.ndarray) -> ShapeKey:
    variants: list[ShapeKey] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((tuple(map(int, flip.shape)), flip.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
    out = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    out[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
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


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    mask = np.zeros_like(forbidden)
    mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(mask)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
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
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            cell = random.choice(cands)
            chosen.add(cell)
            frontier.discard(cell)
            frontier.update(
                {
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                }
            )
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        base.append((rot_patch, rot_mask))
        base.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[ShapeKey] = set()
    for patch_v, mask_v in base:
        key = (tuple(map(int, mask_v.shape)), mask_v.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((patch_v, mask_v))
    return uniq


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if k <= 0 or hh % k != 0 or ww % k != 0:
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


def _parse_frame_for_k(comp: dict, grid: np.ndarray, bg: int, k: int) -> dict | None:
    if int(comp["area"]) <= 1:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    color = int(comp["color"])
    mask = patch == int(color)
    cell_mask = _downsample_blocks(mask, int(k))
    if cell_mask is None or cell_mask.shape[0] < 3 or cell_mask.shape[1] < 3:
        return None
    if not (
        bool(cell_mask[0, :].all())
        and bool(cell_mask[-1, :].all())
        and bool(cell_mask[:, 0].all())
        and bool(cell_mask[:, -1].all())
    ):
        return None

    layout_holes = ~cell_mask[1:-1, 1:-1]
    if not bool(layout_holes.any()):
        return None

    slots: list[dict] = []
    for coords in components_mask(layout_holes, connectivity=4):
        sm = _tight_mask_from_coords(coords)
        rr0, cc0, _, _ = map(int, bbox_coords(coords))
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "frame_r0": int(r0),
                "frame_c0": int(c0),
            }
        )
    if not slots:
        return None
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "patch": patch,
        "mask": mask,
        "slots": slots,
    }


def _solve_slots_for_k(frames: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates: list[dict] = []
    for pi, piece in enumerate(pieces):
        for patch_v, mask_v in _dihedral_variants(piece["patch"], piece["mask"]):
            cell_mask = _downsample_blocks(mask_v, int(k))
            if cell_mask is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": cell_mask,
                    "vp": patch_v,
                    "vm": mask_v,
                }
            )

    slots = [slot for frame in frames for slot in frame["slots"]]
    slot_candidates: list[list[dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        usable = [
            cand
            for cand in candidates
            if int(cand["area"]) == int(need_area)
            and cand["um"].shape == slot["mask"].shape
            and np.array_equal(cand["um"], slot["mask"])
        ]
        if not usable:
            return None
        slot_candidates.append(usable)

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(slot_candidates[i]), -int(slots[i]["area"]), int(slots[i]["frame_r0"]), int(slots[i]["frame_c0"])),
    )
    assignment: list[dict | None] = [None] * len(slots)
    used_piece: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        si = order[pos]
        for cand in slot_candidates[si]:
            pi = int(cand["pi"])
            if pi in used_piece:
                continue
            used_piece.add(pi)
            assignment[si] = cand
            if dfs(pos + 1):
                return True
            assignment[si] = None
            used_piece.remove(pi)
        return False

    if not dfs(0):
        return None
    return [(slots[i], assignment[i]) for i in range(len(slots)) if assignment[i] is not None]


def _make_piece_from_mask(cell_mask: np.ndarray, color: int, scale: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(cell_mask, np.ones((scale, scale), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    if apply_dihedral and int(cell_mask.sum()) > 1:
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
        "area": int(out_mask.sum()),
        "cell_mask": cell_mask,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several already-scaled frames in place by selecting a global subset of matching sprites.

    Input structure:
    - bg is the most common color. The canvas contains 2-3 separate monochrome frames, each already drawn as k-by-k blocks with bg holes inside.
    - Outside the frames are scattered monochrome sprites plus isolated singleton noise; there are more candidate sprites than total holes.

    Rule:
    - Identify every frame, infer the common block scale k, and read each frame's hole regions at the cell level.
    - From all outside sprites, choose an injective subset whose shapes match the frame holes up to rotation/flip, then paste them back into the corresponding holes.
    - Keep the full input canvas size; preserve the frame walls and background; remove all external sprites and singleton noise.

    Key insight: The frames themselves are already scaled, so the task is not "crop and enlarge" anymore. Downsampling each frame and each sprite by the correct k reveals a global hole-matching problem across multiple frames with spare candidates.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "input must contain foreground components"

    pieces_by_k: dict[int, tuple[list[dict], list[tuple[dict, dict]]]] = {}
    possible_ks = sorted(
        {
            int(h)
            for comp in comps
            for h, w in [crop_bbox(g, tuple(map(int, comp["bbox"]))).shape]
            if h == w and int(comp["area"]) == int(h * w) and int(comp["area"]) > 1
        },
        reverse=True,
    )
    assert possible_ks, "at least one solid square piece is required"

    chosen_frames = None
    chosen_pieces = None
    chosen_placements = None
    chosen_k = None
    for k in possible_ks:
        parsed_frames: list[dict] = []
        frame_ids: set[int] = set()
        for idx, comp in enumerate(comps):
            parsed = _parse_frame_for_k(comp, g, bg, int(k))
            if parsed is not None:
                parsed_frames.append(parsed)
                frame_ids.add(int(idx))
        if len(parsed_frames) < 2:
            continue

        pieces: list[dict] = []
        for idx, comp in enumerate(comps):
            if int(idx) in frame_ids or int(comp["area"]) <= 1:
                continue
            r0, c0, r1, c1 = map(int, comp["bbox"])
            patch = crop_bbox(g, (r0, c0, r1, c1))
            pm = patch == int(comp["color"])
            pieces.append({"area": int(comp["area"]), "mask": pm, "patch": patch})
        if not pieces:
            continue

        placements = _solve_slots_for_k(parsed_frames, pieces, int(k))
        if placements is None:
            continue

        chosen_frames = parsed_frames
        chosen_pieces = pieces
        chosen_placements = placements
        chosen_k = int(k)
        break

    assert chosen_frames is not None and chosen_pieces is not None
    assert chosen_placements is not None and chosen_k is not None

    out = np.full_like(g, int(bg))
    for frame in chosen_frames:
        r0, c0, _, _ = frame["bbox"]
        paste(out, frame["patch"], int(r0), int(c0), mask=frame["mask"], overwrite=True)

    for slot, cand in chosen_placements:
        top = int(slot["frame_r0"]) + (int(slot["r0"]) + 1) * int(chosen_k)
        left = int(slot["frame_c0"]) + (int(slot["c0"]) + 1) * int(chosen_k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


def _gen_layout(
    h: int,
    w: int,
    n_slots: int,
    n_units: int,
    used_shape_keys: set[ShapeKey],
) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
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
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            mask = _tight_mask_from_coords(coords)
            hh, ww = map(int, mask.shape)
            if hh == ww and int(mask.sum()) == int(hh * ww):
                continue
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _partition_positive(total: int, parts: int, max_part: int) -> list[int] | None:
    if total < parts or total > parts * max_part:
        return None
    out = [1] * parts
    for _ in range(total - parts):
        choices = [i for i in range(parts) if out[i] < max_part]
        if not choices:
            return None
        out[random.choice(choices)] += 1
    random.shuffle(out)
    return out


def _assign_unit_counts(slot_counts: list[int], total_units: int) -> list[int] | None:
    counts = [0] * len(slot_counts)
    for _ in range(total_units):
        choices = [i for i, slot_n in enumerate(slot_counts) if counts[i] < slot_n]
        if not choices:
            return None
        counts[random.choice(choices)] += 1
    return counts


def _build_frame_patch(layout_h: int, layout_w: int, slots: list[dict], frame_color: int) -> tuple[np.ndarray, np.ndarray]:
    holes = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        holes[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    frame_cells = np.ones((layout_h + 2, layout_w + 2), dtype=bool)
    frame_cells[1:-1, 1:-1] = ~holes
    frame_mask = np.kron(frame_cells, np.ones((TRUE_K, TRUE_K), dtype=bool))
    frame_patch = np.where(frame_mask, int(frame_color), 0).astype(int)
    return frame_patch, frame_mask


def _build_expected_output(canvas_shape: tuple[int, int], frames: list[dict], true_pieces: list[dict]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    for frame in frames:
        paste(out, frame["patch"], int(frame["top"]), int(frame["left"]), mask=frame["mask"], overwrite=True)
    for piece in true_pieces:
        slot = piece["slot"]
        top = int(slot["frame_top"]) + (int(slot["slot_r0"]) + 1) * TRUE_K
        left = int(slot["frame_left"]) + (int(slot["slot_c0"]) + 1) * TRUE_K
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _random_nonmatching_mask(target_size: int, forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(EXTRA_SHAPE_RETRIES):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w < target_size:
            continue
        coords = _grow_slot(np.zeros((h, w), dtype=bool), int(target_size))
        if coords is None:
            continue
        mask = _tight_mask_from_coords(coords)
        hh, ww = map(int, mask.shape)
        if hh == ww and int(mask.sum()) == int(hh * ww):
            continue
        key = _canonical_shape_key(mask)
        if key in forbidden_keys:
            continue
        return mask
    return None


def _choose_canvas_dims(frame_patches: list[np.ndarray], pieces: list[dict]) -> tuple[int, int] | None:
    needed_cells = sum(int(patch.shape[0] * patch.shape[1]) for patch in frame_patches)
    needed_cells += AREA_BUFFER_MULT * sum(int(piece["in_mask"].sum()) for piece in pieces) + AREA_BUFFER_CONST
    h = random.randint(*CANVAS_DIM_RANGE)
    w = random.randint(*CANVAS_DIM_RANGE)
    while h * w < needed_cells and (h < CANVAS_DIM_RANGE[1] or w < CANVAS_DIM_RANGE[1]):
        if h <= w and h < CANVAS_DIM_RANGE[1]:
            h += 1
        elif w < CANVAS_DIM_RANGE[1]:
            w += 1
        else:
            break
    if h * w < needed_cells:
        return None
    return int(h), int(w)


def _place_frames(canvas_h: int, canvas_w: int, frames: list[dict]) -> bool:
    occ = np.zeros((canvas_h, canvas_w), dtype=int)
    order = sorted(range(len(frames)), key=lambda i: -(frames[i]["patch"].shape[0] * frames[i]["patch"].shape[1]))
    for idx in order:
        frame = frames[idx]
        rect = np.ones(frame["patch"].shape, dtype=int)
        try:
            rr, cc = random_free_location(occ, rect, transparent=0, border=0, padding=1, padding_connectivity=8)
        except ValueError:
            return False
        paste(occ, rect, int(rr), int(cc), overwrite=True)
        frame["top"] = int(rr)
        frame["left"] = int(cc)
    return True


def _build_input_canvas(
    canvas_shape: tuple[int, int],
    frames: list[dict],
    all_pieces: list[dict],
    distractor_color: int,
) -> np.ndarray | None:
    canvas = np.zeros(canvas_shape, dtype=int)
    occ = np.zeros(canvas_shape, dtype=int)

    for frame in frames:
        rr, cc = int(frame["top"]), int(frame["left"])
        paste(canvas, frame["patch"], rr, cc, mask=frame["mask"], overwrite=True)
        paste(occ, np.ones(frame["patch"].shape, dtype=int), rr, cc, overwrite=True)

    for piece in sorted(all_pieces, key=lambda item: -int(item["in_mask"].sum())):
        try:
            rr, cc = random_free_location(
                occ,
                piece["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, piece["in_patch"], int(rr), int(cc), mask=piece["in_mask"], overwrite=True)
        paste(occ, np.ones(piece["in_patch"].shape, dtype=int), int(rr), int(cc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_shape[0] - 1)
        cc = random.randint(0, canvas_shape[1] - 1)
        if occ[rr, cc] != 0:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_shape[0], rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_shape[1], cc + 2)
        if np.any(occ[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        occ[rr, cc] = 1
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*N_FRAMES_RANGE)
    total_slots = random.randint(max(n_frames, TOTAL_SLOTS_RANGE[0]), TOTAL_SLOTS_RANGE[1])
    total_units = random.randint(TOTAL_UNITS_RANGE[0], min(TOTAL_UNITS_RANGE[1], total_slots - 1))

    slot_counts = _partition_positive(total_slots, n_frames, FRAME_SLOT_MAX)
    if slot_counts is None:
        return None
    unit_counts = _assign_unit_counts(slot_counts, total_units)
    if unit_counts is None:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_colors = [palette.pop() for _ in range(n_frames)]
    distractor_color = palette.pop()
    n_non_unit = total_slots - total_units
    if len(palette) < n_non_unit + 1:
        return None
    non_unit_colors = [palette.pop() for _ in range(n_non_unit)]
    unit_color = palette.pop()

    used_shape_keys: set[ShapeKey] = set()
    frames: list[dict] = []
    for frame_idx in range(n_frames):
        made = None
        for _ in range(FRAME_LAYOUT_RETRIES):
            layout_h, layout_w = random.choice(FRAME_LAYOUT_CHOICES)
            trial_shape_keys = set(used_shape_keys)
            slots = _gen_layout(layout_h, layout_w, slot_counts[frame_idx], unit_counts[frame_idx], trial_shape_keys)
            if slots is None:
                continue
            used_shape_keys = trial_shape_keys
            patch, mask = _build_frame_patch(layout_h, layout_w, slots, frame_colors[frame_idx])
            made = {"layout_h": layout_h, "layout_w": layout_w, "slots": slots, "patch": patch, "mask": mask}
            break
        if made is None:
            return None
        frames.append(made)

    frame_patches = [frame["patch"] for frame in frames]

    true_pieces: list[dict] = []
    non_unit_i = 0
    for frame in frames:
        for slot in frame["slots"]:
            if int(slot["area"]) == 1:
                piece = _make_piece_from_mask(slot["mask"], unit_color, TRUE_K, apply_dihedral=False)
            else:
                piece = _make_piece_from_mask(slot["mask"], non_unit_colors[non_unit_i], TRUE_K, apply_dihedral=True)
                non_unit_i += 1
            true_pieces.append(piece)

    shape_keys = {
        slot["shape_key"]
        for frame in frames
        for slot in frame["slots"]
        if "shape_key" in slot
    }
    real_colors = [piece["color"] for piece in true_pieces]
    extra_masks = [np.ones((1, 1), dtype=bool)]
    n_extra = random.randint(*N_EXTRA_RANGE)
    extra_keys: set[ShapeKey] = set()
    while len(extra_masks) < n_extra:
        mask = _random_nonmatching_mask(random.choice(NON_UNIT_SIZE_CHOICES), shape_keys | extra_keys)
        if mask is None:
            return None
        extra_masks.append(mask)
        extra_keys.add(_canonical_shape_key(mask))

    extra_pieces = [
        _make_piece_from_mask(mask, random.choice(real_colors), EXTRA_K, apply_dihedral=(int(mask.sum()) > 1))
        for mask in extra_masks
    ]

    canvas_shape = _choose_canvas_dims(frame_patches, true_pieces + extra_pieces)
    if canvas_shape is None:
        return None
    if not _place_frames(canvas_shape[0], canvas_shape[1], frames):
        return None

    ordered_true_pieces: list[dict] = []
    ti = 0
    for frame in frames:
        for slot in frame["slots"]:
            piece = true_pieces[ti]
            ti += 1
            slot_meta = {
                "frame_top": int(frame["top"]),
                "frame_left": int(frame["left"]),
                "slot_r0": int(slot["coords"][:, 0].min()),
                "slot_c0": int(slot["coords"][:, 1].min()),
            }
            piece = dict(piece)
            piece["slot"] = slot_meta
            ordered_true_pieces.append(piece)

    expected = _build_expected_output(canvas_shape, frames, ordered_true_pieces)
    input_canvas = _build_input_canvas(canvas_shape, frames, ordered_true_pieces + extra_pieces, distractor_color)
    if input_canvas is None:
        return None

    try:
        produced = main(input_canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return input_canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
