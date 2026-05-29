from __future__ import annotations

# Combo variant of 898e7135 (index 12)
# combo_spec: D3/1+D7/2
# Name: 多框架选择匹配

"""
Design analysis for the combo variant
=====================================

source_task:
  - Base task 898e7135: one frame contains several hole-slots; each external
    sprite is a block-scaled copy of one slot and the output is the enlarged
    filled frame.
  - D3/1 contributes multiple independent frames.
  - D7/2 contributes surplus sprites, so only a subset should be used.

concretization_and_behavior_trace:
  input_visual:
    - The canvas now contains 2-3 separate monochrome rectangular frames.
      Each frame has a 1-cell border and an interior layout whose background
      cells form 2-3 isolated hole regions.
    - Outside the frames are more block sprites than the total number of holes.
      True sprites still use one common scale k=2. Extra sprites are also valid
      k=2 block-objects, but their downsampled shapes match no hole in any
      frame, so they must be discarded.
    - Distractors remain isolated single pixels in one dedicated color.
  small_trace:
    - Original: one frame has slots {domino, L, unit}; outside pieces are
      exactly {2x domino, 2x L, 2x2 square}. Solver fills the only frame.
    - Combo: frame A has {domino, unit}, frame B has {L, bar}. Outside pieces
      contain those four true sprites plus extras such as a T-shape and a zig.
      The solver must parse both frames, collect all slots across them, and
      choose a subset of the external sprites whose D4-normalized block shapes
      cover every slot exactly once.
    - Difference: the target is no longer a single frame, and the external
      inventory is intentionally larger than what the union of holes requires.
  visible_difference:
    - A human can see multiple frame targets at once and can also see that the
      sprite inventory is oversized. The correct output shows every frame filled
      while some external sprites vanish entirely.

complexity_calibration:
  - Frame count is limited to 2-3 so the multi-frame aspect is obvious without
    overloading the canvas.
  - Each frame has only 2-3 slots and inner size 3-4, which keeps the final
    enlarged union within ARC's 30x30 limit while still showing nontrivial hole
    silhouettes.
  - Non-unit slot shapes are globally D4-distinct across all frames, so subset
    selection is necessary but not ambiguous.
  - Extras reuse real piece colors instead of using a special "trash" color, so
    the discard decision comes from shape compatibility, not color filtering.

implicit_assumptions_review:
  - Original assumption "`max(area)` foreground component is the frame" breaks:
    there are now several frames, and a large piece could rival one frame. The
    new solver detects frame components structurally: full rectangular border,
    interior holes, monochrome connectivity.
  - Original assumption "there is one layout to crop and fill" breaks. The new
    solver extracts slots from every detected frame and solves one global
    injective assignment over the union of all slots.
  - Original assumption "the output is exactly one enlarged frame" breaks. The
    new solver preserves the input arrangement of frame bboxes, scales that
    arrangement by k, fills every frame, and crops to the union.
  - Original greedy matching was safe under 1:1 inventory. With surplus sprites
    it is safer to build all compatible candidates first and solve an injective
    assignment by DFS.
  - The new solver assumes at least one true unit slot exists so a solid square
    sprite reveals k. The generator enforces 1-2 unit slots globally.

placement_legality:
  - Frames are packed into a compact non-overlapping arrangement with at least a
    1-cell gap between frame bboxes, so they remain separate components.
  - Pieces are placed outside the frames with `random_free_location`, which
    keeps at least 1 cell of 8-neighborhood separation from other objects.
  - Expected output is assembled from the prepacked frame arrangement rather
    than the full input canvas, so no out-of-bounds scaling occurs.

visual_cleanliness:
  - All frames share one frame color, which makes the multi-frame target read as
    one puzzle family while still keeping the independent boxes obvious.
  - Single-pixel distractors stay isolated and never overlap frame interiors or
    piece clusters.
  - Extras are few (2-3), so the selection pressure is visible but not noisy.

inference_notes:
  - No hidden ordering or color-id rule is used. The meaningful cues are the
    visible hole silhouettes in each frame, the common block scale of the true
    sprites, and the fact that only some external sprites structurally fit.
"""

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# --- tunables ----------------------------------------------------------------

TRUE_K = 2
LAYOUT_H_RANGE = (3, 4)
LAYOUT_W_RANGE = (3, 4)
FRAME_COUNT_RANGE = (2, 3)
TOTAL_SLOTS_RANGE_BY_FRAMES = {
    2: (4, 6),
    3: (6, 6),
}
PER_FRAME_SLOT_RANGE = (2, 3)
TOTAL_UNITS_RANGE = (1, 2)
NON_UNIT_SIZE_CHOICES = [2, 3, 4]

HOLE_DENSITY_RANGE = (0.18, 0.55)
UNIT_SLOT_RETRIES = 24
NON_UNIT_SLOT_RETRIES = 40
EXTRA_SHAPE_RETRIES = 80
GROW_SEED_TRIES = 10

FRAME_GAP = 1
CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

N_EXTRA_RANGE = (2, 3)
DISTRACTOR_RANGE = (8, 12)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


# --- shared helpers ----------------------------------------------------------

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
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = start
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
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[ShapeKey] = set()
    for p, m in base:
        key = (tuple(map(int, m.shape)), m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


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


def _is_frame_component(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return False
    if not (bool(mask[0].all()) and bool(mask[-1].all()) and bool(mask[:, 0].all()) and bool(mask[:, -1].all())):
        return False
    inner = mask[1:-1, 1:-1]
    return bool(inner.any()) and not bool(inner.all())


def _solve_slots_for_k(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates: list[dict] = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "vp": vp,
                    "vm": vm,
                    "um": um,
                }
            )

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
        key=lambda i: (
            len(slot_candidates[i]),
            -int(slots[i]["area"]),
            int(slots[i]["frame_idx"]),
            int(slots[i]["r0"]),
            int(slots[i]["c0"]),
        ),
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


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge several independent frames and fill every hole with a selected sprite subset.

    Input structure:
    - bg is the most common color. There are 2-3 separate monochrome rectangular
      frames, each with a 1-cell border and interior bg holes separated by frame-color walls.
    - Outside the frames are k-scaled monochrome sprites plus singleton noise;
      there are more sprites than total holes, so some sprites are unused.

    Rule:
    - Detect every frame, extract all hole components from each frame interior,
      and treat the union of those holes as the target slot set.
    - Infer the common block scale k from square sprites, then match a subset of
      external sprites to the slots by block-level shape under rotation/flip.
    - Output the input frame arrangement enlarged by k, with each frame filled at
      the positions of its own matched slots; unused sprites disappear.

    Key insight:
    - The multiple frames do not change the local matching rule: every true
      sprite is still a kxk-block copy of one hole shape. The challenge is to
      solve one global subset assignment over slots distributed across several frames.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "input must contain non-background objects"

    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        color = int(g[int(coords[0, 0]), int(coords[0, 1])])
        if _is_frame_component(mask):
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "patch_h": int(patch.shape[0]),
                    "patch_w": int(patch.shape[1]),
                    "color": int(color),
                    "mask": mask,
                }
            )
        elif int(mask.sum()) > 1:
            pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})

    assert 2 <= len(frames) <= 3, "variant expects 2-3 frames"
    assert pieces, "at least one external sprite is required"

    frames.sort(key=lambda f: (int(f["bbox"][0]), int(f["bbox"][1])))

    slots: list[dict] = []
    for fi, frame in enumerate(frames):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        hole = patch[1:-1, 1:-1] == bg
        for coords in components_mask(hole, connectivity=4):
            sm = _tight_mask_from_coords(coords)
            rr0, cc0, _rr1, _cc1 = map(int, bbox_coords(coords))
            slots.append(
                {
                    "frame_idx": int(fi),
                    "area": int(sm.sum()),
                    "r0": int(rr0 + 1),
                    "c0": int(cc0 + 1),
                    "mask": sm,
                }
            )

    assert slots, "frames must contain holes"

    possible_ks = sorted(
        {
            int(h)
            for piece in pieces
            for h, w in [tuple(map(int, piece["mask"].shape))]
            if h == w and int(piece["mask"].sum()) == int(h * w)
        },
        reverse=True,
    )
    assert possible_ks, "at least one solid square sprite is required"

    placements = None
    k = None
    for cand_k in possible_ks:
        trial = _solve_slots_for_k(slots, pieces, int(cand_k))
        if trial is not None:
            placements = trial
            k = int(cand_k)
            break
    assert placements is not None and k is not None, "some square-derived scale must explain all slots"

    min_r = min(int(f["bbox"][0]) for f in frames)
    min_c = min(int(f["bbox"][1]) for f in frames)
    max_r = max(int(f["bbox"][0]) + int(f["patch_h"]) for f in frames)
    max_c = max(int(f["bbox"][1]) + int(f["patch_w"]) for f in frames)

    out = np.zeros(((max_r - min_r) * k, (max_c - min_c) * k), dtype=int)
    for frame in frames:
        fr0, fc0, _fr1, _fc1 = map(int, frame["bbox"])
        top = (int(fr0) - int(min_r)) * int(k)
        left = (int(fc0) - int(min_c)) * int(k)
        out[top : top + int(frame["patch_h"]) * int(k), left : left + int(frame["patch_w"]) * int(k)] = int(frame["color"])

    for slot, cand in placements:
        frame = frames[int(slot["frame_idx"])]
        fr0, fc0, _fr1, _fc1 = map(int, frame["bbox"])
        top = (int(fr0) - int(min_r) + int(slot["r0"])) * int(k)
        left = (int(fc0) - int(min_c) + int(slot["c0"])) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- layout generation -------------------------------------------------------

def _partition_slot_counts(n_frames: int, total_slots: int) -> list[int]:
    counts = [PER_FRAME_SLOT_RANGE[0]] * n_frames
    remaining = total_slots - sum(counts)
    idxs = list(range(n_frames))
    random.shuffle(idxs)
    for idx in idxs:
        if remaining <= 0:
            break
        counts[idx] += 1
        remaining -= 1
    return counts


def _gen_layout(h: int, w: int, n_slots: int, n_units: int, used_keys: set[ShapeKey]) -> list[dict] | None:
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
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            sm = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_keys:
                continue
            used_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _random_nonmatching_mask(target_size: int, forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(EXTRA_SHAPE_RETRIES):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w < target_size:
            continue
        coords = _grow_slot(np.zeros((h, w), dtype=bool), target_size)
        if coords is None:
            continue
        sm = _tight_mask_from_coords(coords)
        key = _canonical_shape_key(sm)
        if key in forbidden_keys:
            continue
        return sm
    return None


def _pack_frames(frame_specs: list[dict]) -> tuple[list[dict], int, int]:
    n_frames = len(frame_specs)
    if n_frames == 2:
        cells = [(0, 0), (0, 1)] if random.random() < 0.5 else [(0, 0), (1, 0)]
    else:
        all_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        missing = random.choice(all_cells)
        cells = [cell for cell in all_cells if cell != missing]
        random.shuffle(cells)

    n_rows = 1 + max(r for r, _ in cells)
    n_cols = 1 + max(c for _, c in cells)
    row_heights = [0] * n_rows
    col_widths = [0] * n_cols
    for spec, (gr, gc) in zip(frame_specs, cells):
        row_heights[gr] = max(int(row_heights[gr]), int(spec["frame_h"]))
        col_widths[gc] = max(int(col_widths[gc]), int(spec["frame_w"]))

    row_offsets = [0] * n_rows
    col_offsets = [0] * n_cols
    for i in range(1, n_rows):
        row_offsets[i] = int(row_offsets[i - 1]) + int(row_heights[i - 1]) + FRAME_GAP
    for i in range(1, n_cols):
        col_offsets[i] = int(col_offsets[i - 1]) + int(col_widths[i - 1]) + FRAME_GAP

    packed: list[dict] = []
    for spec, (gr, gc) in zip(frame_specs, cells):
        new_spec = dict(spec)
        new_spec["local_r"] = int(row_offsets[gr])
        new_spec["local_c"] = int(col_offsets[gc])
        packed.append(new_spec)

    total_h = int(max(spec["local_r"] + spec["frame_h"] for spec in packed))
    total_w = int(max(spec["local_c"] + spec["frame_w"] for spec in packed))
    return packed, total_h, total_w


def _build_expected_output(frame_specs: list[dict], frame_region_h: int, frame_region_w: int, frame_color: int) -> np.ndarray:
    out = np.zeros((int(frame_region_h) * TRUE_K, int(frame_region_w) * TRUE_K), dtype=int)
    for spec in frame_specs:
        top = int(spec["local_r"]) * TRUE_K
        left = int(spec["local_c"]) * TRUE_K
        out[top : top + int(spec["frame_h"]) * TRUE_K, left : left + int(spec["frame_w"]) * TRUE_K] = int(frame_color)
        for piece in spec["true_pieces"]:
            coords = piece["slot"]["coords"]
            pr = top + (int(coords[:, 0].min()) + 1) * TRUE_K
            pc = left + (int(coords[:, 1].min()) + 1) * TRUE_K
            paste(out, piece["out_patch"], pr, pc, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    frame_specs: list[dict],
    frame_region_h: int,
    frame_region_w: int,
    all_pieces: list[dict],
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in all_pieces)
    needed_cells = int(frame_region_h) * int(frame_region_w) + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], int(frame_region_h) + 2)
    min_w = max(CANVAS_DIM_RANGE[0], int(frame_region_w) + 2)
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
    max_r = canvas_h - int(frame_region_h) - 1
    max_c = canvas_w - int(frame_region_w) - 1
    if max_r < 1 or max_c < 1:
        return None
    base_r = random.randint(1, max_r)
    base_c = random.randint(1, max_c)

    for spec in frame_specs:
        frame_img = np.full((int(spec["frame_h"]), int(spec["frame_w"])), int(frame_color), dtype=int)
        frame_img[1:-1, 1:-1] = np.where(spec["layout_mask"], 0, int(frame_color))
        paste(canvas, frame_img, base_r + int(spec["local_r"]), base_c + int(spec["local_c"]))

    for piece in sorted(all_pieces, key=lambda p: -int(p["in_mask"].sum())):
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
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


# --- main generator attempt --------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    total_slots = random.randint(*TOTAL_SLOTS_RANGE_BY_FRAMES[n_frames])
    slot_counts = _partition_slot_counts(n_frames, total_slots)

    total_units = random.randint(*TOTAL_UNITS_RANGE)
    unit_frames = random.sample(range(n_frames), k=min(total_units, n_frames))
    units_per_frame = [1 if i in unit_frames else 0 for i in range(n_frames)]
    if sum(units_per_frame) == 0:
        units_per_frame[random.randrange(n_frames)] = 1

    frame_color = random.randint(1, 9)
    distractor_color = random.choice([c for c in range(1, 10) if c != frame_color])

    used_shape_keys: set[ShapeKey] = set()
    frame_specs_raw: list[dict] = []
    for i in range(n_frames):
        layout_h = random.randint(*LAYOUT_H_RANGE)
        layout_w = random.randint(*LAYOUT_W_RANGE)
        slots = _gen_layout(layout_h, layout_w, int(slot_counts[i]), int(units_per_frame[i]), used_shape_keys)
        if slots is None:
            return None
        layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
        for slot in slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
        frame_specs_raw.append(
            {
                "layout_h": int(layout_h),
                "layout_w": int(layout_w),
                "frame_h": int(layout_h + 2),
                "frame_w": int(layout_w + 2),
                "slots": slots,
                "layout_mask": layout_mask,
            }
        )

    packed_specs, frame_region_h, frame_region_w = _pack_frames(frame_specs_raw)
    if int(frame_region_h) * TRUE_K > 30 or int(frame_region_w) * TRUE_K > 30:
        return None

    n_non_units = sum(int(len(spec["slots"]) - units_per_frame[i]) for i, spec in enumerate(packed_specs))
    palette = [c for c in range(1, 10) if c not in {frame_color, distractor_color}]
    if len(palette) < n_non_units + 1:
        return None
    random.shuffle(palette)
    non_unit_colors = palette[:n_non_units]
    unit_color = palette[n_non_units]

    true_pieces_all: list[dict] = []
    non_unit_idx = 0
    for i, spec in enumerate(packed_specs):
        unit_slots = spec["slots"][: units_per_frame[i]]
        non_unit_slots = spec["slots"][units_per_frame[i] :]

        true_pieces = []
        for slot in non_unit_slots:
            piece = _make_piece_from_mask(slot["mask"], non_unit_colors[non_unit_idx], TRUE_K, apply_dihedral=True)
            piece["slot"] = slot
            true_pieces.append(piece)
            true_pieces_all.append(piece)
            non_unit_idx += 1
        for slot in unit_slots:
            piece = _make_piece_from_mask(slot["mask"], unit_color, TRUE_K, apply_dihedral=False)
            piece["slot"] = slot
            true_pieces.append(piece)
            true_pieces_all.append(piece)
        spec["true_pieces"] = true_pieces

    slot_keys = {slot["shape_key"] for spec in packed_specs for slot in spec["slots"] if "shape_key" in slot}
    non_unit_slot_areas = [int(slot["area"]) for spec in packed_specs for slot in spec["slots"] if int(slot["area"]) > 1]
    real_piece_colors = [int(piece["color"]) for piece in true_pieces_all]
    n_extra = random.randint(*N_EXTRA_RANGE)

    extra_keys: set[ShapeKey] = set()
    extra_pieces: list[dict] = []
    for _ in range(n_extra):
        target_size = random.choice(non_unit_slot_areas)
        mask = _random_nonmatching_mask(target_size, slot_keys | extra_keys)
        if mask is None:
            return None
        extra_keys.add(_canonical_shape_key(mask))
        piece = _make_piece_from_mask(mask, random.choice(real_piece_colors), TRUE_K, apply_dihedral=True)
        extra_pieces.append(piece)

    expected_out = _build_expected_output(packed_specs, frame_region_h, frame_region_w, frame_color)
    canvas = _build_input_canvas(
        packed_specs,
        frame_region_h,
        frame_region_w,
        true_pieces_all + extra_pieces,
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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
