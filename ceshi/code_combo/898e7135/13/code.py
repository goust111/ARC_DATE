# Combo variant of 898e7135 (index 13)
# combo_spec: D4/2+D8/1
# Name: 边框读取缩放因子

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo variant D4/2 + D8/1
==============================================

concretization_and_behavior_trace:
  small_example:
    original_task:
      - Suppose the logical hole layout is 5x4 and the true scale is k=2.
      - The frame bbox is 7x6 because the border is always 1 cell thick.
      - A unit hole generates a 2x2 solid-square piece, so the solver can recover k by scanning pieces.
    combo_variant:
      - The same 5x4 logical layout sits inside a 9x8 frame bbox because the border is now 2 pixels thick on all sides.
      - There are no unit holes, and every slot mask is non-square or non-solid, so no outside piece becomes a 2x2 clue.
      - The solver first crops the frame, measures the four margins from the hole bbox to the frame bbox, reads k=2 from that thickness, then performs the usual D4 slot matching.
    difference:
      - The old inference path was piece-driven: "find the smallest solid square".
      - The fused path is frame-driven: "measure the visible border width", with k fixed to 2 across all samples.

complexity_calibration:
  - Keeping k in a single visible value {2} avoids extra search while still making the new cue legible.
  - The border is thick enough to notice immediately in 2-4 ARC examples.
  - Slots remain 4-connected and pairwise D4-distinct, so shape matching stays meaningful instead of collapsing into color-only lookup.
  - Removing unit slots is not enough by itself; full-square non-unit slots are also forbidden so the old solid-square heuristic cannot silently survive.

implicit_assumption_audit:
  original_main_lines:
    - `frame_comp = max(comps, key=area)` still works because the thick frame remains the largest foreground component.
    - `hole_box = bbox(sub == bg)` still works, but only if holes touch all four sides of the logical layout; the generator preserves that.
    - `layout = hole[...]` still assumes the tight hole bbox equals the intended logical layout.
    - Old `k` inference from solid-square pieces is invalid and is replaced by border-thickness measurement.
    - `downsample_blocks` still assumes every valid piece is tiled by exact 2x2 blocks; generator builds pieces with `np.kron`, so this remains true.
    - Greedy slot matching by area plus D4-normalized logical mask still works because slot shapes are unique under D4.
  new_assumptions_introduced:
    - The frame border must be uniform on all four sides.
    - That uniform thickness must equal 2 in this combo.
    - No piece component may merge with the frame or another piece; generator enforces >=1 bg-cell separation.

placement_legality:
  - The input frame outer size becomes `(layout_h + 4, layout_w + 4)`, so canvas sizing must account for the thicker border.
  - Pieces are placed with `random_free_location(..., padding=1, padding_connectivity=8)`, preventing overlaps and accidental touching.
  - Distractors stay isolated singleton pixels in a fresh color, so they cannot be mistaken for pieces.

visual_observability:
  - A human can directly read the 2-pixel frame thickness without counting piece internals.
  - The output scale agrees with that visible thickness: each logical cell expands to a 2x2 block.
  - Because all examples lack unit slots and solid-square slot masks, the border is the dominant and intended cue.

tidiness_and_inferability:
  - The visual layers are clean: one thick monochrome frame, separated monochrome pieces, isolated noise.
  - Color values are not used for hidden ordering; matching depends on visible shape under rotation/flip.
  - The number of colors stays modest: frame color, distractor color, and one color per piece.

variant_metric_for_validation:
  - Metric 1: inferred frame-border thickness over generated samples; target distribution is always `{2}`.
  - Metric 2: count of non-frame solid-square piece components that would revive the old heuristic; target is always `0`.
"""


K_FIXED = 2

LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 3
AREA_BUFFER_CONST = 40

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _frame_and_layout(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, dict, int, np.ndarray, tuple[int, int, int, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one background hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return g, bg, frame_comp, frame_color, layout, hole_box


def _infer_k_from_border(frame_bbox_shape: tuple[int, int], hole_box: tuple[int, int, int, int]) -> int:
    fh, fw = map(int, frame_bbox_shape)
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    margins = [
        int(lr0),
        int(lc0),
        int(fh - 1 - lr1),
        int(fw - 1 - lc1),
    ]
    assert all(m == margins[0] for m in margins), "frame border thickness must be uniform"
    k = int(margins[0])
    assert k == K_FIXED, f"combo expects border thickness {K_FIXED}, got {k}"
    return k


def _collect_slots(layout: np.ndarray) -> list[dict]:
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"
    return slots


def _collect_pieces(g: np.ndarray, bg: int, frame_comp: dict) -> list[dict]:
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rp = np.rot90(patch, int(kk))
        rm = np.rot90(mask, int(kk))
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))

    uniq = []
    seen = set()
    for vp, vm in base:
        key = (vm.shape, vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the scaled frame interior by inserting each loose piece into its matching hole.

    Input structure:
    - The most common color is the background. The largest foreground component is a monochrome rectangular frame whose border is exactly two pixels thick.
    - Background cells inside that frame form several 4-connected hole regions, and separate outside pieces are 2x upscaled versions of those hole shapes under rotation or reflection.

    Rule:
    - Tight-crop the frame, measure the border thickness to recover k=2, and read the logical hole layout from the frame interior.
    - For each outside piece, enumerate all D4 orientations, downsample by 2x2 blocks, and match the resulting logical mask to one hole region with the same shape.
    - Output the cropped enlarged frame with the matched pieces pasted into the corresponding slot positions, using the frame color as the background.

    Key insight:
    - The scale clue no longer comes from a smallest square sprite: unit-slot clues are absent and square slot masks are forbidden. The visible two-pixel frame border is the intended way to infer the 2x block scale shared by every piece.
    """
    g, bg, frame_comp, frame_color, layout, hole_box = _frame_and_layout(np.array(grid, dtype=int))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    k = _infer_k_from_border(sub.shape, hole_box)

    slots = _collect_slots(layout)
    pieces = _collect_pieces(g, bg, frame_comp)

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
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
        need_area = int(slot["area"]) * k * k
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
        assert found is not None, "every slot must match exactly one piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full(((int(layout.shape[0]) + 2) * k, (int(layout.shape[1]) + 2) * k), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
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
            return np.array(sorted(chosen))
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


def _is_full_square(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    return h == w and bool(mask.all())


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_slots):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if _is_full_square(sm):
                continue
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / (h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece(slot: dict, color: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((K_FIXED, K_FIXED), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
    if random.random() < 0.5:
        in_mask = np.fliplr(in_mask)
        in_patch = np.fliplr(in_patch)

    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * K_FIXED, (layout_w + 2) * K_FIXED), frame_color, dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * K_FIXED
        left = (int(coords[:, 1].min()) + 1) * K_FIXED
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
    frame_h, frame_w = layout_h + 2 * K_FIXED, layout_w + 2 * K_FIXED
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

    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[K_FIXED : K_FIXED + layout_h, K_FIXED : K_FIXED + layout_w] = np.where(layout_mask, 0, frame_color)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
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
        rr, cc = random.randint(0, canvas_h - 1), random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    piece_colors = palette[:n_slots]

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(slot, piece_colors[i]) for i, slot in enumerate(slots)]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, frame_color)

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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
