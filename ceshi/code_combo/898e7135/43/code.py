# Combo variant of 898e7135 (index 43)
# combo_spec: D4/2+D5/1+D8/1
# Name: 边框线索消失型输出

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
Design analysis for combo variant D4/2+D5/1+D8/1
=================================================

spec_delta:
  preserved_core:
    - Input still contains one large monochrome frame plus scattered piece sprites.
    - Each sprite is still a k-block-scaled version of one hole-region, up to D4 rotation/flip.
    - Matching still happens by downsampling piece masks to slot masks.
  fused_changes:
    - k is fixed to 2 for every sample, but the solver must read it from the frame border thickness.
    - The frame border in the input is exactly 2 pixels thick on all four sides.
    - Output removes the outer frame completely and keeps only the scaled interior layout.
    - To stop regression to the original heuristic, the generator creates no unit slots and no solid-square slot masks.

behavior_trace:
  small_example:
    - Let the logical interior layout be 4x5 with k=2.
    - Original task:
      - input frame bbox would be 6x7 because the border was always 1 cell thick.
      - output would be (4+2)*2 by (5+2)*2 = 12x14, so the enlarged border remained visible.
      - a 2x2 solid square piece could reveal k immediately.
    - Fused variant:
      - input frame bbox is 8x9 because the border itself is 2 cells thick.
      - solver crops the bg holes, measures equal margins [2,2,2,2], and sets k=2 from geometry.
      - output is only 4*2 by 5*2 = 8x10: pieces are pasted into the scaled interior, while the clue-giving thick border disappears.
    - Difference:
      - k moves from piece-size evidence to border-thickness evidence.
      - the final canvas is smaller because the frame is no longer rendered in output.

complexity_calibration:
  - k has a single value, so the search space is simpler, but the border-thickness clue stays visually explicit.
  - Slots remain 3-6 in count, 4-connected, and D4-distinct, which prevents piece-slot ambiguity.
  - Removing unit slots avoids accidental recreation of the original "smallest square reveals k" shortcut.
  - Keeping only one clue source makes the fusion legible in 2-4 examples instead of mixing multiple independent cues.

implicit_assumption_audit:
  main_line_dependencies:
    - `frame_comp = max(comps, key=area)` assumes the frame is the largest foreground component; generator keeps piece areas well below thick-frame area.
    - `hole_box = bbox(sub == bg)` assumes holes are bg cells inside the frame; unchanged and guaranteed.
    - `layout = hole[lr0:lr1+1, lc0:lc1+1]` assumes holes touch all four sides of the logical interior crop; generator enforces this.
    - border-based k inference assumes the four margins from frame bbox to hole bbox are equal and positive; generator builds the frame exactly that way.
    - `downsample_blocks` assumes every piece is tiled by exact 2x2 blocks with no partial blocks; generator builds pieces with `np.kron`.
    - slot matching assumes each slot has exactly one D4-equivalent piece; generator uses D4-unique slot shapes.
  new_conflicts_checked:
    - D5/1 changes the output anchor: piece top-left must shift from `(slot+1)*k` to `slot*k` because the outer border is omitted.
    - D8/1 and D4/2 together would be redundant if square clues remained, so square slots are forbidden.

placement_legality:
  - The thick frame has size `(layout_h + 4, layout_w + 4)` because k is fixed at 2.
  - Pieces are placed with `random_free_location(..., padding=1, padding_connectivity=8)`, so they stay separated from the frame and from each other.
  - Distractors remain isolated singleton pixels in a dedicated color and never merge into candidate pieces.

visual_observability:
  - A human can see the input frame border is two pixels thick rather than one.
  - The output keeps the interior wall pattern and inserted sprites but entirely drops that thick border, so the disappearing clue is externally visible.
  - Because there are no unit-square pieces, the border is the dominant visible reason the scale is 2.

tidiness_and_inference:
  - Frame color is reused only for walls/background of the output interior; distractors use a separate color.
  - Piece colors are distinct, which keeps filled slots easy to track visually.
  - No color-value ordering is used anywhere; all reasoning is geometric and visible.

variant_metrics_for_validation:
  - border_thickness should always be 2 on all four sides.
  - solid_square_piece_count should stay 0, proving the original k-clue was removed.
  - output_shape should always equal `(layout_h*2, layout_w*2)`, not `((layout_h+2)*2, (layout_w+2)*2)`.

main_assumption_audit:
  - Every slot loop assumes at least one slot exists; generator uses 3-6 slots.
  - Every candidate loop assumes each piece has at least one valid D4 orientation whose mask downsamples cleanly; true because pieces are exact block upscales.
  - The greedy matching order assumes no two unused pieces share the same D4-normalized slot mask; generator guarantees D4-unique shapes.
"""


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rp = np.rot90(patch, int(kk))
        rm = np.rot90(mask, int(kk))
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for vp, vm in base:
        key = (tuple(map(int, vm.shape)), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _extract_frame_layout(grid: np.ndarray) -> tuple[np.ndarray, int, object, np.ndarray, tuple[int, int, int, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))

    frame_vals = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_vals).most_common(1)[0][0])

    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return g, frame_color, frame_comp, layout, hole_box


def _infer_k_from_border(frame_shape: tuple[int, int], hole_box: tuple[int, int, int, int]) -> int:
    fh, fw = map(int, frame_shape)
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    margins = [int(lr0), int(lc0), int(fh - 1 - lr1), int(fw - 1 - lc1)]
    k = int(min(margins))
    assert k >= 1, "frame border thickness must be positive"
    assert all(int(m) == k for m in margins), "frame border thickness must be uniform"
    return int(k)


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"
    return slots


def _extract_pieces(g: np.ndarray, frame_comp: object, bg: int) -> list[dict]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    pieces: list[dict] = []
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
        pieces.append({"area": int(pm.sum()), "patch": patch, "mask": pm})
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the frame's interior layout, match each outside sprite to a hole shape, and render only the enlarged interior.

    Input structure:
    - The largest foreground component is a monochrome frame whose border thickness equals the scale k.
    - Inside that frame, background-colored holes form 4-connected slots; outside it, pieces are exact 2x2-block upscales of those slots under D4 transforms.

    Rule:
    - Infer k from the uniform frame border thickness, crop to the hole layout, and match each slot with the uniquely corresponding piece by block-level shape.
    - Build the output as the scaled interior layout only: keep interior walls in the frame color, paste matched pieces into their slot locations, and omit the outer frame entirely.

    Key insight:
    - The same thick border that reveals the scaling factor in the input vanishes in the output, so the solver must use that geometric clue before rendering the borderless enlarged interior.
    """
    g, frame_color, frame_comp, layout, hole_box = _extract_frame_layout(np.array(grid, dtype=int))
    bg = int(get_bg(g))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    frame_shape = crop_bbox(g, (fr0, fc0, fr1, fc1)).shape
    k = _infer_k_from_border(frame_shape, hole_box)

    slots = _extract_slots(layout)
    pieces = _extract_pieces(g, frame_comp, bg)

    candidates: list[dict] = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
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
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full((int(layout.shape[0]) * k, int(layout.shape[1]) * k), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = int(slot["r0"]) * k
        left = int(slot["c0"]) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --- generator ---------------------------------------------------------------


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


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


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


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
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


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full((layout_h * k, layout_w * k), frame_color, dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = int(coords[:, 0].min()) * k
        left = int(coords[:, 1].min()) * k
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
    k = K_FIXED
    frame_h, frame_w = layout_h + 2 * k, layout_w + 2 * k
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
    frame_img[k : k + layout_h, k : k + layout_w] = np.where(layout_mask, 0, frame_color)
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
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
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
    k = K_FIXED
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

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

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
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
