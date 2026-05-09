# Variant of 898e7135
# Dimension: D8 — 缩放因子 k 的推断线索 (meta)
# Change: k 从最小实心方块精灵（单元洞对应的 k×k 方块）的边长推断；每个示例保证有 1-2 个单元洞... → k 通过框架边框厚度编码

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for D8 variant
================================

spec:
  invariant_structure:
    - Background is still 0 and must stay the most common color.
    - The largest foreground component is still the monochrome frame.
    - The frame still encloses a hole-layout made of bg cells (slots) and frame-colored walls.
    - Each outside piece is still a k-block-scaled version of one slot, up to D4 rotation/flip.
    - The output is still the cropped, scaled frame with pieces inserted into their matching slots.
  changed_structure:
    - Input frame border thickness is no longer 1 pixel; it is exactly k pixels.
    - k is read from that border thickness, not from unit-slot square pieces.
    - To avoid degenerating back to the old clue, the generator no longer creates unit slots and also forbids full-square slot masks.

behavior_trace:
  small_example:
    original_input:
      - Suppose logical layout is 5x4 and k=3.
      - Original frame bbox would be 7x6 because border thickness was fixed at 1.
      - Solver looked for the smallest solid square piece, found a 3x3 unit-slot sprite, and set k=3.
    variant_input:
      - The same logical layout uses a frame bbox of 11x10 because the border is 3 pixels thick on every side.
      - Hole bbox inside the frame starts 3 cells from the top/left and ends 3 cells before the bottom/right.
      - Solver reads that margin value 3 as k, then performs the same block-downsampling and slot matching as before.
    difference:
      - The old task encoded k in a piece.
      - This variant encodes k in the frame geometry itself.

complexity_calibration:
  - k remains in the small set {2, 3}; this keeps the clue visible in 2-4 ARC examples.
  - The thick frame is visually obvious and does not require counting piece sizes.
  - Non-unit slots remain D4-distinct to avoid slot-matching ambiguity.
  - Removing unit slots prevents the new samples from silently satisfying the old inference path.

implicit_assumption_audit:
  main_related_lines:
    - `frame_comp = max(comps, key=area)`: still valid because the thick frame is even larger than before.
    - `hole_box = bbox(sub == bg)`: still valid because holes remain bg and there is at least one slot.
    - `layout = hole[...]`: still valid only if holes touch all four sides of the logical layout; generator preserves this.
    - old `k` inference from smallest solid square piece: invalid under D8 and replaced.
    - `downsample_blocks` assumes each piece is perfectly tiled by kxk blocks: still valid.
    - slot/piece matching by area + D4-normalized block mask: unchanged and still valid.
  new_assumptions_introduced:
    - The frame has uniform border thickness on all four sides.
    - That thickness equals k and is at least 1.
    - The generator must avoid unit-slot/full-square clues so k actually depends on the border.

placement_legality:
  - Thick frame size becomes `(layout_h + 2*k, layout_w + 2*k)`, so canvas feasibility must account for the larger frame.
  - Pieces remain separated from the frame and each other by at least one bg cell via `random_free_location(..., padding=1, padding_connectivity=8)`.
  - Distractors remain isolated single pixels in a dedicated color.

visual_observability:
  - Humans can directly see that some examples have border thickness 2 and others 3.
  - Output scaling remains consistent with that thickness, so the cue is externally visible.
  - Because unit-square pieces are absent, the border is the dominant route to infer k.

tidiness_and_colors:
  - Only one frame color, one distractor color, and distinct piece colors are used.
  - No extra markers overlap the frame; the thicker border changes only one clean visual layer.

variant_metric_for_validation:
  - Measure 1: inferred border thickness distribution across generated samples.
  - Measure 2: count of solid-square piece components that would support the old heuristic; target is always 0.
"""


# --- solver ------------------------------------------------------------------


def _frame_and_layout(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, tuple[int, int, int, int]]:
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
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return g, frame_color, layout, hole_box


def _infer_k_from_border(frame_bbox_shape: tuple[int, int], hole_box: tuple[int, int, int, int]) -> int:
    fh, fw = map(int, frame_bbox_shape)
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    margins = [
        int(lr0),
        int(lc0),
        int(fh - 1 - lr1),
        int(fw - 1 - lc1),
    ]
    k = min(margins)
    assert k >= 1, "frame border thickness must be positive"
    assert all(m == k for m in margins), "frame border thickness must be uniform"
    return int(k)


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        p = np.rot90(patch, int(kk))
        m = np.rot90(mask, int(kk))
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq = []
    seen = set()
    for p, m in base:
        key = (m.shape, m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole-layout and fill each hole with its matching piece.

    D8 variant:
    - The frame border thickness equals k.
    - k is inferred from that thickness instead of from unit-square pieces.
    """
    g, frame_color, layout, hole_box = _frame_and_layout(np.array(grid, dtype=int))
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    k = _infer_k_from_border(sub.shape, hole_box)

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
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

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
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

    out = np.full(((int(layout.shape[0]) + 2) * k, (int(layout.shape[1]) + 2) * k), int(frame_color), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * k
        left = (int(s["c0"]) + 1) * k
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


# --- generator ---------------------------------------------------------------


K_CHOICES = [2, 3]
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
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
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
                p
                for p in frontier
                if 0 <= p[0] < H and 0 <= p[1] < W and not forbidden[p[0], p[1]] and p not in chosen
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

    density = sum(s["area"] for s in slots) / (h * w)
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
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
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
    frame_h, frame_w = layout_h + 2 * k, layout_w + 2 * k
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

    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[k : k + layout_h, k : k + layout_w] = np.where(layout_mask, 0, frame_color)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda piece: -int(piece["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

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
    k = random.choice(K_CHOICES)
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
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(s, piece_colors[i], k) for i, s in enumerate(slots)]
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
