# Combo variant of 898e7135 (index 36)
# combo_spec: D5/1+D8/1
# Name: 读取边框后去除

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
Design analysis for combo variant D5/1 + D8/1
==============================================

spec_delta_summary:
  preserved_core:
    - The largest foreground component is still a monochrome frame enclosing a logical hole-layout.
    - Each outside piece is still an exact k-by-k block inflation of one slot shape, up to D4 rotation/flip.
    - Matching is still done by downsampling piece masks back to logical-cell resolution.
  fused_changes:
    - The frame border is no longer 1 pixel thick; its uniform thickness is exactly k.
    - The output no longer includes any border pixels; only the scaled interior layout is returned.
    - Therefore the only visible source for k is in the input frame geometry, and that source disappears in the output.

behavior_trace:
  concrete_example:
    - Let the logical layout be 5x4 and let k=3.
    - Original task: input frame bbox is 7x6 because the border is fixed to 1. Output is 21x18 and still includes the border.
    - D8-only change: input frame bbox becomes 11x10 because each side contributes a 3-pixel border. Solver reads k=3 from the four margins.
    - D5-only change on top of that: output is no longer 21x18. The solver drops the border and returns only the 15x12 scaled interior.
    - Net effect: the solver must first read k from geometry that will not survive into the answer.
  observable_difference:
    - Across examples, people can see border thickness switch between 2 and 3 in the input.
    - In the paired outputs, the filled arrangement has the same internal walls/holes but lacks the outer border entirely.

complexity_calibration:
  - Keep k in the small set {2, 3}; this is visually countable in 2-4 examples.
  - Keep 3-6 slots so the matching task remains substantial but not noisy.
  - Remove unit slots and forbid full-square logical slot masks. Otherwise the old heuristic "smallest solid square piece gives k" would still solve many cases, weakening the fusion.
  - Keep non-unit slot shapes D4-distinct so matching stays unambiguous even after output cropping removes the frame.

implicit_assumption_audit:
  original_main_lines:
    - `frame_comp = max(comps, key=area)`: still valid because the thick frame remains the largest non-bg component.
    - `hole_box = bbox(sub == bg)`: still valid because holes remain bg inside the frame.
    - `layout = hole[lr0:lr1+1, lc0:lc1+1]`: valid only if holes touch all four sides of the logical layout; generator enforces that.
    - Original k-inference from solid square pieces is invalid under this combo and must be removed entirely.
    - `downsample_blocks` still assumes every valid piece is perfectly tiled by kxk blocks; generator preserves that.
    - Slot matching by area plus D4-equivalent logical mask still works after fusion because only k-source and output crop changed.
  new_assumptions_introduced:
    - The frame border thickness is uniform on all four sides and equals k.
    - Pieces do not contain stray partial blocks after rotation/flip.
    - Output coordinates must be expressed in interior-layout coordinates rather than frame coordinates.
    - Generator must avoid square clues so the intended k-source is necessary.

placement_legality:
  - Input frame size is `(layout_h + 2*k, layout_w + 2*k)`, so canvas budgeting must account for the thicker border.
  - Pieces are placed with at least one bg cell of 8-neighborhood clearance from the frame and from each other.
  - Distractors remain isolated single pixels in a dedicated color and are never merged into multi-cell components.
  - Because the output omits the border, expected-output assembly pastes each piece at `(slot_r*k, slot_c*k)` instead of offsetting by one border cell.

visual_observability:
  - The thick border is a direct, human-visible code for k.
  - The output's missing border is also direct: the frame color still appears as internal walls/background, but the enclosing ring is absent.
  - Since the border is the only k clue and is removed from the answer, the fusion is visible as an information-transfer step rather than a cosmetic crop.

tidiness_and_inference:
  - Only one frame color, one distractor color, and one color per piece are used.
  - No markers overlap the frame or slot interiors beyond the intended pieces.
  - Colors are not remapped by hidden numeric order; shape-position correspondence is the only matching key.

variant_metric_for_validation:
  - Metric 1: border-thickness distribution over generated samples should include both k=2 and k=3.
  - Metric 2: zero generated piece components should be solid squares, preventing fallback to the original k heuristic.
  - Metric 3: for every sample, `output.shape == (layout_h*k, layout_w*k)` while `frame_bbox.shape == (layout_h+2*k, layout_w+2*k)`, proving the border is truly removed.
"""


def _find_frame(grid: np.ndarray) -> tuple[np.ndarray, int, dict, np.ndarray, tuple[int, int, int, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    frame_vals = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_vals).most_common(1)[0][0])
    return g, bg, frame_comp, sub, (fr0, fc0, fr1, fc1), frame_color


def _extract_layout_and_k(frame_subgrid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    hole = frame_subgrid == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    fh, fw = map(int, frame_subgrid.shape)
    margins = [int(lr0), int(lc0), int(fh - 1 - lr1), int(fw - 1 - lc1)]
    k = min(margins)
    assert k >= 1, "frame border thickness must be positive"
    assert all(m == k for m in margins), "frame border thickness must be uniform"
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return layout, int(k)


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
    Reconstruct the scaled interior layout by fitting outside pieces into frame holes.

    Input structure:
    - `0` is still the global background, and the largest non-background component is a monochrome frame.
    - The frame encloses background holes and frame-colored walls; its border thickness is uniform and equals the scale factor `k`.
    - Outside the frame are separated monochrome pieces made of exact `k x k` blocks, plus singleton distractor pixels.

    Rule:
    - Read `k` from the frame border thickness, then crop the enclosed hole-layout at logical-cell resolution.
    - Match each outside piece to one hole region by comparing block-downsampled shapes under rotation/flip.
    - Render only the scaled interior layout: fill matching holes with their pieces, keep internal walls in frame color, and discard the surrounding border entirely.

    Key insight:
    - The border is a temporary codebook for `k`: it tells you how to decode the pieces, but it is not part of the answer.
    """
    g, bg, frame_comp, frame_sub, _frame_bbox, frame_color = _find_frame(np.array(grid, dtype=int))
    layout, k = _extract_layout_and_k(frame_sub, bg)

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

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
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = downsample_blocks(vm)
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
        top = int(slot["r0"]) * int(k)
        left = int(slot["c0"]) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


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

    density = sum(slot["area"] for slot in slots) / (h * w)
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
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
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
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
        return None

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
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
