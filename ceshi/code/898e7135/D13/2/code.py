# Variant of 898e7135
# Dimension: D13 — 多步处理与排序 (multi_step)
# Change: 单步处理 — 所有精灵同时匹配并填入，无顺序依赖 → 先匹配后变换再填入

from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np

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
Design analysis for the D13 variant
===================================

Reference-task note
-------------------
- The prompt names `d35bdbdc` as the multi-step inspiration, but this workspace only
  exposes its evaluation JSON, not a solver file. I therefore use the original
  `898e7135` code plus the summarized "match first, then transfer/transform an
  attribute" mechanism as the design baseline.

a) Concrete behavior and process trace
--------------------------------------
- Original task:
  1. Read the frame and its hole-components.
  2. Read outside pieces as solid k-by-k block sprites.
  3. Match each piece to a slot by block-level shape under D4.
  4. Paste the matched piece directly into the scaled frame.
- Variant:
  1. Keep the same frame/slot extraction and shape matching.
  2. Change pieces from monochrome solids to two-color sprites: every occupied
     k-by-k block contains exactly one "accent" pixel in its top-left corner.
     The whole piece may still be globally rotated/flipped in the input.
  3. After the correct piece-slot pairing is known, inspect the matched slot's
     horizontal position inside the frame. Slots in the left half undergo one
     extra block-local transform: every occupied k-by-k block is mirrored
     horizontally, moving its accent pixel from top-left to top-right. Slots in
     the right half stay canonical.
  4. Paste the transformed sprite into the scaled frame.
- Small trace:
  - Suppose a 2-cell L-slot lies on the left side of the frame.
  - Input contains the matching scattered piece globally rotated 90 degrees; its
    per-block accent pixels rotate with it.
  - Step 1 reverses that global rotation so the piece's occupancy matches the
    slot mask and every occupied block is back to canonical top-left accent.
  - Step 2 sees the slot is on the left half, so each occupied block is
    mirrored horizontally, moving the accent to top-right.
  - Step 3 pastes that locally mirrored piece into the output frame.
- Difference from the original:
  - The original output is determined immediately after shape matching.
  - The variant requires an intermediate recovered orientation, then a second
    transformation that depends on the matched slot rather than on the input
    piece alone.

b) Complexity calibration
-------------------------
- A plain recolor on monochrome pieces would be visually invisible, so pieces
  now carry an asymmetric internal marker.
- The extra rule uses only two branches (`left half` vs `right half`) so humans
  can infer it from 2-4 examples, but every sample is forced to contain both
  kinds of slots so the rule does not collapse into a single case.
- The transform does not depend on raw color ids or on arbitrary hidden order;
  it depends on visible slot position.

c) Old implicit assumptions that no longer hold
-----------------------------------------------
- Original `dihedral_variants` deduplicated by mask bytes only. That was safe
  for monochrome pieces because equal masks implied equal visible pieces. It is
  unsafe now: symmetric masks can support different internal marker
  orientations, so the solver must preserve patch-distinct variants.
- Original direct paste after matching assumed the matched oriented patch was
  already final. That is no longer true; we must insert a second transform step.
- Original pieces were visually symmetric inside each occupied k-by-k block.
  Now the solver must recover a canonical orientation before applying the
  slot-driven block-local mirror.

d) Placement legality
---------------------
- The frame still occupies a single monochrome rectangular border and is placed
  first on a zero background canvas.
- Pieces remain outside the frame and each other with at least one background
  cell of separation via `random_free_location(..., padding=1, connectivity=8)`.
- Distractors remain isolated singletons in one dedicated color.
- The output still has size `(layout_h + 2) * k` by `(layout_w + 2) * k` and
  only pastes inside bounds.

e) Visual observability
-----------------------
- The new behavior is human-visible because every occupied block contains a
  single corner marker. After placement, left-half slots show that marker on the
  top-right of each block, right-half slots keep it on the top-left.
- The needed rule is inferable from few examples:
  1. Match piece to hole by outer shape.
  2. Reorient piece to fit.
  3. Compare final block-corner markers against slot position.

f) Visual cleanliness
---------------------
- Only two layers are visible inside pieces: a base piece color and one shared
  accent color. This keeps the frame readable and avoids clutter.
- The transform is local to occupied blocks, so shapes stay crisp and hole
  silhouettes remain identical to the original task.

g) Color-change inferability
----------------------------
- No arbitrary color remapping is used. Colors stay the same; only the accent
  pixel's corner changes.
- The accent color is shared across pieces, giving repeated evidence that the
  rule is about geometric transformation rather than per-color remapping.

9) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumption: every piece has exactly one canonical orientation in which
  each occupied k-by-k block has its unique accent pixel at the block's top-left.
  Generator guarantee: all aligned pieces are built that way, then globally
  transformed by a random dihedral.
- Solver assumption: every sample contains at least one left-half slot and one
  right-half slot so both post-match branches appear. Generator guarantee:
  layouts failing that condition are rejected.
- Solver assumption: slot centers are not exactly on the horizontal midline that
  separates left from right. Generator guarantee: such layouts are rejected.
"""


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


# --- shared helpers -----------------------------------------------------------

def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = hh // k, ww // k
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    variants = []
    seen = set()
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        for pp, mm in ((p, m), (np.fliplr(p), np.fliplr(m))):
            key = (pp.shape, pp.tobytes(), mm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append((pp, mm))
    return variants


def _slot_is_left(coords: np.ndarray, layout_w: int) -> bool:
    center_c = float(coords[:, 1].mean())
    return center_c < (float(layout_w) - 1.0) / 2.0


def _slot_is_centered(coords: np.ndarray, layout_w: int) -> bool:
    center_c = float(coords[:, 1].mean())
    return abs(center_c - (float(layout_w) - 1.0) / 2.0) < 1e-9


def _decorate_scaled_piece(block_mask: np.ndarray, color: int, accent_color: int, k: int) -> np.ndarray:
    h, w = map(int, block_mask.shape)
    patch = np.zeros((h * k, w * k), dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(block_mask[r, c]):
                continue
            rr = r * k
            cc = c * k
            patch[rr : rr + k, cc : cc + k] = int(color)
            patch[rr, cc] = int(accent_color)
    return patch


def _is_canonical_marker_orientation(patch: np.ndarray, mask: np.ndarray, k: int) -> bool:
    vals, counts = np.unique(patch[mask], return_counts=True)
    if len(vals) != 2:
        return False
    accent = int(vals[np.argmin(counts)])
    block_mask = _downsample_blocks(mask, k)
    if block_mask is None:
        return False
    for r in range(block_mask.shape[0]):
        for c in range(block_mask.shape[1]):
            if not bool(block_mask[r, c]):
                continue
            block = patch[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if int(block[0, 0]) != accent:
                return False
            if int(np.count_nonzero(block == accent)) != 1:
                return False
    return True


def _apply_left_slot_transform(patch: np.ndarray, mask: np.ndarray, k: int, *, is_left: bool) -> np.ndarray:
    if not is_left:
        return patch.copy()
    out = patch.copy()
    block_mask = _downsample_blocks(mask, k)
    assert block_mask is not None
    for r in range(block_mask.shape[0]):
        for c in range(block_mask.shape[1]):
            if not bool(block_mask[r, c]):
                continue
            rr = r * k
            cc = c * k
            out[rr : rr + k, cc : cc + k] = np.fliplr(out[rr : rr + k, cc : cc + k])
    return out


def _find_frame_component(g: np.ndarray, comps: list[dict], bg: int) -> dict:
    frame_like = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        cols = {int(g[int(r), int(c)]) for r, c in coords}
        if len(cols) != 1:
            continue
        color = next(iter(cols))
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if not np.any(sub == int(bg)):
            continue
        if not bool(np.all(sub[0, :] == color) and np.all(sub[-1, :] == color)):
            continue
        if not bool(np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
            continue
        frame_like.append(comp)
    assert frame_like, "could not identify frame"
    return max(frame_like, key=lambda d: int(d["area"]))


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    if k is not None:
        return int(k)
    dims = []
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        dims.extend([int(h), int(w)])
    gg = 0
    for d in dims:
        gg = gcd(int(gg), int(d))
    return int(max(gg, 1))


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


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1})
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
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": size, "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _annotate_slots(slots: list[dict], layout_w: int) -> bool:
    left_count = 0
    right_count = 0
    for s in slots:
        coords = s["coords"]
        if _slot_is_centered(coords, layout_w):
            return False
        is_left = _slot_is_left(coords, layout_w)
        s["is_left"] = bool(is_left)
        if is_left:
            left_count += 1
        else:
            right_count += 1
    return left_count > 0 and right_count > 0


def _make_piece(slot: dict, color: int, accent_color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    aligned_patch = _decorate_scaled_piece(slot["mask"], int(color), int(accent_color), int(k))

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(aligned_patch, rot)
    if random.random() < 0.5:
        in_mask = np.fliplr(in_mask)
        in_patch = np.fliplr(in_patch)

    return {
        "color": int(color),
        "slot": slot,
        "aligned_patch": aligned_patch,
        "aligned_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        placed = _apply_left_slot_transform(
            p["aligned_patch"], p["aligned_mask"], int(k), is_left=bool(p["slot"]["is_left"])
        )
        paste(out, placed, int(top), int(left), mask=p["aligned_mask"], overwrite=True)
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
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
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
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda pp: -int(pp["in_mask"].sum())):
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
        paste(canvas, p["in_patch"], int(pr), int(pc), mask=p["in_mask"], overwrite=True)

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


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame and pieces"
    frame_comp = _find_frame_component(g, comps, bg)
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "coords": coords,
                "is_left": _slot_is_left(coords, layout.shape[1]),
            }
        )
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
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = _infer_k(pieces)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            if not _is_canonical_marker_orientation(vp, vm, k):
                continue
            candidates.append(
                {"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm}
            )

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

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        placed = _apply_left_slot_transform(cand["vp"], cand["vm"], int(k), is_left=bool(s["is_left"]))
        paste(out, placed, int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- generator ----------------------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    accent_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = list(map(int, palette[: n_slots - n_units]))
    unit_color = int(palette[n_slots - n_units])

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        r0, c0 = s["coords"][:, 0], s["coords"][:, 1]
        layout_mask[r0, c0] = True
    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
        return None
    if not _annotate_slots(slots, layout_w):
        return None

    for s in unit_slots:
        s["mask"] = np.ones((1, 1), dtype=bool)

    pieces = [_make_piece(s, non_unit_colors[i], accent_color, k) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, accent_color, k) for s in unit_slots]

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
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
