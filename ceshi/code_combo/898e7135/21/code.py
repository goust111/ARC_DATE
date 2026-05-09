# Combo variant of 898e7135 (index 21)
# combo_spec: D4/4+D13/2
# Name: 匹配-重着色-填充三步

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
Design analysis for combo variant D4/4 + D13/2
==============================================

a) Concrete behavior and process trace
--------------------------------------
- Original task:
  1. Read one monochrome rectangular frame with bg holes inside.
  2. Read outside pieces that are k-by-k block-scaled hole shapes.
  3. Match each piece to one hole-region by D4 shape.
  4. Paste the matched piece directly into the scaled frame, keeping its color.
- Combo variant:
  1. Keep the same shape-matching problem and the same inference of k.
  2. Change the frame from monochrome to structured multicolor walls:
     every slot is surrounded by one consistent wall color, while the rest of
     the frame uses a base wall color.
  3. After a piece is matched to its slot, ignore the piece's input color and
     recolor the aligned piece to that slot's surrounding wall color.
  4. Fill the recolored piece into the k-scaled frame, so piece and wall blend.
- Small trace:
  - Suppose k=2 and a 3-cell L-shaped slot is surrounded by teal walls.
  - The input piece for that slot appears outside the frame as an orange 2x2
    block sprite rotated by 90 degrees.
  - Step 1 reorients the orange sprite so its downsampled occupancy matches the
    L-slot.
  - Step 2 reads the wall ring around that L-slot and gets teal.
  - Step 3 recolors the now-aligned sprite from orange to teal and pastes it
    into the scaled frame.
- Difference from the original:
  - The original output follows immediately from shape matching.
  - Here shape determines the pairing first, but the visible output color is a
    second-stage attribute taken from the matched hole boundary.

b) Complexity calibration
-------------------------
- Pure recolor on a monochrome frame would be weak because the boundary cue
  would be invisible. The frame therefore contains exactly two repeated slot
  wall colors plus one base wall color. This is enough variation to reveal the
  recolor rule while keeping the color logic simple.
- The input piece colors stay unrelated to wall colors, so the output cannot be
  explained by "keep piece color" or by a fixed global mapping.
- Non-unit slots remain D4-distinct, while unit slots may repeat as in the
  source task. This preserves the original matching difficulty without adding
  unnecessary ambiguity.

c) Implicit assumptions that change
-----------------------------------
- Original `main()` assumed the largest foreground component was a monochrome
  frame and used its dominant color as the entire output background. That no
  longer holds: the frame is multicolor, so the solver must identify the frame
  by rectangular-border structure rather than monochrome area, and it must
  scale the whole frame image instead of filling a single background color.
- Original direct paste after matching is invalid because the matched oriented
  piece is not yet final; we must recolor it after matching and before filling.
- Original wall extraction only needed the hole mask. The combo solver also
  needs the original frame crop because slot wall colors are read from the
  4-neighbor ring around each hole component.
- Generator-side new assumption: a slot may share some boundary walls with
  nearby slots, but it must still own enough exclusive boundary walls for its
  designated ring color to be the majority among its 4-neighbor wall contacts.
  The generator enforces this majority condition before accepting a layout.

d) Placement legality
---------------------
- The frame is still placed first on a zero background canvas.
- Outside pieces are still placed with `random_free_location(..., padding=1,
  padding_connectivity=8)` so they stay separated from the frame and from each
  other.
- Ring-coloring is only applied to wall cells inside the frame rectangle, never
  outside bounds. Shared boundary walls stay in the base frame color, and the
  generator only accepts layouts where each slot's exclusive ring color remains
  the local majority cue.
- The output is built from a k-scaled frame image, then recolored matched
  pieces are pasted exactly into scaled hole positions.

e) Visual observability
-----------------------
- A human can infer the rule from a few examples because each output piece now
  inherits the local wall color of its matched hole, not its input color.
- The required three-step chain is visible:
  1. use shape to find the right hole,
  2. read the hole's surrounding wall color,
  3. recolor and fill.
- Using exactly two repeated slot wall colors per sample provides repeated
  evidence that the recolor cue is local wall context, not arbitrary per-piece
  memorization.

f) Visual cleanliness
---------------------
- The frame has only three wall-color roles in one sample: one base wall color
  and two slot-ring colors. This keeps the scene readable.
- Piece sprites remain plain monochrome blocks in the input, so the extra
  difficulty comes from the interaction between shape and wall color, not from
  clutter inside the pieces.

g) Color-change inferability
----------------------------
- Color change is structured, not arbitrary: the filled piece color equals the
  wall color touching its matched slot.
- The generator keeps piece colors disjoint from the slot wall colors, so the
  recolor step is real and easy to observe.
- Both slot wall colors are guaranteed to appear in every sample, so the recolor
  rule has repeated support instead of a one-off exception.

9) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumption: the frame is the unique foreground component whose bbox has
  a complete non-bg border and contains bg holes.
  Generator guarantee: the frame is built exactly that way.
- Solver assumption: each slot's orthogonal wall contact set has one dominant
  color.
  Generator guarantee: exclusive ring cells receive the slot color, shared wall
  cells stay base-colored, and acceptance requires the slot color to remain the
  majority around that slot.
- Solver assumption: at least one solid k-by-k square piece exists so k can be
  inferred robustly.
  Generator guarantee: each sample contains 1-2 unit slots, producing such
  square pieces.
"""


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


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if hh % k != 0 or ww % k != 0:
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq = []
    seen = set()
    for p, m in base:
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


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


def _build_layout_mask(slots: list[dict], layout_h: int, layout_w: int) -> np.ndarray:
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    return layout_mask


def _holes_touch_all_layout_sides(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _slot_rings(slots: list[dict], layout_h: int, layout_w: int) -> list[list[tuple[int, int]]] | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    hole_cells = np.zeros((frame_h, frame_w), dtype=bool)
    for s in slots:
        hole_cells[s["coords"][:, 0] + 1, s["coords"][:, 1] + 1] = True

    cell_to_slots: dict[tuple[int, int], set[int]] = {}
    slot_edges: list[list[tuple[int, int]]] = []
    for si, s in enumerate(slots):
        edges: list[tuple[int, int]] = []
        for r, c in s["coords"]:
            rr = int(r) + 1
            cc = int(c) + 1
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if hole_cells[nr, nc]:
                    continue
                cell = (int(nr), int(nc))
                edges.append(cell)
                cell_to_slots.setdefault(cell, set()).add(int(si))
        if not edges:
            return None
        slot_edges.append(edges)

    rings: list[list[tuple[int, int]]] = []
    for si, edges in enumerate(slot_edges):
        exclusive_cells = sorted({cell for cell in edges if len(cell_to_slots[cell]) == 1})
        exclusive_edges = sum(1 for cell in edges if len(cell_to_slots[cell]) == 1)
        shared_edges = len(edges) - exclusive_edges
        if not exclusive_cells or exclusive_edges <= shared_edges:
            return None
        rings.append(exclusive_cells)
    return rings


def _find_frame_component(g: np.ndarray, comps: list[dict], bg: int) -> dict:
    frame_like = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if not np.any(sub == int(bg)):
            continue
        if not bool(np.all(sub[0, :] != int(bg)) and np.all(sub[-1, :] != int(bg))):
            continue
        if not bool(np.all(sub[:, 0] != int(bg)) and np.all(sub[:, -1] != int(bg))):
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
    gg = 0
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        gg = gcd(int(gg), int(h))
        gg = gcd(int(gg), int(w))
    return int(max(gg, 1))


def _slot_wall_color(sub: np.ndarray, slot_coords: np.ndarray, bg: int) -> int:
    cols = []
    for r, c in slot_coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            val = int(sub[nr, nc])
            if val != int(bg):
                cols.append(int(val))
    assert cols, "slot must touch frame walls"
    return int(Counter(cols).most_common(1)[0][0])


def _recolor_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), 0).astype(int)


def _make_piece(slot: dict, input_color: int, k: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    aligned_patch = np.where(out_mask, int(input_color), 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(aligned_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask = out_mask
        in_patch = aligned_patch

    return {
        "slot": slot,
        "input_color": int(input_color),
        "aligned_mask": out_mask,
        "aligned_patch": aligned_patch,
        "in_mask": in_mask,
        "in_patch": in_patch,
    }


def _build_frame_image(
    slots: list[dict],
    layout_h: int,
    layout_w: int,
    base_frame_color: int,
    slot_colors: list[int],
    rings: list[list[tuple[int, int]]],
) -> np.ndarray:
    frame = np.full((layout_h + 2, layout_w + 2), int(base_frame_color), dtype=int)
    for s in slots:
        frame[s["coords"][:, 0] + 1, s["coords"][:, 1] + 1] = 0
    for color, ring in zip(slot_colors, rings):
        for r, c in ring:
            frame[int(r), int(c)] = int(color)
    return frame


def _build_expected_output(frame_img: np.ndarray, pieces: list[dict], k: int) -> np.ndarray:
    out = np.kron(frame_img, np.ones((k, k), dtype=int))
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        patch = _recolor_patch(p["aligned_mask"], int(p["slot"]["wall_color"]))
        paste(out, patch, int(top), int(left), mask=p["aligned_mask"], overwrite=True)
    return out


def _build_input_canvas(pieces: list[dict], frame_img: np.ndarray, distractor_color: int) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
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

    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match block sprites to frame holes, recolor them from local wall cues, then fill a scaled frame.

    Input structure:
    - `bg` is the most common color.
    - One rectangular foreground component is a multicolor frame whose interior contains bg holes.
    - Outside the frame are monochrome k-by-k block sprites plus singleton distractors.

    Rule:
    - Extract each hole component as a slot and infer k from the smallest solid square sprite.
    - Match every sprite to a slot by block-level shape under rotation/flip.
    - After matching, recolor the aligned sprite to the color of that slot's surrounding walls, then paste it into the k-scaled frame.

    Key insight: Shape identifies the correct hole, but the final color comes from the matched hole's boundary, not from the sprite itself.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame and pieces"
    frame_comp = _find_frame_component(g, comps, bg)
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    assert bbox(hole) is not None, "frame must enclose at least one bg hole"

    slots = []
    for coords in components_mask(hole, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "coords": coords,
                "wall_color": _slot_wall_color(sub, coords, bg),
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
        pm = np.zeros_like(patch, dtype=bool)
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
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vm": vm})

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

    out = np.kron(sub, np.ones((k, k), dtype=int))
    for s, cand in placements:
        top = int(s["r0"]) * int(k)
        left = int(s["c0"]) * int(k)
        patch = _recolor_patch(cand["vm"], int(s["wall_color"]))
        paste(out, patch, int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    base_frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    wall_colors = [int(palette.pop()), int(palette.pop())]
    non_unit_count = n_slots - n_units
    if len(palette) < non_unit_count:
        return None
    non_unit_colors = list(map(int, palette[:non_unit_count]))
    unit_color = int(random.choice(non_unit_colors))

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    rings = _slot_rings(slots, layout_h, layout_w)
    if rings is None:
        return None

    slot_wall_colors = wall_colors.copy()
    while len(slot_wall_colors) < n_slots:
        slot_wall_colors.append(random.choice(wall_colors))
    random.shuffle(slot_wall_colors)
    for s, wall_color in zip(slots, slot_wall_colors):
        s["wall_color"] = int(wall_color)

    frame_img = _build_frame_image(slots, layout_h, layout_w, base_frame_color, slot_wall_colors, rings)

    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]
    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    expected_out = _build_expected_output(frame_img, pieces, k)
    canvas = _build_input_canvas(pieces, frame_img, distractor_color)
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
