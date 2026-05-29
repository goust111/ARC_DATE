# Variant of 898e7135
# Dimension: D6 — 干扰项的类型与作用 (meta)
# Change: 8-13 个单像素干扰项，使用专用颜色（与框架色和所有精灵色不同），散布在画布上；输出中完全移除... → 干扰项使用精灵颜色

from __future__ import annotations

from collections import Counter
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


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   unchanged_core:
#     - The largest foreground component is still the monochrome frame.
#     - Slots are still bg holes inside that frame.
#     - Every true piece is still a k-by-k block upscale of one slot, matched by
#       block-level shape under D4 transforms.
#     - The output is still the cropped, scaled frame filled with matched pieces.
#   changed_dimension:
#     - Distractors remain isolated single pixels, count 8-13.
#     - They no longer use a fresh dedicated color.
#     - Each distractor color is sampled from the actual piece-color set, so the
#       same visible color now appears both on real pieces and on removable noise.
#   concrete_pixel_behavior:
#     - Input: a large piece may be color 4, and several isolated color-4 pixels
#       may appear elsewhere on the canvas. Those isolated pixels are not pieces.
#     - Output: only the scaled frame and matched pieces remain; all isolated
#       singletons disappear, even though their colors are reused by real pieces.
#   small_trace:
#     - Original generator: place a green piece plus red singleton noise. A solver
#       could hypothetically drop all red cells by color.
#     - Variant generator: place a green piece plus several green singleton noise
#       pixels. Color-only filtering is now invalid; the solver must distinguish
#       objects by connectivity and area.
#     - Solver impact: when scanning non-frame foreground components, area-1
#       components must be treated as distractors regardless of their color, while
#       area>1 components remain candidate pieces.
#   complexity_calibration:
#     - The variant stays visually legible because the only new ambiguity is
#       color-sharing between real pieces and noise.
#     - We keep distractors as isolated singletons, so the human-visible cue is
#       still simple: tiny lone pixels vanish, larger blocky sprites are used.
#     - To make the change observable, the generator forces every piece color to
#       appear on at least one distractor whenever there are enough distractors.
#   implicit_assumption_audit_main:
#     - find_components(..., monochromatic=False): assumes bg is still 0-most-
#       common and that frame stays the largest foreground component. Generator
#       preserves both.
#     - frame_comp = max(..., area): assumes noise never exceeds frame area.
#       Generator keeps distractors singleton and pieces much smaller than frame.
#     - slot extraction from bg holes: assumes distractors never enter the frame
#       interior. Generator only places them on free canvas background.
#     - piece collection: the original code ignored area-1 components. That was
#       harmless before because distractors had a unique color; it remains correct
#       now only if singleton-vs-piece is the intended discriminator. The new code
#       makes this explicit and documents that color is irrelevant here.
#     - k inference from smallest solid square piece: assumes at least one unit
#       slot exists, so at least one real k-by-k solid piece exists. Generator
#       still guarantees 1-2 unit slots.
#     - candidate matching by area and downsampled block mask: assumes each slot
#       has a unique D4 shape among non-unit slots and unit slots are matched only
#       by remaining unused solid k-by-k squares. Generator preserves this.
#   placement_legality:
#     - Frame is pasted first.
#     - Pieces are scattered with >=1-cell 8-neighborhood clearance from frame and
#       from each other using random_free_location(..., padding=1).
#     - Distractors are then placed only on empty cells whose full 3x3 neighborhood
#       is empty, so they remain isolated single-pixel components and cannot merge
#       with pieces of the same color.
#   visual_observability:
#     - Across examples, the same colors now appear in both large sprites and tiny
#       removable dots. This is visible from raw grids and forces a connectivity-
#       based interpretation rather than a color-exclusive one.
#   cleanliness:
#     - No overlaps are introduced.
#     - Distractors stay sparse and isolated, so the scene remains readable.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    Variant-specific note:
    - Single-pixel distractors may reuse actual piece colors, so pieces cannot be
      separated from noise by color exclusivity.
    - The discriminator is connectivity + size: isolated singleton foreground
      components are distractors and are discarded; larger components are pieces.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Largest foreground component is the frame, as in the base task.
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    # Tight-cropped hole layout inside the frame bbox.
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    # Non-frame singleton components are distractors even if their colors are
    # shared with true pieces. Real pieces are the remaining non-singletons.
    singleton_distractors = []
    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area <= 1:
            singleton_distractors.append(comp)
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": area, "mask": pm, "patch": patch})

    assert pieces, "at least one non-frame piece must exist"
    _ = singleton_distractors  # documented variant behavior; no direct output role

    # k remains identifiable from the guaranteed unit-slot k-by-k solid piece.
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((int(hh // k), int(ww // k)), dtype=bool)
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
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

    out = np.full(((int(layout.shape[0]) + 2) * k, (int(layout.shape[1]) + 2) * k), int(bgc), dtype=int)
    for s, cand in placements:
        paste(out, cand["vp"], (int(s["r0"]) + 1) * k, (int(s["c0"]) + 1) * k, mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

"""Generator for ARC-AGI-2 task 898e7135, D6 variant.

Output-first strategy:
  1. Build a hole/wall layout of 4-connected slots (8-separated, includes units).
  2. Create one monochrome piece per slot, with random D4 transform on input.
  3. Assemble the frame.
  4. Scatter pieces outside the frame on a bg=0 canvas.
  5. Scatter 8-13 isolated singleton distractors, but color them with actual
     piece colors rather than a fresh dedicated distractor color.

The generator validates main(input) == expected_output before returning.
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
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a tight-cropped bool mask."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow one 4-connected slot on currently free cells."""
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
    """Dilate a bool mask by 1 cell in 8-connectivity."""
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
    """Mark coords and their 8-neighbors as forbidden in-place."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """Generate n_slots 4-connected hole components, with units first."""
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
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

    density = sum(s["area"] for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Create one piece patch for a slot."""
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {"color": color, "slot": slot, "out_patch": patch, "out_mask": mask, "in_patch": patch, "in_mask": mask}

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": color,
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    """Assemble the expected output grid."""
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _place_singleton_distractors(canvas: np.ndarray, piece_colors: list[int], n_distractors: int) -> bool:
    """Place isolated singleton distractors that reuse actual piece colors."""
    required = piece_colors[:]
    random.shuffle(required)
    if n_distractors < len(required):
        required = required[:n_distractors]
    distractor_colors = required + [random.choice(piece_colors) for _ in range(n_distractors - len(required))]
    random.shuffle(distractor_colors)

    coords: list[tuple[int, int]] = []
    H, W = canvas.shape
    mark = -1
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if len(coords) >= n_distractors:
            break
        rr = random.randint(0, H - 1)
        cc = random.randint(0, W - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(H, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(W, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = mark
        coords.append((rr, cc))

    if len(coords) < max(MIN_DISTRACTORS, len(required)):
        for rr, cc in coords:
            canvas[rr, cc] = 0
        return False

    for (rr, cc), color in zip(coords, distractor_colors):
        canvas[rr, cc] = int(color)
    return True


def _build_input_canvas(
    pieces: list[dict], layout_mask: np.ndarray, layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray | None:
    """Build the input grid: frame + scattered pieces + color-camouflaged singleton noise."""
    frame_h, frame_w = layout_h + 2, layout_w + 2
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
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
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

    piece_colors = sorted({int(p["color"]) for p in pieces})
    n_distractors = random.randint(*DISTRACTOR_RANGE)
    if not _place_singleton_distractors(canvas, piece_colors, n_distractors):
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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color)
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
