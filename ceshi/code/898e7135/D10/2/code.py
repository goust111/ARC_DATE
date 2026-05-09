# Variant of 898e7135
# Dimension: D10 — 颜色分配策略 (meta)
# Change: 非单元精灵颜色互不相同，单元精灵共享一种颜色；框架用一种专用色，干扰项用另一种专用色；共需 slot_count + 2... → 精灵颜色与洞边缘墙壁颜色对应

from __future__ import annotations

# Design analysis for the D10 variant
# ----------------------------------
# Source task check:
# - The prompt names source_task=d35bdbdc, but this workspace only contains its
#   evaluation JSON and no solver file. Per instruction, the reference-solver
#   step is therefore skipped rather than approximated from other variants.
#
# a) Pixel-level behavior and flow impact
# - Original input: one large frame component with a monochrome wall color and
#   several monochrome pieces outside the frame. Slot-to-piece matching is
#   purely by block-shape under D4 transforms.
# - Variant input: the frame stays a single connected component, but the wall
#   cells immediately touching a hole are recolored to that hole's piece color.
#   Example: if an L-shaped hole is surrounded by wall color 7, then the
#   matching scattered piece is also color 7. The output is still the scaled
#   frame with pieces inserted, but now the wall colors are preserved instead
#   of collapsing to one uniform frame color.
# - Flow change:
#   original main():
#     1. treat frame as multicolor component but recover one dominant wall color
#     2. scale to a uniform background of that color
#     3. match pieces only by shape
#   variant main():
#     1. keep the full cropped frame image with all wall colors
#     2. recover each slot's boundary color from the wall cells adjacent to it
#     3. scale the full multicolor frame image
#     4. match pieces by boundary color first, then shape under D4
#
# b) Complexity calibration
# - The variant should be visible, not cosmetic. Therefore the output preserves
#   multicolor walls, and each piece color appears twice in the input:
#   once on the scattered piece, once on the target hole boundary.
# - To avoid ambiguity from many independent colors hidden in tiny regions, each
#   slot uses exactly one boundary color on all of its adjacent wall cells.
# - Shape matching remains necessary because pieces are still randomly rotated
#   and/or flipped before scattering.
#
# c) Old implicit assumptions that break
# - `bgc = dominant frame color` is no longer valid: the frame is intentionally
#   multicolor.
# - `out = full(frame_color)` is invalid because it would erase the colored wall
#   labels that carry the new rule.
# - Matching without color is no longer the intended behavior. The old loop that
#   accepts the first equal block mask would ignore the D10 signal.
# - The old generator assumed the frame interior walls were all one color, so it
#   could paint holes with 0 and everything else with `frame_color`. That must
#   be replaced with per-slot wall coloring.
#
# d) Placement legality
# - The generator builds the frame first, then colors every wall cell that is
#   4-neighbor-adjacent to a given hole with that hole's color.
# - To keep this legal, boundary wall sets for different holes must be disjoint;
#   otherwise a single wall cell would need two colors. Layouts violating that
#   are rejected during generation.
# - Pieces are still scattered with >= 1 background cell of 8-neighbor padding,
#   and distractors remain isolated single pixels.
#
# e) Visual observability
# - A solver looking at 2-4 examples can now see that each outside sprite color
#   reappears as a colored wall ring around exactly one hole.
# - The rule does not depend on color-id ordering or any hidden numeric property.
#   The color cue is visible directly in the input and persists in the output.
#
# f) Visual cleanliness
# - Only one layer of annotation is added: colored wall cells directly adjacent
#   to holes. No extra markers are overlaid on top of pieces or holes.
# - Remaining frame walls stay in one neutral color, so the layout remains easy
#   to parse.
#
# g) Color-change inferability
# - Slot colors are reused structurally: each slot color is seen on the piece and
#   on the matching hole boundary. This repeated pairing is enough to infer the
#   mapping from few examples.
# - The total palette remains bounded by the ARC palette:
#   1 neutral frame filler + 1 distractor + slot_count unique slot colors.
#
# New main() assumptions introduced by the variant and enforced by generate_input()
# - The largest non-background connected component is still the frame.
# - Every slot has at least one adjacent wall cell, and all adjacent wall cells
#   of that slot share exactly one color.
# - Different slots have disjoint adjacent-wall sets, so slot boundary color is
#   unambiguous.
# - Each non-distractor piece is monochrome and equals its slot shape scaled by
#   k under a dihedral transform.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the multicolor frame and insert each outside piece into the hole whose
    boundary wall color matches the piece color and whose block-shape matches
    under rotation/flip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    # Keep the entire frame crop so the output preserves the wall recoloring.
    frame_img = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_mask = frame_img == bg
    slot_coords_list = components_mask(hole_mask, connectivity=4)
    assert slot_coords_list, "frame must contain at least one hole"

    all_holes = {tuple(map(int, rc)) for rc in np.argwhere(hole_mask)}
    slots = []
    for coords in slot_coords_list:
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True

        boundary_cols = []
        for r, c in coords:
            for nr, nc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)):
                if not (0 <= nr < frame_img.shape[0] and 0 <= nc < frame_img.shape[1]):
                    continue
                if (nr, nc) in all_holes:
                    continue
                boundary_cols.append(int(frame_img[nr, nc]))

        assert boundary_cols, "every slot must touch at least one wall cell"
        boundary_set = set(boundary_cols)
        assert len(boundary_set) == 1, "slot boundary color must be unambiguous"
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "color": int(boundary_cols[0]),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))

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
        piece_colors = set(map(int, patch[pm]))
        assert len(piece_colors) == 1, "pieces must be monochrome"
        pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(next(iter(piece_colors))),
            }
        )

    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

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

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            p = np.rot90(patch, kk)
            m = np.rot90(mask, kk)
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
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(p["area"]),
                    "color": int(p["color"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["color"]) != int(s["color"]):
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a same-color piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out = np.kron(frame_img, np.ones((k, k), dtype=int))
    for s, cand in placements:
        top = int(s["r0"]) * int(k)
        left = int(s["c0"]) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

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

GENERATE_MAX_RETRIES = 400


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = free_cells.tolist()
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
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


def _boundary_sets(slots: list[dict], layout_h: int, layout_w: int) -> list[set[tuple[int, int]]] | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    all_holes = set()
    shifted_slots: list[set[tuple[int, int]]] = []
    for s in slots:
        shifted = {tuple(map(int, rc)) for rc in (s["coords"] + 1)}
        shifted_slots.append(shifted)
        all_holes |= shifted

    owners: dict[tuple[int, int], int] = {}
    all_boundaries: list[set[tuple[int, int]]] = []
    for idx, shifted in enumerate(shifted_slots):
        bset: set[tuple[int, int]] = set()
        for r, c in shifted:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < frame_h and 0 <= nc < frame_w):
                    continue
                if (nr, nc) in all_holes:
                    continue
                bset.add((int(nr), int(nc)))
        if not bset:
            return None
        for cell in bset:
            if cell in owners and owners[cell] != idx:
                return None
            owners[cell] = idx
        all_boundaries.append(bset)
    return all_boundaries


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            sm = _tight_mask(coords)
            slots.append({"coords": coords, "area": 1, "mask": sm, "shape_key": _canonical_shape_key(sm)})
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
            sm = _tight_mask(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    boundaries = _boundary_sets(slots, h, w)
    if boundaries is None:
        return None
    for s, bset in zip(slots, boundaries):
        s["boundary"] = bset

    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral and slot["area"] > 1:
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
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_frame_image(slots: list[dict], layout_h: int, layout_w: int, filler_color: int, slot_colors: list[int]) -> np.ndarray:
    frame = np.full((layout_h + 2, layout_w + 2), int(filler_color), dtype=int)
    for s in slots:
        shifted = s["coords"] + 1
        frame[shifted[:, 0], shifted[:, 1]] = 0
    for s, color in zip(slots, slot_colors):
        for r, c in s["boundary"]:
            frame[int(r), int(c)] = int(color)
    return frame


def _build_expected_output(pieces: list[dict], frame_img: np.ndarray, k: int) -> np.ndarray:
    out = np.kron(frame_img, np.ones((k, k), dtype=int))
    for p in pieces:
        shifted = p["slot"]["coords"] + 1
        top = int(shifted[:, 0].min()) * int(k)
        left = int(shifted[:, 1].min()) * int(k)
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(pieces: list[dict], frame_img: np.ndarray, distractor_color: int) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
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
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
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
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    needed_colors = n_slots + 2
    if len(palette) < needed_colors:
        return None
    filler_color = palette.pop()
    distractor_color = palette.pop()
    slot_colors = [palette.pop() for _ in range(n_slots)]

    frame_img = _build_frame_image(slots, layout_h, layout_w, filler_color, slot_colors)
    pieces = [
        _make_piece(slot, slot_colors[i], k, apply_dihedral=(slot["area"] > 1))
        for i, slot in enumerate(slots)
    ]
    expected_out = _build_expected_output(pieces, frame_img, k)

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
