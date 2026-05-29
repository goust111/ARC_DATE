# Combo variant of 898e7135 (index 2)
# combo_spec: D1/2+D10/0
# Name: 纯几何方向匹配

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and flow impact
#    - The frame, hole layout, scaling factor k, and singleton distractors stay in the original
#      family. The output is still the tight enlarged frame with all slots filled.
#    - D1/2 changes the matching rule: a scattered piece must already have the exact block-level
#      orientation of its destination slot. Rotation and reflection are forbidden.
#    - D10/0 changes the color regime: every real piece, including unit pieces, uses one shared
#      piece_color. This removes color as an identity cue.
#    - Small trace:
#      suppose the layout contains two L-triomino slots that are the same under D4 but point in
#      different directions, plus a unit slot. Original task: distinct colors and allowed dihedral
#      transforms let the solver identify each complex piece by color and rotate if needed.
#      Combo variant: both L pieces have the same color, so the only visible distinction is their
#      exact orientation. The strict solver must compare the downsampled masks as seen.
#
# b) Complexity calibration
#    - If all non-unit slots were D4-unique, then removing rotation/flip would be visually weak:
#      the samples would still look solvable by rough shape class alone.
#    - To make the strict-orientation rule observable, the generator always inserts one pair of
#      non-unit slots from the same D4 family but with different exact oriented masks.
#    - To avoid ambiguity, every other non-unit slot is unique under D4, and exact-shape duplicates
#      are forbidden among non-unit slots. Unit slots may repeat because they are k-by-k squares and
#      are interchangeable in both the original and combo tasks.
#    - Shared color is also made visible by requiring at least two non-unit slots per sample.
#
# c) Implicit-assumption review of main()
#    - find_components(..., monochromatic=False): still valid because same-colored pieces remain
#      disconnected by >=1 background cell, so each piece is still its own component.
#    - frame_comp = largest foreground component: the generator now explicitly rejects samples where
#      any piece area reaches or exceeds the frame component area.
#    - hole extraction via tight crop: unchanged, and the generator still forces holes to touch all
#      four borders of the cropped layout so the crop is exact.
#    - piece extraction skips singleton components: still valid because distractors remain isolated
#      one-pixel components and all real pieces have area > 1.
#    - k inference from the smallest solid square piece: still valid because every sample keeps at
#      least one unit slot, hence at least one k-by-k solid piece.
#    - candidate generation is the key change: only the original piece orientation is considered.
#      This introduces a new assumption that each non-unit slot has at most one exact oriented match;
#      the generator enforces that by disallowing repeated exact non-unit masks.
#    - Shared color removes the old color-identity signal, but main never relied on it directly; it
#      already used connected components and shape comparison.
#
# d) Placement legality
#    - Layout slots are 8-separated, so their enlarged pieces do not overlap inside the output.
#    - The frame is pasted first, then every piece is placed outside it with padding=1 and 8-neighbor
#      separation, so same-colored pieces never merge with each other or with the frame.
#    - Distractors are isolated singletons in a dedicated color and cannot touch any other object.
#
# e) Visual observability
#    - Humans can see one frame, one shared piece color, and at least one pair of same-color shapes
#      that belong to the same rotation/flip family but point in different directions.
#    - This means the input visibly supports the intended reasoning mode: orientation-sensitive visual
#      comparison with no color cue.
#    - Important interaction: once all pieces share one color, a relaxed solver that is allowed to
#      rotate/flip often produces the same final raster, because every filled slot ends up in that same
#      color anyway. Therefore the meaningful non-degeneracy check for this combo is input-structural:
#      the sample must contain an orientation-sensitive same-color family pair, not necessarily an
#      output difference against the original relaxed solver.
#
# f) Visual cleanliness
#    - Only three semantic colors remain: frame color, shared piece color, distractor color.
#    - Objects stay separated, the frame remains singular, and no extra overlays or hidden markers are
#      introduced. The board is simpler than the original, not noisier.
#
# g) Change inferability
#    - There is no complex color mapping to infer because every real piece shares one color.
#    - The remaining inferable structure is geometric: k is visible from the smallest square piece,
#      slots are visible inside the frame, and exact piece orientation is visible from the outer shapes.

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


# --- solver ------------------------------------------------------------------

def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
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


def _solve(grid: np.ndarray, *, allow_dihedral: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

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
        pm = np.zeros(tuple(map(int, patch.shape)), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    candidates = []
    for pi, p in enumerate(pieces):
        variants = _dihedral_variants(p["patch"], p["mask"]) if allow_dihedral else [(p["patch"], p["mask"])]
        for vp, vm in variants:
            um = _downsample_blocks(vm, k)
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

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame's hole layout and fill each slot with the same-colored piece that already has the exact slot orientation.

    Input structure:
    - Background is the most common color. The largest foreground component is a monochrome frame around a layout of background holes.
    - Outside the frame sit separated monochrome pieces built from solid k-by-k blocks, and every real piece uses the same color.
    - At least one piece is a solid k-by-k square, which reveals the block scale k; singleton distractors use another color.

    Rule:
    - Read the connected hole regions inside the frame as slots.
    - Compare each outside piece to the slots at block resolution, but only in the piece's given orientation: no rotation and no reflection are allowed.
    - Rebuild the cropped frame at scale k and paste each matched piece into its slot position, using the frame color as the background.

    Key insight:
    - Color no longer identifies pieces, so the reliable cue is purely geometric.
    - Each real piece is a k-scaled copy of one slot, and downsampling back to block masks exposes the exact orientation match needed for placement.
    """
    return _solve(grid, allow_dihedral=False)


# --- generator ---------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
MIN_NON_UNIT_SLOTS = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8
PAIR_PLACE_RETRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300
FAMILY_ID = 0

ORIENTATION_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
]


def _shape_key(mask: np.ndarray) -> bytes:
    return bytes([int(mask.shape[0]), int(mask.shape[1])]) + mask.astype(np.uint8).tobytes()


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(_shape_key(r))
        variants.append(_shape_key(np.fliplr(r)))
    return min(variants)


def _unique_d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for kk in range(4):
        r = np.rot90(mask, kk)
        for v in (r, np.fliplr(r)):
            key = _shape_key(v)
            if key in seen:
                continue
            seen.add(key)
            out.append(v.copy())
    return out


def _mask_to_slot(coords: np.ndarray, *, family_id: int | None = None) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "r0": int(r0),
        "c0": int(c0),
        "mask": sm,
        "shape_key": _shape_key(sm),
        "canon_key": _canonical_shape_key(sm),
        "family_id": family_id,
    }


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = list(map(tuple, free_cells.tolist()))
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
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
            chosen.add((int(nr), int(nc)))
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


def _place_mask_slot(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    ones = np.argwhere(mask)
    candidates = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            coords = ones + np.array([[int(r0), int(c0)]], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    return ones + np.array([[int(r0), int(c0)]], dtype=int)


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _orientation_pair_count(slots: list[dict]) -> int:
    non_units = [s for s in slots if int(s["area"]) > 1]
    count = 0
    for i in range(len(non_units)):
        for j in range(i + 1, len(non_units)):
            if non_units[i]["canon_key"] == non_units[j]["canon_key"] and non_units[i]["shape_key"] != non_units[j]["shape_key"]:
                count += 1
    return int(count)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_exact: set[bytes] = set()
    used_canon: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            slots.append(slot)
            used_exact.add(slot["shape_key"])
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    pair_slots = None
    for _ in range(PAIR_PLACE_RETRIES):
        base = random.choice(ORIENTATION_BASE_SHAPES)
        variants = _unique_d4_variants(base)
        if len(variants) < 2:
            continue
        m1, m2 = random.sample(variants, 2)
        trial_forbidden = forbidden.copy()
        coords1 = _place_mask_slot(trial_forbidden, m1)
        if coords1 is None:
            continue
        _forbid_coords(trial_forbidden, coords1)
        coords2 = _place_mask_slot(trial_forbidden, m2)
        if coords2 is None:
            continue
        _forbid_coords(trial_forbidden, coords2)
        s1 = _mask_to_slot(coords1, family_id=FAMILY_ID)
        s2 = _mask_to_slot(coords2, family_id=FAMILY_ID)
        if s1["shape_key"] == s2["shape_key"]:
            continue
        pair_slots = [s1, s2]
        forbidden = trial_forbidden
        used_exact.update({s1["shape_key"], s2["shape_key"]})
        used_canon.add(s1["canon_key"])
        break
    if pair_slots is None:
        return None
    slots.extend(pair_slots)

    for _ in range(n_slots - n_units - 2):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            slot = _mask_to_slot(coords)
            if slot["shape_key"] in used_exact:
                continue
            if slot["canon_key"] in used_canon:
                continue
            slots.append(slot)
            used_exact.add(slot["shape_key"])
            used_canon.add(slot["canon_key"])
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None
    if _orientation_pair_count(slots) < 1:
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "family_id": slot.get("family_id"),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], int(top), int(left), mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
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
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda q: -int(q["in_mask"].sum())):
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
    max_units = min(N_UNITS_MAX, n_slots - MIN_NON_UNIT_SLOTS)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if not palette:
        return None
    piece_color = palette.pop()

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if len(non_unit_slots) < MIN_NON_UNIT_SLOTS:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    non_unit_sorted = sorted(non_unit_slots, key=_slot_sort_key)
    pieces = [_make_piece(s, piece_color, k) for s in non_unit_sorted]
    pieces += [_make_piece(s, piece_color, k) for s in unit_slots]

    frame_area = int((layout_h + 2) * (layout_w + 2) - sum(int(s["area"]) for s in slots))
    if any(int(p["in_mask"].sum()) >= frame_area for p in pieces):
        return None

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
    if _orientation_pair_count(non_unit_sorted) < 1:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
