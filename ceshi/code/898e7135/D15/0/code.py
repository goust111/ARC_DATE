# Variant of 898e7135
# Dimension: D15 — 洞形状的唯一性约束 (meta)
# Change: 非单元洞的形状在二面体群下互不相同（canonical shape key 去重），单元洞（面积=1）可重复... → 允许相同形状的洞（颜色消歧）

from __future__ import annotations

"""
Design analysis for the D15 variant
===================================

a) Concrete behavior and trace
spec_variant:
  unchanged_core:
    - There is still one large frame enclosing a hole layout.
    - Each outside piece is still a k*k-block-scaled copy of one hole shape, up to D4 transform.
    - Output is still the scaled frame with pieces pasted into the holes.
  changed_input_signal:
    - Non-unit hole shapes are now allowed to repeat.
    - Every hole gets a 1-pixel wall marker touching it in 4-neighborhood.
    - The marker color equals the color of the outside piece that belongs to that hole.
    - Therefore shape finds the candidate set, and marker color resolves duplicate-shape ambiguity.
  example_trace:
    - Original: two L-shaped outside pieces could never exist, because two L-shaped holes were forbidden.
    - Variant: two L-shaped holes may both exist. One hole touches a red wall marker, the other touches blue.
      Outside there is one red L piece and one blue L piece. Shape alone yields two candidates for each hole;
      adding the touching marker color makes the match unique.

b) Complexity calibration
  - The variant keeps only one new signal channel: a single adjacent marker pixel per hole.
  - Non-unit piece colors remain distinct, so the new ambiguity is controlled rather than combinatorial.
  - Each generated sample is forced to contain at least one duplicate non-unit shape pair, so the variant is
    visible rather than a rare corner case.

c) Implicit-assumption audit of the original main()
  - `slots.sort(...)` + greedy first-match relied on a unique shape per non-unit hole. That is invalid now.
  - Candidate matching by `area + exact downsampled mask` is still valid, but now may yield multiple pieces.
  - `dihedral_variants()` originally deduplicated by `mask.tobytes()` only, which can collapse 1xN vs Nx1
    orientations. That was accidentally tolerable before; with duplicate-shape groups it is riskier, so the
    new solver keys variants by `(shape, bytes)` to preserve all orientations.
  - Frame detection still works because the colored markers are embedded in the same connected frame object and
    the dominant color of that object remains the true frame color.

d) Placement legality
  - Slots remain 8-separated, so pieces and markers stay visually separable.
  - Markers are placed on wall cells that are 4-adjacent to exactly one hole component, preventing one marker
    from accidentally signaling multiple holes.
  - Pieces are still scattered with >=1 background cell of clearance using `random_free_location`.

e) Visual observability
  - Duplicate shapes are forced in every generated example.
  - The resolving clue is directly visible: a hole touches a colored wall pixel, and a same-colored outside
    piece exists.
  - The rule can be inferred from 2-3 examples without hidden numeric ordering.

f) Visual cleanliness
  - Only one marker pixel per hole is added, so the frame is not overloaded.
  - Markers live on walls, not on top of holes or pieces, so there is no layering ambiguity.

g) Color-change inferability
  - Colors do not undergo arbitrary remapping. The marker color is copied directly from the corresponding piece.
  - Because non-unit piece colors are distinct within a sample, repeated shapes have an explicit visual key.

h) New solver assumptions introduced here
  - Each hole has at most one non-frame marker color in its 4-neighborhood.
  - For repeated non-unit shapes, piece color equals the hole's marker color.
  - The generator enforces both conditions explicitly.
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Canonical D4 key for a tight bool mask."""
    best = None
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for var in (rot, np.fliplr(rot)):
            key = (tuple(map(int, var.shape)), var.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return (tight_mask, r0, c0) for coords in layout coordinates."""
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    Variant rule:
    - Hole shapes may repeat.
    - Each hole has a touching wall marker whose color matches the correct piece.
    - Shape still determines geometric fit; marker color breaks ties among repeated shapes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Largest multicolor foreground component is the frame, now including colored wall markers.
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    def slot_marker_color(coords: np.ndarray) -> int | None:
        seen: set[int] = set()
        for rr, cc in coords:
            sr = int(rr + lr0)
            sc = int(cc + lc0)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = sr + dr
                nc = sc + dc
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                    col = int(sub[nr, nc])
                    if col != int(bg) and col != int(bgc):
                        seen.add(col)
        if not seen:
            return None
        assert len(seen) == 1, "each slot should have exactly one marker color"
        return next(iter(seen))

    slots = []
    for coords in components_mask(layout, connectivity=4):
        sm, rr0, cc0 = _tight_mask_from_coords(coords)
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker_color": slot_marker_color(coords),
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
        pm, _dummy_r0, _dummy_c0 = _tight_mask_from_coords(coords - np.array([pr0, pc0]))
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(Counter(vals).most_common(1)[0][0]),
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
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out_h = int(hh // k)
        out_w = int(ww // k)
        out = np.zeros((out_h, out_w), dtype=bool)
        for r in range(out_h):
            for c in range(out_w):
                block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            p = np.rot90(patch, int(kk))
            m = np.rot90(mask, int(kk))
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = (tuple(map(int, m.shape)), m.tobytes())
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
        shape_hits = []
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                shape_hits.append(cand)

        assert shape_hits, "every slot must match at least one piece by shape"
        found = None
        marker = s["marker_color"]
        if marker is not None:
            color_hits = [cand for cand in shape_hits if int(cand["color"]) == int(marker)]
            if color_hits:
                found = color_hits[0]
        if found is None:
            assert len(shape_hits) == 1, "ambiguous slot must be resolved by marker color"
            found = shape_hits[0]

        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --------------- generator ---------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_SIZE_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
DUPLICATE_PAIR_RETRIES = 40
COPY_PLACEMENT_TRIES = 80
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected slot of given size from random free cells."""
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    starts = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(starts)
    for r0, c0 in starts[:GROW_SEED_TRIES]:
        chosen = {(int(r0), int(c0))}
        frontier = {(int(r0 - 1), int(c0)), (int(r0 + 1), int(c0)), (int(r0), int(c0 - 1)), (int(r0), int(c0 + 1))}
        while len(chosen) < int(target_size) and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((int(nr), int(nc)))
            frontier.discard((int(nr), int(nc)))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == int(target_size):
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


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    mask, _r0, _c0 = _tight_mask_from_coords(coords)
    return mask


def _place_copy_of_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Place an exact translated copy of `mask` into free layout cells."""
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    rel = np.argwhere(mask)
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for top, left in positions[:COPY_PLACEMENT_TRIES]:
        coords = rel + np.array([top, left])
        if np.all(~forbidden[coords[:, 0], coords[:, 1]]):
            return coords.astype(int)
    return None


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """
    Generate slots with:
    - unit slots first
    - exactly one duplicate non-unit shape pair
    - all remaining non-unit shapes unique under D4
    """
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1})
                _forbid_coords(forbidden, coords)
                break
        else:
            return None

    n_non_units = int(n_slots - n_units)
    if n_non_units < 2:
        return None

    dup_slots: list[dict] | None = None
    for _ in range(DUPLICATE_PAIR_RETRIES):
        size = random.choice(DUPLICATE_SIZE_CHOICES)
        coords1 = _grow_slot(forbidden, size)
        if coords1 is None:
            continue
        mask1 = _mask_from_coords(coords1)
        temp_forbidden = forbidden.copy()
        _forbid_coords(temp_forbidden, coords1)
        coords2 = _place_copy_of_mask(temp_forbidden, mask1)
        if coords2 is None:
            continue
        dup_key = _canonical_shape_key(mask1)
        dup_slots = [
            {"coords": coords1, "area": int(size), "mask": mask1, "shape_key": dup_key},
            {"coords": coords2, "area": int(size), "mask": mask1.copy(), "shape_key": dup_key},
        ]
        forbidden = temp_forbidden
        _forbid_coords(forbidden, coords2)
        break
    if dup_slots is None:
        return None
    slots.extend(dup_slots)

    used_shape_keys = {dup_slots[0]["shape_key"]}
    for _ in range(n_non_units - 2):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            sm = _mask_from_coords(coords)
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

    return slots


def _choose_marker_cells(slots: list[dict], layout_h: int, layout_w: int) -> list[tuple[int, int]] | None:
    """
    Choose one frame-wall cell per slot.
    Each marker cell must be 4-adjacent to exactly one slot, so colors never signal two holes.
    Returned coords are in frame-image coordinates (including the 1-cell outer border).
    """
    owner = np.full((layout_h, layout_w), -1, dtype=int)
    for idx, slot in enumerate(slots):
        coords = slot["coords"]
        owner[coords[:, 0], coords[:, 1]] = int(idx)

    chosen: list[tuple[int, int]] = []
    for idx, slot in enumerate(slots):
        cands = set()
        for rr, cc in slot["coords"]:
            fr = int(rr + 1)
            fc = int(cc + 1)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = fr + dr
                nc = fc + dc
                if 0 <= nr < layout_h + 2 and 0 <= nc < layout_w + 2:
                    if 1 <= nr <= layout_h and 1 <= nc <= layout_w and owner[nr - 1, nc - 1] != -1:
                        continue
                    adj_owners = set()
                    for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ar = nr + ddr
                        ac = nc + ddc
                        if 1 <= ar <= layout_h and 1 <= ac <= layout_w:
                            sid = int(owner[ar - 1, ac - 1])
                            if sid != -1:
                                adj_owners.add(sid)
                    if adj_owners == {int(idx)}:
                        cands.add((int(nr), int(nc)))
        if not cands:
            return None
        chosen.append(sorted(cands)[0])
    return chosen


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

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
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_frame_image(
    slots: list[dict],
    layout_h: int,
    layout_w: int,
    frame_color: int,
    marker_by_slot: list[int],
) -> np.ndarray | None:
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    frame_img = np.full((layout_h + 2, layout_w + 2), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)

    marker_cells = _choose_marker_cells(slots, layout_h, layout_w)
    if marker_cells is None:
        return None
    for color, (rr, cc) in zip(marker_by_slot, marker_cells):
        frame_img[int(rr), int(cc)] = int(color)
    return frame_img


def _build_input_canvas(
    pieces: list[dict],
    frame_img: np.ndarray,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
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
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
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
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo = max(0, rr - 1)
        r_hi = min(canvas_h, rr + 2)
        c_lo = max(0, cc - 1)
        c_hi = min(canvas_w, cc + 2)
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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    if len(palette) < len(non_unit_slots) + 1:
        return None
    non_unit_colors = palette[: len(non_unit_slots)]
    unit_color = palette[len(non_unit_slots)]

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]

    slot_color_map = {}
    for piece in pieces:
        slot_color_map[id(piece["slot"])] = int(piece["color"])
    marker_by_slot = [int(slot_color_map[id(slot)]) for slot in slots]

    frame_img = _build_frame_image(slots, layout_h, layout_w, frame_color, marker_by_slot)
    if frame_img is None:
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, frame_img, distractor_color)
    if canvas is None:
        return None

    dup_counts: Counter[tuple[tuple[int, int], bytes]] = Counter()
    for slot in non_unit_slots:
        dup_counts[_canonical_shape_key(slot["mask"])] += 1
    if max(dup_counts.values(), default=0) < 2:
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
