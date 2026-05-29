# Variant of 898e7135
# Dimension: D9 — 洞区域的空间约束 (meta)
# Change: 洞区域 4-连通且彼此 8-隔离（相邻洞之间至少隔 1 像素墙壁，含对角线方向），面积 1-7 像素... → 洞区域允许 8-连通

"""
Design analysis for the D9 variant.

a) Concrete behavior and trace
   - In the original task, a hole shape like
       1 0
       0 1
     was not a single slot: under 4-connectivity it split into two unit holes.
   - In this variant, the same diagonal pair is one slot because holes are grouped
     with 8-connectivity. Its outside piece is likewise one 8-connected sprite:
     two k-by-k squares touching only at a corner.
   - Old solver failure mode:
     1. `components_mask(layout, connectivity=4)` splits the diagonal hole into
        two slots.
     2. `find_components(... connectivity=4, monochromatic=False)` splits the
        matching piece into separate square components as well.
     3. The slot/piece counts and areas no longer line up.
   - Variant solver behavior:
     1. Parse foreground objects with 8-connectivity so a diagonal-only piece is
        extracted as one component.
     2. Parse frame holes with 8-connectivity so the diagonal hole is one slot.
     3. Downsample each rotated/flipped piece and match the 8-connected block mask.

b) Complexity calibration
   - Only D9 changes: slots are still distinct regions, areas stay 1-7, k stays 2-3,
     pieces remain monochrome block-scaled sprites, and matching is still by D4 shape.
   - To make the variant observable, every generated example contains at least one
     non-unit slot that is 8-connected but not 4-connected. Other slots may remain
     ordinary 4-connected shapes, which keeps the puzzle close to the source task.
   - Non-unit slot shapes remain D4-unique to avoid matching ambiguity. Unit slots
     are still allowed and preserve the original k-inference trick.

c) Implicit-assumption review of the original solver
   - `find_components(... connectivity=4, monochromatic=False)` relied on every
     piece being 4-connected at pixel level. That is false once diagonal block
     contact is allowed, so it must become 8-connectivity.
   - `components_mask(layout, connectivity=4)` relied on every hole slot being
     4-connected. That is the main broken assumption and must become 8.
   - Candidate matching by area/shape still works because block scaling preserves
     both area and exact tight-cropped mask under D4 transforms.
   - The frame detection still works as long as generator keeps pieces/distractors
     8-separated from the frame, preventing accidental foreground merges.

d) Placement legality
   - Distinct slots are still kept 8-separated in the layout generator, so each
     8-component corresponds to exactly one intended hole region.
   - Outside pieces are scattered with 8-neighborhood padding, ensuring the solver's
     8-connected foreground extraction sees one component per piece and does not
     merge pieces with the frame or distractors.
   - Distractors remain isolated singleton pixels in a dedicated color.

e) Visual observability
   - The mandatory diagonal slot makes the change visible in both input and output:
     the frame contains a diagonally linked hole pattern and the matching piece is a
     diagonally linked arrangement of square blocks.
   - Because the original rule would interpret that pattern as multiple slots/pieces,
     the difference is recoverable from a few examples.

f) Visual cleanliness
   - Only one or two diagonal slots are forced; the rest of the scene keeps the
     original clean frame + scattered pieces layout, so examples remain readable.
   - Slots are still separated by frame-colored walls, preventing hole regions from
     visually bleeding together.

g) Color inference
   - No new color rule is introduced. Colors remain purely object-local labels:
     frame color becomes output background, piece colors are preserved, distractor
     color is discarded. The variant depends on geometry, not hidden color order.
"""

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D9 variant:
    - Hole slots are extracted with 8-connectivity.
    - Piece components are also extracted with 8-connectivity so diagonal-only
      block contact still forms one sprite.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Foreground objects must be parsed with 8-connectivity: a diagonal-block piece
    # is one intended sprite in this variant.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=8):
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
                block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
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
        top = (int(s["r0"]) + 1) * k
        left = (int(s["c0"]) + 1) * k
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DIAGONAL_SLOT_SIZE_CHOICES = [2, 3, 4]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 24
NON_UNIT_SLOT_RETRIES = 48
GROW_SEED_TRIES = 12
GROW_BUILD_RETRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a bool mask (tight-cropped)."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _is_diagonal_slot(coords: np.ndarray) -> bool:
    """True iff coords form one 8-component but not one 4-component."""
    mask = _mask_from_coords(coords)
    return len(components_mask(mask, connectivity=8)) == 1 and len(components_mask(mask, connectivity=4)) > 1


def _neighbor_cells(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _is_diag_only_neighbor(cell: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
    r, c = cell
    has8 = False
    has4 = False
    for nr, nc in _neighbor_cells(r, c):
        if (nr, nc) not in chosen:
            continue
        has8 = True
        if nr == r or nc == c:
            has4 = True
    return has8 and not has4


def _grow_slot(forbidden: np.ndarray, target_size: int, require_diagonal: bool) -> np.ndarray | None:
    """Grow an 8-connected slot; optionally force the final shape to be non-4-connected."""
    h, w = forbidden.shape
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        for _ in range(GROW_BUILD_RETRIES):
            chosen = {start}
            frontier = {p for p in _neighbor_cells(*start) if 0 <= p[0] < h and 0 <= p[1] < w}
            used_diag_only = False
            while len(chosen) < target_size and frontier:
                cands = [p for p in frontier if not forbidden[p[0], p[1]] and p not in chosen]
                if not cands:
                    break
                diag_cands = [p for p in cands if _is_diag_only_neighbor(p, chosen)]
                if require_diagonal and not used_diag_only:
                    if not diag_cands and len(chosen) + 1 == target_size:
                        break
                    pool = diag_cands if diag_cands else cands
                else:
                    pool = cands
                nr, nc = random.choice(pool)
                if (nr, nc) in diag_cands:
                    used_diag_only = True
                chosen.add((nr, nc))
                frontier.discard((nr, nc))
                frontier.update(
                    p
                    for p in _neighbor_cells(nr, nc)
                    if 0 <= p[0] < h and 0 <= p[1] < w and p not in chosen
                )
            if len(chosen) != target_size:
                continue
            coords = np.array(sorted(chosen), dtype=int)
            if require_diagonal and not _is_diagonal_slot(coords):
                continue
            return coords
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a bool mask by 1 cell (8-connectivity)."""
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
    """Mark coords and their 8-neighbors as forbidden (in-place)."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int, n_diagonal_slots: int) -> list[dict] | None:
    """
    Generate a layout with n_slots total 8-connected slots.
    Distinct slots remain 8-separated; at least n_diagonal_slots non-unit slots are
    forced to be 8-connected but not 4-connected.
    """
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1, require_diagonal=False)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for idx in range(n_slots - n_units):
        require_diagonal = idx < n_diagonal_slots
        size_choices = DIAGONAL_SLOT_SIZE_CHOICES if require_diagonal else NON_UNIT_SIZE_CHOICES
        for _ in range(NON_UNIT_SLOT_RETRIES):
            size = random.choice(size_choices)
            coords = _grow_slot(forbidden, int(size), require_diagonal=require_diagonal)
            if coords is None:
                continue
            sm = _mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "mask": sm,
                    "shape_key": key,
                    "is_diagonal": bool(_is_diagonal_slot(coords)),
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    if sum(int(s.get("is_diagonal", False)) for s in slots) < n_diagonal_slots:
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Build output and input patches for one slot."""
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": color,
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
        "color": color,
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
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

    for p in sorted(pieces, key=lambda obj: -int(obj["in_mask"].sum())):
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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    n_diagonal_slots = random.randint(1, min(2, n_slots - n_units))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units, n_diagonal_slots)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    if sum(int(s.get("is_diagonal", False)) for s in non_unit_slots) < n_diagonal_slots:
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

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
