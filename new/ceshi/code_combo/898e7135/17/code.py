# Combo variant of 898e7135 (index 17)
# combo_spec: D6/2+D11/0
# Name: 干扰混淆空间重叠

from __future__ import annotations

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
)


# Design analysis for the combo variant
# -------------------------------------
# fused_spec:
#   core_task_kept:
#     - The output is still the k-scaled frame layout filled with the matched
#       monochrome pieces.
#     - Each true piece is still a k-by-k block upscale of one slot under a D4
#       transform.
#     - There are still 1-2 unit slots, so at least one solid k-by-k square
#       piece exists and keeps k observable.
#   changed_dimensions:
#     - D6/2: singleton distractors no longer use a dedicated color; every
#       distractor borrows a real piece color.
#     - D11/0: at least one piece now occupies hole cells inside the frame bbox
#       and touches the frame, while still never overwriting frame-colored cells.
#   concrete_trace:
#     - Original path:
#       1. largest multicolor foreground component = frame
#       2. inside frame bbox, `sub == bg` gives the hole layout
#       3. every non-frame multicolor component with area>1 is a piece
#     - Combo path:
#       1. step 1 fails because a piece can touch the frame while remaining a
#          different color, so the frame can no longer be recovered from the
#          largest multicolor foreground component.
#       2. step 2 must become `sub != frame_color`, because hole cells may be
#          visible bg or may already be occupied by an intruding piece.
#       3. step 3 must become monochrome-component based, because a piece bbox
#          may overlap the frame bbox, and singleton same-color pixels are now
#          distractors rather than a separate distractor color class.
#     - Small pixel example:
#       a. A green L-piece places one arm inside a top-left hole and one arm
#          outside the frame, touching the blue border but not covering it.
#       b. Elsewhere, isolated green single pixels appear on background.
#       c. The solver must keep the connected green L as a candidate piece,
#          discard the isolated green singletons, and recover the hole layout
#          from all non-blue cells inside the frame bbox.
#   complexity_calibration:
#     - The fusion is stronger than either single variant alone because color is
#       no longer useful for filtering noise, and geometry alone can no longer
#       separate frame from pieces by connected foreground size.
#     - We keep distractors as isolated singletons and preserve distinct non-unit
#       shapes under D4, so the puzzle remains visually legible and matching
#       stays shape-based rather than search-heavy.
#     - The generator forces both behaviors to be visible in every accepted
#       sample: at least one piece has positive occupancy inside the frame bbox,
#       and every unique piece color appears on at least one singleton distractor
#       whenever the distractor budget allows.
#   implicit_assumption_audit_main:
#     - `find_components(... monochromatic=False)` for frame detection is no
#       longer valid and is removed.
#     - `sub == bg` for hole recovery is no longer valid because covered holes
#       are not bg; `sub != frame_color` replaces it.
#     - raw bbox cropping for pieces is no longer safe because an overlapping
#       piece bbox may include frame pixels; each patch is rebuilt from the
#       component coords only.
#     - singleton removal can no longer rely on a fresh distractor color; it is
#       now purely component-size based.
#   placement_legality:
#     - Pieces never overlap nonzero canvas cells, so they do not cover frame
#       pixels or each other.
#     - Forced overlap means actual piece pixels lie inside the frame bbox on
#       zero-valued hole cells and are 4-neighbor-adjacent to frame pixels.
#     - Distractors stay outside the frame bbox and are placed only on cells with
#       an empty 3x3 neighborhood, so they remain isolated singletons.
#   observability_and_cleanliness:
#     - Humans can see the same color appearing both as a real piece and as tiny
#       removable dots.
#     - Humans can also see at least one piece partly entering the frame's
#       rectangle and hugging the border.
#     - Because distractors stay singleton and overlap uses hole cells rather
#       than wall overwrite, the frame remains readable instead of turning into
#       clutter.
#   new_solver_assumptions_checked_against_generator:
#     - The frame is the unique monochrome component with a solid rectangular
#       border and a nontrivial non-frame interior; generator preserves this.
#     - Every non-singleton monochrome non-frame component is a real piece;
#       generator keeps distractors singleton and piece colors distinct from the
#       frame color.
#     - At least one piece is a solid k-by-k square; generator still guarantees
#       1-2 unit slots.
#     - Slot matching remains unambiguous because non-unit slot masks are unique
#       up to D4, and unit slots are matched by unused solid squares.


def _solid_square_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _dihedral_variants(
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


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
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


def _recover_layout_from_frame_bbox(sub: np.ndarray, frame_color: int) -> np.ndarray:
    non_frame = sub != int(frame_color)
    hole_box = bbox(non_frame)
    assert hole_box is not None, "frame must enclose at least one non-frame cell"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = non_frame[lr0 : lr1 + 1, lc0 : lc1 + 1]
    assert layout[0].any() and layout[-1].any()
    assert layout[:, 0].any() and layout[:, -1].any()
    return layout


def _find_frame_component(g: np.ndarray, bg: int) -> dict:
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    candidates: list[tuple[tuple[int, int, int, int], dict]] = []
    for comp in mono:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        sub = crop_bbox(g, (r0, c0, r1, c1))
        if not (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        interior = sub[1:-1, 1:-1]
        if interior.size == 0 or not np.any(interior != color):
            continue
        try:
            layout = _recover_layout_from_frame_bbox(sub, color)
        except AssertionError:
            continue
        if not components_mask(layout, connectivity=4):
            continue
        rank = (h * w, int(comp["area"]), h + w, color)
        candidates.append((rank, comp))
    assert candidates, "no valid frame component found"
    return max(candidates, key=lambda item: item[0])[1]


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"
    return slots


def _extract_pieces(g: np.ndarray, bg: int, frame_comp: dict) -> list[dict]:
    frame_bbox = tuple(map(int, frame_comp["bbox"]))
    frame_color = int(frame_comp["color"])
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces = []
    for comp in mono:
        color = int(comp["color"])
        if color == frame_color and tuple(map(int, comp["bbox"])) == frame_bbox:
            continue
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        pm = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        patch = np.zeros(pm.shape, dtype=int)
        patch[pm] = color
        pieces.append({"area": area, "mask": pm, "patch": patch, "color": color})
    assert pieces, "at least one non-frame piece must exist"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a framed hole-layout and refill each slot with its matching piece.

    Input structure:
    - The frame is a monochrome rectangular-border component whose interior mixes frame-color walls and hole cells.
    - Real pieces are monochrome k-by-k block sprites, but some may intrude into hole cells inside the frame bbox.
    - Distractors are isolated single pixels that can reuse the same colors as the real pieces.

    Rule:
    - Detect the frame by its own monochrome rectangular border rather than by largest multicolor connectivity.
    - Inside the frame bbox, treat every non-frame-color cell as part of the slot layout, then extract 4-connected slots.
    - Keep only non-singleton monochrome components as candidate pieces, match them to slots by block shape under rotation/flip, and output the scaled filled frame.

    Key insight:
    - Covered holes are still holes, so the layout is recovered from `inside != frame_color`; same-color distractors disappear because only connected non-singleton monochrome components can be true pieces.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame_comp = _find_frame_component(g, bg)
    frame_color = int(frame_comp["color"])
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    layout = _recover_layout_from_frame_bbox(sub, frame_color)
    slots = _extract_slots(layout)
    pieces = _extract_pieces(g, bg, frame_comp)
    k = _solid_square_k(pieces)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append({"pi": pi, "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != need_area:
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


# --------------- generator ---------------

"""Generator for ARC-AGI-2 task 898e7135, combo D6/2 + D11/0.

Output-first strategy:
  1. Build a hole/wall layout of 4-connected slots (8-separated, includes units).
  2. Create one monochrome piece per slot, with random D4 transform on input.
  3. Assemble the frame.
  4. Place pieces on a bg=0 canvas, forcing at least one piece to occupy cells
     inside the frame bbox without covering frame pixels.
  5. Add isolated singleton distractors that reuse actual piece colors.

The generator validates main(input) == expected_output before returning.
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
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

FORCED_OVERLAPPERS_RANGE = (1, 1)
OPTIONAL_OVERLAP_PROB = 0.30
GENERATE_MAX_RETRIES = 500


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
    free_list = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
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


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
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


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
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
    out_patch = np.where(out_mask, int(color), 0).astype(int)

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


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _occupied_inside_bbox(mask: np.ndarray, top: int, left: int, frame_bbox: tuple[int, int, int, int]) -> int:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    r0 = max(int(top), fr0)
    c0 = max(int(left), fc0)
    r1 = min(int(top) + int(mask.shape[0]) - 1, fr1)
    c1 = min(int(left) + int(mask.shape[1]) - 1, fc1)
    if r0 > r1 or c0 > c1:
        return 0
    mr0 = r0 - int(top)
    mc0 = c0 - int(left)
    mr1 = r1 - int(top) + 1
    mc1 = c1 - int(left) + 1
    return int(mask[mr0:mr1, mc0:mc1].sum())


def _candidate_positions(
    canvas: np.ndarray,
    piece_mask: np.ndarray,
    piece_occ: np.ndarray,
    frame_occ: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    require_overlap: bool,
) -> list[tuple[int, int, int, bool]]:
    canvas_h, canvas_w = canvas.shape
    ph, pw = piece_mask.shape
    padded_piece_occ = _dilate8(piece_occ)
    touch_frame_occ = _dilate4(frame_occ)
    candidates = []
    for top in range(canvas_h - ph + 1):
        for left in range(canvas_w - pw + 1):
            view_nonzero = canvas[top : top + ph, left : left + pw] != 0
            if bool(np.logical_and(view_nonzero, piece_mask).any()):
                continue
            if bool(np.logical_and(padded_piece_occ[top : top + ph, left : left + pw], piece_mask).any()):
                continue
            inside_pixels = _occupied_inside_bbox(piece_mask, top, left, frame_bbox)
            touches_frame = bool(
                np.logical_and(touch_frame_occ[top : top + ph, left : left + pw], piece_mask).any()
            )
            if require_overlap and not (inside_pixels > 0 and touches_frame):
                continue
            candidates.append((int(top), int(left), int(inside_pixels), bool(touches_frame)))
    return candidates


def _choose_position(
    candidates: list[tuple[int, int, int, bool]], require_overlap: bool
) -> tuple[int, int, int] | None:
    if not candidates:
        return None
    if require_overlap:
        best = max(int(c[2]) for c in candidates)
        strong = [c for c in candidates if int(c[2]) >= max(1, best - 1)]
        top, left, inside_pixels, _ = random.choice(strong)
        return int(top), int(left), int(inside_pixels)
    overlap_cands = [c for c in candidates if int(c[2]) > 0 and bool(c[3])]
    if overlap_cands and random.random() < OPTIONAL_OVERLAP_PROB:
        top, left, inside_pixels, _ = random.choice(overlap_cands)
        return int(top), int(left), int(inside_pixels)
    top, left, inside_pixels, _ = random.choice(candidates)
    return int(top), int(left), int(inside_pixels)


def _place_singleton_distractors(
    canvas: np.ndarray,
    piece_colors: list[int],
    n_distractors: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    required = piece_colors[:]
    random.shuffle(required)
    if n_distractors < len(required):
        required = required[:n_distractors]
    distractor_colors = required + [random.choice(piece_colors) for _ in range(n_distractors - len(required))]
    random.shuffle(distractor_colors)

    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    coords: list[tuple[int, int]] = []
    mark = -1
    h, w = canvas.shape
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if len(coords) >= n_distractors:
            break
        rr = random.randint(0, h - 1)
        cc = random.randint(0, w - 1)
        if fr0 <= rr <= fr1 and fc0 <= cc <= fc1:
            continue
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(w, cc + 2)
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
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
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
    frame_top = random.randint(1, fr_max)
    frame_left = random.randint(1, fc_max)
    paste(canvas, frame_img, frame_top, frame_left)
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    frame_occ = canvas != 0

    piece_occ = np.zeros_like(canvas, dtype=bool)
    inside_counts: list[int] = []
    order = list(range(len(pieces)))
    order.sort(key=lambda idx: (int(pieces[idx]["slot"]["area"]) == 1, -int(pieces[idx]["in_mask"].sum())))

    n_forced = min(len(order), random.randint(*FORCED_OVERLAPPERS_RANGE))
    forced_sequence: list[int] = []
    if n_forced > 0:
        feasible_forced: list[tuple[int, int, int, list[tuple[int, int, int, bool]]]] = []
        for idx in order:
            cands = _candidate_positions(canvas, pieces[idx]["in_mask"], piece_occ, frame_occ, frame_bbox, True)
            if not cands:
                continue
            best_inside = max(int(c[2]) for c in cands)
            feasible_forced.append((best_inside, int(pieces[idx]["in_mask"].sum()), idx, cands))
        if len(feasible_forced) < n_forced:
            return None
        feasible_forced.sort(key=lambda item: (-int(item[0]), -int(item[1]), int(order.index(item[2]))))
        forced_sequence = [int(item[2]) for item in feasible_forced[:n_forced]]

    placement_sequence = forced_sequence + [idx for idx in order if idx not in set(forced_sequence)]
    forced_set = set(forced_sequence)
    for idx in placement_sequence:
        p = pieces[idx]
        require_overlap = idx in forced_set
        candidates = _candidate_positions(
            canvas, p["in_mask"], piece_occ, frame_occ, frame_bbox, require_overlap
        )
        pos = _choose_position(candidates, require_overlap)
        if pos is None:
            return None
        top, left, inside_pixels = pos
        inside_counts.append(int(inside_pixels))
        paste(canvas, p["in_patch"], top, left, mask=p["in_mask"], overwrite=True)
        piece_occ[top : top + p["in_mask"].shape[0], left : left + p["in_mask"].shape[1]] |= p["in_mask"]

    if not any(v > 0 for v in inside_counts):
        return None

    piece_colors = sorted({int(p["color"]) for p in pieces})
    n_distractors = random.randint(*DISTRACTOR_RANGE)
    if not _place_singleton_distractors(canvas, piece_colors, n_distractors, frame_bbox):
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
    frame_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = int(palette[n_slots - n_units])

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
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
