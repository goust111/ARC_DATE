# Combo variant of 898e7135 (index 20)
# combo_spec: D10/2+D15/0
# Name: 同形颜色消歧

from __future__ import annotations

"""
Design analysis for the combo variant
====================================

a) Concrete behavior and trace
  - Original task:
    one monochrome frame encloses several bg holes, each outside piece is a
    k-by-k scaled copy of one hole, and shape under D4 is enough because
    non-unit hole shapes are unique.
  - Fused variant:
    repeated non-unit hole shapes are now allowed, and every hole is surrounded
    by wall cells of exactly one slot-specific color. The matching outside piece
    uses that same color. The output preserves these colored walls, then fills
    the hole with the correctly oriented piece.
  - Small example trace:
    if two L-shaped holes appear inside the same frame, one ringed by color 2
    and one ringed by color 8, then the input also contains one color-2 L piece
    and one color-8 L piece outside. Shape-only matching returns two candidates
    for each L hole; adding boundary color makes the pairing unique.
  - Difference from the original:
    the original collapsed the frame to one color and relied on unique shapes;
    the combo keeps a multicolor frame and requires color+shape jointly.

b) Complexity calibration
  - The color signal is strong but controlled: one color per slot, reused on the
    outside piece and on all 4-neighbor wall cells of that slot.
  - The generator forces at least one duplicate non-unit shape pair, so color
    disambiguation is actually needed in typical samples rather than being a
    rare edge case.
  - Remaining non-unit slots stay unique under D4, which prevents uncontrolled
    combinatorics while still making the new rule visually obvious.

c) Implicit-assumption audit of the original main()
  - `bgc = dominant frame color` as the full output background is invalid because
    the frame is intentionally multicolor now.
  - Tight-cropping to the hole layout and discarding wall detail is invalid,
    because the wall colors are part of the rule and must persist to the output.
  - Greedy first shape match is invalid, because duplicate hole shapes are now
    legal and would create multiple shape-only matches.
  - Deduplicating D4 variants only by `mask.tobytes()` is unsafe for 1xN versus
    Nx1 masks; this combo uses `(shape, bytes)` keys instead.
  - Generator-side uniqueness filtering of non-unit hole shapes must be relaxed
    selectively: at least one repeated shape is required, but not arbitrary many.

d) Placement legality
  - Holes remain 8-separated so hole components stay visually distinct.
  - Because entire boundary rings are recolored, different holes must not share
    any wall cell in their 4-neighbor boundary sets. Layouts violating this are
    rejected before frame construction.
  - Pieces are still placed outside the frame with at least one bg cell of
    8-neighbor clearance, and distractors remain isolated single pixels.

e) Visual observability
  - A human sees duplicate shapes inside the frame and duplicate-shape pieces
    outside, but each hole boundary repeats its correct piece color.
  - The distinguishing clue is fully visible and local: no ordering, counting,
    or hidden numeric property is needed.
  - Across 2-3 examples, the repeated pattern "same hole shape, different wall
    color, different same-colored piece" is enough to infer the fused rule.

f) Visual cleanliness
  - The added signal lives only on the frame walls adjacent to holes, so there is
    no overlap on top of holes or pieces.
  - Non-boundary frame walls keep one neutral filler color, which preserves the
    frame silhouette and avoids turning the whole board into noise.

g) Color-change inferability
  - Colors are not arbitrarily remapped: each slot color is copied directly from
    the boundary ring to the matching piece and stays visible in the output.
  - The palette stays bounded by the ARC palette:
    1 filler wall color + 1 distractor color + 1 unique color per slot.

h) New solver assumptions introduced here, and how the generator guarantees them
  - Every slot has at least one adjacent wall cell, and all of those adjacent
    wall cells share exactly one color. `_boundary_sets()` and
    `_build_frame_image()` enforce this.
  - Different slots have disjoint boundary rings, so a wall cell never signals
    two holes. `_boundary_sets()` rejects overlaps.
  - At least one non-unit D4 shape is duplicated. `_gen_layout()` constructs an
    exact translated copy of one generated non-unit mask.
  - Each piece is monochrome and is a k-scaled slot mask under a D4 transform.
    `_make_piece()` constructs pieces directly from the slot masks.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


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

GENERATE_MAX_RETRIES = 500


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
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
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, int(r0), int(c0)


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        base.append((rot_patch, rot_mask))
        base.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
    uniq = []
    seen = set()
    for var_patch, var_mask in base:
        key = (tuple(map(int, var_mask.shape)), var_mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((var_patch, var_mask))
    return uniq


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
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


def _extract_frame_slots_and_pieces(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_img = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_mask = frame_img == bg
    slot_coords_list = components_mask(hole_mask, connectivity=4)
    assert slot_coords_list, "frame must contain at least one hole"

    all_holes = {tuple(map(int, rc)) for rc in np.argwhere(hole_mask)}
    slots = []
    for coords in slot_coords_list:
        mask, rr0, cc0 = _tight_mask_from_coords(coords)
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
                "area": int(mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": mask,
                "color": int(boundary_cols[0]),
            }
        )
    slots.sort(key=lambda slot: (-int(slot["area"]), int(slot["r0"]), int(slot["c0"])))

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        piece_cols = set(map(int, patch[mask]))
        assert len(piece_cols) == 1, "pieces must be monochrome"
        pieces.append(
            {
                "area": int(mask.sum()),
                "mask": mask,
                "patch": patch,
                "color": int(next(iter(piece_cols))),
            }
        )

    assert pieces, "at least one non-frame piece must exist"
    return frame_img, slots, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a colored frame template and fill each hole with the outside piece
    that matches both its boundary color and its block-level shape.

    Input structure:
    - `0` is the background. The largest foreground component is a frame whose
      hole-adjacent wall cells may use multiple colors instead of one uniform wall color.
    - Outside the frame sit monochrome pieces made from solid `k x k` blocks,
      plus isolated singleton distractors in another color.

    Rule:
    - Treat each background region inside the frame as a slot and read the
      unique wall color touching that slot.
    - Match scattered pieces to slots by downsampling block structure under all
      rotations/flips, then use piece color to disambiguate duplicate shapes.
    - Output the entire frame crop scaled by `k`, preserving colored walls, and
      paste each matched piece into its slot position.

    Key insight:
    - Shape still recovers the geometric fit, but repeated hole shapes make
      shape-only matching ambiguous; the surrounding wall color and piece color
      form the tie-breaker channel that identifies the correct pairing.
    """
    frame_img, slots, pieces = _extract_frame_slots_and_pieces(grid)
    k = _infer_k(pieces)

    candidates = []
    for piece_index, piece in enumerate(pieces):
        for var_patch, var_mask in _dihedral_variants(piece["patch"], piece["mask"]):
            unit_mask = _downsample_blocks(var_mask, k)
            if unit_mask is None:
                continue
            candidates.append(
                {
                    "pi": int(piece_index),
                    "area": int(piece["area"]),
                    "color": int(piece["color"]),
                    "um": unit_mask,
                    "vp": var_patch,
                    "vm": var_mask,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        shape_hits: dict[int, dict] = {}
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                shape_hits.setdefault(int(cand["pi"]), cand)

        assert shape_hits, "every slot must match at least one piece by shape"
        color_hits = [cand for cand in shape_hits.values() if int(cand["color"]) == int(slot["color"])]
        assert len(color_hits) == 1, "slot color must resolve to exactly one unused piece"
        found = color_hits[0]
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.kron(frame_img, np.ones((k, k), dtype=int))
    for slot, cand in placements:
        top = int(slot["r0"]) * int(k)
        left = int(slot["c0"]) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
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
                pos
                for pos in frontier
                if 0 <= pos[0] < h and 0 <= pos[1] < w and not forbidden[pos[0], pos[1]] and pos not in chosen
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


def _place_copy_of_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
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


def _boundary_sets(slots: list[dict], layout_h: int, layout_w: int) -> list[set[tuple[int, int]]] | None:
    frame_h = int(layout_h + 2)
    frame_w = int(layout_w + 2)
    all_holes = set()
    shifted_slots: list[set[tuple[int, int]]] = []
    for slot in slots:
        shifted = {tuple(map(int, rc)) for rc in (slot["coords"] + 1)}
        shifted_slots.append(shifted)
        all_holes |= shifted

    owners: dict[tuple[int, int], int] = {}
    all_boundaries: list[set[tuple[int, int]]] = []
    for idx, shifted in enumerate(shifted_slots):
        boundary: set[tuple[int, int]] = set()
        for r, c in shifted:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < frame_h and 0 <= nc < frame_w):
                    continue
                if (nr, nc) in all_holes:
                    continue
                boundary.add((int(nr), int(nc)))
        if not boundary:
            return None
        for cell in boundary:
            if cell in owners and owners[cell] != idx:
                return None
            owners[cell] = idx
        all_boundaries.append(boundary)
    return all_boundaries


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            mask, _r0, _c0 = _tight_mask_from_coords(coords)
            slots.append({"coords": coords, "area": 1, "mask": mask, "shape_key": _canonical_shape_key(mask)})
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
        mask1, _r0, _c0 = _tight_mask_from_coords(coords1)
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
            mask, _r0, _c0 = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    boundaries = _boundary_sets(slots, h, w)
    if boundaries is None:
        return None
    for slot, boundary in zip(slots, boundaries):
        slot["boundary"] = boundary

    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if int(slot["area"]) > 1:
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
    for slot in slots:
        shifted = slot["coords"] + 1
        frame[shifted[:, 0], shifted[:, 1]] = 0
    for slot, color in zip(slots, slot_colors):
        for r, c in slot["boundary"]:
            frame[int(r), int(c)] = int(color)
    return frame


def _build_expected_output(pieces: list[dict], frame_img: np.ndarray, k: int) -> np.ndarray:
    out = np.kron(frame_img, np.ones((k, k), dtype=int))
    for piece in pieces:
        shifted = piece["slot"]["coords"] + 1
        top = int(shifted[:, 0].min()) * int(k)
        left = int(shifted[:, 1].min()) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(pieces: list[dict], frame_img: np.ndarray, distractor_color: int) -> np.ndarray | None:
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
    pieces = [_make_piece(slot, slot_colors[i], k) for i, slot in enumerate(slots)]
    expected_out = _build_expected_output(pieces, frame_img, k)

    canvas = _build_input_canvas(pieces, frame_img, distractor_color)
    if canvas is None:
        return None

    dup_counts: dict[tuple[tuple[int, int], bytes], int] = {}
    for slot in slots[n_units:]:
        key = slot["shape_key"]
        dup_counts[key] = int(dup_counts.get(key, 0)) + 1
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
