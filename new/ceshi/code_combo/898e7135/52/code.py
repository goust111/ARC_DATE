# Combo variant of 898e7135 (index 52)
# combo_spec: D4/0+D8/0+D9/2
# Name: 反向缩放大洞纯推断

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Original task: small hole masks live inside the frame, outside pieces are kxk-block
#      enlargements, and the output enlarges the whole frame by k before pasting the large pieces.
#    - This combo reverses the output direction (D4/0): the frame stays at its original size and
#      each outside piece must be shrunk back into its matching small hole.
#    - It also removes all area-1 holes (D8/0), so the usual "smallest solid kxk square reveals k"
#      clue disappears. The only stable global relation is piece_area = slot_area * k^2.
#    - Finally, non-unit holes now range from area 2 up to 12 and every sample contains at least
#      one irregular large hole with area >= 8 (D9/2), making the area-ratio inference and shape
#      matching more visible than in the base task.
#    - Small trace:
#        slot areas [3, 5, 9], k = 2
#        outside piece areas become [12, 20, 36]
#        original task would enlarge the frame and paste those large sprites directly;
#        this combo keeps the frame small, infers k=2 from the multiset relation, downsamples each
#        rotated/flipped piece, and fills the 3/5/9-cell holes.
#
# b) Complexity calibration
#    - k stays in {2, 3}; that keeps the latent scale choice learnable while still requiring real
#      inference because no unit holes remain.
#    - Slot count stays modest (3-5 for k=2, 3-4 for k=3), shapes remain D4-unique, and colors are
#      one-per-piece. This prevents ambiguous matchings while leaving repeated evidence that the
#      same global shrink factor governs every piece.
#    - Large holes are forced to actually appear, and at least one such hole must be irregular in
#      its bbox. Without that constraint, the D9 contribution would often be visually invisible.
#
# c) Implicit assumption review on main()
#    - largest foreground component is the frame:
#      still valid because generator keeps the frame and all pieces 8-separated.
#    - tight hole crop equals the intended layout:
#      still valid because generator first places slots on a work grid, then crops by the union bbox.
#    - k can be read from a smallest solid square:
#      no longer valid; main() must instead infer the unique k whose slot-area multiset scaled by
#      k^2 equals the piece-area multiset.
#    - every piece is exactly divisible into uniform kxk blocks under some D4 orientation:
#      generator still guarantees that by constructing every input piece from a slot mask and then
#      applying only rotation / reflection.
#    - enlarged output canvas:
#      no longer valid; output must be the original frame size, with downsampled pieces pasted into
#      the small holes.
#
# d) Placement legality
#    - Hole slots are generated as 4-connected components and then forbidden together with their
#      8-neighborhoods, so slots never touch or merge.
#    - Pieces are placed largest-first on the outer canvas using padding=1 under 8-connectivity, so
#      they remain separate from the frame and from one another.
#    - Distractors remain isolated singleton pixels in a dedicated color, so the solver can ignore
#      them by discarding size-1 foreground components.
#
# e) Visual observability
#    - Across examples, the outside sprites are visibly much larger than the holes while the output
#      frame is not enlarged. That makes the reverse-scaling rule directly observable.
#    - Because there are no unit holes, a human cannot rely on a tiny square clue; instead the only
#      consistent explanation is that all pieces shrink by one shared factor inferred from area.
#    - Large irregular holes ensure that the combo is not mistaken for a trivial "copy colored
#      rectangles back" task.
#
# f) Visual cleanliness
#    - The frame stays monochrome, pieces stay monochrome and separated, and distractors stay single
#      pixels. No overlays or marker systems are introduced.
#    - Distinct piece colors keep large irregular sprites easy to parse even when multiple areas are
#      similar.
#
# g) Color inferability
#    - Color is not transformed by rule logic; it is merely carried by each matched piece into its
#      slot. One distinct color per piece is enough structure for humans to see correspondences
#      without turning color into a hidden lookup table.
#
# h) Quantified anti-degeneracy targets enforced by generate_input()
#    - zero unit slots in every sample;
#    - at least one slot with area >= 8 and irregular bbox occupancy;
#    - old square-side heuristic never equals the true k;
#    - oracle check main(input) == expected_output.
#
# i) New solver assumptions introduced by the combo
#    - The global area multiset identifies a unique shrink factor k >= 2.
#    - Each slot has exactly one matching piece under D4 after block-perfect downsampling.
#    - generate_input() enforces both conditions via shared-k construction and D4-shape dedup.

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


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants: list[bytes] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(bytes([rot.shape[0], rot.shape[1]]) + rot.tobytes())
        flip = np.fliplr(rot)
        variants.append(bytes([flip.shape[0], flip.shape[1]]) + flip.tobytes())
    return min(variants)


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    variants: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        for vp, vm in ((rot_patch, rot_mask), (np.fliplr(rot_patch), np.fliplr(rot_mask))):
            key = (tuple(map(int, vm.shape)), vm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append((vp, vm))
    return variants


def _downsample_piece(
    patch: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None

    out_h, out_w = hh // k, ww // k
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    out_patch = np.zeros((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            block_mask = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block_mask.all()):
                vals = patch[r * k : (r + 1) * k, c * k : (c + 1) * k][block_mask]
                if vals.size == 0:
                    return None
                color = int(vals[0])
                if np.any(vals != color):
                    return None
                out_mask[r, c] = True
                out_patch[r, c] = color
            elif bool(block_mask.any()):
                return None
    return out_patch, out_mask


def _infer_k_from_area_multisets(slots: list[dict], pieces: list[dict]) -> int:
    slot_areas = sorted(int(s["area"]) for s in slots)
    piece_areas = sorted(int(p["area"]) for p in pieces)
    max_dim = max(max(map(int, p["mask"].shape)) for p in pieces)
    candidates = []
    for k in range(2, max_dim + 1):
        if sorted(int(a) * int(k) * int(k) for a in slot_areas) == piece_areas:
            candidates.append(int(k))
    assert len(candidates) == 1, "k must be uniquely determined by area ratios"
    return int(candidates[0])


def _old_square_heuristic_k_from_pieces(pieces: list[dict]) -> int:
    guess = None
    for p in pieces:
        h, w = map(int, p["in_mask"].shape)
        if h == w and int(p["in_mask"].sum()) == int(h * w):
            guess = int(h) if guess is None else min(int(guess), int(h))
    return int(guess or 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink scattered large sprites back into the frame's small hole layout.

    Input structure:
    - The most common color is the background. The largest foreground component is a monochrome
      frame that encloses several background-colored holes.
    - Outside the frame sit one-piece-per-hole monochrome sprites that are k-times larger than the
      holes in both dimensions, under rotation or reflection. There are no area-1 holes.

    Rule:
    - Extract the small hole components from the frame as slots.
    - Infer the shared shrink factor k from the multiset relation piece_area = slot_area * k^2.
    - For each piece, test all D4 orientations, downsample by k, and place the matching shrunken
      sprite into its hole.
    - Return the cropped frame at original scale, not a k-enlarged frame.

    Key insight:
    - With no unit-hole clue, k can only be recovered from the global area ratio; once that k is
      known, valid pieces are exactly those whose kxk blocks collapse cleanly to a slot mask.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame plus scattered pieces"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_colors = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_colors).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_mask = sub == int(bg)
    hole_box = bbox(hole_mask)
    assert hole_box is not None, "frame must enclose holes"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        slot_mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        slot_mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(slot_mask.sum()), "r0": rr0, "c0": cc0, "mask": slot_mask})
    slots.sort(key=lambda item: (-int(item["area"]), int(item["r0"]), int(item["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        piece_mask = np.zeros(patch.shape, dtype=bool)
        piece_mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(piece_mask.sum()), "mask": piece_mask, "patch": patch})

    assert pieces, "expected non-frame pieces"
    assert len(pieces) == len(slots), "piece / slot count mismatch"

    k = _infer_k_from_area_multisets(slots, pieces)

    candidates: list[dict] = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            down = _downsample_piece(vp, vm, k)
            if down is None:
                continue
            dp, dm = down
            candidates.append(
                {"pi": int(pi), "area": int(piece["area"]), "down_patch": dp, "down_mask": dm}
            )

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
            if cand["down_mask"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["down_mask"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match one downsampled piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full((int(layout.shape[0]) + 2, int(layout.shape[1]) + 2), int(frame_color), dtype=int)
    for slot, cand in placements:
        paste(out, cand["down_patch"], int(slot["r0"]) + 1, int(slot["c0"]) + 1, mask=cand["down_mask"], overwrite=True)
    return out


# --------------- generator ---------------

"""Generator for combo variant D4/0+D8/0+D9/2 of ARC-AGI-2 task 898e7135.

Output-first strategy:
  1. Build a hole layout with only non-unit slots, forcing at least one irregular large slot.
  2. The expected output is the unscaled frame with each small slot filled directly.
  3. The input pieces are k-block enlargements of those slots under random dihedral transforms.
  4. Scatter the large pieces outside the frame and add isolated singleton distractors.

The generator validates main(input) == expected_output before returning.
"""


K_CHOICES = [2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE_K2 = (3, 5)
N_SLOTS_RANGE_K3 = (3, 4)

SMALL_NON_UNIT_CHOICES = [2, 3, 4, 5, 6, 7]
BIG_NON_UNIT_CHOICES = [8, 9, 10, 11, 12]
SECOND_BIG_CHOICES = [8, 9, 10]
TOTAL_SLOT_AREA_BUDGET = {2: 30, 3: 18}

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 120
GROW_SEED_TRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 600


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w


def _grow_slot(forbidden: np.ndarray, target_size: int, *, want_irregular_large: bool) -> np.ndarray | None:
    H, W = map(int, forbidden.shape)
    free_cells = [tuple(map(int, cell)) for cell in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)

    for r0, c0 in free_cells[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = []
            for nr, nc in frontier:
                if 0 <= nr < H and 0 <= nc < W and not forbidden[nr, nc] and (nr, nc) not in chosen:
                    freedom = 0
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if 0 <= ar < H and 0 <= ac < W and not forbidden[ar, ac] and (ar, ac) not in chosen:
                            freedom += 1
                    cands.append((freedom, nr, nc))
            if not cands:
                break
            cands.sort(reverse=True)
            _, nr, nc = random.choice(cands[: min(4, len(cands))])
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

        if len(chosen) != target_size:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        if want_irregular_large:
            mask, _ = _tight_mask_from_coords(coords)
            if not _is_irregular_large_shape(mask):
                continue
        return coords
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


def _choose_slot_sizes(k: int) -> list[int] | None:
    n_slots = random.randint(*(N_SLOTS_RANGE_K2 if int(k) == 2 else N_SLOTS_RANGE_K3))
    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])

    sizes = [random.choice(BIG_NON_UNIT_CHOICES)]
    if int(k) == 2 and n_slots >= 4 and random.random() < 0.35:
        candidate = random.choice(SECOND_BIG_CHOICES)
        if candidate + sum(sizes) + 2 * (n_slots - 2) <= budget:
            sizes.append(candidate)

    while len(sizes) < n_slots:
        remaining = n_slots - len(sizes)
        max_allowed = budget - sum(sizes) - 2 * (remaining - 1)
        if max_allowed < 2:
            return None
        choices = [s for s in SMALL_NON_UNIT_CHOICES if s <= max_allowed]
        if not choices:
            return None
        if int(k) == 3:
            choices = choices[: min(4, len(choices))]
        sizes.append(random.choice(choices))

    random.shuffle(sizes)
    return sizes


def _generate_slots(slot_sizes: list[int]) -> list[dict] | None:
    forbidden = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()
    need_irregular_large = True

    for size in sorted(slot_sizes, reverse=True):
        want_irregular = need_irregular_large and int(size) >= 8
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size), want_irregular_large=want_irregular)
            if coords is None:
                continue
            mask, _ = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key})
            _forbid_coords(forbidden, coords)
            if want_irregular:
                need_irregular_large = False
            break
        else:
            return None

    if need_irregular_large:
        return None
    return slots


def _crop_layout(slots: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    full_mask = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    for slot in slots:
        full_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    hole_box = bbox(full_mask)
    if hole_box is None:
        return None
    r0, c0, r1, c1 = map(int, hole_box)
    layout_mask = full_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, layout_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(layout_mask.sum()) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots: list[dict] = []
    for slot in slots:
        coords = slot["coords"].copy()
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        mask, _ = _tight_mask_from_coords(coords)
        cropped_slots.append(
            {"coords": coords, "area": int(slot["area"]), "mask": mask, "shape_key": _canonical_shape_key(mask)}
        )
    return cropped_slots, layout_mask


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = slot["mask"]
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    in_mask = np.kron(out_mask, np.ones((k, k), dtype=bool))
    in_patch = np.where(in_mask, int(color), 0).astype(int)
    rot = random.randint(0, 3)
    in_mask = np.rot90(in_mask, rot)
    in_patch = np.rot90(in_patch, rot)
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


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, frame_color: int
) -> np.ndarray:
    out = np.full((layout_h + 2, layout_w + 2), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        paste(
            out,
            piece["out_patch"],
            int(coords[:, 0].min()) + 1,
            int(coords[:, 1].min()) + 1,
            mask=piece["out_mask"],
            overwrite=True,
        )
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
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
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
    slot_sizes = _choose_slot_sizes(int(k))
    if slot_sizes is None:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < len(slot_sizes):
        return None
    piece_colors = palette[: len(slot_sizes)]

    slots = _generate_slots(slot_sizes)
    if slots is None:
        return None
    cropped = _crop_layout(slots)
    if cropped is None:
        return None
    slots, layout_mask = cropped

    if any(int(slot["area"]) == 1 for slot in slots):
        return None
    if not any(int(slot["area"]) >= 8 for slot in slots):
        return None
    if not any(_is_irregular_large_shape(slot["mask"]) for slot in slots if int(slot["area"]) >= 8):
        return None

    pieces = [_make_piece(slot, piece_colors[i], int(k)) for i, slot in enumerate(slots)]
    if _old_square_heuristic_k_from_pieces(pieces) == int(k):
        return None

    layout_h, layout_w = map(int, layout_mask.shape)
    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(frame_color))
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, int(frame_color), int(distractor_color))
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
