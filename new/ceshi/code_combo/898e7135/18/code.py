# Combo variant of 898e7135 (index 18)
# combo_spec: D8/0+D9/2
# Name: 大洞纯推断k

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Core object grammar is unchanged: one monochrome rectangular frame encloses several bg hole
#      regions, and each hole still has one outside sprite that is the same shape after kxk block
#      upscaling and some D4 transform.
#    - D8/0 removes all unit holes, so the input no longer contains any direct kxk solid clue. D9/2
#      expands non-unit hole sizes to 2-12 and requires visibly large holes.
#    - Small trace:
#        original-style sample: slot areas [1, 3, 6], k=3 -> piece areas [9, 27, 54].
#        The 3x3 solid block reveals k immediately.
#        combo sample: slot areas [3, 8, 10], k=3 -> piece areas [27, 72, 90].
#        Now there is no area-1 hole and no side-length-k square. The solver must recover one global
#        k from the area multiset relation piece_area = slot_area * k^2, then downsample each sprite
#        by that k and match the recovered masks against the large irregular slots.
#    - Pixelwise difference from the base task is therefore twofold at the same time:
#      the smallest-hole clue disappears, and at least one visible hole / sprite pair is large enough
#      that exact cell counting and irregular-shape matching become necessary.
#
# b) Complexity calibration
#    - k stays in the tiny set {2, 3}; this keeps the inference target small even though the visual
#      clue is weaker.
#    - All slots are non-unit, total slot count is kept small, and non-unit D4 shape keys are unique.
#      This avoids ambiguous reuse of the same shape while still leaving enough evidence across pieces.
#    - The generator forces at least one area-8..12 hole and makes that hole irregular in its bbox,
#      so the "large-hole" change is visible instead of being a rare edge case.
#    - Total hole area is budgeted by k so k=3 samples still fit on a 15-30 canvas with spacing.
#
# c) Implicit assumption review on main()
#    - The original "largest foreground component is the frame" assumption becomes weaker once pieces
#      can have area 8..12 scaled by k=3, so this combo solver identifies the frame by structure:
#      full rectangular border, monochrome foreground, and interior bg holes.
#    - Slot extraction still assumes holes are 4-connected and pairwise 8-separated; the generator
#      preserves that with an 8-neighborhood forbid mask.
#    - Matching still assumes each piece is assembled from clean kxk solid blocks. The generator builds
#      every sprite by kron-upscaling a boolean slot mask, so block downsampling remains exact.
#    - Pure area-ratio inference adds one new assumption: the non-singleton pieces are exactly the slot
#      copies and all share one global k. generate_input() guarantees one piece per slot at a single k.
#    - D4 matching still relies on non-unit shapes being unique up to rotation/reflection; the generator
#      enforces unique canonical D4 keys.
#
# d) Placement legality
#    - Slot placement happens on a fixed 12x8 work grid; every accepted slot and all of its 8-neighbors
#      become forbidden, so holes never touch and walls remain visible.
#    - The work grid is tight-cropped afterward, which keeps the final frame within the original spec
#      while allowing larger holes to be packed safely before cropping.
#    - Outside sprites are placed with random_free_location(..., padding=1, padding_connectivity=8), so
#      they never touch each other or the frame. Singleton distractors are also isolated.
#
# e) Visual observability
#    - Across 2-4 examples a human can see that the outside shapes are enlarged copies of the inside
#      holes, but cannot rely on a tiny square clue because there are no unit holes at all.
#    - The large-hole requirement makes the combo visually different from the base task: at least one
#      slot has 8-12 cells and an irregular silhouette, so the solver visibly has to reason about area
#      ratios and exact polyomino shape, not just spot a minimal kxk block.
#    - No hidden ordering by numeric color ids is used. Colors only separate objects visually.
#
# f) Visual cleanliness
#    - The scene still uses one frame color, one distractor color, and one color per piece. Objects are
#      separated by background gaps, and no markers are layered on top of one another.
#    - Large slots are required to be irregular but hole-free as polyominoes, preventing donut-like
#      sprites that would visually compete with frame detection.
#
# g) Quantified anti-degeneracy targets
#    - Every generated sample has zero unit slots.
#    - Every generated sample has at least one slot of area >= 8 and at least one large irregular slot.
#    - Every generated sample satisfies old_square_heuristic_k != true_k, where the old heuristic means
#      "smallest solid square piece side length" (or 1 if no solid square exists).
#
# h) New solver assumptions introduced here
#    - Frame recognition now assumes exactly one component has a complete rectangular border and interior
#      bg holes; the generator guarantees that by banning holey piece shapes.
#    - Pure area inference assumes len(pieces) == len(slots) after singleton distractors are removed.
#      The generator builds exactly one non-singleton piece per slot.

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
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(bytes([rot.shape[0], rot.shape[1]]) + rot.tobytes())
        flip = np.fliplr(rot)
        variants.append(bytes([flip.shape[0], flip.shape[1]]) + flip.tobytes())
    return min(variants)


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    inv = ~mask
    seen = np.zeros_like(inv, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and inv[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(inv & ~seen))


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w and not _mask_has_hole(mask)


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


def _grow_slot(forbidden: np.ndarray, target_size: int, *, want_irregular_large: bool) -> np.ndarray | None:
    height, width = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, cell)) for cell in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = []
            for nr, nc in frontier:
                if 0 <= nr < height and 0 <= nc < width and not forbidden[nr, nc] and (nr, nc) not in chosen:
                    freedom = 0
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if 0 <= ar < height and 0 <= ac < width and not forbidden[ar, ac] and (ar, ac) not in chosen:
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
        mask, _ = _tight_mask_from_coords(coords)
        if _mask_has_hole(mask):
            continue
        if want_irregular_large and not _is_irregular_large_shape(mask):
            continue
        return coords
    return None


def _old_square_heuristic_k_from_pieces(pieces: list[dict]) -> int:
    guess = None
    for piece in pieces:
        h, w = map(int, piece["in_mask"].shape)
        if h == w and int(piece["in_mask"].sum()) == h * w:
            guess = h if guess is None else min(int(guess), h)
    return int(guess or 1)


def _is_frame_component(comp: dict, grid: np.ndarray, bg: int) -> bool:
    coords = comp["coords"]
    if coords.size == 0:
        return False
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    if len(colors) != 1:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    mask = np.zeros_like(patch, dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not (
        bool(mask[0, :].all())
        and bool(mask[-1, :].all())
        and bool(mask[:, 0].all())
        and bool(mask[:, -1].all())
    ):
        return False
    return bool(np.any(patch[1:-1, 1:-1] == int(bg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame's hole blueprint and refill it with the scattered sprites.

    Input structure:
    - Background is the most frequent color. One component is a monochrome rectangular frame whose
      outer border is complete and whose interior contains several bg hole regions.
    - Every non-singleton component outside that frame is one monochrome sprite made from kxk blocks,
      but there are no area-1 holes and therefore no direct kxk square clue.
    - Hole regions remain 4-connected and 8-separated, and at least one of them can be a larger
      irregular polyomino with area up to 12.

    Rule:
    - Detect the frame, extract and tight-crop its bg hole layout, and split that layout into slot masks.
    - Infer the shared scale k from the global area identity piece_area = slot_area * k^2.
    - For each sprite, test all rotations/reflections, downsample by k blockwise, and place the sprite
      into the slot whose recovered boolean mask matches exactly.

    Key insight:
    The missing unit-hole clue means k must be recovered mathematically from the slot/piece area
    multisets. Once k is known, each outside sprite collapses back to its original polyomino shape, so
    larger irregular holes are still solved by the same D4 shape match after exact block downsampling.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame plus pieces"
    frame_candidates = [comp for comp in comps if _is_frame_component(comp, g, bg)]
    assert len(frame_candidates) == 1, "expected a unique structural frame"
    frame_comp = frame_candidates[0]
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
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(mask.sum()), "r0": rr0, "c0": cc0, "mask": mask})
    slots.sort(key=lambda item: (-int(item["area"]), int(item["r0"]), int(item["c0"])))
    assert slots, "frame must contain slots"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})

    assert pieces, "expected non-frame pieces"
    assert len(pieces) == len(slots), "piece/slot count mismatch"

    slot_areas = sorted(int(slot["area"]) for slot in slots)
    piece_areas = sorted(int(piece["area"]) for piece in pieces)
    max_k = max(max(map(int, piece["mask"].shape)) for piece in pieces)
    k_candidates = []
    for k in range(2, max_k + 1):
        scaled = sorted(int(area) * int(k) * int(k) for area in slot_areas)
        if scaled == piece_areas:
            k_candidates.append(int(k))
    assert len(k_candidates) == 1, "k must be uniquely determined by area ratios"
    k = int(k_candidates[0])

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
            rot_patch = np.rot90(patch, kk)
            rot_mask = np.rot90(mask, kk)
            base.append((rot_patch, rot_mask))
            base.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
        uniq = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for var_patch, var_mask in base:
            key = (tuple(map(int, var_mask.shape)), var_mask.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((var_patch, var_mask))
        return uniq

    candidates = []
    for piece_index, piece in enumerate(pieces):
        for var_patch, var_mask in dihedral_variants(piece["patch"], piece["mask"]):
            unit_mask = downsample_blocks(var_mask)
            if unit_mask is None:
                continue
            candidates.append(
                {
                    "pi": int(piece_index),
                    "area": int(piece["area"]),
                    "um": unit_mask,
                    "vp": var_patch,
                    "vm": var_mask,
                }
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
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match one piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


K_CHOICES = [2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE_K2 = (3, 5)
N_SLOTS_RANGE_K3 = (3, 4)

SMALL_NON_UNIT_CHOICES = [2, 3, 4, 5, 6, 7]
BIG_NON_UNIT_CHOICES = [8, 9, 10, 11, 12]
SECOND_BIG_CHOICES = [8, 9, 10]
TOTAL_SLOT_AREA_BUDGET = {2: 30, 3: 20}

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 160
GROW_SEED_TRIES = 24

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 700


def _choose_slot_sizes(k: int) -> list[int] | None:
    n_slots = random.randint(*(N_SLOTS_RANGE_K2 if k == 2 else N_SLOTS_RANGE_K3))
    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])
    sizes = [random.choice(BIG_NON_UNIT_CHOICES)]

    if k == 2 and n_slots >= 4 and random.random() < 0.35:
        extra_big = random.choice(SECOND_BIG_CHOICES)
        if extra_big + sum(sizes) + 2 * (n_slots - 2) <= budget:
            sizes.append(extra_big)

    while len(sizes) < n_slots:
        remaining = n_slots - len(sizes)
        max_allowed = budget - sum(sizes) - 2 * (remaining - 1)
        if max_allowed < 2:
            return None
        choices = [size for size in SMALL_NON_UNIT_CHOICES if size <= max_allowed]
        if not choices:
            return None
        if k == 3:
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
        want_irregular = need_irregular_large and size >= 8
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size, want_irregular_large=want_irregular)
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

    cropped_slots = []
    for slot in slots:
        coords = slot["coords"].copy()
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        mask, _ = _tight_mask_from_coords(coords)
        if _mask_has_hole(mask):
            return None
        cropped_slots.append(
            {
                "coords": coords,
                "area": int(slot["area"]),
                "mask": mask,
                "shape_key": _canonical_shape_key(mask),
            }
        )
    return cropped_slots, layout_mask


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
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


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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
    slot_sizes = _choose_slot_sizes(k)
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

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    if _old_square_heuristic_k_from_pieces(pieces) == int(k):
        return None

    layout_h, layout_w = map(int, layout_mask.shape)
    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    if frame_area <= max(int(piece["in_mask"].sum()) for piece in pieces):
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
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
