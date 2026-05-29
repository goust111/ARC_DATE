# Combo variant of 898e7135 (index 46)
# combo_spec: D1/4+D6/3+D7/2
# Name: 排序筛选双重过滤

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the combo variant
=====================================

fusion_summary:
  - D6/3 keeps shape matching alive, but only as a truth filter: a multi-pixel
    outside component is considered a true sprite iff it can still realize some
    hole shape under one global block scale k.
  - D1/4 changes the final correspondence rule: after truth filtering, slots are
    not matched by geometry anymore. They are sorted by area, true sprites are
    sorted by observed input position inside each area tier, and placement uses
    that order directly.
  - D7/2 adds surplus true sprites, so the solver must pick an area-compatible
    subsequence instead of using every true sprite.

concrete_trace:
  - Example layout: slot areas are [3, 3, 2, 1]. The two area-3 holes are D4-
    distinct, say an L and a T.
  - Outside the frame we place:
    1. the true T piece,
    2. the true L piece,
    3. one extra true copy of the L piece,
    4. the true area-2 piece,
    5. the true unit square,
    6. a fake 3x-scaled zig-zag that matches no hole,
    7. a fake 2x2 square.
  - Original 898e7135 logic would try to use all non-singletons, infer k from
    the smallest square, and fail on the fake 2x2 square or the extra inventory.
  - This combo solver instead:
    1. tests candidate k values from solid squares,
    2. discards fake sprites by asking whether a sprite matches any slot shape,
    3. sorts remaining true sprites by (-area, top, left),
    4. scans that order to extract the earliest subsequence whose pixel areas
       equal the sorted slot-area sequence,
    5. pastes those chosen sprites into the sorted slots without re-orienting
       them to fit the slot geometry.
  - Visible consequence:
    - the fake sprites disappear entirely,
    - one true sprite is dropped because there are more true sprites than holes,
    - among the two area-3 holes, the T and L can be swapped because ordering,
      not shape, decides which selected sprite goes into which area-3 slot.

complexity_calibration:
  - True scale is fixed at k=3. That makes the fake 2x2 square a stable wrong-k
    lure while keeping output size within 30x30 by shrinking the inner layout
    height range to 4-8.
  - The generator always creates exactly one duplicated non-unit area group with
    two D4-distinct hole shapes. That keeps the D1 change visually active.
  - There is exactly one surplus true sprite. This is enough to force subset
    selection without making the outside inventory unreadably crowded.
  - There are exactly two fake sprites:
    - one plausible 3x-scaled block sprite with a non-slot shape,
    - one 2x2 solid square that makes naive k inference wrong.
  - Pseudo-piece colors are unique and never reused by true sprites, so humans
    can see they are separate objects, but solver logic still ignores colors.

implicit_assumption_audit:
  - `frame_comp = max(comps, key=area)`:
    still valid only if the frame remains larger than every outside component.
    The generator rejects layouts where the frame-color area is not larger than
    every true or fake sprite area.
  - collecting all non-singleton foreground components as pieces:
    no longer means "all of them are used". The new solver explicitly allows
    fake leftovers and surplus true leftovers.
  - `k = smallest solid square piece`:
    no longer valid. A fake 2x2 square is always present, so the solver tries
    every square-derived k and requires exactly one k to pass all filters.
  - the original per-slot greedy shape match:
    no longer valid because D1 removes shape-based correspondence. We still use
    shape only in `_piece_matches_any_slot`, not in final placement.
  - zipping all true pieces with all slots:
    no longer valid because D7 allows more true sprites than slots. The new
    `_select_real_pieces_by_area` extracts the earliest area-compatible
    subsequence from the sorted true list.
  - using input orientation directly in output:
    this is now an intentional new assumption. The generator therefore allows
    non-unit pieces to appear rotated/flipped in input and validates the exact
    observed-orientation output.

placement_legality:
  - The canvas is fixed to 30x30. The frame is anchored at the bottom-right,
    leaving a large free zone for ordered sprite placement.
  - All outside objects are placed by mask-aware scanning with 8-neighbor gap
    checks, so nothing overlaps or touches the frame or another object.
  - Tight-cropping the hole layout still works because the hole union is forced
    to touch all four sides of the inner frame layout.

visual_observability:
  - D6 is visible because there are clearly extra colored multi-pixel sprites
    outside the frame that never appear in the output.
  - D7 is visible because, even after ignoring fakes, there are still more
    plausible block sprites than holes.
  - D1 is visible because the chosen duplicate-area group is reversed in input
    order, so the area-sorted output disagrees with shape-aware filling.
  - No rule depends on raw color ids, hidden object creation order, or any
    non-visual property. Ties only use top-left position, which humans can see.

cleanliness_and_inference:
  - Objects do not overlap, do not touch, and stay in the normal ARC palette.
  - Extra reasoning burden comes from one coherent outside-sprite inventory
    rather than from additional annotations or markers.
  - The generator later verifies three measurable signals on every accepted
    sample: at least one fake sprite exists, at least one true sprite is dropped,
    and the combo output differs from a shape-aware baseline on the same input.
"""


TRUE_K = 3
FALSE_K = 2

LAYOUT_H_RANGE = (4, 8)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
TRUE_EXTRA_COUNT = 1
PSEUDO_COUNT = 2

NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_AREA_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 35
PSEUDO_SHAPE_RETRIES = 80
GROW_SEED_TRIES = 8

CANVAS_SIZE = 30
GENERATE_MAX_RETRIES = 500


ShapeKey = tuple[tuple[int, int], bytes]


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _piece_sort_key(piece: dict) -> tuple[int, int, int]:
    return (-int(piece["area"]), int(piece["r0"]), int(piece["c0"]))


def _canonical_shape_key(mask: np.ndarray) -> ShapeKey:
    variants: list[ShapeKey] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((tuple(map(int, flip.shape)), flip.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
    out = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    out[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return out


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


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)
    for r0, c0 in free_cells[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < int(target_size) and frontier:
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
        if len(chosen) == int(target_size):
            return np.array(sorted(chosen), dtype=int)
    return None


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[ShapeKey] = set()
    for p, m in base:
        key = (tuple(map(int, m.shape)), m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if k <= 0 or hh % k != 0 or ww % k != 0:
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


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=_slot_sort_key)
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
        pm = np.zeros_like(patch, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append(
            {
                "area": int(pm.sum()),
                "r0": int(pr0),
                "c0": int(pc0),
                "patch": patch,
                "mask": pm,
            }
        )
    assert pieces, "at least one non-frame multi-pixel component is required"
    return g, bg, frame_color, layout, slots, pieces


def _piece_matches_any_slot(piece: dict, slots: list[dict], k: int) -> bool:
    for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
        um = _downsample_blocks(vm, int(k))
        if um is None:
            continue
        area = int(vm.sum())
        for slot in slots:
            if area != int(slot["area"]) * int(k) * int(k):
                continue
            if um.shape == slot["mask"].shape and np.array_equal(um, slot["mask"]):
                return True
    return False


def _filter_true_pieces(slots: list[dict], pieces: list[dict], k: int) -> list[dict]:
    real = [piece for piece in pieces if _piece_matches_any_slot(piece, slots, int(k))]
    real.sort(key=_piece_sort_key)
    return real


def _select_real_pieces_by_area(slots: list[dict], real_pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    placements: list[tuple[dict, dict]] = []
    idx = 0
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        while idx < len(real_pieces) and int(real_pieces[idx]["area"]) > int(need_area):
            idx += 1
        if idx >= len(real_pieces):
            return None
        if int(real_pieces[idx]["area"]) < int(need_area):
            return None
        placements.append((slot, real_pieces[idx]))
        idx += 1
    return placements


def _shape_match_placements(slots: list[dict], real_pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    piece_order = sorted(range(len(real_pieces)), key=lambda i: _piece_sort_key(real_pieces[i]))
    rank = {pi: idx for idx, pi in enumerate(piece_order)}
    candidates_by_slot: list[list[dict]] = [[] for _ in slots]

    for pi, piece in enumerate(real_pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, int(k))
            if um is None:
                continue
            area = int(vm.sum())
            for si, slot in enumerate(slots):
                if area != int(slot["area"]) * int(k) * int(k):
                    continue
                if um.shape != slot["mask"].shape:
                    continue
                if np.array_equal(um, slot["mask"]):
                    candidates_by_slot[si].append({"pi": int(pi), "vp": vp, "vm": vm})

    for cands in candidates_by_slot:
        if not cands:
            return None
        cands.sort(key=lambda cand: rank[int(cand["pi"])])

    order = sorted(
        range(len(slots)),
        key=lambda si: (len(candidates_by_slot[si]), -int(slots[si]["area"]), int(slots[si]["r0"]), int(slots[si]["c0"])),
    )
    chosen: list[dict | None] = [None] * len(slots)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        si = order[pos]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            chosen[si] = cand
            if dfs(pos + 1):
                return True
            chosen[si] = None
            used.remove(pi)
        return False

    if not dfs(0):
        return None
    return [(slots[i], chosen[i]) for i in range(len(slots)) if chosen[i] is not None]


def _build_output(
    layout_shape: tuple[int, int],
    frame_color: int,
    k: int,
    placements: list[tuple[dict, dict]],
    *,
    use_variant_patch: bool,
) -> np.ndarray:
    out_h = (int(layout_shape[0]) + 2) * int(k)
    out_w = (int(layout_shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, item in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        if use_variant_patch:
            patch = item["vp"]
            mask = item["vm"]
        else:
            patch = item["patch"]
            mask = item["mask"]
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)
    return out


def _solve_with_k(slots: list[dict], pieces: list[dict], k: int) -> tuple[list[dict], list[tuple[dict, dict]]] | None:
    real_pieces = _filter_true_pieces(slots, pieces, int(k))
    if len(real_pieces) < len(slots):
        return None
    placements = _select_real_pieces_by_area(slots, real_pieces, int(k))
    if placements is None:
        return None
    return real_pieces, placements


def _infer_k_and_placements(slots: list[dict], pieces: list[dict]) -> tuple[int, list[dict], list[tuple[dict, dict]]]:
    candidate_ks = sorted(
        {
            int(piece["mask"].shape[0])
            for piece in pieces
            if piece["mask"].shape[0] == piece["mask"].shape[1]
            and int(piece["mask"].sum()) == int(piece["mask"].shape[0] * piece["mask"].shape[1])
            and int(piece["mask"].shape[0]) > 1
        },
        reverse=True,
    )
    assert candidate_ks, "at least one solid square piece is required to infer k"

    successes: list[tuple[int, list[dict], list[tuple[dict, dict]]]] = []
    for k in candidate_ks:
        solved = _solve_with_k(slots, pieces, int(k))
        if solved is None:
            continue
        real_pieces, placements = solved
        successes.append((int(k), real_pieces, placements))

    assert len(successes) == 1, "exactly one scale must survive the three-stage filtering"
    return successes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame layout and refill it with a filtered, size-ordered subset of sprites.

    Input structure:
    - The largest foreground component is a monochrome frame around background holes.
    - Outside the frame are k-scaled block sprites, but some are fake and some are extra.
    - Solid-square sprites may suggest multiple k candidates, including a wrong one.

    Rule:
    - Try each k implied by a solid square.
    - For that k, keep only sprites whose block-level shape can match some hole under rotation/flip.
    - Sort holes by area descending, sort surviving sprites by area then input position, and extract the earliest area-compatible subsequence. Paste that chosen subsequence into the sorted holes using each sprite's observed orientation.

    Key insight:
    - Shape is only a validity filter in this combo, not the final correspondence rule.
    - The correct solution is the unique k for which fake sprites disappear and a size-sorted true subset exactly covers the hole-area sequence.
    """
    _g, _bg, frame_color, layout, slots, pieces = _parse_scene(grid)
    k, _real_pieces, placements = _infer_k_and_placements(slots, pieces)
    return _build_output(layout.shape, frame_color, k, placements, use_variant_patch=False)


def _choose_non_unit_sizes(n_non_unit: int) -> tuple[list[int], int]:
    assert n_non_unit >= 2
    dup_area = int(random.choice(DUPLICATE_AREA_CHOICES))
    pool = [size for size in NON_UNIT_SIZE_CHOICES if int(size) != int(dup_area)]
    random.shuffle(pool)
    if len(pool) < n_non_unit - 2:
        raise AssertionError("size pool too small")
    sizes = [dup_area, dup_area] + pool[: n_non_unit - 2]
    random.shuffle(sizes)
    return [int(size) for size in sizes], int(dup_area)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> tuple[list[dict], int] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[ShapeKey] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            sm = np.ones((1, 1), dtype=bool)
            r0, c0 = map(int, coords[0])
            slots.append(
                {
                    "coords": coords,
                    "area": 1,
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": sm,
                    "shape_key": _canonical_shape_key(sm),
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    non_unit_sizes, dup_area = _choose_non_unit_sizes(n_slots - n_units)
    for size in non_unit_sizes:
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            sm = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            rr0, cc0, _rr1, _cc1 = map(int, bbox_coords(coords))
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "r0": int(rr0),
                    "c0": int(cc0),
                    "mask": sm,
                    "shape_key": key,
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    non_unit_groups = defaultdict(list)
    for slot in slots:
        if int(slot["area"]) > 1:
            non_unit_groups[int(slot["area"])].append(slot)
    if len(non_unit_groups[int(dup_area)]) != 2:
        return None

    return slots, int(dup_area)


def _make_scaled_piece(mask: np.ndarray, color: int, scale: int, *, allow_dihedral: bool) -> tuple[np.ndarray, np.ndarray]:
    full_mask = np.kron(mask, np.ones((scale, scale), dtype=bool))
    patch = np.where(full_mask, int(color), 0).astype(int)
    if allow_dihedral and int(mask.sum()) > 1:
        rot = random.randint(0, 3)
        patch = np.rot90(patch, rot)
        full_mask = np.rot90(full_mask, rot)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
            full_mask = np.fliplr(full_mask)
    return patch, full_mask


def _make_true_piece(slot: dict, color: int) -> dict:
    patch, mask = _make_scaled_piece(slot["mask"], int(color), TRUE_K, allow_dihedral=True)
    return {
        "kind": "true",
        "slot": slot,
        "color": int(color),
        "area": int(mask.sum()),
        "patch": patch,
        "mask": mask,
    }


def _random_nonmatching_mask(target_size: int, forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(PSEUDO_SHAPE_RETRIES):
        base_h = random.randint(2, min(5, target_size + 2))
        base_w = random.randint(2, min(5, target_size + 2))
        if base_h * base_w < int(target_size):
            continue
        coords = _grow_slot(np.zeros((base_h, base_w), dtype=bool), int(target_size))
        if coords is None:
            continue
        mask = _tight_mask_from_coords(coords)
        hh, ww = map(int, mask.shape)
        if hh == ww and int(mask.sum()) == int(hh * ww):
            continue
        key = _canonical_shape_key(mask)
        if key in forbidden_keys:
            continue
        return mask
    return None


def _make_block_pseudo(color: int, forbidden_keys: set[ShapeKey], target_size: int) -> dict | None:
    base_mask = _random_nonmatching_mask(int(target_size), forbidden_keys)
    if base_mask is None:
        return None
    patch, mask = _make_scaled_piece(base_mask, int(color), TRUE_K, allow_dihedral=True)
    return {
        "kind": "pseudo",
        "color": int(color),
        "area": int(mask.sum()),
        "patch": patch,
        "mask": mask,
        "shape_key": _canonical_shape_key(base_mask),
    }


def _make_small_square_pseudo(color: int) -> dict:
    patch = np.full((FALSE_K, FALSE_K), int(color), dtype=int)
    mask = np.ones((FALSE_K, FALSE_K), dtype=bool)
    return {
        "kind": "pseudo",
        "color": int(color),
        "area": int(mask.sum()),
        "patch": patch,
        "mask": mask,
        "shape_key": ((FALSE_K, FALSE_K), mask.tobytes()),
    }


def _frame_area(layout_h: int, layout_w: int, slots: list[dict]) -> int:
    return int((layout_h + 2) * (layout_w + 2) - sum(int(slot["area"]) for slot in slots))


def _can_place_with_gap(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    hh, ww = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    probe = np.zeros_like(occupied)
    probe[r0 : r0 + h, c0 : c0 + w] = mask
    return not bool(np.any(_dilate8(probe) & occupied))


def _build_input_canvas(
    ordered_true_pieces: list[dict],
    pseudo_pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    frame_h, frame_w = layout_h + 2, layout_w + 2
    frame_top = CANVAS_SIZE - frame_h - 1
    frame_left = CANVAS_SIZE - frame_w - 1
    if frame_top < 1 or frame_left < 1:
        return None

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    paste(canvas, frame_img, int(frame_top), int(frame_left))

    occupied = canvas != 0
    start_idx = 0
    for obj in ordered_true_pieces + pseudo_pieces:
        placed = False
        for idx in range(start_idx, CANVAS_SIZE * CANVAS_SIZE):
            rr = idx // CANVAS_SIZE
            cc = idx % CANVAS_SIZE
            if not _can_place_with_gap(occupied, obj["mask"], int(rr), int(cc)):
                continue
            paste(canvas, obj["patch"], int(rr), int(cc), mask=obj["mask"], overwrite=True)
            occupied[rr : rr + obj["mask"].shape[0], cc : cc + obj["mask"].shape[1]] |= obj["mask"]
            obj["r0"] = int(rr)
            obj["c0"] = int(cc)
            start_idx = int(idx + 1)
            placed = True
            break
        if not placed:
            return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


def _swap_count(slot_order: list[dict], selected_true_pieces: list[dict]) -> int:
    by_area_slots: dict[int, list[dict]] = defaultdict(list)
    by_area_sources: dict[int, list[dict]] = defaultdict(list)
    for slot in slot_order:
        by_area_slots[int(slot["area"])].append(slot)
    for piece in selected_true_pieces:
        by_area_sources[int(piece["slot"]["area"])].append(piece["slot"])

    count = 0
    for area, slots in by_area_slots.items():
        if int(area) <= 1 or len(slots) < 2:
            continue
        slot_ids = [id(slot) for slot in slots]
        src_ids = [id(slot) for slot in by_area_sources[int(area)]]
        if slot_ids != src_ids:
            count += 1
    return int(count)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    _g, _bg, frame_color, layout, slots, pieces = _parse_scene(grid)
    k, real_pieces, combo_placements = _infer_k_and_placements(slots, pieces)
    shape_placements = _shape_match_placements(slots, real_pieces, k)
    combo_out = _build_output(layout.shape, frame_color, k, combo_placements, use_variant_patch=False)
    shape_diff = 0
    if shape_placements is not None:
        shape_out = _build_output(layout.shape, frame_color, k, shape_placements, use_variant_patch=True)
        shape_diff = int(not np.array_equal(combo_out, shape_out))
    return {
        "k": int(k),
        "total_piece_count": int(len(pieces)),
        "true_piece_count": int(len(real_pieces)),
        "selected_piece_count": int(len(combo_placements)),
        "pseudo_piece_count": int(len(pieces) - len(real_pieces)),
        "dropped_true_piece_count": int(len(real_pieces) - len(combo_placements)),
        "shape_diff": int(shape_diff),
    }


def _gen_attempt() -> np.ndarray | None:
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    pseudo_colors = [int(palette.pop()), int(palette.pop())]

    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = [int(palette[i]) for i in range(n_slots - n_units)]
    unit_color = int(palette[n_slots - n_units])

    layout_info = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if layout_info is None:
        return None
    slots, dup_area = layout_info
    slot_order = sorted(slots, key=_slot_sort_key)

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    unit_slots = [slot for slot in slot_order if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slot_order if int(slot["area"]) > 1]

    frame_area = _frame_area(layout_h, layout_w, slots)
    if frame_area <= int(max(int(slot["area"]) * TRUE_K * TRUE_K for slot in slots)):
        return None

    base_true_pieces: list[dict] = []
    color_by_slot_id: dict[int, int] = {}
    for idx, slot in enumerate(non_unit_slots):
        piece = _make_true_piece(slot, non_unit_colors[idx])
        base_true_pieces.append(piece)
        color_by_slot_id[id(slot)] = int(non_unit_colors[idx])
    for slot in unit_slots:
        piece = _make_true_piece(slot, unit_color)
        base_true_pieces.append(piece)
        color_by_slot_id[id(slot)] = int(unit_color)

    dup_slots = [slot for slot in slot_order if int(slot["area"]) == int(dup_area)]
    if len(dup_slots) != 2:
        return None

    extra_source = dup_slots[0]
    extra_true_piece = _make_true_piece(extra_source, color_by_slot_id[id(extra_source)])
    true_pieces = base_true_pieces + [extra_true_piece]

    pieces_by_slot: dict[int, list[dict]] = defaultdict(list)
    for piece in true_pieces:
        pieces_by_slot[id(piece["slot"])].append(piece)

    ordered_true_pieces: list[dict] = []
    for area in sorted({int(slot["area"]) for slot in slot_order}, reverse=True):
        group_slots = [slot for slot in slot_order if int(slot["area"]) == int(area)]
        if int(area) == int(dup_area):
            slot_a, slot_b = group_slots
            primary_b = pieces_by_slot[id(slot_b)][0]
            primary_a = pieces_by_slot[id(slot_a)][0]
            extras_a = pieces_by_slot[id(slot_a)][1:]
            ordered_true_pieces.extend([primary_b, primary_a, *extras_a])
        else:
            for slot in group_slots:
                ordered_true_pieces.extend(pieces_by_slot[id(slot)])

    selected_preview = _select_real_pieces_by_area(slot_order, ordered_true_pieces, TRUE_K)
    if selected_preview is None:
        return None
    selected_preview_pieces = [piece for _slot, piece in selected_preview]
    if _swap_count(slot_order, selected_preview_pieces) == 0:
        return None

    slot_shape_keys = {slot["shape_key"] for slot in slots}
    block_pseudo = _make_block_pseudo(pseudo_colors[0], slot_shape_keys, dup_area)
    if block_pseudo is None:
        return None
    if int(block_pseudo["area"]) >= int(frame_area):
        return None
    small_square_pseudo = _make_small_square_pseudo(pseudo_colors[1])
    pseudo_pieces = [block_pseudo, small_square_pseudo]

    canvas = _build_input_canvas(ordered_true_pieces, pseudo_pieces, layout_mask, layout_h, layout_w, frame_color)
    if canvas is None:
        return None

    sorted_true_after_placement = sorted(true_pieces, key=_piece_sort_key)
    expected_placements = _select_real_pieces_by_area(slot_order, sorted_true_after_placement, TRUE_K)
    if expected_placements is None:
        return None
    expected_out = _build_output((layout_h, layout_w), frame_color, TRUE_K, expected_placements, use_variant_patch=False)

    shape_placements = _shape_match_placements(slot_order, sorted_true_after_placement, TRUE_K)
    if shape_placements is None:
        return None
    shape_out = _build_output((layout_h, layout_w), frame_color, TRUE_K, shape_placements, use_variant_patch=True)
    if np.array_equal(expected_out, shape_out):
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    metrics = _behavior_metrics(canvas)
    if metrics["pseudo_piece_count"] < 2:
        return None
    if metrics["dropped_true_piece_count"] < TRUE_EXTRA_COUNT:
        return None
    if metrics["shape_diff"] == 0:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
