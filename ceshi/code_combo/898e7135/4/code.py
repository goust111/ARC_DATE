# Combo variant of 898e7135 (index 4)
# combo_spec: D1/4+D7/2
# Name: 排序选择匹配

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

"""
Design analysis for combo D1/4 + D7/2
=====================================

concretization_and_behavior_trace:
  input_visual:
    - The largest foreground component is still a monochrome frame that encloses
      background holes and frame-colored walls.
    - Outside the frame, every real piece is still a monochrome sprite made from
      solid k x k blocks, and unit slots still provide at least one k x k square
      so k stays visually recoverable.
    - The change is twofold: there is one extra same-scale sprite that does not
      correspond to any hole area, and the true non-unit holes contain at least
      one duplicated area group with different D4 shapes.
  small_trace:
    - Example holes after sorting: [area 4 L-shape at left, area 4 bar at right,
      area 1 unit].
    - True pieces are generated from those holes, but the two area-4 pieces are
      scattered so the bar appears earlier in the input ordering than the L-shape.
      An extra area-5 piece is also scattered among them.
    - Original task: use shape + D4 matching and 1:1 inventory, so the bar goes
      back to the bar hole and the L goes back to the L hole.
    - This combo: sort holes by area, sort all pieces by observed pixel area, then
      choose the length-3 subsequence whose areas are closest to [4, 4, 1] * k^2.
      The area-5 extra is discarded because [4, 4, 1] is an exact zero-cost match.
      Among the kept pieces, the earlier area-4 piece is pasted into the earlier
      area-4 hole without any shape check, so the two same-area shapes can swap.
    - The visible difference is therefore both "one outside sprite is ignored" and
      "same-area holes may receive the wrong silhouette".

complexity_calibration:
  - The generator always creates at least two non-unit holes with the same area
    but different D4 shapes, so area-only matching has a visible consequence.
  - Exactly one extra sprite is added. That is enough to force subset selection
    while keeping the scene readable.
  - The extra sprite uses the same block scale k as the true pieces. This avoids
    a degenerate shortcut where the solver can discard it purely by scale.
  - The extra area is chosen from an unused non-unit size close to one of the
    true slot sizes, so the selection cue is numeric proximity rather than an
    obviously unrelated outlier.

implicit_assumption_audit:
  - `frame_comp = max(comps, key=area)` still assumes the frame is the largest
    4-connected foreground component. The generator rejects any sample where a
    piece area meets or exceeds the frame area.
  - Tight-cropping the hole layout still assumes the union of holes touches all
    four borders of the logical layout. The generator keeps that constraint.
  - `k = smallest solid square piece` no longer tolerates arbitrary extras. The
    generator therefore guarantees at least one true unit slot and forbids extra
    area-1 pieces, so the smallest full square remains a real k x k unit piece.
  - Sorting pieces by `(-area, r0, c0)` introduces a visible-order assumption.
    The generator places pieces in monotone scan order so observed positions match
    the intended ranking.
  - Subsequence selection assumes the minimum-cost area match is decisive. The
    generator ensures the true sorted piece subsequence matches slot areas exactly,
    while every alternative subsequence has strictly positive area error.
  - Because shape is ignored after selection, pasted pieces may exceed a hole or
    the cropped output boundary. The solver uses `clip=True`, and the generator
    intentionally allows this as the visual signal of area-only matching.

placement_legality:
  - A fixed 30 x 30 canvas is used. The frame is anchored near the bottom-right,
    leaving a large top-left packing region for pieces.
  - Piece placement checks real masks with 8-neighborhood dilation so pieces do
    not overlap or touch each other or the frame.
  - Distractors remain isolated single pixels in one dedicated color.

visual_observability:
  - A human can count that there are more candidate sprites than holes, so some
    sprite must be discarded.
  - Because the extra sprite shares the same scale family as the real sprites, the
    discard rule is not reducible to "ignore the differently-sized one".
  - Repeated hole areas make the surviving subset insufficient by itself; the
    solver must also respect rank order, which becomes visible when same-area
    shapes swap.

cleanliness_and_inference:
  - Only frame, scaled sprites, and singleton distractors are present.
  - Colors are not used as hidden sort keys or labels. Non-unit true pieces keep
    distinct colors, unit pieces share one color, and the extra reuses an existing
    real-piece color so color alone cannot mark it as disposable.
"""


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 8)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_AREA_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
EXTRA_SHAPE_RETRIES = 60
GROW_SEED_TRIES = 8

CANVAS_SIZE = 30

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


ShapeKey = tuple[tuple[int, int], bytes]


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _piece_sort_key(piece: dict) -> tuple[int, int, int]:
    return (-int(piece["area"]), int(piece["r0"]), int(piece["c0"]))


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return int(r0), int(c0), mask


def _canonical_shape_key(mask: np.ndarray) -> ShapeKey:
    variants: list[ShapeKey] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((tuple(map(int, flip.shape)), flip.tobytes()))
    return min(variants)


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
    mask = np.zeros_like(forbidden)
    mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(mask)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_list)
    for start in free_list[:GROW_SEED_TRIES]:
        r0, c0 = start
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


def _choose_non_unit_sizes(n_non_unit: int) -> list[int]:
    assert n_non_unit >= 2
    dup_area = int(random.choice(DUPLICATE_AREA_CHOICES))
    sizes = [dup_area, dup_area]
    while len(sizes) < n_non_unit:
        sizes.append(int(random.choice(NON_UNIT_SIZE_CHOICES)))
    random.shuffle(sizes)
    return sizes


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[ShapeKey] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            r0, c0, mask = _tight_mask_from_coords(coords)
            slots.append(
                {
                    "coords": coords,
                    "area": 1,
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": _canonical_shape_key(mask),
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for size in _choose_non_unit_sizes(n_slots - n_units):
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            r0, c0, mask = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": key,
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    groups = defaultdict(list)
    for slot in slots:
        if int(slot["area"]) > 1:
            groups[int(slot["area"])].append(slot)
    if not any(len(group) >= 2 for group in groups.values()):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int, allow_dihedral: bool) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    patch = np.where(out_mask, int(color), 0).astype(int)
    mask = out_mask

    if allow_dihedral and int(slot["area"]) > 1:
        rot = random.randint(0, 3)
        patch = np.rot90(patch, rot)
        mask = np.rot90(mask, rot)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
            mask = np.fliplr(mask)

    return {
        "slot": slot,
        "color": int(color),
        "patch": patch,
        "mask": mask,
        "area": int(mask.sum()),
    }


def _random_extra_mask(target_size: int, forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(EXTRA_SHAPE_RETRIES):
        h = random.randint(2, min(5, target_size + 2))
        w = random.randint(2, min(5, target_size + 2))
        if h * w < target_size:
            continue
        coords = _grow_slot(np.zeros((h, w), dtype=bool), target_size)
        if coords is None:
            continue
        _r0, _c0, mask = _tight_mask_from_coords(coords)
        key = _canonical_shape_key(mask)
        if key in forbidden_keys:
            continue
        return mask
    return None


def _make_extra_piece(mask: np.ndarray, color: int, k: int) -> dict:
    slot = {"mask": mask, "area": int(mask.sum())}
    return _make_piece(slot, int(color), int(k), allow_dihedral=True)


def _desired_true_piece_order(pieces: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for piece in pieces:
        groups[int(piece["slot"]["area"])].append(piece)

    ordered: list[dict] = []
    for area in sorted(groups.keys(), reverse=True):
        group = sorted(groups[area], key=lambda p: (int(p["slot"]["r0"]), int(p["slot"]["c0"])))
        if int(area) > 1 and len(group) >= 2:
            group = list(reversed(group))
        ordered.extend(group)
    return ordered


def _selection_costs(ordered_areas: list[int], need_areas: list[int]) -> list[tuple[int, tuple[int, ...]]]:
    costs: list[tuple[int, tuple[int, ...]]] = []
    m = len(ordered_areas)
    n = len(need_areas)
    for idxs in combinations(range(m), n):
        score = 0
        for pick_i, need_area in zip(idxs, need_areas):
            score += abs(int(ordered_areas[pick_i]) - int(need_area))
        costs.append((int(score), tuple(int(i) for i in idxs)))
    costs.sort()
    return costs


def _select_piece_subsequence(ordered_pieces: list[dict], need_areas: list[int]) -> list[dict]:
    m = len(ordered_pieces)
    n = len(need_areas)
    assert m >= n, "piece count must be at least slot count"

    inf = 10**9
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    choose = [[False] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][n] = 0

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            skip = dp[i + 1][j]
            take = inf
            if m - i >= n - j:
                take = abs(int(ordered_pieces[i]["area"]) - int(need_areas[j])) + dp[i + 1][j + 1]
            if take <= skip:
                dp[i][j] = take
                choose[i][j] = True
            else:
                dp[i][j] = skip

    out: list[dict] = []
    i = 0
    j = 0
    while j < n:
        assert i < m and m - i >= n - j, "selection reconstruction failed"
        if choose[i][j]:
            out.append(ordered_pieces[i])
            i += 1
            j += 1
        else:
            i += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout by k and fill slots by ranked-area subset selection.

    Input structure:
    - The most common color is the background. The largest foreground component is
      a monochrome frame that encloses several background hole regions.
    - Outside the frame are k-scaled monochrome sprites plus single-pixel noise.
      There are more candidate sprites than holes, and at least one k x k square
      still reveals the common block scale k.

    Rule:
    - Extract the hole regions inside the frame and sort them by area descending,
      breaking ties by top-to-bottom, then left-to-right order.
    - Extract every non-singleton external sprite, sort them by observed pixel area
      descending with the same positional tiebreak, and choose the length-matched
      subsequence whose areas are closest to the sorted hole areas times k^2.
    - Paste the chosen sprites into the scaled hole anchors in that ranked order.
      No rotation, flip, or shape verification is performed after selection.

    Key insight:
    - The task is no longer geometric matching. First infer k from the smallest
      solid square, then do numeric reasoning: rank hole sizes, discard the extra
      sprite by best area fit, and keep the surviving ranked pieces in their input
      orientation.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        r0, c0, mask = _tight_mask_from_coords(coords)
        slots.append({"area": int(mask.sum()), "r0": int(r0), "c0": int(c0), "mask": mask})
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
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append(
            {
                "area": int(mask.sum()),
                "r0": int(pr0),
                "c0": int(pc0),
                "patch": patch,
                "mask": mask,
            }
        )
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    need_areas = [int(slot["area"]) * int(k) * int(k) for slot in slots]
    ordered_pieces = sorted(pieces, key=_piece_sort_key)
    assert len(ordered_pieces) >= len(slots), "piece inventory must cover all slots"

    chosen_pieces = _select_piece_subsequence(ordered_pieces, need_areas)

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, piece in zip(slots, chosen_pieces):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)

    return out


def _can_place_with_gap(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    probe = np.zeros_like(occupied)
    probe[r0 : r0 + h, c0 : c0 + w] = mask
    return not bool(np.any(_dilate8(probe) & occupied))


def _build_input_canvas(
    ordered_pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> tuple[np.ndarray, int] | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    frame_h, frame_w = layout_h + 2, layout_w + 2
    frame_top = CANVAS_SIZE - frame_h - 1
    frame_left = CANVAS_SIZE - frame_w - 1
    if frame_top < 1 or frame_left < 1:
        return None

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    frame_area = int(np.sum(frame_img == int(frame_color)))
    paste(canvas, frame_img, int(frame_top), int(frame_left))

    occupied = canvas != 0
    start_idx = 0
    for piece in ordered_pieces:
        placed = False
        for idx in range(start_idx, CANVAS_SIZE * CANVAS_SIZE):
            rr = idx // CANVAS_SIZE
            cc = idx % CANVAS_SIZE
            if not _can_place_with_gap(occupied, piece["mask"], int(rr), int(cc)):
                continue
            paste(canvas, piece["patch"], int(rr), int(cc), mask=piece["mask"], overwrite=True)
            occupied[rr : rr + piece["mask"].shape[0], cc : cc + piece["mask"].shape[1]] |= piece["mask"]
            piece["r0"] = int(rr)
            piece["c0"] = int(cc)
            start_idx = int(idx + 1)
            placed = True
            break
        if not placed:
            return None

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, CANVAS_SIZE - 1)
        cc = random.randint(0, CANVAS_SIZE - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(CANVAS_SIZE, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(CANVAS_SIZE, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas, int(frame_area)


def _build_expected_output(
    slot_order: list[dict],
    selected_pieces: list[dict],
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for slot, piece in zip(slot_order, selected_pieces):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)
    return out


def _swap_count(slot_order: list[dict], selected_true_pieces: list[dict]) -> int:
    by_area_slots: dict[int, list[dict]] = defaultdict(list)
    by_area_pieces: dict[int, list[dict]] = defaultdict(list)
    for slot in slot_order:
        if int(slot["area"]) > 1:
            by_area_slots[int(slot["area"])].append(slot)
    for piece in selected_true_pieces:
        area = int(piece["slot"]["area"])
        if area > 1:
            by_area_pieces[area].append(piece)

    count = 0
    for area, slots in by_area_slots.items():
        if len(slots) < 2:
            continue
        slot_ids = [id(slot) for slot in slots]
        piece_ids = [id(piece["slot"]) for piece in by_area_pieces[area]]
        if piece_ids != slot_ids:
            count += 1
    return int(count)


def _pick_extra_area(slot_areas: list[int]) -> int | None:
    used = set(int(a) for a in slot_areas if int(a) > 1)
    cands = [int(a) for a in NON_UNIT_SIZE_CHOICES if int(a) not in used]
    if not cands:
        return None
    return min(cands, key=lambda a: (min(abs(int(a) - int(b)) for b in used), -int(a)))


def _ordered_areas(pieces: list[dict]) -> list[int]:
    return [int(piece["area"]) for piece in pieces]


def _gen_attempt(return_meta: bool = False):
    k = int(random.choice(K_CHOICES))
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    needed_piece_colors = len(non_unit_slots) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = [int(c) for c in palette[: len(non_unit_slots)]]
    unit_color = int(palette[len(non_unit_slots)])

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    true_pieces = [_make_piece(slot, non_unit_colors[i], k, allow_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    true_pieces += [_make_piece(slot, unit_color, k, allow_dihedral=False) for slot in unit_slots]
    slot_order = sorted(slots, key=_slot_sort_key)
    true_order = _desired_true_piece_order(true_pieces)

    extra_area = _pick_extra_area([int(slot["area"]) for slot in slots])
    if extra_area is None:
        return None
    forbidden_keys = {slot["shape_key"] for slot in slots}
    extra_mask = _random_extra_mask(int(extra_area), forbidden_keys)
    if extra_mask is None:
        return None
    extra_piece = _make_extra_piece(extra_mask, random.choice([int(p["color"]) for p in true_pieces]), k)

    full_order = sorted(true_order + [extra_piece], key=lambda piece: (-int(piece["area"]), int(piece["slot"]["area"] == 1)))

    selection_scores = _selection_costs(_ordered_areas(full_order), [int(slot["area"]) * k * k for slot in slot_order])
    if not selection_scores or int(selection_scores[0][0]) != 0:
        return None
    if len(selection_scores) < 2 or int(selection_scores[1][0]) <= int(selection_scores[0][0]):
        return None

    canvas_result = _build_input_canvas(full_order, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas_result is None:
        return None
    canvas, frame_area = canvas_result

    max_piece_area = max(int(piece["area"]) for piece in full_order)
    if int(frame_area) <= int(max_piece_area):
        return None

    actual_order = sorted(full_order, key=_piece_sort_key)
    selected_actual = _select_piece_subsequence(actual_order, [int(slot["area"]) * k * k for slot in slot_order])
    if any(piece is extra_piece for piece in selected_actual):
        return None

    selected_true = [piece for piece in selected_actual if piece is not extra_piece]
    if len(selected_true) != len(slot_order):
        return None

    swap_count = _swap_count(slot_order, selected_true)
    if swap_count <= 0:
        return None

    expected_out = _build_expected_output(slot_order, selected_true, layout_h, layout_w, k, frame_color)
    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    if return_meta:
        return {
            "input": canvas,
            "selection_margin": int(selection_scores[1][0] - selection_scores[0][0]),
            "best_selection_cost": int(selection_scores[0][0]),
            "swap_count": int(swap_count),
            "extra_area_cells": int(extra_area),
            "slot_areas": [int(slot["area"]) for slot in slot_order],
        }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
