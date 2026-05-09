from __future__ import annotations

# Variant of 898e7135
# Dimension: D7 — 精灵-洞基数关系 (meta)
# Change: 严格 1:1 — 每个精灵恰好对应一个洞，每个洞恰好对应一个精灵，总数相等... → 精灵数量多于洞数量（需选择）

"""
Design analysis for the D7 variant
==================================

source_task:
  none: the variant is derived directly from 898e7135, so only the original
  solver/generator/spec are used as the baseline.

concretization_and_behavior_trace:
  input_visual:
    - The frame is still the largest foreground component: a monochrome 1-cell
      border around an inner layout of background holes and frame-color walls.
    - The correct pieces are still monochrome block sprites made from one common
      scale k_true, and they still match the hole shapes up to rotation/flip.
    - The new visual change is that there are extra external sprites. At least
      one extra sprite is a smaller solid square built from a different block
      scale k_extra=2, so the outside inventory is visibly larger than the hole
      inventory and also visually mixed in scale.
  small_trace:
    - Original: slots = {L-shape area 3, unit area 1}, pieces = {3x-scaled L,
      3x3 square}. Solver takes the smallest solid square (3) as k and matches
      both pieces greedily.
    - Variant: add an extra 2x2 square and optionally another 2x-scaled shape.
      Now the outside pieces are {3x-scaled L, 3x3 square, 2x2 square, ... }.
      The old "smallest solid square => k" rule would pick k=2 and fail.
      The new solver instead tries every square-derived candidate k, asks
      "which k allows every slot to be matched injectively by some subset of
      pieces?", and keeps the successful subset while discarding the rest.
  visible_difference:
    - The difference is not hidden in color ids or ordering. A human can see
      that there are more outside sprites than holes, and that only the larger
      family of sprites consistently fits the hole silhouettes.

complexity_calibration:
  - Extra sprites are limited to 1-2 per instance, so the selection dimension
    is clear without clutter.
  - True non-unit slot shapes remain D4-distinct to avoid ambiguous many-way
    matches.
  - At least one extra shares the role of a plausible piece (solid square), so
    the subset choice is not reducible to "ignore obviously malformed noise".
  - Colors are not used for selection: extra pieces reuse real piece colors, so
    shape/scale consistency is the only stable cue.

implicit_assumptions_review:
  - `frame_comp = max(comps, key=area)` still holds because the frame is kept as
    the largest connected foreground component and all pieces remain separated.
  - Tight-cropping the hole layout still works because holes are forced to touch
    all four sides of the inner layout.
  - The original `k = smallest solid square piece` no longer holds. It relied on
    the old 1:1 inventory having only true pieces plus singleton noise. This
    variant intentionally violates that by adding a smaller solid-square extra.
  - The original greedy slot loop assumed that finding one unused matching piece
    per slot was enough. In the variant, unused pieces are allowed, so the
    solver must solve an injective slot->piece assignment while tolerating
    leftovers.
  - D4-variant deduplication must include shape dimensions, not just raw bytes:
    otherwise 1x2 and 2x1 full masks collide in `tobytes()`, which becomes more
    harmful once multiple scales and spare pieces are present.

placement_legality:
  - Frame and pieces are still placed on a bg=0 canvas with at least one empty
    cell of 8-neighborhood separation, so components do not merge.
  - Extras are built exactly like ordinary sprites (scaled connected masks), so
    the solver can parse them through the same component pipeline.
  - Output size is kept within 30x30 by fixing the true scale to 3 and capping
    the inner layout at 8x8.

visual_cleanliness:
  - Distractors remain isolated single pixels in one dedicated color.
  - Spare sprites are not layered over the frame or other sprites.
  - The number of extras is small, so the canvas remains readable.

color_inference:
  - No color mapping rule is introduced. Colors are simply carried by the chosen
    subset pieces into the output, exactly as in the original task.
"""

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


# --- tunables ----------------------------------------------------------------

TRUE_K = 3                        # fixed so extra 2x2 square can invalidate old k inference
EXTRA_K = 2                       # spare sprites use a smaller block size
LAYOUT_H_RANGE = (4, 8)           # keeps output within 30 rows when scaled by TRUE_K
LAYOUT_W_RANGE = (4, 8)           # keeps output within 30 cols when scaled by TRUE_K
N_SLOTS_RANGE = (3, 6)            # total slot count (incl. unit)
N_UNITS_MAX = 2                   # max unit-slot count per example
N_EXTRA_RANGE = (1, 2)            # strictly more sprites than holes
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
EXTRA_SHAPE_RETRIES = 60
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 5
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


# --- shared helpers ----------------------------------------------------------

ShapeKey = tuple[tuple[int, int], bytes]


def _canonical_shape_key(mask: np.ndarray) -> ShapeKey:
    """Canonical representation under D4 for a tight bool mask."""
    variants: list[ShapeKey] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((tuple(map(int, flip.shape)), flip.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
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
    H, W = forbidden.shape
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
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = hh // k, ww // k
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _solve_slots_for_k(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates: list[dict] = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(p["area"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    slot_candidates: list[list[dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        usable = [
            cand
            for cand in candidates
            if int(cand["area"]) == int(need_area)
            and cand["um"].shape == s["mask"].shape
            and np.array_equal(cand["um"], s["mask"])
        ]
        if not usable:
            return None
        slot_candidates.append(usable)

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(slot_candidates[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
    )
    assignment: list[dict | None] = [None] * len(slots)
    used_piece: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        si = order[pos]
        for cand in slot_candidates[si]:
            pi = int(cand["pi"])
            if pi in used_piece:
                continue
            used_piece.add(pi)
            assignment[si] = cand
            if dfs(pos + 1):
                return True
            assignment[si] = None
            used_piece.remove(pi)
        return False

    if not dfs(0):
        return None
    return [(slots[i], assignment[i]) for i in range(len(slots)) if assignment[i] is not None]


def _make_piece_from_mask(cell_mask: np.ndarray, color: int, scale: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(cell_mask, np.ones((scale, scale), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    if apply_dihedral and int(cell_mask.sum()) > 1:
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
        "area": int(out_mask.sum()),
        "cell_mask": cell_mask,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with a selected subset of pieces.

    Variant rule:
    - Outside the frame there are more candidate sprites than holes.
    - True pieces all share one block scale k and match hole shapes under D4.
    - Extra sprites may use another block scale and must be discarded.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
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

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = _tight_mask_from_coords(coords)
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
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
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    possible_ks = sorted(
        {
            int(h)
            for p in pieces
            for h, w in [tuple(map(int, p["mask"].shape))]
            if h == w and int(p["mask"].sum()) == int(h * w)
        },
        reverse=True,
    )
    assert possible_ks, "at least one solid square piece is required to infer scale candidates"

    placements = None
    k = None
    for cand_k in possible_ks:
        trial = _solve_slots_for_k(slots, pieces, int(cand_k))
        if trial is not None:
            placements = trial
            k = int(cand_k)
            break
    assert placements is not None and k is not None, "some scale must explain all slots"

    outH = (int(layout.shape[0]) + 2) * int(k)
    outW = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((outH, outW), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- layout generation -------------------------------------------------------

def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[ShapeKey] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            sm = np.ones((1, 1), dtype=bool)
            slots.append(
                {
                    "coords": coords,
                    "area": 1,
                    "mask": sm,
                    "shape_key": _canonical_shape_key(sm),
                }
            )
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
            sm = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": size, "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _random_nonmatching_mask(target_size: int, forbidden_keys: set[ShapeKey]) -> np.ndarray | None:
    for _ in range(EXTRA_SHAPE_RETRIES):
        h = random.randint(2, min(5, target_size + 2))
        w = random.randint(2, min(5, target_size + 2))
        if h * w < target_size:
            continue
        coords = _grow_slot(np.zeros((h, w), dtype=bool), target_size)
        if coords is None:
            continue
        sm = _tight_mask_from_coords(coords)
        hh, ww = map(int, sm.shape)
        if hh == ww and int(sm.sum()) == int(hh * ww):
            continue
        key = _canonical_shape_key(sm)
        if key in forbidden_keys:
            continue
        return sm
    return None


def _build_expected_output(pieces: list[dict], slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * TRUE_K, (layout_w + 2) * TRUE_K), int(frame_color), dtype=int)
    for p, slot in zip(pieces, slots):
        coords = slot["coords"]
        top = (int(coords[:, 0].min()) + 1) * TRUE_K
        left = (int(coords[:, 1].min()) + 1) * TRUE_K
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    all_pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in all_pieces)
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

    for p in sorted(all_pieces, key=lambda piece: -int(piece["in_mask"].sum())):
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


# --- main generator attempt --------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    n_extra = random.randint(*N_EXTRA_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    true_pieces = [
        _make_piece_from_mask(s["mask"], non_unit_colors[i], TRUE_K, apply_dihedral=True)
        for i, s in enumerate(non_unit_slots)
    ]
    true_pieces += [_make_piece_from_mask(s["mask"], unit_color, TRUE_K, apply_dihedral=False) for s in unit_slots]

    real_piece_colors = [p["color"] for p in true_pieces]
    slot_keys = {s["shape_key"] for s in slots}
    non_unit_slot_areas = [int(s["area"]) for s in non_unit_slots]

    extra_masks: list[np.ndarray] = [np.ones((1, 1), dtype=bool)]
    extra_keys: set[ShapeKey] = set()
    while len(extra_masks) < n_extra:
        target_size = random.choice(non_unit_slot_areas)
        mask = _random_nonmatching_mask(target_size, slot_keys | extra_keys)
        if mask is None:
            return None
        extra_masks.append(mask)
        extra_keys.add(_canonical_shape_key(mask))

    extra_pieces = [
        _make_piece_from_mask(mask, random.choice(real_piece_colors), EXTRA_K, apply_dihedral=(int(mask.sum()) > 1))
        for mask in extra_masks
    ]

    expected_out = _build_expected_output(true_pieces, non_unit_slots + unit_slots, layout_h, layout_w, frame_color)

    canvas = _build_input_canvas(true_pieces + extra_pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
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
