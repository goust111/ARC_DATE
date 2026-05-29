# Combo variant of 898e7135 (index 37)
# combo_spec: D7/2+D10/0
# Name: 同色选择匹配

from __future__ import annotations

"""
Design analysis for combo D7/2 + D10/0
======================================

concretization_and_behavior_trace:
  fused_visual:
    - The input still contains one monochrome rectangular frame around a tight
      inner hole/wall layout, plus scattered external sprites and singleton
      noise on background 0.
    - The crucial fused change is outside inventory: there are more sprites than
      holes, and every non-singleton sprite uses the same visible color.
    - A human now sees several same-colored candidates and must select the
      subset whose block geometry fits the holes after rotation/flip.
  small_trace:
    - Suppose the frame contains three slots: an L triomino, a zigzag tetromino,
      and one unit slot. True pieces are their 3x-scaled sprites, all in one
      shared color.
    - Add one spare 2x2 square and optionally one spare 2x-scaled nonmatching
      shape, also in that same shared color.
    - Original task: color or exact inventory count implicitly helps, and
      `k = smallest solid square` is safe.
    - Fused task: the smallest solid square is the spare 2x2, so naive `k=2`
      fails; color cannot distinguish anything; the solver must try candidate k
      values and keep only the subset that injectively explains every hole.
  visible_difference:
    - Compared with the base task, the board plainly shows too many same-colored
      sprites for the available holes, so both "selection" and "shape-only
      matching" are visually exposed.

complexity_calibration:
  - True scale is fixed to 3 and spare scale to 2. This keeps the selection
    dimension strong without introducing many possible scales.
  - Each example has 3-6 slots, 1-2 extras, and at least two non-unit slots so
    same-color ambiguity is visible on multiple complex shapes.
  - Non-unit hole shapes are D4-distinct, which avoids ambiguous many-to-one
    assignments after color information is removed.
  - Extras are plausible sprites rather than singleton noise, but their count is
    small enough that the board stays readable.

implicit_assumptions_review:
  - `frame_comp = max(comps, key=area)` requires the frame to remain the largest
    foreground component. The generator explicitly checks frame area against all
    piece areas before accepting a sample.
  - Tight-cropping the hole layout assumes holes touch all four sides of the
    intended inner layout. The generator enforces this exactly as in the base
    task.
  - `k = smallest solid square` is invalid under this combo because a same-color
    spare 2x2 square is intentionally present. The solver now enumerates square-
    derived k candidates and asks which one yields a full injective cover.
  - Greedy first-fit matching is no longer enough because unused pieces are now
    legal. The solver therefore performs a small DFS assignment over candidate
    piece-slot matches.
  - D4 deduplication cannot key only by raw bytes: rectangular masks like 1x2
    and 2x1 can collide. The solver and generator both key by shape + bytes.

placement_legality:
  - Frame, true pieces, spare pieces, and singleton distractors are all placed
    with at least one background cell of 8-neighborhood separation.
  - Extras are ordinary connected sprites, so the solver can parse them through
    the same component pipeline as true pieces.
  - Layout size is capped at 8x8 because true scale is fixed to 3; this keeps
    the enlarged output within the ARC 30x30 limit.

visual_observability_and_cleanliness:
  - Only four semantic regions remain: background, frame, same-colored sprites,
    and singleton distractors. This is simpler than the original palette while
    still cleanly separable by spacing and shape.
  - Because there are multiple complex same-colored sprites and more sprites
    than holes, a human can infer that color is irrelevant and subset selection
    is required.
  - No hidden ordering or numeric-color convention is used. The only stable cue
    is visible geometry, which is exactly the intended reasoning channel.
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

TRUE_K = 3
EXTRA_K = 2
LAYOUT_H_RANGE = (4, 8)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
MIN_NON_UNIT_SLOTS = 2
N_EXTRA_RANGE = (1, 2)
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
    """Canonical representation of a tight bool mask under D4."""
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
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    slot_candidates: list[list[dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        usable = [
            cand
            for cand in candidates
            if int(cand["area"]) == int(need_area)
            and cand["um"].shape == slot["mask"].shape
            and np.array_equal(cand["um"], slot["mask"])
        ]
        if not usable:
            return None
        slot_candidates.append(usable)

    order = sorted(
        range(len(slots)),
        key=lambda i: (
            len(slot_candidates[i]),
            -int(slots[i]["area"]),
            int(slots[i]["r0"]),
            int(slots[i]["c0"]),
        ),
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


def _frame_component_area(layout_mask: np.ndarray) -> int:
    frame_h, frame_w = map(int, (layout_mask.shape[0] + 2, layout_mask.shape[1] + 2))
    return int(frame_h * frame_w - int(layout_mask.sum()))


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame layout and fill its holes using a selected same-colored subset.

    Input structure:
    - Background is the most common color, and the largest foreground component
      is a monochrome frame enclosing background holes.
    - Outside the frame are several disconnected sprites built from solid k-by-k
      blocks, but there are more sprites than holes and all sprites share one color.
    - At least one spare solid square may use a smaller block scale than the true pieces.

    Rule:
    - Extract the hole components inside the frame as slot masks.
    - Collect every non-singleton external sprite, try each square-derived scale
      candidate, and find an injective slot-to-piece assignment under rotation/flip.
    - Discard leftover sprites, place the matched subset into the enlarged frame,
      and return the cropped scaled frame with the frame color as background.

    Key insight:
    - Color is intentionally useless and the smallest square can be a decoy, so
      the invariant is the one scale whose block-downsampled sprite shapes can
      cover all slot masks exactly under D4 symmetry.
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
        rr0, cc0, _, _ = map(int, bbox_coords(coords))
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
            for piece in pieces
            for h, w in [tuple(map(int, piece["mask"].shape))]
            if h == w and int(piece["mask"].sum()) == int(h * w)
        },
        reverse=True,
    )
    assert possible_ks, "at least one solid square piece is required"

    placements = None
    k = None
    for cand_k in possible_ks:
        trial = _solve_slots_for_k(slots, pieces, int(cand_k))
        if trial is not None:
            placements = trial
            k = int(cand_k)
            break
    assert placements is not None and k is not None, "one scale must explain all slots"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)

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

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
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
    for piece, slot in zip(pieces, slots):
        coords = slot["coords"]
        top = (int(coords[:, 0].min()) + 1) * TRUE_K
        left = (int(coords[:, 1].min()) + 1) * TRUE_K
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in all_pieces)
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

    for piece in sorted(all_pieces, key=lambda item: -int(item["in_mask"].sum())):
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


# --- main generator attempt --------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - MIN_NON_UNIT_SLOTS)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)
    n_extra = random.randint(*N_EXTRA_RANGE)

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
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    true_pieces = [
        _make_piece_from_mask(slot["mask"], piece_color, TRUE_K, apply_dihedral=True)
        for slot in non_unit_slots
    ]
    true_pieces += [
        _make_piece_from_mask(slot["mask"], piece_color, TRUE_K, apply_dihedral=False)
        for slot in unit_slots
    ]

    frame_area = _frame_component_area(layout_mask)
    if frame_area <= max(int(piece["area"]) for piece in true_pieces):
        return None

    slot_keys = {slot["shape_key"] for slot in slots}
    non_unit_slot_areas = [int(slot["area"]) for slot in non_unit_slots]

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
        _make_piece_from_mask(mask, piece_color, EXTRA_K, apply_dihedral=(int(mask.sum()) > 1))
        for mask in extra_masks
    ]
    if frame_area <= max(int(piece["area"]) for piece in true_pieces + extra_pieces):
        return None

    expected_out = _build_expected_output(
        true_pieces,
        non_unit_slots + unit_slots,
        layout_h,
        layout_w,
        frame_color,
    )

    canvas = _build_input_canvas(
        true_pieces + extra_pieces,
        layout_mask,
        layout_h,
        layout_w,
        frame_color,
        distractor_color,
    )
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
