# Combo variant of 898e7135 (index 34)
# combo_spec: D3/2+D15/0
# Name: 彩色同形洞消歧

from __future__ import annotations

"""
Design analysis for combo D3/2 + D15/0
======================================

a) Concrete behavior and trace
  - Original task:
    - The frame enclosed background-colored holes.
    - The solver extracted those holes as boolean masks.
    - Outside pieces could be rotated/flipped, so matching was done by shape under D4.
    - Non-unit hole shapes were unique, so greedy shape matching was safe.
  - Combo task:
    - The frame interior no longer contains background holes. Each slot region is filled
      with its own solid marker color.
    - Several non-unit slots may share the same shape. A small example is two red/blue
      L-shaped slots inside the frame and one green 1-cell slot elsewhere.
    - Outside the frame sit one red L piece, one blue L piece, and one green k*k square.
    - The solver must crop the colored layout, read each slot as a monochrome component,
      map color -> slot position, and then place the outside piece of the same color.
  - Difference:
    - The original `sub == bg` hole recovery fails immediately because slots are not bg.
    - Pure shape matching is intentionally the wrong abstraction because duplicate shapes
      are present and color is the visible correspondence key.

b) Complexity calibration
  - The generator forces at least one duplicate non-unit D4 shape pair in every sample,
    so the D15 change is always observable instead of being a rare corner case.
  - Slot colors are unique within a sample. This keeps the color mapping one-to-one and
    makes two identical shapes still visually distinguishable.
  - Outside pieces keep the same orientation as their slots. Allowing random D4 transforms
    here would introduce a second hidden problem: color would say which slot to use, but
    shape would still be needed to recover orientation. That would dilute the intended rule.
  - Color budget stays within ARC limits: bg + frame + distractor + up to 6 slot colors.

c) Implicit-assumption audit of the original main()
  - `hole = sub == bg` assumed slots were background holes. Invalid now: slots are colored.
  - `components_mask(layout)` on a boolean hole mask assumed slot identity came from shape.
    Invalid now: the solver must preserve per-slot colors.
  - Dihedral candidate search assumed shape was the pairing signal. Invalid now because
    identical slot shapes may repeat and color is the actual identifier.
  - Greedy first shape match assumed non-unit shapes were unique. Invalid now because the
    generator deliberately creates duplicate-shape groups.
  - The frame-color inference from the dominant color of the largest multicolor component
    still works: frame-color cells are more numerous than any single slot color.

d) Placement legality
  - Slots remain 4-connected and pairwise 8-separated in the inner layout.
  - The duplicate slot is placed as an exact translated copy of an earlier non-unit mask,
    and both copies are checked against the same forbidden map, so they never overlap or
    touch diagonally.
  - Pieces are scattered with `random_free_location`, preserving at least one bg cell of
    clearance from the frame and from one another.
  - Distractors are isolated singletons in a dedicated color distinct from frame and slots.

e) Visual observability
  - A human can immediately see that hole positions are already color-coded inside the frame.
  - A human can also see repeated shapes with different colors, so shape-only matching is
    visibly ambiguous while color-position correspondence is visibly sufficient.
  - Because samples always contain at least one duplicate non-unit shape pair, 2-3 examples
    are enough to show that the task is not the original shape-matching one.

f) Visual cleanliness
  - There is no layered marker system. The slot itself carries the color, so the frame is
    simpler than the wall-marker variant and remains easy to read.
  - All semantic elements occupy distinct roles: frame walls use one color, each slot uses
    one solid color, pieces reuse those colors, and distractors use a fresh singleton color.

g) Color-change inferability
  - Colors are copied directly: slot color equals piece color equals output fill color for
    that slot. There is no arbitrary recoloring table to infer.
  - The rule uses few colors per sample and ties each color to a concrete region and sprite,
    so the mapping is inferable from structure rather than numeric color ids.

h) Fusion-specific non-degeneracy target
  - The generator must not silently fall back to the original behavior.
  - Two enforced signals prevent that:
    - the inner frame contains no background holes at all, only colored slot regions;
    - at least one non-unit canonical slot shape appears more than once.
  - Therefore the original solver's hole-extraction pipeline is incompatible with these inputs.

i) New solver assumptions introduced here and generator support
  - Assumption: each slot color appears exactly once inside the frame.
    Support: generator assigns distinct colors to all slots.
  - Assumption: each slot color appears on exactly one outside piece.
    Support: generator builds one piece per slot with the same unique color.
  - Assumption: each piece is already in the slot's orientation.
    Support: generator never applies dihedral transforms in this combo variant.
  - Assumption: cropping `sub != frame_color` yields the exact inner layout bbox.
    Support: generator rejects layouts unless colored slots touch all four sides of the layout.
"""

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox, bbox_coords, crop_bbox, find_components, get_bg, paste, random_free_location


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


def _slot_from_coords(coords: np.ndarray, color: int) -> dict:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    h = int(rr1 - rr0 + 1)
    w = int(cc1 - cc0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return {
        "color": int(color),
        "coords": coords.astype(int),
        "area": int(coords.shape[0]),
        "r0": int(rr0),
        "c0": int(cc0),
        "mask": mask,
    }


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
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a color-coded frame layout and place each outside sprite into the
    same-colored slot, even when multiple slots share the same shape.

    Input structure:
    - bg is the most common color.
    - The largest 4-connected non-bg component is a rectangular frame whose interior
      contains frame-color walls plus colored slot regions.
    - Each slot color is unique inside the frame and appears on exactly one outside piece.
    - Outside pieces are already oriented k×k-block-scaled copies; k is the side of the
      smallest solid square piece.

    Rule:
    - Crop the colored layout from inside the frame by locating all non-frame-color cells.
    - Extract monochrome slot components, map each slot color to its layout position, and
      find the outside piece of the same color.
    - Scale the framed layout by k and paste each same-color piece into its slot position.

    Key insight:
    - Shape is no longer the identity signal. Duplicate hole shapes are allowed, so color
      alone determines which outside sprite belongs to which destination slot.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "need a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    slot_mask = sub != int(frame_color)
    slot_box = bbox(slot_mask)
    assert slot_box is not None, "frame must contain colored slots"
    lr0, lc0, lr1, lc1 = map(int, slot_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slot_comps = find_components(layout, bg=frame_color, connectivity=4, monochromatic=True, return_format="dict")
    assert slot_comps, "frame must contain at least one slot"
    slots_by_color: dict[int, dict] = {}
    for comp in slot_comps:
        color = int(comp["color"])
        slot = _slot_from_coords(comp["coords"], color)
        assert color not in slots_by_color, "slot colors must be unique"
        slots_by_color[color] = slot

    mono_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces_by_color: dict[int, dict] = {}
    for comp in mono_comps:
        coords = comp["coords"]
        if int(coords.shape[0]) <= 1:
            continue
        inside_frame = bool(
            np.all(
                (coords[:, 0] >= fr0)
                & (coords[:, 0] <= fr1)
                & (coords[:, 1] >= fc0)
                & (coords[:, 1] <= fc1)
            )
        )
        if inside_frame:
            continue
        color = int(comp["color"])
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert color not in pieces_by_color, "piece colors must be unique"
        pieces_by_color[color] = {
            "color": int(color),
            "area": int(coords.shape[0]),
            "patch": patch,
            "mask": mask,
        }

    assert pieces_by_color, "at least one outside piece must exist"
    assert set(slots_by_color) == set(pieces_by_color), "slot colors and piece colors must match exactly"

    pieces = list(pieces_by_color.values())
    k = _infer_k(pieces)

    placements: list[tuple[dict, dict]] = []
    for color, slot in slots_by_color.items():
        piece = pieces_by_color[color]
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must equal scaled slot area"
        um = _downsample_blocks(piece["mask"], k)
        assert um is not None, "piece must be composed of full k-by-k blocks"
        assert np.array_equal(um, slot["mask"]), "same-color piece must match same-color slot orientation"
        placements.append((slot, piece))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, piece in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True)

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
AREA_BUFFER_CONST = 70

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
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
    slot = _slot_from_coords(coords, color=-1)
    return slot["mask"]


def _place_copy_of_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
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
    - unit slots first,
    - at least one duplicate non-unit D4-shape pair,
    - all remaining non-unit shapes unique under D4.
    """
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    n_non_units = int(n_slots - n_units)
    if n_non_units < 2:
        return None

    dup_slots = None
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
        shape_key = _canonical_shape_key(mask1)
        dup_slots = [
            {"coords": coords1, "area": int(size), "mask": mask1, "shape_key": shape_key},
            {"coords": coords2, "area": int(size), "mask": mask1.copy(), "shape_key": shape_key},
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
            mask = _mask_from_coords(coords)
            shape_key = _canonical_shape_key(mask)
            if shape_key in used_shape_keys:
                continue
            used_shape_keys.add(shape_key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": shape_key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_layout_colors(slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    layout = np.full((layout_h, layout_w), int(frame_color), dtype=int)
    for slot in slots:
        coords = slot["coords"]
        layout[coords[:, 0], coords[:, 1]] = int(slot["color"])
    return layout


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        slot = piece["slot"]
        top = (int(slot["coords"][:, 0].min()) + 1) * int(k)
        left = (int(slot["coords"][:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_colors: np.ndarray,
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
    frame_img[1:-1, 1:-1] = layout_colors
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
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != 0:
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
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    slot_colors = [palette.pop() for _ in range(n_slots)]
    for slot, color in zip(slots, slot_colors):
        slot["color"] = int(color)

    layout_colors = _build_layout_colors(slots, layout_h, layout_w, frame_color)
    slot_mask = layout_colors != int(frame_color)
    if not (slot_mask[0].any() and slot_mask[-1].any() and slot_mask[:, 0].any() and slot_mask[:, -1].any()):
        return None

    non_unit_keys = Counter(_canonical_shape_key(slot["mask"]) for slot in slots if int(slot["area"]) > 1)
    if max(non_unit_keys.values(), default=0) < 2:
        return None

    pieces = [_make_piece(slot, int(slot["color"]), k) for slot in slots]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    canvas = _build_input_canvas(pieces, layout_colors, layout_h, layout_w, frame_color, distractor_color)
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
