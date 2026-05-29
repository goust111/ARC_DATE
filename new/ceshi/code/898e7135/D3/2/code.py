# Variant of 898e7135
# Dimension: D3 — 目标对象：框架与洞的结构 (dst)
# Change: hole_filling — 单色矩形框架（1 像素厚边框），内部布局由洞（背景色单元格）和墙壁（框架色单元格）组成，3... → 洞区域带颜色标记

"""
Design analysis for the D3 variant
==================================

variant_spec:
  preserved_structure:
    grid: 15-30 rows x 15-30 cols
    background: 0 and remains the most common color
    frame:
      outer_shape: monochrome rectangle with 1-cell-thick border
      inner_layout_size: 4-12 rows x 4-8 cols
      slot_count: 3-6
      slot_connectivity: each slot is 4-connected and 8-separated from others
      slot_area: 1-7 cells
      unit_slots: 1-2
    pieces:
      count: one outside piece per slot
      scale_factor_k: 2 or 3, still inferred from the smallest solid square piece
      placement: all pieces outside the frame with >= 1 background-cell gap
    distractors:
      type: isolated singleton pixels in one fresh color
      count: 8-13
  core_D3_change:
    old_layout: holes inside the frame were background-colored, so the solver had to
      recover them as `sub == bg` and then shape-match outside pieces to slot masks.
    new_layout: each slot is filled by its own unique marker color inside the frame.
      The outside piece uses the same color as its slot marker.
    pairing_rule: color -> position mapping, not shape -> position mapping.
    necessary_supporting_adjustment:
      Outside pieces are no longer randomly rotated/flipped. If orientation were still
      randomized, color would reveal only which slot to use, not which orientation to
      paste, creating an extra hidden rule unrelated to the D3 change.

behavior_trace:
  small_example:
    layout_before_scaling:
      - frame_color = 8
      - inside the frame, a red L-slot sits at top-left, a blue 1-cell slot sits at bottom-right
      - outside the frame there is one red 2x-scaled L piece and one blue 2x2 square piece
    original_logic_on_old_task:
      - detect holes as background cells
      - extract each hole mask
      - try all dihedral transforms of every piece
      - pair by downsampled shape equality
    new_logic_on_variant:
      - detect the frame color as the dominant color of the largest multicolor component
      - crop the inner layout by taking the bbox of non-frame-color cells
      - read each monochrome colored region as one slot; color gives identity
      - find the outside piece of the same color
      - paste that piece into the enlarged frame at the colored slot position
    difference:
      - the old solver fails because there are no background holes inside the frame
      - the new solver treats color as the correspondence key and shape only as a
        sanity check that the piece still fits its own slot

complexity_calibration:
  - Every slot color is unique. This is required because colors now carry identity.
  - Unit slots may now have different colors even though they share the same shape.
    This is desirable: when there are two unit slots, shape alone is insufficient and
    the color-position rule becomes visually necessary.
  - Non-unit shapes are allowed to repeat; the generator no longer enforces D4-shape
    uniqueness because the variant should not rely on shape uniqueness for pairing.
  - Color count stays bounded: bg + frame + distractor + up to 6 slot colors fits the ARC palette.

implicit_assumption_audit:
  original_main_lines_that_break:
    - `hole = sub == bg` assumed slots were background-colored. Invalid now because
      slots are colored markers, not background.
    - `layout = hole[...]` produced a boolean hole mask. Invalid now because the solver
      needs the colored layout, not just hole occupancy.
    - slot extraction with `components_mask(layout)` assumed a boolean mask. Invalid now
      because slots must be extracted as monochrome connected components by color.
    - candidate matching iterated over all piece dihedral variants and matched by shape.
      That is no longer the correspondence mechanism and would reintroduce the old rule.
  new_main_guards:
    - frame color is still taken as the dominant color of the largest multicolor component;
      the generator guarantees frame-color cells outnumber any marker color.
    - inner layout is recovered as the bbox of `sub != frame_color`; the generator keeps
      marked slots touching all four sides of the inner layout so this crop is exact.
    - each slot color must appear exactly once inside the frame and exactly once among
      outside pieces; the generator constructs that bijection directly.
    - piece shape is checked only against its own same-color slot after pairing, which
      prevents accidental malformed examples without using cross-piece shape search.

placement_legality_and_cleanliness:
  - Frame and pieces are placed with at least one background-cell gap.
  - Distractors are isolated singletons in a color not used by frame or slots.
  - The frame interior contains only frame-color walls and solid colored slot markers,
    so there is no ambiguous overlap layer.
  - Output remains visually clean: frame-color background, then colored enlarged pieces
    exactly occupying the enlarged slot areas.

variant_observability:
  - A human can see that the hole regions already carry colors inside the frame.
  - The outside sprites use those same colors, so color-position correspondence is visible.
  - When two unit slots appear, they share the same geometry but differ in color, making
    the color mapping indispensable from only a few examples.
"""

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox, bbox_coords, crop_bbox, find_components, get_bg, paste, random_free_location


def _slot_from_coords(coords: np.ndarray, color: int) -> dict:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    h = int(rr1 - rr0 + 1)
    w = int(cc1 - cc0 + 1)
    sm = np.zeros((h, w), dtype=bool)
    sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return {
        "color": int(color),
        "coords": coords,
        "area": int(coords.shape[0]),
        "r0": int(rr0),
        "c0": int(cc0),
        "mask": sm,
    }


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a color-marked frame layout and place each outside piece into the
    slot of the same color.

    Input structure:
    - bg = most common color.
    - Largest 4-connected multicolor component is the frame plus its colored slots.
    - Inner walls use the frame color; each slot region is a solid unique marker color.
    - Outside pieces are k×k-block-scaled sprites in those same slot colors.

    Rule:
    - Recover the inner layout from colored slot markers inside the frame.
    - Pair each outside piece to the slot of the same color.
    - Output the layout scaled by k, with each same-color piece pasted into place.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "need a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    marker_mask = sub != int(bgc)
    marker_box = bbox(marker_mask)
    assert marker_box is not None, "frame must contain colored slots"
    lr0, lc0, lr1, lc1 = map(int, marker_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slot_comps = find_components(layout, bg=bgc, connectivity=4, monochromatic=True, return_format="dict")
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
        pm = np.zeros_like(patch, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert color not in pieces_by_color, "piece colors must be unique"
        pieces_by_color[color] = {
            "color": color,
            "area": int(coords.shape[0]),
            "mask": pm,
            "patch": patch,
        }

    assert pieces_by_color, "at least one outside piece must exist"
    assert set(slots_by_color) == set(pieces_by_color), "slot colors and piece colors must match"

    pieces = list(pieces_by_color.values())
    k = _infer_k(pieces)

    placements: list[tuple[dict, dict]] = []
    for color, slot in slots_by_color.items():
        piece = pieces_by_color[color]
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must equal scaled slot area"
        um = _downsample_blocks(piece["mask"], k)
        assert um is not None and np.array_equal(um, slot["mask"]), "piece must fit its same-color slot"
        placements.append((slot, piece))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, piece in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], top, left, mask=piece["mask"], overwrite=True)

    return out


# --------------- generator ---------------

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
AREA_BUFFER_CONST = 70

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = free_cells.tolist()
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
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


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append(_slot_from_coords(coords, color=-1))
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
            slots.append(_slot_from_coords(coords, color=-1))
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)
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
        color = int(slot["color"])
        coords = slot["coords"]
        layout[coords[:, 0], coords[:, 1]] = color
    return layout


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        slot = piece["slot"]
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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
    frame_img[1:-1, 1:-1] = layout_colors
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
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
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    slot_colors = [palette.pop() for _ in range(n_slots)]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    for slot, color in zip(slots, slot_colors):
        slot["color"] = int(color)

    layout_colors = _build_layout_colors(slots, layout_h, layout_w, frame_color)
    marker_mask = layout_colors != int(frame_color)
    if not (
        marker_mask[0].any()
        and marker_mask[-1].any()
        and marker_mask[:, 0].any()
        and marker_mask[:, -1].any()
    ):
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
