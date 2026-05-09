# Variant of 898e7135
# Dimension: D4 — 变换：缩放因子 k 的推断与应用 (transform)
# Change: scale_to_fit — 缩放因子 k（2 或 3）从最小实心方块精灵的边长推断；框架放大 k 倍后精灵恰好填入对应... → 缩小而非放大（反向缩放）

from __future__ import annotations

# Design analysis for the D4 inversion variant
# ============================================
# a) Concrete behavior
#    - Input still contains one large monochrome frame with bg-colored holes plus several
#      scattered monochrome pieces outside the frame.
#    - The outside pieces remain k-block-scaled sprites (k in {2, 3}), optionally rotated
#      or mirrored.
#    - The change is inside/output semantics: frame holes now show the unscaled slot shapes,
#      while the outside pieces are larger by factor k. The output is the original frame
#      size (layout + 1-cell border), not an enlarged frame. Each large piece is first
#      downsampled by k and then pasted into its matching hole.
#    - Example: if a slot is an L-shape of 3 cells and k=2, the input piece is a 6-cell
#      2x2-block L. Original task would output a 2x-enlarged frame and paste the large L;
#      this variant outputs the normal-sized frame and fills the 3 small hole cells.
# b) Complexity calibration
#    - k stays binary-valued {2, 3}; this keeps the scale inference visible but controlled.
#    - 3-6 slots with 1-2 unit slots are preserved so the smallest solid square still
#      reliably reveals k, and multiple matches support the rule visually.
#    - Non-unit slot D4 shapes remain unique to avoid ambiguous piece-slot matching.
# c) Implicit assumptions audited against the original solver
#    - Largest multicolor foreground component is still the frame: generator keeps pieces
#      separated from the frame and from each other.
#    - Tight hole crop still depends on holes touching all four sides of the inner layout:
#      generator preserves that constraint.
#    - k inference from the smallest solid square still assumes at least one unit slot:
#      generator enforces 1-2 unit slots, producing kxk square pieces.
#    - Candidate matching by downsampling D4 variants still assumes every valid piece is
#      partitionable into uniform kxk blocks with no partial occupancy: generator builds
#      pieces exactly that way.
#    - The original output step pasted full-size variants into an enlarged canvas; that
#      assumption no longer holds. Here we paste the downsampled variant into an unscaled
#      canvas, otherwise the solver would silently revert to the original task.
# d) Placement legality
#    - Frame placement stays arbitrary on a 15-30 canvas.
#    - Pieces are placed largest-first with >=1 bg-cell 8-neighborhood separation from the
#      frame and from each other, so connected-component extraction remains stable.
#    - Distractors remain isolated singletons in a dedicated color and are filtered out by
#      the solver's non-singleton piece extraction.
# e) Visual observability
#    - Across examples, the outside pieces are visibly larger than the frame holes, and the
#      output frame is not enlarged. This makes the inversion from "expand frame" to
#      "shrink pieces" directly observable from 2-3 examples.
# f) Visual cleanliness
#    - No overlays are introduced. The output is simply the cropped frame with holes filled.
#      Piece colors stay distinct and the frame color stays uniform.
# g) Color inferability
#    - Colors preserve the original structure: frame color is the background of the output,
#      each non-unit piece has its own color, unit pieces may share one color, and
#      distractors use one separate color. No hidden color ordering is used.

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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, int(kk))
        m = np.rot90(mask, int(kk))
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))

    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for p, m in base:
        key = m.tobytes() + bytes([m.shape[0], m.shape[1]])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _downsample_piece(
    patch: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None

    out_h, out_w = int(hh // k), int(ww // k)
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    out_patch = np.zeros((out_h, out_w), dtype=int)

    for r in range(out_h):
        for c in range(out_w):
            block_mask = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
            if bool(block_mask.all()):
                vals = patch[
                    int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)
                ][block_mask]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match large k-block pieces to small frame holes, then shrink the pieces into the holes.

    k is still inferred from the smallest solid square piece, but unlike the original task
    the frame is not enlarged. Each matched piece is downsampled by k and pasted into an
    output canvas of the frame's original size.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
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

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
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
    assert pieces, "expected at least one non-frame piece"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and int(k) >= 2, "expected unit square pieces revealing shrink factor"
    k = int(k)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            down = _downsample_piece(vp, vm, k)
            if down is None:
                continue
            dp, dm = down
            candidates.append(
                {"pi": int(pi), "area": int(p["area"]), "dp": dp, "dm": dm, "vp": vp, "vm": vm}
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["dm"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["dm"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match one shrunken piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out = np.full((int(layout.shape[0]) + 2, int(layout.shape[1]) + 2), int(frame_color), dtype=int)
    for s, cand in placements:
        top = int(s["r0"]) + 1
        left = int(s["c0"]) + 1
        paste(out, cand["dp"], top, left, mask=cand["dm"], overwrite=True)
    return out


# --------------- generator ---------------

"""Generator for the D4 reverse-scaling variant of ARC-AGI-2 task 898e7135.

Output-first strategy:
  1. Build a small hole/wall layout of 4-connected slots (includes unit slots).
  2. The expected output is the unscaled frame with each slot filled by a 1x-scale piece.
  3. The input pieces are k-block-scaled versions of those slot shapes, optionally under
     a dihedral transform.
  4. Scatter the large pieces outside the frame and add singleton distractors.

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
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(free_cells)

    for r0, c0 in free_cells[:GROW_SEED_TRIES]:
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
    out_mask = slot["mask"] if slot["area"] > 1 else np.ones((1, 1), dtype=bool)
    out_patch = np.where(out_mask, color, 0).astype(int)

    in_mask = np.kron(out_mask, np.ones((k, k), dtype=bool))
    in_patch = np.where(in_mask, color, 0).astype(int)

    if apply_dihedral and slot["area"] > 1:
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
    for p in pieces:
        coords = p["slot"]["coords"]
        top = int(coords[:, 0].min()) + 1
        left = int(coords[:, 1].min()) + 1
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
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
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
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
    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, frame_color)
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
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
