# Combo variant of 898e7135 (index 10)
# combo_spec: D2/0+D4/4
# Name: 多色输入单色输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# ------------------------------------
# a) Concretization:
#    Original pieces were monochrome k-by-k-block sprites and their colors survived into the
#    output. Here each piece is visibly multicolor (2-3 colors). Matching still depends only on
#    the occupied block mask after D4 transforms, but once a piece is placed into its slot, every
#    occupied pixel is recolored to the frame-wall color. Example: an L-slot of 3 blocks may be
#    seen in input as red/yellow/teal blocks; after fitting into the enlarged frame, those pixels
#    become the wall color, so the output loses the internal palette.
# b) Complexity calibration:
#    If every piece shared the same palette, the input would look noisy but less informative. The
#    generator therefore gives each piece a unique primary color plus shared accents, so pieces are
#    easy to distinguish by eye while still requiring shape-based matching. We keep k in {2,3} and
#    2-3 colors per piece to make the color-drop behavior obvious without exploding the palette.
# c) Broken assumptions from the original solver:
#    - "Piece patch colors matter in output" is false now; solver must separate shape matching from
#      render color.
#    - "Monochrome piece" is false; component extraction must stay color-agnostic.
#    - "Smallest solid square piece is monochrome" is false; only its occupancy mask stays solid.
#    The generator guarantees at least one unit slot so mask-based k inference still works.
# d) Placement legality:
#    The frame is placed first; pieces are scattered outside it with 8-neighbor padding via
#    random_free_location, so they never touch the frame or each other. Distractors remain isolated.
# e) Visual observability:
#    Across examples, inputs show clearly multicolor pieces while outputs collapse to a single frame
#    color rectangle of the appropriate scaled size. The asymmetry is visible because all piece
#    colors disappear, but k and the frame-derived dimensions still control output size.
# f) Cleanliness:
#    Colors appear only inside pieces; the frame and distractors use separate colors, and objects
#    remain spatially separated. That keeps the input readable despite the richer piece textures.
# g) Inferability:
#    Color changes are structured, not arbitrary: all piece colors are discarded in the same way and
#    replaced by the surrounding wall color. This repeated "many colors in, one wall color out"
#    pattern is learnable from a few examples.


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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, int(kk))
        m = np.rot90(mask, int(kk))
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for p, m in base:
        key = (m.shape, m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame layout and absorb each matching multicolor piece into the wall color.

    Input structure:
    - The most common color is background. The largest non-background component is a monochrome frame
      enclosing background holes, while scattered outside it are 2-3-color pieces built from k×k blocks.
    - At least one piece is a solid k×k square in occupancy, so k can still be inferred from shape rather
      than from color.

    Rule:
    - Read the hole components inside the frame as slots.
    - Match each outside piece to a slot by comparing only its block occupancy under rotations/reflections.
    - Produce the scaled frame and fill each matched slot, but recolor the filled pixels to the frame color
      instead of preserving the piece's internal palette.

    Key insight:
    The multicolor patterns are useful only to identify and separate input pieces; the invariant that survives
    into the output is the k-scaled occupancy mask, and even that is rendered in a single wall color.
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
    assert hole_box is not None, "frame must enclose at least one background hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
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
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "expected at least one non-frame piece"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and k >= 2, "could not infer scale from a solid square piece"

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, int(k))
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vm": vm, "vp": vp})

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
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece by occupancy"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        recolored = np.full(cand["vm"].shape, int(frame_color), dtype=int)
        paste(out, recolored, int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --------------- generator ---------------

"""Generator for the fused ARC-AGI-2 variant of 898e7135.

Output-first strategy:
  1. Build a slot layout inside a monochrome frame.
  2. Convert each slot to a k-scaled multicolor piece (2-3 colors), then apply a dihedral transform.
  3. Scatter those pieces outside the frame and add singleton distractors.
  4. The expected output is the scaled frame after every slot has been filled and recolored to the
     frame-wall color, so the result is monochrome.

The generator validates main(input) == expected_output before returning.
"""

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_RANGE = (1, 2)
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

GENERATE_MAX_RETRIES = 400


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
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
            return np.array(sorted(chosen))
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
            if coords is None:
                continue
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


def _color_order_for_piece(piece_idx: int, primaries: list[int], shared_accents: list[int]) -> list[int]:
    base = [int(primaries[piece_idx]), int(shared_accents[0])]
    if len(shared_accents) > 1 and random.random() < 0.5:
        base.append(int(shared_accents[1]))
    elif len(primaries) > 1 and random.random() < 0.5:
        base.append(int(primaries[(piece_idx + 1) % len(primaries)]))
    out: list[int] = []
    for c in base:
        if c not in out:
            out.append(int(c))
    if len(out) < 2 and len(primaries) > 1:
        extra = int(primaries[(piece_idx + 1) % len(primaries)])
        if extra not in out:
            out.append(extra)
    return out[:3]


def _paint_unit_patch(k: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros((k, k), dtype=int)
    for r in range(k):
        for c in range(k):
            idx = (2 * r + c) % len(colors)
            patch[r, c] = int(colors[idx])
    if len(set(map(int, patch.flatten()))) < len(colors):
        vals = np.array(colors, dtype=int)
        patch = vals[(np.arange(k * k).reshape(k, k)) % len(colors)]
    mask = np.ones((k, k), dtype=bool)
    return patch, mask


def _paint_block_patch(slot_mask: np.ndarray, k: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    out_mask = np.kron(slot_mask, np.ones((k, k), dtype=bool))
    patch = np.zeros(out_mask.shape, dtype=int)
    coords = np.argwhere(slot_mask)
    used = set()
    for idx, (r, c) in enumerate(coords):
        color = int(colors[(2 * int(r) + int(c) + idx) % len(colors)])
        used.add(color)
        patch[int(r) * k : (int(r) + 1) * k, int(c) * k : (int(c) + 1) * k] = color
    if len(used) < min(2, len(colors)) and coords.shape[0] >= 2:
        for idx, (r, c) in enumerate(coords):
            color = int(colors[idx % len(colors)])
            patch[int(r) * k : (int(r) + 1) * k, int(c) * k : (int(c) + 1) * k] = color
    return patch, out_mask


def _apply_random_dihedral(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = random.randint(0, 3)
    out_patch = np.rot90(patch, rot)
    out_mask = np.rot90(mask, rot)
    if random.random() < 0.5:
        out_patch = np.fliplr(out_patch)
        out_mask = np.fliplr(out_mask)
    return out_patch, out_mask


def _make_piece(slot: dict, k: int, colors: list[int]) -> dict:
    if int(slot["area"]) == 1:
        base_patch, base_mask = _paint_unit_patch(int(k), colors)
    else:
        base_patch, base_mask = _paint_block_patch(slot["mask"], int(k), colors)
    in_patch, in_mask = _apply_random_dihedral(base_patch, base_mask)
    return {
        "slot": slot,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "n_colors": len(set(map(int, in_patch[in_mask]))),
    }


def _build_expected_output(layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    return np.full(((int(layout_h) + 2) * int(k), (int(layout_w) + 2) * int(k)), int(frame_color), dtype=int)


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h) + 2, int(layout_w) + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
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
    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
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
        paste(canvas, p["in_patch"], int(pr), int(pc), mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr, cc = random.randint(0, canvas_h - 1), random.randint(0, canvas_w - 1)
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
    n_units = random.randint(N_UNITS_RANGE[0], min(N_UNITS_RANGE[1], n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    if len(palette) < n_slots + 1:
        return None
    primaries = [int(c) for c in palette[:n_slots]]
    shared_accents = [int(palette[n_slots])]
    if len(palette) > n_slots + 1 and random.random() < 0.6:
        shared_accents.append(int(palette[n_slots + 1]))

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())

    pieces = []
    for piece_idx, slot in enumerate(slots):
        colors = _color_order_for_piece(piece_idx, primaries, shared_accents)
        piece = _make_piece(slot, int(k), colors)
        if int(piece["n_colors"]) < 2:
            return None
        pieces.append(piece)

    max_piece_area = max(int(p["in_mask"].sum()) for p in pieces)
    if frame_area <= max_piece_area:
        return None

    expected_out = _build_expected_output(layout_h, layout_w, int(k), int(frame_color))
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, int(k), int(frame_color), int(distractor_color))
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
