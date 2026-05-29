# Combo variant of 898e7135 (index 27)
# combo_spec: D2/0+D15/0
# Name: 多色同形消歧

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
spec_variant:
  unchanged_core:
    - There is still one large frame enclosing a layout of background holes.
    - Each outside sprite is still a k*k-scaled copy of one hole shape, up to a D4 transform.
    - The output is still the cropped enlarged frame with each hole filled in place.
  changed_signals:
    - Non-unit hole shapes are now allowed to repeat, so shape alone no longer identifies a destination.
    - Every sprite now uses 2-3 colors on connected block-level regions instead of being monochrome.
    - Every hole carries three touching wall markers. Each marker copies the color of a fixed anchor block
      from the correctly oriented sprite for that hole.
  example_trace:
    - Original: two identical L holes could not coexist, and a red L piece was identified only by shape.
    - Here: two identical L holes do coexist. Outside there are two L pieces using the same three colors,
      but the colors are arranged differently on the L. Around the first hole the top/right/bottom markers
      read blue-yellow-red; around the second they read yellow-blue-red. Shape finds both L pieces for both
      holes; the three-color boundary signature picks the correct piece and orientation for each.

b) Complexity calibration
  - The ambiguity is real rather than cosmetic: every generated sample contains at least one repeated
    non-unit shape, and its repeated-shape pieces share the same color set.
  - The extra clue is controlled: only three marker cells per hole are added, using a small shared palette.
  - Multicolor regions stay connected, so the sprites are legible rather than noisy.

c) Implicit-assumption audit of the original main()
  - Greedy first-fit by shape was safe only because non-unit hole shapes were unique. That is invalid now.
  - Inferring k from the smallest solid-square piece is invalid because no piece is required to be monochrome
    or a solid square anymore. The new solver tries the allowed k values and keeps the one that yields a full
    shape+signature matching.
  - Downsampling can no longer inspect only occupancy. It must also verify that every occupied k*k block is
    monochrome and record the block color grid.
  - D4 deduplication by mask bytes alone can collapse 1xN and Nx1 cases. The new code keys by shape+bytes.
  - The frame is no longer monochrome, but it remains the largest multicolor foreground component and its
    dominant color is still the true frame color.

d) Placement legality
  - Holes remain 8-separated, so pieces, markers, and wall cells do not merge accidentally.
  - Marker cells are deterministic wall neighbors of their own hole and are checked for uniqueness.
  - Outside pieces are still placed with at least one background cell of clearance via random_free_location.

e) Visual observability
  - A solver looking at 2-3 samples can see repeated hole shapes, multicolor outside pieces, and matching
    three-color clues touching the holes.
  - The tie-break does not depend on numeric color ordering or invisible metadata: the visible marker colors
    are copied directly from visible sprite blocks.

f) Visual cleanliness
  - Only three wall pixels per hole are recolored, so the frame remains readable.
  - Markers stay on walls; sprites stay outside the frame; the output contains only the enlarged filled frame.

g) Color-change inferability
  - Colors are not remapped arbitrarily. Marker colors are sampled from fixed anchor blocks of the target
    sprite in its output orientation.
  - Repeated-shape pieces deliberately reuse the same color set but permute where those colors appear, making
    the multicolor layout itself part of the rule.

h) New solver assumptions introduced here
  - Every hole has exactly three designated marker cells derived from the same anchor rule.
  - Each designated marker color equals the color of the corresponding anchor block in the correct filled-in
    sprite.
  - At least one repeated-shape group exists, and in that group the shared shape still becomes uniquely
    solvable once the three-color signature is applied.
  - The generator enforces all of these conditions explicitly.
"""

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_SIZE_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 30
DUPLICATE_PAIR_RETRIES = 40
COPY_PLACEMENT_TRIES = 80
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

COLORING_TRIES = 80
PARTITION_TRIES = 50

GENERATE_MAX_RETRIES = 450


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Canonical D4 key for a tight bool mask."""
    best = None
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for var in (rot, np.fliplr(rot)):
            key = (tuple(map(int, var.shape)), var.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return (tight_mask, r0, c0) for coords."""
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, int(r0), int(c0)


def _anchor_specs_from_coords(coords: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Return three deterministic anchor specs for a slot.
    Each entry is ((cell_r, cell_c), (wall_r, wall_c)) in layout coordinates.
    """
    pts = [tuple(map(int, rc)) for rc in coords]
    min_r = min(r for r, _c in pts)
    max_r = max(r for r, _c in pts)
    max_c = max(c for _r, c in pts)

    top_cell = min((p for p in pts if p[0] == min_r), key=lambda p: (p[1], p[0]))
    right_cell = min((p for p in pts if p[1] == max_c), key=lambda p: (p[0], p[1]))
    bottom_cell = max((p for p in pts if p[0] == max_r), key=lambda p: (p[1], p[0]))

    anchors = [
        (top_cell, (top_cell[0] - 1, top_cell[1])),
        (right_cell, (right_cell[0], right_cell[1] + 1)),
        (bottom_cell, (bottom_cell[0] + 1, bottom_cell[1])),
    ]
    return anchors


def _slot_signature_from_grid(slot: dict, color_grid: np.ndarray) -> tuple[int, ...]:
    """Read the ordered anchor signature from a block-level color grid."""
    return tuple(int(color_grid[int(r), int(c)]) for r, c in slot["anchor_locals"])


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected slot of given size from random free cells."""
    h, w = forbidden.shape
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
    mask, _r0, _c0 = _tight_mask_from_coords(coords)
    return mask


def _place_copy_of_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Place an exact translated copy of `mask` into free layout cells."""
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    rel = np.argwhere(mask)
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for top, left in positions[:COPY_PLACEMENT_TRIES]:
        coords = rel + np.array([top, left])
        if np.all(~forbidden[coords[:, 0], coords[:, 1]]):
            return coords.astype(int)
    return None


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    """
    Generate only non-unit slots with:
    - exactly one duplicate non-unit shape pair
    - all remaining non-unit shapes unique under D4
    """
    if n_slots < 3:
        return None

    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    dup_slots: list[dict] | None = None
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
        dup_key = _canonical_shape_key(mask1)
        dup_slots = [
            {"coords": coords1, "area": int(size), "mask": mask1, "shape_key": dup_key, "dup_group": 0},
            {"coords": coords2, "area": int(size), "mask": mask1.copy(), "shape_key": dup_key, "dup_group": 0},
        ]
        forbidden = temp_forbidden
        _forbid_coords(forbidden, coords2)
        break
    if dup_slots is None:
        return None
    slots.extend(dup_slots)

    used_shape_keys = {dup_slots[0]["shape_key"]}
    for _ in range(n_slots - 2):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            mask = _mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key, "dup_group": None})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _partition_mask(mask: np.ndarray, n_parts: int) -> np.ndarray | None:
    """Partition a connected mask into `n_parts` connected labeled regions."""
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(cells) < n_parts:
        return None

    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cell_set = set(cells)
    for r, c in cells:
        adj = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set:
                adj.append((nr, nc))
        neigh[(r, c)] = adj

    for _ in range(PARTITION_TRIES):
        seeds = random.sample(cells, n_parts)
        label_of = {seed: idx for idx, seed in enumerate(seeds)}
        assigned = set(seeds)
        while len(assigned) < len(cells):
            frontier = []
            for cell in cells:
                if cell in assigned:
                    continue
                adj_labels = sorted({label_of[nbr] for nbr in neigh[cell] if nbr in label_of})
                if adj_labels:
                    frontier.append((cell, adj_labels))
            if not frontier:
                break
            cell, adj_labels = random.choice(frontier)
            label_of[cell] = random.choice(adj_labels)
            assigned.add(cell)
        if len(assigned) != len(cells):
            continue
        out = np.full(mask.shape, -1, dtype=int)
        for (r, c), label in label_of.items():
            out[int(r), int(c)] = int(label)
        return out
    return None


def _color_grid_from_labels(mask: np.ndarray, labels: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    for idx, color in enumerate(colors):
        out[labels == idx] = int(color)
    out[~mask] = 0
    return out


def _make_piece(slot: dict, out_colors: np.ndarray, k: int, apply_dihedral: bool = True) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.kron(out_colors, np.ones((k, k), dtype=int))

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask.copy(), out_patch.copy()

    return {
        "slot": slot,
        "out_colors": out_colors,
        "color_set": tuple(sorted(map(int, np.unique(out_colors[out_colors > 0])))),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "signature": _slot_signature_from_grid(slot, out_colors),
    }


def _make_random_piece(slot: dict, k: int, color_pool: list[int]) -> dict | None:
    max_colors = min(3, int(slot["area"]))
    for _ in range(COLORING_TRIES):
        n_colors = random.randint(2, max_colors)
        labels = _partition_mask(slot["mask"], n_colors)
        if labels is None:
            continue
        colors = tuple(random.sample(color_pool, n_colors))
        out_colors = _color_grid_from_labels(slot["mask"], labels, colors)
        if len(set(int(v) for v in _slot_signature_from_grid(slot, out_colors))) < 1:
            continue
        return _make_piece(slot, out_colors, k, apply_dihedral=True)
    return None


def _make_duplicate_pair(slot_a: dict, slot_b: dict, k: int, color_pool: list[int]) -> tuple[dict, dict] | None:
    max_colors = min(3, int(slot_a["area"]))
    for _ in range(COLORING_TRIES):
        n_colors = random.randint(2, max_colors)
        labels = _partition_mask(slot_a["mask"], n_colors)
        if labels is None:
            continue
        colors = tuple(random.sample(color_pool, n_colors))
        out_a = _color_grid_from_labels(slot_a["mask"], labels, colors)
        sig_a = _slot_signature_from_grid(slot_a, out_a)

        perms = list(itertools.permutations(colors))
        random.shuffle(perms)
        for perm in perms:
            if tuple(perm) == colors:
                continue
            out_b = _color_grid_from_labels(slot_b["mask"], labels, tuple(int(v) for v in perm))
            sig_b = _slot_signature_from_grid(slot_b, out_b)
            if sig_b == sig_a:
                continue
            piece_a = _make_piece(slot_a, out_a, k, apply_dihedral=True)
            piece_b = _make_piece(slot_b, out_b, k, apply_dihedral=True)
            if piece_a["color_set"] != piece_b["color_set"]:
                continue
            return piece_a, piece_b
    return None


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for piece in pieces:
        slot = piece["slot"]
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_frame_image(slots: list[dict], pieces: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray | None:
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    frame_img = np.full((layout_h + 2, layout_w + 2), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)

    used_marker_cells: set[tuple[int, int]] = set()
    piece_by_slot = {id(piece["slot"]): piece for piece in pieces}
    for slot in slots:
        piece = piece_by_slot[id(slot)]
        for color, (mr, mc) in zip(piece["signature"], slot["marker_cells"]):
            cell = (int(mr), int(mc))
            if cell in used_marker_cells:
                return None
            used_marker_cells.add(cell)
            if not (0 <= mr < layout_h + 2 and 0 <= mc < layout_w + 2):
                return None
            if frame_img[int(mr), int(mc)] == 0:
                return None
            frame_img[int(mr), int(mc)] = int(color)
    return frame_img


def _build_input_canvas(pieces: list[dict], frame_img: np.ndarray, distractor_color: int) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
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
        r_lo = max(0, rr - 1)
        r_hi = min(canvas_h, rr + 2)
        c_lo = max(0, cc - 1)
        c_hi = min(canvas_w, cc + 2)
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


def _extract_scene(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "grid must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        mask, rr0, cc0 = _tight_mask_from_coords(coords)
        anchors = _anchor_specs_from_coords(coords)
        anchor_locals = tuple((int(cr - rr0), int(cc - cc0)) for (cr, cc), _wall in anchors)
        marker_cells = tuple((int(wr + 1), int(wc + 1)) for _cell, (wr, wc) in anchors)
        marker_sig = tuple(int(sub[int(lr0 + wr), int(lc0 + wc)]) for _cell, (wr, wc) in anchors)
        assert all(int(v) != bg and int(v) != frame_color for v in marker_sig), "markers must be non-bg, non-frame"
        slots.append(
            {
                "area": int(mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": mask,
                "coords": coords.astype(int),
                "shape_key": _canonical_shape_key(mask),
                "anchor_locals": anchor_locals,
                "marker_cells": marker_cells,
                "marker_sig": marker_sig,
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
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
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    return bg, frame_color, sub, layout, slots, pieces


def _downsample_colored_blocks(patch: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out_h = int(hh // k)
    out_w = int(ww // k)
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    out_colors = np.zeros((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            rb0 = int(r * k)
            rb1 = int((r + 1) * k)
            cb0 = int(c * k)
            cb1 = int((c + 1) * k)
            mblock = mask[rb0:rb1, cb0:cb1]
            pblock = patch[rb0:rb1, cb0:cb1]
            if bool(mblock.all()):
                vals = set(int(v) for v in pblock[mblock])
                if len(vals) != 1:
                    return None
                color = next(iter(vals))
                if color == 0:
                    return None
                out_mask[r, c] = True
                out_colors[r, c] = int(color)
            elif bool(mblock.any()):
                return None
            elif np.any(pblock != 0):
                return None
    return out_mask, out_colors


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq = []
    seen = set()
    for p, m in base:
        key = (tuple(map(int, m.shape)), m.tobytes(), p.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _solve_for_k(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]]:
    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            down = _downsample_colored_blocks(vp, vm, k)
            if down is None:
                continue
            um, uc = down
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": um,
                    "uc": uc,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        exact = []
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if not np.array_equal(cand["um"], slot["mask"]):
                continue
            if _slot_signature_from_grid(slot, cand["uc"]) != slot["marker_sig"]:
                continue
            exact.append(cand)

        piece_ids = {int(cand["pi"]) for cand in exact}
        assert piece_ids, "no piece matches slot shape+signature"
        assert len(piece_ids) == 1, "multiple pieces share the same slot signature"
        chosen_piece = next(iter(piece_ids))
        found = next(cand for cand in exact if int(cand["pi"]) == chosen_piece)
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    assert len(used_piece) == len(slots), "each slot must consume one distinct piece"
    return placements


def _solve_scene(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray, int, list[tuple[dict, dict]]]:
    _bg, frame_color, _sub, layout, slots, pieces = _extract_scene(grid)

    solutions = []
    for k in K_CHOICES:
        try:
            placements = _solve_for_k(slots, pieces, int(k))
        except Exception:
            continue
        solutions.append((int(k), placements))
    assert solutions, "no valid scale factor found"
    assert len(solutions) == 1, "scale factor must be uniquely identifiable"
    k, placements = solutions[0]
    return frame_color, slots, layout, k, placements


def analyze_instance(grid: np.ndarray) -> dict:
    """
    Return metrics used to verify that generation really exercises the combo behavior.
    """
    frame_color, slots, _layout, k, placements = _solve_scene(grid)
    shape_groups: dict[tuple[tuple[int, int], bytes], list[tuple[dict, dict]]] = {}
    for slot, cand in placements:
        shape_groups.setdefault(slot["shape_key"], []).append((slot, cand))

    duplicate_groups = [group for group in shape_groups.values() if len(group) > 1]
    repeated_same_colorset = 0
    repeated_distinct_signatures = 0
    for group in duplicate_groups:
        color_sets = []
        signatures = []
        for slot, cand in group:
            uc = cand["uc"]
            color_set = tuple(sorted(int(v) for v in np.unique(uc[uc > 0])))
            color_sets.append(color_set)
            signatures.append(_slot_signature_from_grid(slot, uc))
        if len(set(color_sets)) == 1:
            repeated_same_colorset += 1
        if len(set(signatures)) > 1:
            repeated_distinct_signatures += 1

    return {
        "frame_color": int(frame_color),
        "k": int(k),
        "n_slots": int(len(slots)),
        "duplicate_shape_groups": int(len(duplicate_groups)),
        "repeated_same_colorset_groups": int(repeated_same_colorset),
        "repeated_distinct_signature_groups": int(repeated_distinct_signatures),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the frame layout and fill each hole with its matching multicolor sprite.

    Input structure:
    - The largest foreground component is a mostly monochrome frame that also contains colored wall markers.
    - Outside the frame are separated sprites made of k*k monochrome blocks, but each sprite now uses 2-3 colors.
    - Multiple holes may share the same shape, so the frame exposes three touching marker cells around every hole.

    Rule:
    - Extract the hole components inside the frame and treat them as block-level slots.
    - For each candidate k in {2, 3}, downsample each sprite under all D4 variants, keeping both shape and block colors.
    - Match a sprite to a hole only when both the block mask and the ordered three-color marker signature agree, then
      paste the matched oriented sprite into the enlarged frame and return the cropped result.

    Key insight:
    - Geometry still identifies which variants can fit a hole, but repeated hole shapes create multiple geometric fits.
      The decisive clue is the ordered boundary color signature copied from fixed anchor blocks of the correct sprite.
    """
    frame_color, _slots, layout, k, placements = _solve_scene(grid)

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    piece_pool_size = random.randint(4, min(5, len(palette)))
    piece_color_pool = random.sample(palette, piece_pool_size)

    raw_slots = _gen_layout(layout_h, layout_w, n_slots)
    if raw_slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    slots = []
    for raw in raw_slots:
        coords = raw["coords"].astype(int)
        layout_mask[coords[:, 0], coords[:, 1]] = True
        mask, r0, c0 = _tight_mask_from_coords(coords)
        anchors = _anchor_specs_from_coords(coords)
        anchor_locals = tuple((int(cr - r0), int(cc - c0)) for (cr, cc), _wall in anchors)
        marker_cells = tuple((int(wr + 1), int(wc + 1)) for _cell, (wr, wc) in anchors)
        slots.append(
            {
                **raw,
                "mask": mask,
                "r0": int(r0),
                "c0": int(c0),
                "shape_key": _canonical_shape_key(mask),
                "anchor_locals": anchor_locals,
                "marker_cells": marker_cells,
            }
        )

    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    dup_slots = [slot for slot in slots if slot["dup_group"] == 0]
    if len(dup_slots) != 2:
        return None

    dup_pair = _make_duplicate_pair(dup_slots[0], dup_slots[1], k, piece_color_pool)
    if dup_pair is None:
        return None
    pieces = [dup_pair[0], dup_pair[1]]

    for slot in slots:
        if slot["dup_group"] == 0:
            continue
        piece = _make_random_piece(slot, k, piece_color_pool)
        if piece is None:
            return None
        pieces.append(piece)

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    frame_img = _build_frame_image(slots, pieces, layout_h, layout_w, frame_color)
    if frame_img is None:
        return None
    canvas = _build_input_canvas(pieces, frame_img, distractor_color)
    if canvas is None:
        return None

    dup_shapes = Counter(slot["shape_key"] for slot in slots)
    if max(dup_shapes.values(), default=0) < 2:
        return None
    if dup_pair[0]["color_set"] != dup_pair[1]["color_set"]:
        return None
    if dup_pair[0]["signature"] == dup_pair[1]["signature"]:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    metrics = analyze_instance(canvas)
    if metrics["duplicate_shape_groups"] < 1:
        return None
    if metrics["repeated_same_colorset_groups"] < 1:
        return None
    if metrics["repeated_distinct_signature_groups"] < 1:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
