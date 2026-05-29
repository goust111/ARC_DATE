# Variant of 898e7135
# Dimension: D6 — 干扰项的类型与作用 (meta)
# Change: 8-13 个单像素干扰项，使用专用颜色（与框架色和所有精灵色不同），散布在画布上；输出中完全移除... → 干扰项为小型多像素色块

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# variant_spec:
#   core_change:
#     original: distractors were 8-13 isolated single pixels, easy for the solver
#       to discard with an area>1 filter and easy for humans to spot as noise.
#     variant: distractors are 8-13 separate 2-4 pixel monochrome blobs.
#       They are still outside the frame and removed in the output, but they now
#       look more like "small objects" than like salt-and-pepper noise.
#   concrete_visualization:
#     frame: unchanged; largest foreground component, monochrome rectangular border
#       with bg holes inside.
#     true_pieces: unchanged; every real piece is made of exact kxk solid blocks
#       (k in {2,3}), optionally rotated/flipped.
#     distractors: each distractor is a tiny 4-connected polyomino of area 2-4,
#       placed outside the frame with at least one bg cell around it. Distractors
#       reuse colors from real pieces, so color alone no longer identifies noise.
#   behavior_trace:
#     original_solver_path:
#       - collect all non-frame components
#       - silently discard components of area <= 1
#       - remaining components are treated as candidate pieces
#     variant_solver_path:
#       - collect all non-frame components, including distractor blobs
#       - keep only components that can be interpreted as exact kxk-block pieces
#         under some dihedral orientation
#       - match those usable candidates to hole slots
#     difference:
#       the classification boundary moves from "singleton vs non-singleton" to
#       "block-scaled piece vs non-piece blob".
#   complexity_calibration:
#     - Distractors reuse piece colors so the D6 change is visible; otherwise a
#       dedicated distractor color would make the new blobs trivial to ignore.
#     - For k=2, a 2x2 distractor would be visually identical to a true unit piece.
#       That would create genuine ambiguity rather than useful challenge, so the
#       generator excludes that one shape when k=2.
#     - k remains restricted to {2,3}; slot layout and piece construction stay
#       unchanged, keeping the task recognizable as 898e7135.
#   implicit_assumption_review:
#     - Old code assumed "area<=1 implies distractor". That is false now.
#     - Old k inference from the smallest solid square remains safe because the
#       generator preserves 1-2 true unit slots and avoids k=2 distractor squares.
#       Even so, the solver now validates k by testing which of {2,3} can explain
#       all slots through block-compatible candidates.
#     - Matching assumes every slot has exactly one real piece and distractors are
#       not themselves valid block pieces; the generator enforces this.
#   placement_legality:
#     - Frame, real pieces, and distractor blobs are all placed with >=1 bg-cell
#       spacing using random_free_location(..., padding=1, padding_connectivity=8).
#       This prevents accidental component merging, even when colors repeat.
#   visual_observability:
#     - In input grids, humans now see many tiny colored blobs mixed among the
#       true sprites. The output still contains only the inflated frame+pieces.
#       Across examples, the stable cue is that real sprites are chunky kxk-block
#       objects, while distractors are undersized irregular blobs.
#   cleanliness:
#     - Distractors remain outside the frame and never overlap or touch other
#       objects, so the scene is busier than the original but still readable.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D6 variant difference:
    - Outside the frame there may also be small 2-4 pixel distractor blobs.
    - A usable piece is therefore identified structurally: some dihedral variant
      must be an exact tiling of kxk solid blocks that downsamples to a slot mask.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground objects"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    raw_parts = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        raw_parts.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert raw_parts, "input must contain at least one non-frame component"

    def downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
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

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        base: list[tuple[np.ndarray, np.ndarray]] = []
        for kk in range(4):
            p = np.rot90(patch, kk)
            m = np.rot90(mask, kk)
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = (tuple(map(int, m.shape)), m.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m))
        return uniq

    def build_candidates_for_k(k: int) -> list[dict]:
        candidates = []
        for pi, part in enumerate(raw_parts):
            for vp, vm in dihedral_variants(part["patch"], part["mask"]):
                um = downsample_blocks(vm, k)
                if um is None:
                    continue
                candidates.append(
                    {
                        "pi": int(pi),
                        "area": int(part["area"]),
                        "um": um,
                        "vp": vp,
                        "vm": vm,
                    }
                )
        return candidates

    def solve_for_k(k: int) -> list[tuple[dict, dict]] | None:
        candidates = build_candidates_for_k(k)
        if not candidates:
            return None

        slot_options: list[list[int]] = []
        for slot in slots:
            need_area = int(slot["area"]) * k * k
            opts = []
            for ci, cand in enumerate(candidates):
                if int(cand["area"]) != need_area:
                    continue
                if cand["um"].shape != slot["mask"].shape:
                    continue
                if np.array_equal(cand["um"], slot["mask"]):
                    opts.append(int(ci))
            if not opts:
                return None
            slot_options.append(opts)

        order = sorted(
            range(len(slots)),
            key=lambda i: (len(slot_options[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
        )
        used_piece: set[int] = set()
        assign: dict[int, int] = {}

        def backtrack(idx: int) -> bool:
            if idx == len(order):
                return True
            si = int(order[idx])
            for ci in slot_options[si]:
                pi = int(candidates[ci]["pi"])
                if pi in used_piece:
                    continue
                used_piece.add(pi)
                assign[si] = int(ci)
                if backtrack(idx + 1):
                    return True
                used_piece.remove(pi)
                del assign[si]
            return False

        if not backtrack(0):
            return None
        return [(slots[i], candidates[assign[i]]) for i in range(len(slots))]

    solutions = []
    for k_try in (2, 3):
        placements = solve_for_k(k_try)
        if placements is not None:
            solutions.append((int(k_try), placements))

    assert len(solutions) == 1, "input must imply a unique block scale"
    k, placements = solutions[0]

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), frame_color, dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --------------- generator ---------------

"""Generator for ARC-AGI-2 task 898e7135, D6 distractor-blob variant.

Output-first strategy:
  1. Build a hole/wall layout of 4-connected slots.
  2. Convert each slot into a monochrome kxk-block piece with a random dihedral.
  3. Assemble the expected output from the inflated frame layout.
  4. Build the input by scattering real pieces outside the frame, then add 8-13
     tiny 2-4 pixel distractor blobs that do not touch anything.

The generator validates main(input) == expected_output before returning.
"""


# --- tunables ----------------------------------------------------------------

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

DISTRACTOR_BLOB_RANGE = (8, 13)
DISTRACTOR_AREA_CHOICES = [2, 3, 4]
MIN_DISTRACTOR_BLOBS = 8

GENERATE_MAX_RETRIES = 300


# --- layout generation -------------------------------------------------------

def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a bool mask (tight-cropped)."""
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected slot of given size from a random free seed."""
    h, w = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
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
            chosen.add((int(nr), int(nc)))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen))
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a bool mask by 1 cell (8-connectivity)."""
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
    """Mark coords and their 8-neighbors as forbidden (in-place)."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """Generate a valid layout; unit slots appear first in the returned list."""
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


# --- piece / distractor construction ----------------------------------------

def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Build output patch and scattered input patch for one slot."""
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    if apply_dihedral:
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
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _distractor_shape_pool(k: int) -> list[np.ndarray]:
    """Small 4-connected blobs. Exclude 2x2 when k=2 to avoid true-piece ambiguity."""
    shapes = [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1], [1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    ]
    if int(k) != 2:
        shapes.append(np.array([[1, 1], [1, 1]], dtype=bool))
    return [s for s in shapes if int(s.sum()) in DISTRACTOR_AREA_CHOICES]


def _make_distractor_blob(color: int, k: int) -> dict:
    mask = random.choice(_distractor_shape_pool(k))
    patch = np.where(mask, int(color), 0).astype(int)
    return {"color": int(color), "mask": mask, "patch": patch, "area": int(mask.sum())}


# --- output assembly ---------------------------------------------------------

def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    """Assemble the expected output grid."""
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


# --- input canvas assembly ---------------------------------------------------

def _build_input_canvas(
    pieces: list[dict], layout_mask: np.ndarray, layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray | None:
    """Build the input grid: frame + scattered pieces + small distractor blobs."""
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    worst_distractor_area = DISTRACTOR_BLOB_RANGE[1] * max(DISTRACTOR_AREA_CHOICES)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST + worst_distractor_area

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

    for piece in sorted(pieces, key=lambda piece: -int(piece["in_mask"].sum())):
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

    piece_colors = sorted({int(piece["color"]) for piece in pieces})
    n_blobs = random.randint(*DISTRACTOR_BLOB_RANGE)
    placed = 0
    for _ in range(n_blobs):
        blob = _make_distractor_blob(random.choice(piece_colors), k)
        try:
            br, bc = random_free_location(
                canvas,
                blob["patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, blob["patch"], br, bc, mask=blob["mask"], overwrite=True)
        placed += 1

    if placed < MIN_DISTRACTOR_BLOBS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


# --- main generator attempt --------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = [int(c) for c in palette[: n_slots - n_units]]
    unit_color = int(palette[n_slots - n_units])

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

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color)
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
