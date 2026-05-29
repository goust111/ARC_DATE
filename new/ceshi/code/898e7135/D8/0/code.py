# Variant of 898e7135
# Dimension: D8 — 缩放因子 k 的推断线索 (meta)
# Change: k 从最小实心方块精灵（单元洞对应的 k×k 方块）的边长推断；每个示例保证有 1-2 个单元洞
# ... → 无单元洞线索（纯面积比推断 k）

from __future__ import annotations

# Design analysis for the D8 variant
# ==================================
# a) Concrete behavior and trace
#    - The frame remains a 1-cell-thick monochrome rectangle enclosing a layout of bg holes.
#    - Each hole component ("slot") still has one outside piece that is its k×k-block-scaled copy,
#      up to rotation / reflection.
#    - The visible change is that there are now no area-1 slots anywhere in the frame, so the input
#      no longer contains the direct k×k solid-square clue used by the original solver.
#    - Small trace:
#        original: slot areas [1, 3, 5], k=3 -> piece areas [9, 27, 45].
#                  The 9-cell 3×3 solid block directly reveals k=3.
#        variant:  slot areas [2, 3, 5], k=3 -> piece areas [18, 27, 45].
#                  No piece has side length k. Solver must test which k makes
#                  sorted(piece_areas) == sorted(slot_areas * k^2).
#    - The rest of the pipeline is unchanged: detect frame, extract slots, match pieces by shape
#      after downsampling by the inferred k, then paste them into the scaled frame.
#
# b) Complexity calibration
#    - k is still drawn from a tiny set {2, 3}, so the task remains learnable.
#    - We keep 3-6 slots and distinct non-unit D4 shapes to avoid ambiguous piece-slot matching.
#    - Removing unit slots is visually obvious: no tiny single-hole islands inside the frame and no
#      standalone k×k clue blocks outside it.
#
# c) Original solver assumptions that break under D8
#    - "pieces excluding singleton distractors" still holds because distractors remain isolated pixels.
#    - "smallest solid square piece has side k" no longer holds; with no unit slots that heuristic is
#      absent or wrong (e.g. a 2×2 slot at k=2 gives a 4×4 solid square, not side 2).
#    - "need_area = slot_area * k^2" still holds and becomes the new inference route for k.
#    - downsample-by-k logic still holds once k is inferred correctly.
#
# d) Placement legality
#    - Slots remain 8-separated in the frame layout, so hole components do not merge.
#    - Pieces are scattered with padding=1 and connectivity=8, so they do not touch one another or the frame.
#    - Distractors remain singleton pixels in a dedicated color and are also isolated.
#
# e) Visual observability
#    - Across samples a solver can still see that each outside piece is a blown-up copy of one interior
#      hole shape, but the direct "smallest square gives k" cue is gone.
#    - The only stable way to recover k in the solver is the area ratio between any piece set and slot set.
#
# f) Visual cleanliness
#    - We preserve the original sparse canvas, dedicated distractor color, and one-pixel spacing rules.
#      No new overlays or marker systems are introduced.
#
# g) Color inference
#    - Colors remain structurally simple: frame color, one distractor color, and one distinct color per piece.
#      No new color logic is added.
#
# h) Quantified anti-degeneracy target
#    - Every generated sample has zero unit slots.
#    - Every generated sample satisfies old_square_heuristic_k != true_k, where old_square_heuristic_k is
#      the original "smallest solid square side" rule (or 1 if no solid square exists).
#
# i) New solver assumptions introduced here
#    - The area multiset of pieces must equal the area multiset of slots multiplied by one global k^2.
#    - generate_input() guarantees this by constructing every piece directly from one slot at a shared k.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D8 variant:
    - There are no unit slots.
    - k is inferred from the global area relation
      piece_area = slot_area * k^2, not from a smallest k×k solid square.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame plus pieces"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose holes"
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

    assert pieces, "at least one non-frame piece must exist"
    assert len(pieces) == len(slots), "piece/slot count mismatch"

    slot_areas = sorted(int(s["area"]) for s in slots)
    piece_areas = sorted(int(p["area"]) for p in pieces)
    max_k = max(max(int(p["mask"].shape[0]), int(p["mask"].shape[1])) for p in pieces)
    k_candidates = []
    for k in range(2, max_k + 1):
        if sorted(int(a) * int(k) * int(k) for a in slot_areas) == piece_areas:
            k_candidates.append(int(k))
    assert len(k_candidates) == 1, "k must be uniquely determined by area ratios"
    k = int(k_candidates[0])

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = map(int, mask.shape)
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((hh // k, ww // k), dtype=bool)
        for r in range(hh // k):
            for c in range(ww // k):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            rp = np.rot90(patch, kk)
            rm = np.rot90(mask, kk)
            base.append((rp, rm))
            base.append((np.fliplr(rp), np.fliplr(rm)))
        uniq = []
        seen = set()
        for vp, vm in base:
            key = (vp.shape, vm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

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
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(bgc), dtype=int)
    for s, cand in placements:
        top = (int(s["r0"]) + 1) * k
        left = (int(s["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
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
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = free_cells.tolist()
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p for p in frontier
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


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_slots):
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
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
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


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
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

    for p in sorted(pieces, key=lambda x: -int(x["in_mask"].sum())):
        try:
            pr, pc = random_free_location(canvas, p["in_patch"], transparent=0, border=0, padding=1, padding_connectivity=8)
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


def _old_square_heuristic_k_from_pieces(pieces: list[dict]) -> int:
    guess = None
    for p in pieces:
        h, w = map(int, p["in_mask"].shape)
        if h == w and int(p["in_mask"].sum()) == h * w:
            guess = h if guess is None else min(int(guess), h)
    return int(guess or 1)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    piece_colors = palette[:n_slots]

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None
    if any(int(s["area"]) == 1 for s in slots):
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    if _old_square_heuristic_k_from_pieces(pieces) == int(k):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
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
