# Combo variant of 898e7135 (index 14)
# combo_spec: D4/0+D8/0
# Name: 反向缩放纯推断

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The frame still encloses several 4-connected bg holes ("slots"), and each outside
#      piece is still a monochrome kxk-block enlargement of one slot up to D4 symmetry.
#    - The combo changes two linked parts of the original task at once:
#      1. Direction reversal: output keeps the original frame scale, so the large outside
#         piece must be shrunk into the small hole instead of enlarging the frame.
#      2. Pure inference: there are no unit slots, so no kxk clue square exists. k must be
#         recovered from the shared area equation piece_area = slot_area * k^2.
#    - Small trace:
#        slots have areas [2, 3, 5], true k = 3
#        input pieces therefore have areas [18, 27, 45]
#        original task: enlarge frame by 3 and paste 18/27/45-cell pieces directly
#        combo task: keep frame unscaled, infer that only k=3 makes
#                    sorted(slot_areas * k^2) == sorted(piece_areas),
#                    then downsample each matched piece by 3 before pasting.
#    - The visual difference is strong: holes inside the frame are smaller than their
#      corresponding outside pieces, and the output frame is not enlarged.
#
# b) Complexity calibration
#    - k remains limited to {2, 3}; this keeps the inference branch compact and observable.
#    - We keep 3-6 slots and require D4-distinct non-unit slot shapes so matching remains
#      unambiguous after downsampling.
#    - No unit slots are allowed anywhere, so every example consistently demonstrates that
#      the old smallest-square heuristic is unavailable or wrong.
#
# c) Implicit assumption audit against the original solver
#    - Largest non-bg component is still the frame: generator places frame and pieces with
#      at least one bg cell of separation.
#    - Tight hole crop still assumes holes touch all four sides of the inner layout:
#      generator preserves that border-touch condition.
#    - Original k inference from the smallest solid square is invalid here and must be
#      removed. We replace it with a unique global area-ratio inference.
#    - Original output scaling is invalid here. The solver must instead downsample the
#      matched piece and paste it into an unscaled frame-sized output.
#    - Downsampling still assumes every piece decomposes cleanly into uniform kxk blocks
#      with no partial occupancy; generator constructs pieces exactly that way.
#
# d) Placement legality
#    - Slots are generated 8-separated inside the frame so connected hole components do not
#      merge.
#    - Pieces are scattered outside the frame with 8-neighborhood padding, preventing
#      piece-piece and piece-frame merges.
#    - Singleton distractors use one dedicated color and remain isolated, so solver-side
#      piece extraction can discard all 1-cell components safely.
#
# e) Visual observability
#    - Across examples the solver can see two facts at once: outside sprites are too large
#      for the holes, and no tiny unit-hole / clue-square exists.
#    - With 2-3 examples, the only consistent explanation is "infer one global shrink
#      factor from area ratios, then shrink each piece into its hole".
#
# f) Visual cleanliness
#    - No overlays or markers are added. The scene stays sparse: one frame, several large
#      separated pieces, and isolated singleton distractors.
#    - Each piece gets its own color to make the slot-piece correspondence easier to spot.
#
# g) Color inferability
#    - Colors remain structural only: one frame color, one distractor color, and one color
#      per piece. No ordering depends on numeric color ids.
#
# h) Quantified anti-degeneracy target
#    - Every generated sample has zero unit slots.
#    - Every generated sample satisfies old_square_heuristic_k != true_k.
#    - Every generated sample has output shape (layout_h + 2, layout_w + 2), while the
#      original task would output ((layout_h + 2) * k, (layout_w + 2) * k).
#
# i) New solver assumptions introduced here
#    - Piece count equals slot count.
#    - Exactly one k >= 2 satisfies the global area-multiset equation.
#    - After choosing that k, each piece has at least one dihedral orientation whose
#      downsampled mask equals exactly one slot mask.
#    - generate_input() enforces all three properties and validates main(input) == output.

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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rp = np.rot90(patch, int(kk))
        rm = np.rot90(mask, int(kk))
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))

    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for vp, vm in base:
        key = (tuple(map(int, vm.shape)), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _downsample_piece(
    patch: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None

    out_h, out_w = int(hh // k), int(ww // k)
    out_patch = np.zeros((out_h, out_w), dtype=int)
    out_mask = np.zeros((out_h, out_w), dtype=bool)

    for r in range(out_h):
        for c in range(out_w):
            r0 = int(r * k)
            r1 = int((r + 1) * k)
            c0 = int(c * k)
            c1 = int((c + 1) * k)
            block_mask = mask[r0:r1, c0:c1]
            if bool(block_mask.all()):
                vals = patch[r0:r1, c0:c1][block_mask]
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


def _infer_k_from_areas(slot_areas: list[int], piece_areas: list[int], max_k: int) -> int:
    candidates = []
    for k in range(2, int(max_k) + 1):
        scaled = sorted(int(a) * int(k) * int(k) for a in slot_areas)
        if scaled == piece_areas:
            candidates.append(int(k))
    assert len(candidates) == 1, "k must be uniquely determined by the area ratio"
    return int(candidates[0])


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(~forbidden)]
    if not free_cells:
        return None
    random.shuffle(free_cells)

    for r0, c0 in free_cells[:GROW_SEED_TRIES]:
        chosen = {(int(r0), int(c0))}
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


def _old_square_heuristic_k_from_pieces(pieces: list[dict]) -> int:
    guess = None
    for p in pieces:
        h, w = map(int, p["in_mask"].shape)
        if h == w and int(p["in_mask"].sum()) == int(h * w):
            guess = int(h) if guess is None else min(int(guess), int(h))
    return int(guess or 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink oversized outside pieces into their matching frame holes and keep the frame unscaled.

    Input structure:
    - Background is the most common color. The largest non-background component is a monochrome
      rectangular frame whose interior contains several small bg hole regions.
    - Outside the frame are monochrome piece sprites that are kxk-block enlargements of those
      hole shapes under rotation and/or reflection. No area-1 hole exists, so there is no direct
      kxk clue square.

    Rule:
    - Extract the hole components inside the frame as slots, and extract the outside non-singleton
      components as candidate pieces.
    - Infer the single global shrink factor k from the area relation
      piece_area = slot_area * k^2. Then test dihedral variants of each piece, downsample by k,
      and match the resulting small mask to one slot.
    - Return the original frame scale: a tight crop of the layout plus its 1-cell border, with
      each matched piece shrunk into the corresponding hole and the frame color as background.

    Key insight:
    - The decisive cue is not a smallest solid square. The only consistent explanation is that
      every large sprite is a uniformly block-scaled copy of one slot, all sharing the same k.
      Once k is recovered from the global area ratio, downsampling reveals the correct hole shape.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected a frame plus separate pieces"
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
    assert len(pieces) == len(slots), "piece and slot counts must agree"

    slot_areas = sorted(int(s["area"]) for s in slots)
    piece_areas = sorted(int(p["area"]) for p in pieces)
    max_k = max(max(int(p["mask"].shape[0]), int(p["mask"].shape[1])) for p in pieces)
    k = _infer_k_from_areas(slot_areas, piece_areas, max_k)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            down = _downsample_piece(vp, vm, k)
            if down is None:
                continue
            dp, dm = down
            candidates.append(
                {"pi": int(pi), "area": int(p["area"]), "dp": dp, "dm": dm}
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
        paste(out, cand["dp"], int(s["r0"]) + 1, int(s["c0"]) + 1, mask=cand["dm"], overwrite=True)
    return out


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
            slots.append(
                {"coords": coords, "area": int(size), "mask": sm, "shape_key": key}
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None
    if any(int(s["area"]) == 1 for s in slots):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = slot["mask"].copy()
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    in_mask = np.kron(out_mask, np.ones((k, k), dtype=bool))
    in_patch = np.where(in_mask, int(color), 0).astype(int)

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
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
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

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    if _old_square_heuristic_k_from_pieces(pieces) == int(k):
        return None

    slot_areas = sorted(int(s["area"]) for s in slots)
    piece_areas = sorted(int(p["in_mask"].sum()) for p in pieces)
    max_piece_side = max(max(int(p["in_mask"].shape[0]), int(p["in_mask"].shape[1])) for p in pieces)
    try:
        inferred_k = _infer_k_from_areas(slot_areas, piece_areas, max_piece_side)
    except AssertionError:
        return None
    if int(inferred_k) != int(k):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, frame_color)
    canvas = _build_input_canvas(
        pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color
    )
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None
    if produced.shape == ((layout_h + 2) * k, (layout_w + 2) * k):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(
        f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts"
    )
