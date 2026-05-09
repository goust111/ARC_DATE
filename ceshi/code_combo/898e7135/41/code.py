# Combo variant of 898e7135 (index 41)
# combo_spec: D1/2+D9/3+D10/0
# Name: 纯几何极限匹配

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Preserved scaffold: one monochrome frame encloses background holes; outside the frame sit
#      disconnected k-by-k block-scaled piece sprites; output is the enlarged cropped frame with
#      the pieces pasted back into the slots.
#    - D1/2 changes matching: a piece can no longer be mentally rotated or mirrored. Its current
#      orientation must already equal the slot orientation after block downsampling.
#    - D9/3 removes area as a cue: every non-unit slot now has the same area, so
#      `slot_area * k * k == piece_area` cannot distinguish them.
#    - D10/0 removes color as a cue: every piece, including unit squares, shares one color.
#    - Small trace:
#      original task: a size-4 L hole and a size-4 T hole could already be filtered by area, and
#      differently colored pieces made identity even easier; rotation/flip could still rescue a
#      mismatched orientation.
#      combo task: several non-unit holes all have area 4 and every outside piece is the same
#      color. Suppose two holes are two orientations of the same L tetromino family. A human can
#      no longer use area or color; the only usable clue is that one outside L already points the
#      right way for the upper hole while the other points the right way for the lower hole.
#
# b) Complexity calibration
#    - If all exact shapes were also unique up to D4, then forbidding rotation/flip would be
#      visually weak: the task would still look like plain same-area shape matching.
#    - To keep the D1 signal visible, the generator always includes one non-unit family with two
#      different exact orientations in the same sample. They share area, color, and D4 family.
#    - To avoid full ambiguity, all remaining non-unit slots come from different D4 families and
#      every exact oriented mask is unique. Unit slots may repeat because they are identical and
#      harmless for the output.
#
# c) Implicit-assumption review of `main()`
#    - `frame_comp = largest component`: still valid because the generator rejects layouts whose
#      frame area is not larger than every outside piece.
#    - tight hole crop inside the frame: still valid because the generator crops the final layout
#      to the union bbox of all holes, so holes touch all four sides of that crop.
#    - singleton skip for distractors: preserved; distractors remain isolated 1-pixel components.
#    - `k` inference from the smallest solid square piece: preserved by always generating at least
#      one unit slot, hence at least one k-by-k solid square piece.
#    - old area filtering is no longer valid because all non-unit holes share one area; the solver
#      must compare exact block masks instead.
#    - old dihedral candidate expansion is intentionally removed; the solver now assumes every
#      non-unit exact slot mask appears at most once among the pieces. The generator guarantees
#      that by enforcing exact-mask uniqueness.
#    - subtle fusion risk: with all pieces the same color, a relaxed rotate/flip solver may often
#      produce the same final output even when the correspondence is orientation-ambiguous. This
#      variant therefore measures structural ambiguity rather than relying only on output mismatch:
#      each generated sample contains at least one D4 family collision whose exact orientations are
#      different, so exact orientation is still the sole discriminator of piece identity.
#
# d) Placement legality
#    - Slots are placed on a boolean stage with 8-neighbor dilation, so they are 4-connected
#      internally and pairwise 8-separated.
#    - Pieces are pasted with `random_free_location(..., padding=1, padding_connectivity=8)`, so
#      same-colored sprites never merge with each other or with the frame.
#    - Distractors are added only on isolated empty cells and remain separate singleton components.
#
# e) Visual observability
#    - A solver looking at 2-4 examples can see: all complex holes have the same size, all outside
#      pieces have the same color, and at least one pair of pieces shares a family but not an
#      orientation. That makes area and color useless by inspection.
#    - Because the repeated family is orientation-sensitive and the shapes are already presented in
#      their final orientation, the natural human inference is “match the exact outline as seen”.
#
# f) Visual cleanliness
#    - Only three semantic colors remain: frame, shared piece color, distractor.
#    - Objects stay disconnected and the hole layout is sparse enough to keep silhouettes legible.
#
# g) Change inferability
#    - No hidden ordering by color id or area value is used.
#    - The only repeated support signal is structural: same-area, same-color non-unit pieces with a
#      forced repeated D4 family. That repeated evidence makes the combo readable across samples.

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


# --- shared helpers ----------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
MIN_NON_UNIT_SLOTS = 2
NON_UNIT_AREA_CHOICES = [3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
LAYOUT_PLACE_RETRIES = 50

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = np.array(mask, dtype=bool, copy=False)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool, copy=True)


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best = None
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            tight = _tight_mask(np.array(var, dtype=bool, copy=False))
            sig = _mask_signature(tight)
            if best is None or sig < best:
                best = sig
    assert best is not None
    return best


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        for var in (rot, np.fliplr(rot)):
            tight = _tight_mask(np.array(var, dtype=bool, copy=False))
            sig = _mask_signature(tight)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(tight.copy())
    return out


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = True
    return out


def _enumerate_free_polyominoes(max_area: int) -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {1: [np.ones((1, 1), dtype=bool)]}
    for area in range(2, max_area + 1):
        next_shapes: dict[tuple[int, int, bytes], np.ndarray] = {}
        for shape in by_area[area - 1]:
            coords = {tuple(map(int, rc)) for rc in np.argwhere(shape)}
            for r, c in list(coords):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in coords:
                        continue
                    grown = set(coords)
                    grown.add((int(nr), int(nc)))
                    mask = _coords_to_mask(grown)
                    key = _canonical_shape_key(mask)
                    if key not in next_shapes:
                        next_shapes[key] = _tight_mask(mask)
        by_area[area] = list(next_shapes.values())
    return by_area


SHAPES_BY_AREA = _enumerate_free_polyominoes(max(NON_UNIT_AREA_CHOICES))
FAMILIES_BY_AREA: dict[int, dict[tuple[int, int, bytes], list[np.ndarray]]] = {}
for _area, _masks in SHAPES_BY_AREA.items():
    fams: dict[tuple[int, int, bytes], list[np.ndarray]] = {}
    for _mask in _masks:
        key = _canonical_shape_key(_mask)
        fams[key] = _dihedral_masks(_mask)
    FAMILIES_BY_AREA[_area] = fams


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


def _place_exact_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    mh, mw = map(int, mask.shape)
    if mh > h or mw > w:
        return None
    rel = np.argwhere(mask)
    locs = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
    random.shuffle(locs)
    for r0, c0 in locs:
        coords = rel + np.array([[int(r0), int(c0)]], dtype=int)
        if np.any(forbidden[coords[:, 0], coords[:, 1]]):
            continue
        return coords
    return None


def _slot_from_coords(coords: np.ndarray) -> dict:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
    sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return {
        "coords": np.array(coords, dtype=int, copy=True),
        "area": int(coords.shape[0]),
        "r0": int(rr0),
        "c0": int(cc0),
        "mask": sm,
        "exact_key": _mask_signature(sm),
        "canon_key": _canonical_shape_key(sm),
    }


def _crop_layout(slots: list[dict]) -> tuple[list[dict] | None, np.ndarray | None]:
    all_coords = np.concatenate([slot["coords"] for slot in slots], axis=0)
    r0, c0, r1, c1 = bbox_coords(all_coords)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    layout_h = int(r1 - r0 + 1)
    layout_w = int(c1 - c0 + 1)
    if not (LAYOUT_H_RANGE[0] <= layout_h <= LAYOUT_H_RANGE[1]):
        return None, None
    if not (LAYOUT_W_RANGE[0] <= layout_w <= LAYOUT_W_RANGE[1]):
        return None, None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    cropped: list[dict] = []
    offset = np.array([[int(r0), int(c0)]], dtype=int)
    for slot in slots:
        coords = slot["coords"] - offset
        cropped_slot = _slot_from_coords(coords)
        layout_mask[coords[:, 0], coords[:, 1]] = True
        cropped.append(cropped_slot)

    density = float(layout_mask.sum()) / float(layout_h * layout_w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None, None
    return cropped, layout_mask


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


def _extract_layout_and_pieces(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
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

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(tuple(map(int, patch.shape)), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"patch": patch, "mask": pm, "area": int(pm.sum())})
    assert pieces, "at least one non-frame piece must exist"

    return layout, frame_color, pieces


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the framed hole layout and refill it using only exact-orientation geometric matches.

    Input structure:
    - The most common color is background, and the largest foreground component is a monochrome
      frame around background holes.
    - Every outside piece is a monochrome k-by-k block expansion of one slot, and all pieces share
      the same color.
    - All non-unit holes have the same area, so neither area nor color identifies a match.

    Rule:
    - Tight-crop the frame's hole layout and split it into connected slots.
    - Downsample each outside piece back to block level and compare its exact mask to slot masks
      without any rotation or flip.
    - Output the k-scaled cropped frame, using the frame color as background and pasting the
      matched piece into each slot position.

    Key insight:
    - This combo deletes the easy cues. Equal non-unit areas remove area filtering, shared piece
      color removes color tracking, and forbidden dihedral transforms mean the only invariant left
      is the exact oriented block silhouette already visible in the input.
    """
    layout, frame_color, pieces = _extract_layout_and_pieces(grid)
    k = _infer_k(pieces)

    slots = []
    for coords in components_mask(layout, connectivity=4):
        slot = _slot_from_coords(coords)
        slots.append(slot)
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    candidates = []
    for pi, piece in enumerate(pieces):
        um = _downsample_blocks(piece["mask"], k)
        if um is None:
            continue
        candidates.append(
            {
                "pi": int(pi),
                "vp": piece["patch"],
                "vm": piece["mask"],
                "exact_key": _mask_signature(um),
            }
        )
    assert candidates, "at least one usable piece candidate is required"

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if cand["exact_key"] != slot["exact_key"]:
                continue
            found = cand
            break
        assert found is not None, "every slot must match a piece by exact oriented shape"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


# --- generator ---------------------------------------------------------------

def _choose_exact_shapes(n_non_unit: int) -> tuple[int, list[np.ndarray], tuple[int, int, bytes]]:
    candidate_areas = []
    for area in NON_UNIT_AREA_CHOICES:
        families = FAMILIES_BY_AREA[area]
        pairable = [key for key, variants in families.items() if len(variants) >= 2]
        if pairable and len(families) >= n_non_unit - 1:
            candidate_areas.append(area)
    assert candidate_areas

    common_area = random.choice(candidate_areas)
    families = FAMILIES_BY_AREA[common_area]
    pair_family = random.choice([key for key, variants in families.items() if len(variants) >= 2])

    chosen = random.sample(families[pair_family], 2)
    other_family_keys = [key for key in families if key != pair_family]
    random.shuffle(other_family_keys)
    for fam_key in other_family_keys[: max(0, n_non_unit - 2)]:
        chosen.append(random.choice(families[fam_key]))
    assert len(chosen) == n_non_unit

    random.shuffle(chosen)
    return int(common_area), [mask.copy() for mask in chosen], pair_family


def _gen_layout(
    stage_h: int,
    stage_w: int,
    exact_masks: list[np.ndarray],
    n_units: int,
) -> tuple[list[dict] | None, np.ndarray | None]:
    shape_order = sorted(exact_masks, key=lambda m: (-int(m.shape[0] * m.shape[1]), -int(m.sum())))
    for _ in range(LAYOUT_PLACE_RETRIES):
        forbidden = np.zeros((stage_h, stage_w), dtype=bool)
        slots: list[dict] = []
        ok = True

        for mask in shape_order:
            coords = _place_exact_mask(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            slots.append(slot)
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        unit_mask = np.ones((1, 1), dtype=bool)
        for _unit_idx in range(n_units):
            coords = _place_exact_mask(forbidden, unit_mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            slots.append(slot)
            _forbid_coords(forbidden, coords)
        if not ok:
            continue

        cropped_slots, layout_mask = _crop_layout(slots)
        if cropped_slots is None or layout_mask is None:
            continue

        non_unit = [slot for slot in cropped_slots if int(slot["area"]) > 1]
        if len(non_unit) != len(shape_order):
            continue
        if len({slot["exact_key"] for slot in non_unit}) != len(non_unit):
            continue
        if len({slot["area"] for slot in non_unit}) != 1:
            continue
        return cropped_slots, layout_mask

    return None, None


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_expected_output(
    pieces: list[dict],
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
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
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
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


def _orientation_metric_from_slots(slots: list[dict]) -> int:
    counts = Counter(slot["canon_key"] for slot in slots if int(slot["area"]) > 1)
    return sum(int(v) for v in counts.values() if int(v) >= 2)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    n_slots = random.randint(*N_SLOTS_RANGE)
    max_units = min(N_UNITS_MAX, n_slots - MIN_NON_UNIT_SLOTS)
    if max_units < 1:
        return None
    n_units = random.randint(1, max_units)
    n_non_unit = int(n_slots - n_units)

    common_area, exact_masks, pair_family = _choose_exact_shapes(n_non_unit)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_color = palette.pop()

    stage_h = random.randint(*LAYOUT_H_RANGE)
    stage_w = random.randint(*LAYOUT_W_RANGE)
    slots, layout_mask = _gen_layout(stage_h, stage_w, exact_masks, n_units)
    if slots is None or layout_mask is None:
        return None

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    if len(non_unit_slots) != n_non_unit or len(unit_slots) != n_units:
        return None
    if {int(slot["area"]) for slot in non_unit_slots} != {int(common_area)}:
        return None
    if len({slot["exact_key"] for slot in non_unit_slots}) != len(non_unit_slots):
        return None
    if _orientation_metric_from_slots(non_unit_slots) < 2:
        return None
    if pair_family not in {slot["canon_key"] for slot in non_unit_slots}:
        return None

    layout_h, layout_w = map(int, layout_mask.shape)
    hole_cells = int(layout_mask.sum())
    frame_area = int((layout_h + 2) * (layout_w + 2) - hole_cells)
    max_piece_area = int(common_area * k * k)
    if frame_area <= max_piece_area:
        return None

    pieces = [_make_piece(slot, piece_color, k) for slot in non_unit_slots]
    pieces += [_make_piece(slot, piece_color, k) for slot in unit_slots]

    expected = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
