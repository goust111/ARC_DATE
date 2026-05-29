from __future__ import annotations

# Variant of 898e7135
# Dimension: D6 — 干扰项的类型与作用 (meta)
# Change: 8-13 个单像素干扰项，使用专用颜色（与框架色和所有精灵色不同），散布在画布上；输出中完全移除... → 干扰项作为额外不匹配的伪精灵
#
# Design analysis for this D6 variant
# -----------------------------------
# spec_variant:
#   input:
#     background: 0 remains the most common color.
#     frame:
#       unchanged from the source task: a monochrome rectangular border encloses
#       an inner layout whose 4-connected bg regions are the slots.
#     real_pieces:
#       unchanged: one block-scaled piece per slot, outside the frame, separated
#       from other objects by at least one bg cell, with D4 rotation/flip allowed.
#     pseudo_pieces:
#       change core:
#         - replace singleton noise with 1-2 extra connected multi-pixel objects.
#         - each pseudo-piece uses its own color, distinct from the frame and all
#           real piece colors.
#         - at least one pseudo-piece is a plausible k-by-k block sprite whose
#           downsampled shape is D4-distinct from every slot.
#         - when k=3, also add a 2x2 solid square pseudo-piece. It is still a
#           multi-pixel object, but it is not a valid 3x3-block sprite. This makes
#           the old "smallest solid square => k" heuristic unsafe.
#   output:
#     unchanged: the scaled frame filled only with the matched real pieces.
#     pseudo-pieces disappear entirely.
#
#   behavior_trace:
#     example:
#       - slots inside the frame are [L-shape area 3, domino area 2, unit area 1].
#       - true scale is k=3, so the real unit piece is a 3x3 square.
#       - outside the frame, add one fake zig-zag piece built from 3x3 blocks and
#         one fake 2x2 solid square.
#     original_solver_path:
#       - collect every non-singleton component as a piece.
#       - infer k from the smallest solid square across all pieces.
#       - the 2x2 fake square forces k=2, so many true pieces fail block
#         downsampling and matching breaks.
#     new_solver_path:
#       - still collect all non-singleton components.
#       - try each square-derived k candidate and ask a stricter question:
#         "is there a one-to-one assignment from some subset of pieces to all slots?"
#       - k=2 fails to cover all slots; k=3 succeeds.
#       - the assignment uses only the real pieces; unmatched components are
#         explicitly treated as pseudo-pieces and ignored in the output.
#
#   complexity_calibration:
#     - keep pseudo-piece count small (1-2) so the scene is readable.
#     - keep real slot count and geometry the same as the source task.
#     - at least one fake remains block-styled, so humans cannot dismiss every
#       distractor just by checking whether it looks "off-grid".
#     - colors stay within the normal ARC palette budget: frame + real pieces +
#       up to 2 fake colors = at most 9 non-bg colors.
#
#   implicit_assumption_audit_of_original_main:
#     - "all non-singleton foreground components are true pieces":
#       no longer valid. The variant adds extra multi-pixel components.
#     - "smallest solid square piece reveals k":
#       no longer valid once a fake square can exist.
#     - "greedy first matching candidate is enough":
#       only safe when the candidate pool contains exactly the real pieces.
#       With extras, we should solve a subset matching problem instead.
#     - "unused pieces never exist":
#       no longer valid. The new solver must tolerate leftovers after matching.
#
#   placement_legality:
#     - generator still uses random_free_location with 8-neighbor padding=1, so
#       frame, real pieces, and pseudo-pieces stay disconnected.
#     - pseudo-pieces are kept small so the frame remains the largest component.
#     - the layout still touches all four inner borders, preserving the original
#       solver's hole-layout cropping convention.
#
#   visual_observability:
#     - across examples, solvers and humans see the same frame-to-piece relation
#       as before, but now there are visibly extra colored shapes with no hole to
#       fill. Because they are few and spatially separate, the "ignore the extras"
#       rule is inferable from 2-4 examples.
#
#   cleanliness:
#     - no overlaps, no touching components, no layered markings.
#     - pseudo-pieces live in the same outside-piece region, so the scene stays
#       coherent instead of introducing a separate annotation system.
#
#   post-change solver assumptions and generator guarantees:
#     - main() assumes at least one true unit slot exists, so a real kxk square
#       is present among the true pieces. generate_input() preserves that.
#     - main() assumes exactly one k candidate yields a full slot assignment.
#       generate_input() preserves that by making fake block shapes D4-distinct
#       from all slot shapes, and by using the special 2x2 fake square only when
#       k=3, where it cannot also support a full k=2 assignment.

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


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if k <= 0 or hh % k != 0 or ww % k != 0:
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rp = np.rot90(patch, kk)
        rm = np.rot90(mask, kk)
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for p, m in base:
        key = (m.tobytes(), tuple(map(int, m.shape)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _match_slots_with_k(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = [[] for _ in slots]
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            area = int(vm.sum())
            for si, slot in enumerate(slots):
                if area != int(slot["area"]) * int(k) * int(k):
                    continue
                if um.shape != slot["mask"].shape:
                    continue
                if np.array_equal(um, slot["mask"]):
                    candidates_by_slot[si].append(
                        {
                            "pi": int(pi),
                            "vp": vp,
                            "vm": vm,
                            "area": area,
                        }
                    )

    if any(not cands for cands in candidates_by_slot):
        return None

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(candidates_by_slot[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
    )
    chosen: list[dict | None] = [None] * len(slots)
    used: set[int] = set()

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        si = order[idx]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            chosen[si] = cand
            if dfs(idx + 1):
                return True
            chosen[si] = None
            used.remove(pi)
        return False

    if not dfs(0):
        return None
    return [(slots[i], chosen[i]) for i in range(len(slots)) if chosen[i] is not None]


def _infer_k_and_match(slots: list[dict], pieces: list[dict]) -> tuple[int, list[tuple[dict, dict]]]:
    square_sizes = sorted(
        {
            int(piece["mask"].shape[0])
            for piece in pieces
            if piece["mask"].shape[0] == piece["mask"].shape[1]
            and int(piece["mask"].sum()) == int(piece["mask"].shape[0] * piece["mask"].shape[1])
            and int(piece["mask"].shape[0]) > 1
        }
    )
    assert square_sizes, "need at least one solid square to infer k candidates"

    best: tuple[int, list[tuple[dict, dict]]] | None = None
    for k in square_sizes:
        placements = _match_slots_with_k(slots, pieces, k)
        if placements is None:
            continue
        if best is None:
            best = (int(k), placements)
            continue
        raise AssertionError("ambiguous k: more than one scale yields a full assignment")

    assert best is not None, "no scale yields a complete slot-to-piece assignment"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole-layout and fill each slot with its matching piece.

    Variant-specific rule:
    - Extra outside multi-pixel pseudo-pieces may exist.
    - Only the subset of pieces that can jointly match all hole shapes under one
      block scale k are used; leftover components are ignored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
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

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
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
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame multi-pixel piece must exist"

    k, placements = _infer_k_and_match(slots, pieces)

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
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
AREA_BUFFER_CONST = 55

DECOY_COUNT_FOR_K2 = [1, 2]
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
    H, W = forbidden.shape
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
                if 0 <= p[0] < H and 0 <= p[1] < W and not forbidden[p[0], p[1]] and p not in chosen
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


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


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
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
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


def _scaled_piece_from_mask(base_mask: np.ndarray, color: int, k: int, apply_dihedral: bool) -> tuple[np.ndarray, np.ndarray]:
    full_mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
    patch = np.where(full_mask, int(color), 0).astype(int)
    if apply_dihedral:
        rot = random.randint(0, 3)
        full_mask = np.rot90(full_mask, rot)
        patch = np.rot90(patch, rot)
        if random.random() < 0.5:
            full_mask = np.fliplr(full_mask)
            patch = np.fliplr(patch)
    return patch, full_mask


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    patch, mask = _scaled_piece_from_mask(slot["mask"], color, k, apply_dihedral and int(slot["area"]) > 1)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": np.where(np.kron(slot["mask"], np.ones((k, k), dtype=bool)), int(color), 0).astype(int),
        "out_mask": np.kron(slot["mask"], np.ones((k, k), dtype=bool)),
        "in_patch": patch,
        "in_mask": mask,
    }


def _random_decoy_base_mask(forbidden_shape_keys: set[bytes]) -> np.ndarray | None:
    for _ in range(80):
        size = random.randint(2, 6)
        coords = _grow_slot(np.zeros((5, 5), dtype=bool), size)
        if coords is None:
            continue
        mask = _tight_mask_from_coords(coords)
        if mask.shape[0] == mask.shape[1] and int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        key = _canonical_shape_key(mask)
        if key in forbidden_shape_keys:
            continue
        return mask
    return None


def _make_block_decoy(color: int, k: int, forbidden_shape_keys: set[bytes]) -> dict | None:
    base_mask = _random_decoy_base_mask(forbidden_shape_keys)
    if base_mask is None:
        return None
    patch, mask = _scaled_piece_from_mask(base_mask, color, k, apply_dihedral=True)
    return {
        "color": int(color),
        "in_patch": patch,
        "in_mask": mask,
        "kind": "block_decoy",
        "base_mask": base_mask,
    }


def _make_small_square_decoy(color: int) -> dict:
    patch = np.full((2, 2), int(color), dtype=int)
    mask = np.ones((2, 2), dtype=bool)
    return {
        "color": int(color),
        "in_patch": patch,
        "in_mask": mask,
        "kind": "small_square_decoy",
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    decoys: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    outside_objects = pieces + decoys
    total_piece_area = sum(int(obj["in_mask"].sum()) for obj in outside_objects)
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

    for obj in sorted(outside_objects, key=lambda item: -int(item["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                obj["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, obj["in_patch"], pr, pc, mask=obj["in_mask"], overwrite=True)

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

    n_decoys = 2 if k == 3 else random.choice(DECOY_COUNT_FOR_K2)
    palette = list(range(1, 10))
    random.shuffle(palette)
    needed_piece_colors = (n_slots - n_units) + 1
    needed_colors = 1 + needed_piece_colors + n_decoys
    if len(palette) < needed_colors:
        return None

    frame_color = palette.pop()
    piece_palette = [palette.pop() for _ in range(needed_piece_colors)]
    decoy_colors = [palette.pop() for _ in range(n_decoys)]
    non_unit_colors = piece_palette[: n_slots - n_units]
    unit_color = piece_palette[-1]

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

    forbidden_shape_keys = {_canonical_shape_key(slot["mask"]) for slot in non_unit_slots}
    decoys: list[dict] = []
    color_idx = 0
    if k == 3:
        decoys.append(_make_small_square_decoy(decoy_colors[color_idx]))
        color_idx += 1

    while len(decoys) < n_decoys:
        decoy = _make_block_decoy(decoy_colors[color_idx], k, forbidden_shape_keys)
        if decoy is None:
            return None
        forbidden_shape_keys.add(_canonical_shape_key(decoy["base_mask"]))
        decoys.append(decoy)
        color_idx += 1

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, decoys, layout_mask, layout_h, layout_w, frame_color)
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
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
