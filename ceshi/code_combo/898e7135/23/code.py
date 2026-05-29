# Combo variant of 898e7135 (index 23)
# combo_spec: D7/0+D9/2
# Name: 大洞拼图组合
#
# Design analysis for combo D7/0 + D9/2
# =======================================
# a) Concrete fused behavior
#    - Source task: each hole region inside the frame has exactly one matching outside piece; the solver
#      downscales each piece by k and checks shape equality under D4.
#    - Fused task: at least one large hole of area 8-12 no longer has a whole-piece counterpart. Instead,
#      the hole is partitioned into 2-3 connected subregions, each emitted as its own outside piece. A
#      human sees a big irregular hole in the frame and several smaller colored blocky sprites outside; in
#      the output those colors reappear together inside the same large hole.
#    - Small trace: if a 9-cell hole shaped like a 4x3 box missing 3 cells used to produce one 9*k*k piece,
#      now it may produce three pieces of areas 4, 3, 2. The old solver would fail because no single piece
#      equals the hole; the new solver must try translations of several oriented pieces whose union exactly
#      covers that hole.
# b) Complexity calibration
#    - To keep the many:1 change visible, every generated puzzle contains exactly one composite large hole.
#      It is never replaced by a single whole-shape piece, so the instance cannot collapse back to the seed
#      behavior.
#    - To avoid uncontrolled ambiguity, only one hole is composite, its parts all have area >= 2, and all
#      non-unit piece shapes are globally unique up to D4. Generator-side exact-cover validation rejects any
#      puzzle with more than one valid piece-to-hole assignment.
#    - The large-hole change remains visually obvious because the composite hole has area 8-12 and is forced
#      to be irregular inside its bbox; the output therefore shows a conspicuous multi-color patchwork inside
#      one noticeably larger cavity.
# c) Implicit assumptions in main()
#    - Largest multicolor foreground component is still the frame. Generator keeps frame area larger than any
#      single piece area and preserves >=1 background-cell separation between objects.
#    - Tight hole bbox still defines the layout. Generator first builds slots in a work grid and then crops
#      by the union of hole cells, matching that assumption exactly.
#    - Slot extraction still relies on 4-connected holes that are 8-separated. Generator continues to mark
#      each hole and its 8-neighborhood as forbidden during placement.
#    - k inference still relies on unit slots producing solid kxk squares. Generator always includes 1-2
#      unit holes, so the smallest solid square piece remains available.
#    - The old one-piece-per-slot matching assumption is intentionally broken. New code replaces it with an
#      exact-cover search over oriented piece placements inside each slot.
# d) Placement legality
#    - Slot generation forbids overlap and diagonal touching inside the layout.
#    - Piece scattering still uses random_free_location with padding=1 and 8-connectivity, so pieces stay
#      separated from the frame and from each other.
#    - Composite parts are connected subsets of one large slot and are pasted back only within that slot's
#      hole cells; exact-cover search rejects overlaps and out-of-slot placements.
# e) Visual observability
#    - Across 2-4 examples, humans can see that one large output hole is filled by several differently
#      colored outside sprites rather than by one monochrome piece. The same block scale k applies to all
#      pieces, preserving the original visual cue while adding the new composition cue.
# f) Visual cleanliness
#    - The frame stays monochrome, distractors stay singleton and isolated, and composite behavior is shown
#      only by piece subdivision rather than extra markers. No region mixes guide symbols with puzzle pieces.
# g) New solver assumptions after the change
#    - Every non-singleton piece must be a clean kxk block-upscaled sprite under some D4 orientation.
#      Generator builds pieces that satisfy this exactly.
#    - The global piece set must admit a unique exact cover of the slot set. Generator checks that at the
#      block level before accepting an example.

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


K_CHOICES = [2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE_K2 = (3, 5)
N_SLOTS_RANGE_K3 = (3, 4)
N_UNITS_MAX = 2

SMALL_NON_UNIT_CHOICES = [2, 3, 4, 5, 6, 7]
BIG_NON_UNIT_CHOICES = [8, 9, 10, 11, 12]
TOTAL_SLOT_AREA_BUDGET = {2: 28, 3: 18}

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 40
NON_UNIT_SLOT_RETRIES = 120
GROW_SEED_TRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    sm = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sm, (r0, c0)


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(bytes([rot.shape[0], rot.shape[1]]) + rot.tobytes())
        flip = np.fliplr(rot)
        variants.append(bytes([flip.shape[0], flip.shape[1]]) + flip.tobytes())
    return min(variants)


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    base = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        base.append(rot)
        base.append(np.fliplr(rot))
    uniq = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for vm in base:
        key = (tuple(map(int, vm.shape)), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(vm)
    return uniq


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rot_patch = np.rot90(patch, kk)
        rot_mask = np.rot90(mask, kk)
        base.append((rot_patch, rot_mask))
        base.append((np.fliplr(rot_patch), np.fliplr(rot_mask)))
    uniq = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for vp, vm in base:
        key = (tuple(map(int, vm.shape)), vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


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


def _extract_slots(layout: np.ndarray) -> list[dict]:
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "coords": coords.copy(),
                "shape_key": _canonical_shape_key(sm),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    return slots


def _extract_pieces(g: np.ndarray, comps: list[dict], frame_comp: dict) -> list[dict]:
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
    return pieces


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == h * w:
            k = h if k is None else min(int(k), h)
    return int(k or 1)


def _piece_candidates_from_observed(pieces: list[dict], k: int) -> list[dict]:
    out = []
    for pi, p in enumerate(pieces):
        variants = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            key = (tuple(map(int, um.shape)), um.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "um": um,
                    "cells": int(um.sum()),
                    "coords": np.argwhere(um),
                    "vp": vp,
                    "vm": vm,
                }
            )
        assert variants, "every non-singleton piece must reduce to block cells"
        out.append({"pi": int(pi), "cells": int(variants[0]["cells"]), "variants": variants})
    return out


def _piece_candidates_from_masks(piece_masks: list[np.ndarray]) -> list[dict]:
    out = []
    for pi, mask in enumerate(piece_masks):
        variants = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for um in _dihedral_masks(mask):
            key = (tuple(map(int, um.shape)), um.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "um": um,
                    "cells": int(um.sum()),
                    "coords": np.argwhere(um),
                    "vp": np.where(um, 1, 0).astype(int),
                    "vm": um.copy(),
                }
            )
        out.append({"pi": int(pi), "cells": int(mask.sum()), "variants": variants})
    return out


def _build_slot_plan(slot: dict, piece_candidates: list[dict]) -> dict:
    slot_mask = slot["mask"]
    slot_cells = [tuple(map(int, rc)) for rc in np.argwhere(slot_mask)]
    cell_to_bit = {rc: idx for idx, rc in enumerate(slot_cells)}
    placements = []
    cover_lists = [[] for _ in slot_cells]

    for cand in piece_candidates:
        pi = int(cand["pi"])
        for variant in cand["variants"]:
            um = variant["um"]
            uh, uw = map(int, um.shape)
            if uh > slot_mask.shape[0] or uw > slot_mask.shape[1]:
                continue
            coords = variant["coords"]
            for dr in range(slot_mask.shape[0] - uh + 1):
                for dc in range(slot_mask.shape[1] - uw + 1):
                    window = slot_mask[dr : dr + uh, dc : dc + uw]
                    if not bool(window[um].all()):
                        continue
                    cover_bits = 0
                    for rr, cc in coords:
                        bit = cell_to_bit[(int(dr + rr), int(dc + cc))]
                        cover_bits |= 1 << bit
                    pid = len(placements)
                    placements.append(
                        {
                            "pi": pi,
                            "cover_bits": int(cover_bits),
                            "used_mask": 1 << pi,
                            "dr": int(dr),
                            "dc": int(dc),
                            "cells": int(variant["cells"]),
                            "variant": variant,
                        }
                    )
                    for rr, cc in coords:
                        cover_lists[cell_to_bit[(int(dr + rr), int(dc + cc))]].append(pid)

    return {
        "area": int(slot["area"]),
        "full_bits": (1 << len(slot_cells)) - 1,
        "placements": placements,
        "cover_lists": cover_lists,
    }


def _enumerate_slot_tilings(slot_plan: dict, available_mask: int, limit: int) -> list[tuple[int, list[dict]]]:
    full_bits = int(slot_plan["full_bits"])
    placements = slot_plan["placements"]
    cover_lists = slot_plan["cover_lists"]
    results: list[tuple[int, list[dict]]] = []
    dead: set[tuple[int, int]] = set()

    def rec(covered_bits: int, avail_mask: int, chosen: list[dict]) -> None:
        if len(results) >= limit:
            return
        state = (int(covered_bits), int(avail_mask))
        if state in dead:
            return
        if covered_bits == full_bits:
            results.append((int(available_mask ^ avail_mask), list(chosen)))
            return

        remaining = full_bits & ~covered_bits
        best_options = None
        best_len = None
        tmp = remaining
        while tmp:
            lsb = tmp & -tmp
            bit_idx = lsb.bit_length() - 1
            options = []
            for pid in cover_lists[bit_idx]:
                pl = placements[pid]
                if not (avail_mask & (1 << int(pl["pi"]))):
                    continue
                if covered_bits & int(pl["cover_bits"]):
                    continue
                options.append(pid)
            if not options:
                dead.add(state)
                return
            if best_len is None or len(options) < best_len:
                best_len = len(options)
                best_options = options
                if best_len == 1:
                    break
            tmp ^= lsb

        assert best_options is not None
        best_options.sort(
            key=lambda pid: (
                -int(placements[pid]["cells"]),
                int(placements[pid]["pi"]),
                int(placements[pid]["dr"]),
                int(placements[pid]["dc"]),
            )
        )
        before = len(results)
        for pid in best_options:
            pl = placements[pid]
            chosen.append(pl)
            rec(
                int(covered_bits | int(pl["cover_bits"])),
                int(avail_mask & ~int(pl["used_mask"])),
                chosen,
            )
            chosen.pop()
            if len(results) >= limit:
                return
        if len(results) == before:
            dead.add(state)

    rec(0, int(available_mask), [])
    return results


def _solve_slot_assignments(slots: list[dict], piece_candidates: list[dict], limit: int) -> list[dict]:
    slot_plans = [_build_slot_plan(slot, piece_candidates) for slot in slots]
    if any(not plan["placements"] for plan in slot_plans):
        return []

    slot_order = sorted(
        range(len(slots)),
        key=lambda si: (-int(slots[si]["area"]), len({int(pl["pi"]) for pl in slot_plans[si]["placements"]})),
    )
    piece_cells = [int(cand["cells"]) for cand in piece_candidates]
    remaining_slot_area = [0] * (len(slot_order) + 1)
    for idx in range(len(slot_order) - 1, -1, -1):
        remaining_slot_area[idx] = remaining_slot_area[idx + 1] + int(slots[slot_order[idx]]["area"])

    results: list[dict] = []
    dead: set[tuple[int, int]] = set()

    def rec(order_pos: int, avail_mask: int, chosen_by_slot: dict[int, list[dict]]) -> None:
        if len(results) >= limit:
            return
        state = (int(order_pos), int(avail_mask))
        if state in dead:
            return
        if order_pos == len(slot_order):
            if avail_mask == 0:
                results.append({si: list(tiling) for si, tiling in chosen_by_slot.items()})
                return
            dead.add(state)
            return

        avail_area = 0
        tmp = avail_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            avail_area += piece_cells[idx]
            tmp ^= lsb
        if avail_area != remaining_slot_area[order_pos]:
            dead.add(state)
            return

        si = slot_order[order_pos]
        tilings = _enumerate_slot_tilings(slot_plans[si], int(avail_mask), max(1, limit - len(results)))
        if not tilings:
            dead.add(state)
            return

        tilings.sort(key=lambda item: (len(item[1]), -sum(int(pl["cells"]) for pl in item[1])))
        before = len(results)
        for used_mask, tiling in tilings:
            chosen_by_slot[si] = list(tiling)
            rec(order_pos + 1, int(avail_mask & ~used_mask), chosen_by_slot)
            chosen_by_slot.pop(si, None)
            if len(results) >= limit:
                return
        if len(results) == before:
            dead.add(state)

    rec(0, (1 << len(piece_candidates)) - 1, {})
    return results


def _build_output(layout_shape: tuple[int, int], frame_color: int, k: int, slots: list[dict], solution: dict[int, list[dict]]) -> np.ndarray:
    out = np.full(((layout_shape[0] + 2) * k, (layout_shape[1] + 2) * k), frame_color, dtype=int)
    for si, slot in enumerate(slots):
        for pl in solution.get(si, []):
            top = (int(slot["r0"]) + int(pl["dr"]) + 1) * k
            left = (int(slot["c0"]) + int(pl["dc"]) + 1) * k
            paste(out, pl["variant"]["vp"], top, left, mask=pl["variant"]["vm"], overwrite=True)
    return out


def _solve_instance(grid: np.ndarray, solution_limit: int = 1) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
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

    slots = _extract_slots(layout)
    assert slots, "frame must contain at least one slot"

    pieces = _extract_pieces(g, comps, frame_comp)
    assert pieces, "at least one non-frame piece must exist"

    k = _infer_k(pieces)
    piece_candidates = _piece_candidates_from_observed(pieces, k)
    solutions = _solve_slot_assignments(slots, piece_candidates, limit=solution_limit)
    assert solutions, "piece set must tile the frame holes"

    out = _build_output(tuple(map(int, layout.shape)), frame_color, k, slots, solutions[0])
    return {
        "output": out,
        "slots": slots,
        "solutions": solutions,
        "k": int(k),
        "frame_color": int(frame_color),
        "layout_shape": tuple(map(int, layout.shape)),
    }


def _measure_composite_behavior(grid: np.ndarray) -> dict:
    info = _solve_instance(grid, solution_limit=2)
    first = info["solutions"][0]
    counts = [len(first.get(si, [])) for si in range(len(info["slots"]))]
    multi_areas = [int(info["slots"][si]["area"]) for si, count in enumerate(counts) if count > 1]
    return {
        "solution_count": int(len(info["solutions"])),
        "max_pieces_per_slot": int(max(counts) if counts else 0),
        "multi_slot_areas": multi_areas,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the frame layout and refill every hole, allowing some large holes to be assembled from multiple pieces.

    Input structure:
    - Background is the most common color. The largest foreground component is a monochrome rectangular frame whose
      interior contains 4-connected background holes and frame-color walls.
    - Outside the frame are monochrome piece sprites built from solid kxk blocks; unit holes still create kxk
      solid squares, while at least one large hole is represented by several smaller sprites.

    Rule:
    - Recover the hole layout inside the frame, infer k from the smallest solid-square piece, and downsample every
      outside sprite to a block-level polyomino under all D4 orientations.
    - For each hole, place one or more oriented pieces at block-grid translations so that the chosen pieces are
      pairwise disjoint and their union exactly covers that hole. Then output the tight frame layout scaled by k
      with those matched pieces pasted back into their solved positions.

    Key insight:
    - The outside sprites are still exact kxk-block magnifications of the cells that must be filled inside the frame;
      the new difficulty is not scaling but exact cover: a large hole may only be reconstructible as the union of
      several smaller oriented block masks rather than as one whole-shape match.
    """
    return _solve_instance(np.array(grid, dtype=int), solution_limit=1)["output"]


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w


def _grow_slot(forbidden: np.ndarray, target_size: int, *, want_irregular_large: bool) -> np.ndarray | None:
    H, W = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, cell)) for cell in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = []
            for nr, nc in frontier:
                if 0 <= nr < H and 0 <= nc < W and not forbidden[nr, nc] and (nr, nc) not in chosen:
                    freedom = 0
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if 0 <= ar < H and 0 <= ac < W and not forbidden[ar, ac] and (ar, ac) not in chosen:
                            freedom += 1
                    cands.append((freedom, nr, nc))
            if not cands:
                break
            cands.sort(reverse=True)
            top = cands[: min(4, len(cands))]
            _, nr, nc = random.choice(top)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) != target_size:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        if want_irregular_large:
            mask, _ = _tight_mask_from_coords(coords)
            if not _is_irregular_large_shape(mask):
                continue
        return coords
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


def _choose_slot_sizes(k: int) -> tuple[int, list[int]] | None:
    n_slots = random.randint(*(N_SLOTS_RANGE_K2 if k == 2 else N_SLOTS_RANGE_K3))
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    non_unit_count = n_slots - n_units
    if non_unit_count <= 0:
        return None

    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])
    sizes = [random.choice(BIG_NON_UNIT_CHOICES)]
    while len(sizes) < non_unit_count:
        remaining = non_unit_count - len(sizes)
        max_allowed = budget - n_units - sum(sizes) - 2 * (remaining - 1)
        if max_allowed < 2:
            return None
        choices = [s for s in SMALL_NON_UNIT_CHOICES if s <= max_allowed]
        if not choices:
            return None
        sizes.append(random.choice(choices))
    random.shuffle(sizes)
    return n_units, sizes


def _generate_slots(slot_sizes: list[int], n_units: int) -> list[dict] | None:
    forbidden = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    non_unit_slots: list[dict] = []
    used_shape_keys: set[bytes] = set()
    need_irregular_large = True

    for size in sorted(slot_sizes, reverse=True):
        want_irregular = need_irregular_large and size >= 8
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size), want_irregular_large=want_irregular)
            if coords is None:
                continue
            mask, _ = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            non_unit_slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key})
            _forbid_coords(forbidden, coords)
            if want_irregular:
                need_irregular_large = False
            break
        else:
            return None

    unit_slots: list[dict] = []
    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1, want_irregular_large=False)
            if coords is None:
                continue
            unit_slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool), "shape_key": b"unit"})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    if need_irregular_large:
        return None
    return unit_slots + non_unit_slots


def _crop_layout(slots: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    full_mask = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    for s in slots:
        full_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    hole_box = bbox(full_mask)
    if hole_box is None:
        return None
    r0, c0, r1, c1 = map(int, hole_box)
    layout_mask = full_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, layout_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(layout_mask.sum()) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots = []
    for s in slots:
        coords = s["coords"].copy()
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        mask, (sr0, sc0) = _tight_mask_from_coords(coords)
        cropped_slots.append(
            {
                "coords": coords,
                "area": int(s["area"]),
                "mask": mask,
                "r0": int(sr0),
                "c0": int(sc0),
                "shape_key": _canonical_shape_key(mask) if int(s["area"]) > 1 else b"unit",
            }
        )
    cropped_slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    return cropped_slots, layout_mask


def _bits_to_part(mask_coords: np.ndarray, bits: int) -> dict:
    idxs = [idx for idx in range(len(mask_coords)) if bits & (1 << idx)]
    sel = mask_coords[idxs]
    part_mask, (r0, c0) = _tight_mask_from_coords(sel)
    return {
        "coords_local": sel,
        "mask": part_mask,
        "r0": int(r0),
        "c0": int(c0),
        "shape_key": _canonical_shape_key(part_mask),
    }


def _partition_large_slot(slot: dict, forbidden_keys: set[bytes]) -> list[dict] | None:
    mask = slot["mask"]
    coords = np.argwhere(mask)
    cell_count = int(coords.shape[0])
    if cell_count < 8:
        return None

    coord_to_idx = {tuple(map(int, rc)): idx for idx, rc in enumerate(coords)}
    neighbor_masks = [0] * cell_count
    for idx, (r, c) in enumerate(coords):
        bits = 0
        for nr, nc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)):
            j = coord_to_idx.get((nr, nc))
            if j is not None:
                bits |= 1 << j
        neighbor_masks[idx] = bits

    def is_connected(bits: int) -> bool:
        if bits == 0:
            return False
        seen = 0
        frontier = bits & -bits
        while frontier:
            lsb = frontier & -frontier
            frontier ^= lsb
            if seen & lsb:
                continue
            seen |= lsb
            idx = lsb.bit_length() - 1
            frontier |= neighbor_masks[idx] & bits & ~seen
        return seen == bits

    full_bits = (1 << cell_count) - 1
    min_part = 2
    max_part = 6
    preferred_counts = [3, 2] if cell_count >= 10 else [2]
    all_solutions: list[list[dict]] = []

    def rec(remaining_bits: int, parts_left: int, chosen: list[dict], used_keys: set[bytes]) -> None:
        if parts_left == 1:
            size = remaining_bits.bit_count()
            if not (min_part <= size <= max_part):
                return
            if not is_connected(remaining_bits):
                return
            part = _bits_to_part(coords, remaining_bits)
            key = part["shape_key"]
            if key in forbidden_keys or key in used_keys:
                return
            all_solutions.append(chosen + [part])
            return

        rem_size = remaining_bits.bit_count()
        if rem_size < min_part * parts_left or rem_size > max_part * parts_left:
            return

        anchor = remaining_bits & -remaining_bits
        sub = remaining_bits
        while sub:
            if sub & anchor:
                size = sub.bit_count()
                if min_part <= size <= max_part and sub != remaining_bits and is_connected(sub):
                    rest = remaining_bits ^ sub
                    if rest:
                        part = _bits_to_part(coords, sub)
                        key = part["shape_key"]
                        if key not in forbidden_keys and key not in used_keys:
                            rec(rest, parts_left - 1, chosen + [part], used_keys | {key})
            sub = (sub - 1) & remaining_bits

    for part_count in preferred_counts:
        all_solutions.clear()
        rec(full_bits, int(part_count), [], set())
        if all_solutions:
            random.shuffle(all_solutions)
            return all_solutions[0]
    return None


def _plan_atomic_pieces(slots: list[dict]) -> tuple[list[dict], int] | None:
    big_candidates = [idx for idx, slot in enumerate(slots) if int(slot["area"]) >= 8]
    if not big_candidates:
        return None
    composite_idx = max(big_candidates, key=lambda idx: int(slots[idx]["area"]))
    forbidden_keys = {slot["shape_key"] for idx, slot in enumerate(slots) if idx != composite_idx and int(slot["area"]) > 1}
    parts = _partition_large_slot(slots[composite_idx], forbidden_keys)
    if parts is None or len(parts) < 2:
        return None

    atoms = []
    for si, slot in enumerate(slots):
        if int(slot["area"]) == 1:
            coords = slot["coords"].copy()
            atoms.append(
                {
                    "slot_index": int(si),
                    "coords": coords,
                    "mask": np.ones((1, 1), dtype=bool),
                    "r0": int(coords[0, 0]),
                    "c0": int(coords[0, 1]),
                    "is_unit": True,
                }
            )
            continue

        if si == composite_idx:
            for part in parts:
                global_coords = part["coords_local"].copy()
                global_coords[:, 0] += int(slot["r0"])
                global_coords[:, 1] += int(slot["c0"])
                atoms.append(
                    {
                        "slot_index": int(si),
                        "coords": global_coords,
                        "mask": part["mask"],
                        "r0": int(slot["r0"] + part["r0"]),
                        "c0": int(slot["c0"] + part["c0"]),
                        "is_unit": False,
                    }
                )
        else:
            atoms.append(
                {
                    "slot_index": int(si),
                    "coords": slot["coords"].copy(),
                    "mask": slot["mask"].copy(),
                    "r0": int(slot["r0"]),
                    "c0": int(slot["c0"]),
                    "is_unit": False,
                }
            )
    return atoms, int(composite_idx)


def _make_piece(atom: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(atom["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

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
        "slot_index": int(atom["slot_index"]),
        "coords": atom["coords"].copy(),
        "r0": int(atom["r0"]),
        "c0": int(atom["c0"]),
        "color": int(color),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "is_unit": bool(atom["is_unit"]),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        paste(out, p["out_patch"], (int(p["r0"]) + 1) * k, (int(p["c0"]) + 1) * k, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
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
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    size_plan = _choose_slot_sizes(k)
    if size_plan is None:
        return None
    n_units, non_unit_sizes = size_plan

    slots = _generate_slots(non_unit_sizes, n_units)
    if slots is None:
        return None
    cropped = _crop_layout(slots)
    if cropped is None:
        return None
    slots, layout_mask = cropped

    atom_plan = _plan_atomic_pieces(slots)
    if atom_plan is None:
        return None
    atoms, composite_idx = atom_plan

    non_unit_atom_masks = [atom["mask"] for atom in atoms if not atom["is_unit"]]
    shape_keys = [_canonical_shape_key(mask) for mask in non_unit_atom_masks]
    if len(shape_keys) != len(set(shape_keys)):
        return None

    block_piece_candidates = _piece_candidates_from_masks([atom["mask"] for atom in atoms])
    block_solutions = _solve_slot_assignments(slots, block_piece_candidates, limit=2)
    if len(block_solutions) != 1:
        return None
    first_solution = block_solutions[0]
    counts = [len(first_solution.get(si, [])) for si in range(len(slots))]
    if counts[composite_idx] < 2 or int(slots[composite_idx]["area"]) < 8:
        return None

    frame_area = (layout_mask.shape[0] + 2) * (layout_mask.shape[1] + 2) - int(layout_mask.sum())
    max_piece_area = max(int(atom["mask"].sum()) * k * k for atom in atoms)
    if frame_area <= max_piece_area:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()

    non_unit_atom_count = sum(1 for atom in atoms if not atom["is_unit"])
    if len(palette) < non_unit_atom_count + 1:
        return None
    non_unit_colors = palette[:non_unit_atom_count]
    unit_color = palette[non_unit_atom_count]

    pieces = []
    color_idx = 0
    for atom in atoms:
        if atom["is_unit"]:
            pieces.append(_make_piece(atom, unit_color, k, apply_dihedral=False))
        else:
            pieces.append(_make_piece(atom, non_unit_colors[color_idx], k, apply_dihedral=True))
            color_idx += 1

    layout_h, layout_w = map(int, layout_mask.shape)
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    behavior = _measure_composite_behavior(canvas)
    if behavior["solution_count"] != 1:
        return None
    if behavior["max_pieces_per_slot"] < 2:
        return None
    if not any(area >= 8 for area in behavior["multi_slot_areas"]):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
