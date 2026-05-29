# Combo variant of 8b9c3697 (index 13)
# combo_spec: D11/0+D13/1
# Name: 视线干扰面积消歧

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The base task casts rays from a comb's interior socket and used to assume that
#    the visible red on that opening side was the true mover. D11 breaks that
#    by allowing line-of-sight distractors; D13 then says that if several reds are
#    visible on that opening, selection is based on area fit to the slot, not on
#    hit count or nearness.
#    Small trace in canonical orientation:
#    - Structure with teeth=3 has slot area 2x3 = 6.
#    - Near decoy: a 2x2 red sits closer to the opening and blocks two rays.
#    - True target: a 2x3 red sits farther away on the same axis and is only hit
#      by one remaining ray.
#    - Old logic picks the 2x2 because hit_count=2 beats 1.
#    - Combo logic still sees both reds on the same line of sight, but chooses the
#      2x3 because |6-6| < |4-6|. The 2x2 disappears instead of docking.
# b) Complexity calibration
#    To keep the rule visible from a few samples, every matched structure gets one
#    visible decoy on its opening side. Matched structures only use teeth=2 or 3,
#    so slot areas repeat as 4 or 6 and the relevant area comparison stays easy to
#    spot. Unrelated background distractors are still limited and kept off rays.
# c) Implicit-assumption audit of the original main()
#    - Original assumption: one visible red per structure. Broken by D11.
#    - Original scoring: hit_count / min_dist identifies the right red. Broken by
#      D13 because the nearer red is deliberately the wrong size.
#    - Original placement: output-only occupancy is enough. Broken here because the
#      input now contains extra visible reds, so structure placement must reserve
#      the union footprint of structure + output trail + all input reds.
#    - Original generator check `_unique_matched_visibility` is no longer valid,
#      because ambiguity is now required rather than forbidden.
# d) Placement legality
#    Structures are still non-overlapping and padded apart. Each matched patch is
#    built output-first, then its input adds a nearer decoy on the same opening
#    axis without touching the intended red. Global placement uses a footprint mask
#    so those added input reds cannot overlap neighboring structures or patches.
# e) Visual observability
#    Each matched structure visibly presents two reds on one opening side: a nearer
#    smaller rectangle that vanishes, and a farther rectangle that moves inward and
#    leaves a black trail. That repeated pattern makes the area-based choice human-
#    observable from 2-3 examples without relying on hidden numeric color ids.
# f) Visual cleanliness
#    The generator keeps exactly one visible decoy per matched structure and blocks
#    all other distractors from structure rays. This avoids dense stacks of reds in
#    the same corridor and keeps the contrast between target, decoy, structure, and
#    black trail legible.
# g) Color reasoning
#    Colors keep their original semantics. Red is the only movable/erasable color,
#    black is only the trail, and all matching decisions depend on geometry and area
#    rather than on any color number or hidden ordering.
# h) New solver assumptions and generator guarantees
#    - If a structure has visible reds, the generator guarantees at least one true
#      target among them; unmatched structures are generated with no visible reds.
#    - Visible candidates are solid rectangles created by the generator, so area is
#      well-defined and the simple bbox-based shift remains valid.
#    - Every matched structure has exactly two visible reds and at least one of them
#      makes the area rule disagree with the legacy hit-count / distance rule.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

MATCHED_WIDTH_TEETH = ((8, 2), (9, 3))
UNMATCHED_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
THREE_STRUCT_GRID_H_RANGE = (23, 27)
THREE_STRUCT_GRID_W_RANGE = (28, 33)
SCENARIOS = ("two_matched", "three_matched", "three_with_unmatched")
SCENARIO_WEIGHTS = (1, 1, 1)
SLIDE_RANGE = (5, 8)
DECOY_SLIDE_RANGE = (2, 3)
STRUCT_PADDING = 4
THREE_STRUCT_PADDING = 3

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 10
DISTRACTOR_MIN = 1
DISTRACTOR_MAX = 4

MAX_RETRIES = 400

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _extract_red_data(
    g: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_comps = components_mask(g == int(Color.RED), connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)
    return red_coords, red_bbox, red_at


def _extract_structures(
    g: np.ndarray,
    bg: int,
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    struct_mask = (g != int(bg)) & (g != int(Color.RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox


def _slot_candidates(
    g: np.ndarray,
    bg: int,
    red_at: dict[tuple[int, int], int],
) -> tuple[list[dict], list[list[int]], list[tuple[int, int, int, int]], list[list[tuple[int, int]]]]:
    h, w = g.shape
    red_coords, red_bbox, _ = _extract_red_data(g)
    red_areas = [len(coords) for coords in red_coords]
    struct_coords, struct_bbox = _extract_structures(g, bg)

    struct_infos: list[dict] = []
    visible_per_struct: list[list[int]] = []

    for sid, s_coords in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best_area = None
        best_legacy = None
        visible_rids: set[int] = set()

        for dr, dc in _DIRS:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            socket_cells: set[tuple[int, int]] = set()

            for r, c in s_coords:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                socket_cells.add((int(nr), int(nc)))
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(Color.RED):
                    rid = int(red_at[(int(rr), int(cc))])
                    visible_rids.add(int(rid))
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits or not socket_cells:
                continue

            max_proj = max(int(r * dr + c * dc) for r, c in socket_cells)
            mouth_width = sum(1 for r, c in socket_cells if int(r * dr + c * dc) == int(max_proj))
            slot_area = int(mouth_width * min(mouth_width, 2))

            def area_key(rid: int) -> tuple[int, int, int]:
                return (
                    -abs(int(red_areas[rid]) - int(slot_area)),
                    int(hits[rid][0]),
                    -int(hits[rid][1]),
                )

            def legacy_key(rid: int) -> tuple[int, int, int]:
                return (
                    int(hits[rid][0]),
                    -int(hits[rid][1]),
                    -abs(int(red_areas[rid]) - int(slot_area)),
                )

            rid_area = max(hits.keys(), key=area_key)
            rid_legacy = max(hits.keys(), key=legacy_key)

            cnt_a, mind_a, anchor_a = hits[rid_area]
            cand_area = {
                "score": (
                    -abs(int(red_areas[rid_area]) - int(slot_area)),
                    int(cnt_a),
                    -int(mind_a),
                ),
                "rid": int(rid_area),
                "dir": (int(dr), int(dc)),
                "anchor": anchor_a,
                "sid": int(sid),
                "slot_area": int(slot_area),
                "area_diff": int(abs(int(red_areas[rid_area]) - int(slot_area))),
                "hit_count": int(cnt_a),
                "min_dist": int(mind_a),
            }

            cnt_l, mind_l, anchor_l = hits[rid_legacy]
            cand_legacy = {
                "score": (
                    int(cnt_l),
                    -int(mind_l),
                    -abs(int(red_areas[rid_legacy]) - int(slot_area)),
                ),
                "rid": int(rid_legacy),
                "dir": (int(dr), int(dc)),
                "anchor": anchor_l,
                "sid": int(sid),
                "slot_area": int(slot_area),
                "area_diff": int(abs(int(red_areas[rid_legacy]) - int(slot_area))),
                "hit_count": int(cnt_l),
                "min_dist": int(mind_l),
            }

            if best_area is None or cand_area["score"] > best_area["score"]:
                best_area = cand_area
            if best_legacy is None or cand_legacy["score"] > best_legacy["score"]:
                best_legacy = cand_legacy

        struct_infos.append(
            {
                "sid": int(sid),
                "best_area": best_area,
                "best_legacy": best_legacy,
            }
        )
        visible_per_struct.append(sorted(int(rid) for rid in visible_rids))

    return struct_infos, visible_per_struct, red_bbox, red_coords


def _assign_structures(struct_infos: list[dict], mode: str) -> list[dict]:
    key_name = "best_area" if mode == "area" else "best_legacy"
    ranked = [info[key_name] for info in struct_infos if info[key_name] is not None]
    ranked.sort(reverse=True, key=lambda item: item["score"])

    used_red: set[int] = set()
    assignments: list[dict] = []
    for item in ranked:
        if int(item["rid"]) in used_red or item["anchor"] is None:
            continue
        used_red.add(int(item["rid"]))
        assignments.append(item)
    return assignments


def _behavior_metrics(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    _red_coords, _red_bbox, red_at = _extract_red_data(grid)
    struct_infos, visible_per_struct, _rb2, _rc2 = _slot_candidates(grid, bg, red_at)

    visible_structs = sum(1 for visible in visible_per_struct if visible)
    multi_visible_structs = sum(1 for visible in visible_per_struct if len(visible) >= 2)
    legacy_conflicts = sum(
        1
        for info in struct_infos
        if info["best_area"] is not None
        and info["best_legacy"] is not None
        and int(info["best_area"]["rid"]) != int(info["best_legacy"]["rid"])
    )
    return {
        "structures": len(struct_infos),
        "visible_structs": int(visible_structs),
        "multi_visible_structs": int(multi_visible_structs),
        "legacy_conflicts": int(legacy_conflicts),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide one red rectangle per comb cavity, choosing among visible aligned reds by the slot area they fit best.

    Input structure:
    - The dominant color is background. Large non-red components are comb / bracket structures with one open side.
    - Red components are separate solid rectangles outside the structures; some openings intentionally show more than one visible red.

    Rule:
    - Ray-cast outward from each structure's interior socket cells and collect the red components visible on each opening direction.
    - Infer that opening's slot area from the socket mouth width, then pick the visible red whose area is closest to that slot area; ties fall back to ray coverage and distance.
    - Greedily reserve the chosen reds across structures, slide each reserved component straight into the cavity while painting the swept path black, and erase every unchosen red.

    Key insight:
    A red block can sit directly on a structure's line of sight and still be the wrong choice; the deciding signal is area fit to the cavity, not simple visibility, hit count, or proximity.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red = int(Color.RED)
    black = int(Color.BLACK)

    assert red in set(map(int, np.unique(g)))

    red_coords, red_bbox, red_at = _extract_red_data(g)
    struct_infos, _visible_per_struct, _rb2, _rc2 = _slot_candidates(g, bg, red_at)
    assignments = _assign_structures(struct_infos, mode="area")

    out = g.copy()
    out[out == red] = int(bg)

    for item in assignments:
        rid = int(item["rid"])
        dr, dc = map(int, item["dir"])
        ar, ac = map(int, item["anchor"])

        coords = red_coords[rid]
        r0, c0, r1, c1 = map(int, red_bbox[rid])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(black)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(red)

    return out


def _matched_patches(
    width: int,
    teeth: int,
    slide: int,
    decoy_slide: int,
    rot_k: int,
    bg: int,
    sc: int,
) -> dict | None:
    red = int(Color.RED)
    black = int(Color.BLACK)
    red_h = min(teeth, 2)
    if int(decoy_slide + red_h) > int(slide):
        return None

    decoy_w = max(1, teeth - 1)
    decoy_side = random.choice((0, 1))
    start = (width - teeth) // 2
    decoy_c0 = int(start if decoy_side == 0 else start + teeth - decoy_w)

    patch_h = 2 + slide + red_h
    structure = np.full((patch_h, width), bg, dtype=int)
    base = _structure_array(width, teeth, bg, sc)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()
    intended_mask = np.zeros_like(structure, dtype=bool)

    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = black
    output_patch[2:2 + red_h, start:start + teeth] = red

    target_r0 = 2 + slide
    input_patch[target_r0:target_r0 + red_h, start:start + teeth] = red
    intended_mask[target_r0:target_r0 + red_h, start:start + teeth] = True

    decoy_r0 = 2 + decoy_slide
    input_patch[decoy_r0:decoy_r0 + red_h, decoy_c0:decoy_c0 + decoy_w] = red

    footprint = np.where(
        (input_patch != bg) | (output_patch != bg) | (structure != bg),
        sc,
        bg,
    )

    return {
        "matched": True,
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "footprint": np.rot90(footprint, k=rot_k),
        "intended_mask": np.rot90(intended_mask, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "matched": False,
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "footprint": structure.copy(),
        "intended_mask": np.zeros_like(structure, dtype=bool),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if struct_grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and struct_grid[rr, cc] == bg:
                    forbidden[rr, cc] = True
                    rr += dr
                    cc += dc
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
    sc: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    occupancy = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(occupancy, spec["footprint"], bg, padding=padding)
        if loc is None:
            return None
        r0, c0 = loc
        paste(occupancy, spec["footprint"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        intended_coords = [
            (r0 + int(rr), c0 + int(cc))
            for rr, cc in np.argwhere(spec["intended_mask"])
        ]

        placed.append(
            {
                "matched": bool(spec["matched"]),
                "coords": coords,
                "coords_key": frozenset(coords),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "intended_coords": intended_coords,
            }
        )

    return inp, out, struct_grid, placed


def _placed_by_structure_order(
    grid: np.ndarray,
    placed: list[dict],
    bg: int,
) -> list[dict] | None:
    struct_coords, _struct_bbox = _extract_structures(grid, bg)
    placed_by_coords = {item["coords_key"]: item for item in placed}
    ordered: list[dict] = []
    for coords in struct_coords:
        item = placed_by_coords.get(frozenset(coords))
        if item is None:
            return None
        ordered.append(item)
    return ordered


def _validate_fusion_grid(grid: np.ndarray, placed: list[dict], bg: int) -> bool:
    _red_coords, _red_bbox, red_at = _extract_red_data(grid)
    struct_infos, visible_per_struct, _rb2, _rc2 = _slot_candidates(grid, bg, red_at)
    placed_in_order = _placed_by_structure_order(grid, placed, bg)
    if placed_in_order is None or len(placed_in_order) != len(struct_infos):
        return False

    matched_count = sum(1 for item in placed if item["matched"])
    visible_decoy_count = 0
    legacy_conflicts = 0

    for sid, item in enumerate(placed_in_order):
        visible = visible_per_struct[sid]
        best_area = struct_infos[sid]["best_area"]
        best_legacy = struct_infos[sid]["best_legacy"]

        if item["matched"]:
            if len(visible) != 2 or best_area is None or best_legacy is None:
                return False
            intended_rids = {int(red_at[(r, c)]) for r, c in item["intended_coords"]}
            if len(intended_rids) != 1:
                return False
            intended_rid = next(iter(intended_rids))
            if int(best_area["rid"]) != int(intended_rid):
                return False
            if int(best_legacy["rid"]) == int(intended_rid):
                return False
            visible_decoy_count += 1
            legacy_conflicts += 1
        else:
            if visible or best_area is not None or best_legacy is not None:
                return False

    if visible_decoy_count != matched_count:
        return False
    return legacy_conflicts == matched_count


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    sc: int,
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True
    red_mask = inp == int(Color.RED)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            red_mask[loc[0]:loc[0] + rh, loc[1]:loc[1] + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    if scenario == "two_matched":
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        n_structs, matched, padding = 2, 2, STRUCT_PADDING
    elif scenario == "three_matched":
        h = random.randint(*THREE_STRUCT_GRID_H_RANGE)
        w = random.randint(*THREE_STRUCT_GRID_W_RANGE)
        n_structs, matched, padding = 3, 3, THREE_STRUCT_PADDING
    else:
        h = random.randint(*THREE_STRUCT_GRID_H_RANGE)
        w = random.randint(*THREE_STRUCT_GRID_W_RANGE)
        n_structs, matched, padding = 3, 2, THREE_STRUCT_PADDING
    unmatched = n_structs - matched

    base_reds = 2 * matched
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - base_reds)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - base_reds)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs: list[dict] = []
    for idx in range(n_structs):
        rot_k = random.randrange(4)
        if idx < matched:
            width, teeth = random.choice(MATCHED_WIDTH_TEETH)
            slide = random.randint(*SLIDE_RANGE)
            decoy_slide = random.randint(*DECOY_SLIDE_RANGE)
            spec = _matched_patches(width, teeth, slide, decoy_slide, rot_k, bg, sc)
            if spec is None:
                return None
            specs.append(spec)
        else:
            width, teeth = random.choice(UNMATCHED_WIDTH_TEETH)
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["footprint"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc, padding)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _validate_fusion_grid(inp, placed, bg):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not _validate_fusion_grid(inp, placed, bg):
        return None

    metrics = _behavior_metrics(inp)
    if (
        metrics["visible_structs"] != matched
        or metrics["multi_visible_structs"] != matched
        or metrics["legacy_conflicts"] != matched
    ):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo 8b9c3697/13")
