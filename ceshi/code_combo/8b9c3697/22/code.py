# Combo variant of 8b9c3697 (index 22)
# combo_spec: D10/0+D11/0
# Name: 多对一视线内干扰

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The original task treated each comb structure as having one large docking
#    rectangle, so at most one red component could move into each structure.
#    This combo changes the granularity: each tooth column defines its own thin
#    lane inside the cavity. A 3-tooth comb therefore exposes three independent
#    candidate slots, each with the same opening direction but its own exact
#    rectangle shape and line-of-sight.
#    Small trace:
#    - Original behavior for a 3-tooth downward comb:
#      one 2x3 red rectangle sits below the opening and slides into the whole
#      cavity as one block.
#    - Combo behavior:
#      two separate 2x1 red strips sit below the left and right tooth lanes and
#      both slide into the same structure, while a 1x1 red decoy sits below the
#      middle lane. All three are visibly aligned to the same comb, but only the
#      two strips fit the inferred slot shape. Output shows two red docked strips
#      plus black trails; the 1x1 decoy disappears.
#    The visible difference is therefore joint: one structure consumes multiple
#    reds, and another aligned red in the same viewing region is rejected.
# b) Complexity calibration
#    The main signal is concentrated in one primary 3-tooth structure per grid.
#    It always has exactly two valid lane matches and one visible wrong-shape
#    decoy on the remaining lane. Additional structures keep the multi-object
#    feel but are limited to at most one matched lane each, which avoids turning
#    every structure into a dense bundle of reds and keeps the multi-to-one cue
#    easy to spot from a few examples.
# c) Implicit-assumption audit of the original main()
#    - The old solver assumed one best red per structure. That fails because a
#      single structure can now own several independent lanes.
#    - The old ray scoring assumed visibility alone identifies the correct red.
#      That fails because a wrong-shape red may now sit directly on a lane.
#    - The old final placement assumed the chosen component already had the full
#      structure-wide extent. That fails because the new matched pieces are thin
#      sub-rectangles, not one full-width block.
#    The new solver therefore parses each comb into per-tooth slot bboxes, then
#    performs exact rectangle matching per slot rather than per structure.
# d) Placement legality
#    Structures are still non-overlapping and padded apart. Every matched or
#    decoy red is constructed by shifting a slot bbox straight outward along the
#    structure opening direction, so motion is axis-aligned and stays faithful to
#    the geometry. Background distractors are forbidden from all exact slot rays
#    to avoid accidental extra matches; only the intentionally constructed visible
#    decoy is allowed on a slot ray.
# e) Visual observability
#    In each sample, the primary comb has three visible candidate lanes on the
#    same opening side: two contain correctly sized red rectangles that move into
#    the comb, and one contains an aligned but smaller rectangle that vanishes.
#    This makes both altered dimensions legible without hidden ordering rules.
# f) Visual cleanliness
#    The multi-match structure uses staggered slide distances 4, 6, and 8, so the
#    separate red components do not merge into one connected blob before moving.
#    Other reds keep 4-neighborhood separation, preventing ambiguous clusters.
# g) Color reasoning
#    No new color semantics are introduced. Red still marks movable or vanishing
#    pieces, black is still the swept trail, and structure colors are unchanged.
# h) New solver assumptions and generator guarantees
#    - Structures remain exact rotations of the original comb template, so local
#      parsing from the full back wall is reliable.
#    - True matches are solid rectangles whose bbox equals one inferred lane slot.
#    - The generator always creates at least one structure with >=2 matched slots
#      and at least one visible wrong-shape red on a slot lane.
#    - The generator rejects any sample where another red could also match a slot
#      or where the solver output disagrees with the independently rendered target.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

PRIMARY_STRUCT = (9, 3)
SECONDARY_STRUCTS = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
STRUCT_PADDING = 4

PRIMARY_MATCH_SLIDES = (4, 8)
PRIMARY_DECOY_SLIDE = 6
SLIDE_RANGE = (4, 8)

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8
VISIBLE_DECOYS_PER_GRID = 1

MAX_RETRIES = 400
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_hw(b: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(b[2] - b[0] + 1), int(b[3] - b[1] + 1)


def _shift_bbox(b: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return int(b[0] + dr), int(b[1] + dc), int(b[2] + dr), int(b[3] + dc)


def _box_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


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
    g[1, start : start + teeth] = sc
    return g


def _analyze_local_structure(local_mask: np.ndarray) -> dict:
    top_full = bool(local_mask[0, :].all())
    bottom_full = bool(local_mask[-1, :].all())
    left_full = bool(local_mask[:, 0].all())
    right_full = bool(local_mask[:, -1].all())

    if top_full:
        open_dir = (1, 0)
        rot_to_base = 0
    elif left_full:
        open_dir = (0, 1)
        rot_to_base = 3
    elif bottom_full:
        open_dir = (-1, 0)
        rot_to_base = 2
    elif right_full:
        open_dir = (0, -1)
        rot_to_base = 1
    else:
        raise AssertionError("structure does not match the comb template")

    canonical = np.rot90(local_mask, k=rot_to_base)
    if canonical.shape[0] != 4:
        raise AssertionError("unexpected comb depth")

    tooth_row = canonical[1].copy()
    tooth_row[1] = False
    tooth_row[-2] = False
    tooth_cols = np.flatnonzero(tooth_row)
    if len(tooth_cols) == 0 or np.any(np.diff(tooth_cols) != 1):
        raise AssertionError("teeth must form one contiguous block")

    red_len = int(min(len(tooth_cols), 2))
    slot_bboxes_local: list[tuple[int, int, int, int]] = []
    for col in tooth_cols:
        slot_canonical = np.zeros_like(canonical, dtype=bool)
        slot_canonical[2 : 2 + red_len, int(col)] = True
        slot_local = np.rot90(slot_canonical, k=(4 - rot_to_base) % 4)
        slot_bboxes_local.append(tuple(map(int, bbox_coords(np.argwhere(slot_local)))))

    return {
        "dir": (int(open_dir[0]), int(open_dir[1])),
        "teeth": int(len(tooth_cols)),
        "slot_bboxes_local": slot_bboxes_local,
    }


def _red_infos(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    red = int(Color.RED)
    infos: list[dict] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(components_mask(g == red, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        h, w = _bbox_hw(box)
        info = {
            "coords": pts,
            "bbox": box,
            "shape": (int(h), int(w)),
            "is_rect": len(pts) == int(h * w),
        }
        infos.append(info)
        for rc in pts:
            red_at[rc] = int(rid)
    return infos, red_at


def _structure_infos(g: np.ndarray, bg: int) -> list[dict]:
    red = int(Color.RED)
    struct_mask = (g != int(bg)) & (g != red)
    infos: list[dict] = []
    for sid, coords in enumerate(components_mask(struct_mask, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = map(int, box)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[int(r - r0), int(c - c0)] = True
        meta = _analyze_local_structure(local_mask)
        slots = []
        for slot_idx, local_box in enumerate(meta["slot_bboxes_local"]):
            abs_box = _shift_bbox(local_box, r0, c0)
            slots.append(
                {
                    "sid": int(sid),
                    "slot_idx": int(slot_idx),
                    "dock_bbox": abs_box,
                    "shape": _bbox_hw(abs_box),
                    "dir": meta["dir"],
                }
            )
        infos.append(
            {
                "sid": int(sid),
                "coords": set(pts),
                "bbox": box,
                "dir": meta["dir"],
                "teeth": meta["teeth"],
                "slots": slots,
            }
        )
    return infos


def _candidate_distance(
    grid: np.ndarray,
    bg: int,
    struct_info: dict,
    slot_info: dict,
    red_bbox: tuple[int, int, int, int],
) -> int | None:
    sr0, sc0, sr1, sc1 = map(int, struct_info["bbox"])
    tr0, tc0, tr1, tc1 = map(int, slot_info["dock_bbox"])
    rr0, rc0, rr1, rc1 = map(int, red_bbox)
    dr, dc = map(int, slot_info["dir"])

    if dr == 1:
        if rc0 != tc0 or rc1 != tc1 or rr0 <= sr1:
            return None
        corridor = grid[tr1 + 1 : rr0, tc0 : tc1 + 1]
    elif dr == -1:
        if rc0 != tc0 or rc1 != tc1 or rr1 >= sr0:
            return None
        corridor = grid[rr1 + 1 : tr0, tc0 : tc1 + 1]
    elif dc == 1:
        if rr0 != tr0 or rr1 != tr1 or rc0 <= sc1:
            return None
        corridor = grid[tr0 : tr1 + 1, tc1 + 1 : rc0]
    else:
        if rr0 != tr0 or rr1 != tr1 or rc1 >= sc0:
            return None
        corridor = grid[tr0 : tr1 + 1, rc1 + 1 : tc0]

    if corridor.size and np.any(corridor != int(bg)):
        return None
    return int(abs(rr0 - tr0) + abs(rc0 - tc0))


def _solve_assignments(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, tuple[int, int], int, int]], list[dict], list[dict]]:
    red_infos, _ = _red_infos(grid)
    struct_infos = _structure_infos(grid, bg)

    ranked: list[tuple[int, int, int, int]] = []
    for sinfo in struct_infos:
        for slot in sinfo["slots"]:
            best: tuple[int, int, int, int] | None = None
            for rid, rinfo in enumerate(red_infos):
                if not rinfo["is_rect"] or rinfo["shape"] != slot["shape"]:
                    continue
                dist = _candidate_distance(grid, bg, sinfo, slot, rinfo["bbox"])
                if dist is None:
                    continue
                cand = (int(dist), int(rid), int(sinfo["sid"]), int(slot["slot_idx"]))
                if best is None or cand < best:
                    best = cand
            if best is not None:
                ranked.append(best)

    ranked.sort()
    used_red: set[int] = set()
    used_slots: set[tuple[int, int]] = set()
    assignments: list[tuple[int, tuple[int, int], int, int]] = []
    for _dist, rid, sid, slot_idx in ranked:
        slot_key = (int(sid), int(slot_idx))
        if int(rid) in used_red or slot_key in used_slots:
            continue
        used_red.add(int(rid))
        used_slots.add(slot_key)
        red_box = red_infos[int(rid)]["bbox"]
        dock_box = struct_infos[int(sid)]["slots"][int(slot_idx)]["dock_bbox"]
        shift = (int(dock_box[0] - red_box[0]), int(dock_box[1] - red_box[1]))
        assignments.append((int(rid), shift, int(sid), int(slot_idx)))

    assignments.sort(key=lambda x: (int(x[2]), int(x[3]), int(x[0])))
    return assignments, red_infos, struct_infos


def _apply_slide(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    shift_r: int,
    shift_c: int,
    red_val: int,
    black_val: int,
) -> None:
    h, w = out.shape
    steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
    step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
    step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)

    for step in range(0, steps):
        off_r = int(step * step_dr)
        off_c = int(step * step_dc)
        for r, c in coords:
            rr = int(r + off_r)
            cc = int(c + off_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(black_val)

    for r, c in coords:
        rr = int(r + shift_r)
        cc = int(c + shift_c)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(red_val)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multiple correctly sized red lane pieces into a comb structure, while aligned wrong-shape reds still disappear.

    Input structure:
    - The dominant color is the background. Large non-background, non-red components are rotated comb/bracket structures.
    - Red connected components are solid rectangles outside the structures. Several may face the same comb opening.

    Rule:
    - Parse each comb into individual per-tooth docking lanes inside the cavity, recovering both the opening direction and the exact rectangle shape for each lane.
    - A red component docks only if it is a solid rectangle of that exact lane shape and lies on the opening side with the same cross-axis span and a clear straight corridor.
    - Every matched red slides straight into its lane, painting a black trail; all unmatched reds, including aligned wrong-shape decoys, are erased.

    Key insight:
    The object-to-structure relation is now many-to-one and shape-filtered: one comb can accept several separate reds, but line-of-sight alone is insufficient because visible decoys can share the same opening region without fitting a lane.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red = int(Color.RED)
    black = int(Color.BLACK)

    assert red in set(map(int, np.unique(g)))
    assert components_mask(g == red, connectivity=4)

    assignments, red_infos, _struct_infos = _solve_assignments(g, bg)

    out = g.copy()
    out[out == red] = int(bg)

    for rid, (shift_r, shift_c), _sid, _slot_idx in assignments:
        _apply_slide(out, red_infos[int(rid)]["coords"], int(shift_r), int(shift_c), red, black)

    assert out.shape == (h, w)
    return out


def _structure_spec(width: int, teeth: int, rot_k: int, bg: int, sc: int, primary: bool) -> dict:
    patch = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    meta = _analyze_local_structure(patch == sc)
    slots = []
    for slot_idx, dock_bbox_rel in enumerate(meta["slot_bboxes_local"]):
        slots.append(
            {
                "slot_idx": int(slot_idx),
                "dock_bbox_rel": tuple(map(int, dock_bbox_rel)),
                "shape": _bbox_hw(dock_bbox_rel),
                "dir": meta["dir"],
            }
        )
    return {
        "patch": patch,
        "shape": tuple(map(int, patch.shape)),
        "dir": meta["dir"],
        "teeth": meta["teeth"],
        "slots_rel": slots,
        "primary": bool(primary),
    }


def _structure_coords_abs(spec: dict, top: int, left: int, sc: int) -> list[tuple[int, int]]:
    coords = np.argwhere(spec["patch"] == int(sc))
    return [(int(top + r), int(left + c)) for r, c in coords]


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int, sc: int) -> tuple[np.ndarray, list[dict]] | None:
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    ordered = sorted(specs, key=lambda s: np.count_nonzero(s["patch"] != bg), reverse=True)
    for spec in ordered:
        loc = _place_patch(struct_grid, spec["patch"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        top, left = map(int, loc)
        paste(struct_grid, spec["patch"], top, left, transparent=bg)

        coords = _structure_coords_abs(spec, top, left, sc)
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        slots_abs = []
        for slot in spec["slots_rel"]:
            slots_abs.append(
                {
                    "sid": len(placed),
                    "slot_idx": int(slot["slot_idx"]),
                    "dock_bbox": _shift_bbox(slot["dock_bbox_rel"], top, left),
                    "shape": slot["shape"],
                    "dir": spec["dir"],
                }
            )

        placed.append(
            {
                "sid": len(placed),
                "coords": set(coords),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "dir": spec["dir"],
                "teeth": spec["teeth"],
                "slots": slots_abs,
                "primary": bool(spec["primary"]),
            }
        )

    return struct_grid, placed


def _bbox_in_bounds(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _box_is_free(inp: np.ndarray, box: tuple[int, int, int, int], bg: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return bool(np.all(inp[r0 : r1 + 1, c0 : c1 + 1] == int(bg)))


def _box_has_red_gap(red_mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    dilated = _dilate4(red_mask)
    return not bool(dilated[r0 : r1 + 1, c0 : c1 + 1].any())


def _fill_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _slot_box_shift(slot_box: tuple[int, int, int, int], dirv: tuple[int, int], slide: int) -> tuple[int, int, int, int]:
    dr, dc = map(int, dirv)
    return _shift_bbox(slot_box, int(dr * slide), int(dc * slide))


def _visible_decoy_box(slot_box: tuple[int, int, int, int], dirv: tuple[int, int], slide: int) -> tuple[int, int, int, int]:
    shifted = _slot_box_shift(slot_box, dirv, slide)
    return (int(shifted[0]), int(shifted[1]), int(shifted[0]), int(shifted[1]))


def _slot_ray_mask(shape: tuple[int, int], structures: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    ray = np.zeros((h, w), dtype=bool)
    for sinfo in structures:
        for slot in sinfo["slots"]:
            tr0, tc0, tr1, tc1 = map(int, slot["dock_bbox"])
            dr, dc = map(int, slot["dir"])
            if dr == 1:
                ray[tr1 + 1 :, tc0 : tc1 + 1] = True
            elif dr == -1:
                ray[:tr0, tc0 : tc1 + 1] = True
            elif dc == 1:
                ray[tr0 : tr1 + 1, tc1 + 1 :] = True
            else:
                ray[tr0 : tr1 + 1, :tc0] = True
    return ray


def _would_match_any_slot(
    structures: list[dict],
    red_bbox: tuple[int, int, int, int],
    red_shape: tuple[int, int],
    grid: np.ndarray,
    bg: int,
) -> bool:
    for sinfo in structures:
        for slot in sinfo["slots"]:
            if tuple(map(int, red_shape)) != tuple(map(int, slot["shape"])):
                continue
            if _candidate_distance(grid, bg, sinfo, slot, red_bbox) is not None:
                return True
    return False


def _try_place_box(
    inp: np.ndarray,
    box: tuple[int, int, int, int],
    bg: int,
    red_mask: np.ndarray,
) -> bool:
    h, w = inp.shape
    if not _bbox_in_bounds(box, h, w):
        return False
    if not _box_is_free(inp, box, bg):
        return False
    if not _box_has_red_gap(red_mask, box):
        return False
    _fill_box(inp, box, int(Color.RED))
    r0, c0, r1, c1 = map(int, box)
    red_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _render_expected_output(inp: np.ndarray, matched: list[dict], bg: int) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    out[out == int(Color.RED)] = int(bg)
    for item in matched:
        shift_r = int(item["dock_bbox"][0] - item["bbox"][0])
        shift_c = int(item["dock_bbox"][1] - item["bbox"][1])
        _apply_slide(out, item["coords"], shift_r, shift_c, int(Color.RED), int(Color.BLACK))
    return out


def _place_primary_matches(inp: np.ndarray, primary: dict, bg: int, red_mask: np.ndarray) -> tuple[list[dict], dict] | None:
    slots = list(primary["slots"])
    if len(slots) != 3:
        return None

    decoy_slot_idx = random.randrange(3)
    matched_slot_ids = [idx for idx in range(3) if idx != decoy_slot_idx]
    matched: list[dict] = []

    for slot_idx, slide in zip(matched_slot_ids, PRIMARY_MATCH_SLIDES):
        slot = slots[int(slot_idx)]
        box = _slot_box_shift(slot["dock_bbox"], slot["dir"], int(slide))
        if not _try_place_box(inp, box, bg, red_mask):
            return None
        matched.append(
            {
                "sid": int(primary["sid"]),
                "slot_idx": int(slot_idx),
                "bbox": box,
                "dock_bbox": slot["dock_bbox"],
                "coords": _box_coords(box),
            }
        )

    decoy_slot = slots[int(decoy_slot_idx)]
    decoy_box = _visible_decoy_box(decoy_slot["dock_bbox"], decoy_slot["dir"], int(PRIMARY_DECOY_SLIDE))
    if not _try_place_box(inp, decoy_box, bg, red_mask):
        return None
    decoy = {
        "sid": int(primary["sid"]),
        "slot_idx": int(decoy_slot_idx),
        "bbox": decoy_box,
        "shape": (1, 1),
    }
    return matched, decoy


def _place_secondary_matches(inp: np.ndarray, structures: list[dict], bg: int, red_mask: np.ndarray) -> list[dict] | None:
    matched: list[dict] = []
    secondaries = [s for s in structures if not s["primary"]]
    random.shuffle(secondaries)
    for sinfo in secondaries:
        if random.random() < 0.35:
            continue
        slot = random.choice(sinfo["slots"])
        slide = random.randint(*SLIDE_RANGE)
        box = _slot_box_shift(slot["dock_bbox"], slot["dir"], int(slide))
        if not _try_place_box(inp, box, bg, red_mask):
            return None
        matched.append(
            {
                "sid": int(sinfo["sid"]),
                "slot_idx": int(slot["slot_idx"]),
                "bbox": box,
                "dock_bbox": slot["dock_bbox"],
                "coords": _box_coords(box),
            }
        )
    return matched


def _place_background_distractors(inp: np.ndarray, structures: list[dict], bg: int, count: int) -> bool:
    if count <= 0:
        return True

    h, w = inp.shape
    ray_mask = _slot_ray_mask(inp.shape, structures)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for sinfo in structures:
        r0, c0, r1, c1 = map(int, sinfo["bbox"])
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    red_mask = inp == int(Color.RED)
    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        random.shuffle(shapes)
        placed = False
        for rh, rw in shapes:
            candidates: list[tuple[int, int]] = []
            blocked = ray_mask | bbox_blocked | (inp != int(bg)) | _dilate4(red_mask)
            for top in range(0, h - rh + 1):
                for left in range(0, w - rw + 1):
                    if bool(blocked[top : top + rh, left : left + rw].any()):
                        continue
                    box = (int(top), int(left), int(top + rh - 1), int(left + rw - 1))
                    if _would_match_any_slot(structures, box, (int(rh), int(rw)), inp, bg):
                        continue
                    candidates.append((int(top), int(left)))
            if not candidates:
                continue
            top, left = random.choice(candidates)
            box = (int(top), int(left), int(top + rh - 1), int(left + rw - 1))
            _fill_box(inp, box, int(Color.RED))
            red_mask[top : top + rh, left : left + rw] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _first_visible_red_id(grid: np.ndarray, bg: int, slot: dict, red_at: dict[tuple[int, int], int]) -> int | None:
    h, w = grid.shape
    red = int(Color.RED)
    tr0, tc0, tr1, tc1 = map(int, slot["dock_bbox"])
    dr, dc = map(int, slot["dir"])

    if dr == 1:
        rr = int(tr1 + 1)
        while rr < h:
            stripe = grid[rr, tc0 : tc1 + 1]
            if np.any(stripe != int(bg)):
                for cc in range(tc0, tc1 + 1):
                    if int(grid[rr, cc]) == red:
                        return int(red_at[(int(rr), int(cc))])
                return None
            rr += 1
    elif dr == -1:
        rr = int(tr0 - 1)
        while rr >= 0:
            stripe = grid[rr, tc0 : tc1 + 1]
            if np.any(stripe != int(bg)):
                for cc in range(tc0, tc1 + 1):
                    if int(grid[rr, cc]) == red:
                        return int(red_at[(int(rr), int(cc))])
                return None
            rr -= 1
    elif dc == 1:
        cc = int(tc1 + 1)
        while cc < w:
            stripe = grid[tr0 : tr1 + 1, cc]
            if np.any(stripe != int(bg)):
                for rr in range(tr0, tr1 + 1):
                    if int(grid[rr, cc]) == red:
                        return int(red_at[(int(rr), int(cc))])
                return None
            cc += 1
    else:
        cc = int(tc0 - 1)
        while cc >= 0:
            stripe = grid[tr0 : tr1 + 1, cc]
            if np.any(stripe != int(bg)):
                for rr in range(tr0, tr1 + 1):
                    if int(grid[rr, cc]) == red:
                        return int(red_at[(int(rr), int(cc))])
                return None
            cc -= 1
    return None


def _behavior_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    assignments, red_infos, struct_infos = _solve_assignments(g, bg)
    red_at: dict[tuple[int, int], int] = {}
    for rid, info in enumerate(red_infos):
        for rc in info["coords"]:
            red_at[(int(rc[0]), int(rc[1]))] = int(rid)

    matched_per_struct: dict[int, int] = {}
    matched_slots = {(int(sid), int(slot_idx)): int(rid) for rid, _shift, sid, slot_idx in assignments}
    for _rid, _shift, sid, _slot_idx in assignments:
        matched_per_struct[int(sid)] = matched_per_struct.get(int(sid), 0) + 1

    visible_wrong_slots = 0
    for sinfo in struct_infos:
        for slot in sinfo["slots"]:
            rid = _first_visible_red_id(g, bg, slot, red_at)
            key = (int(sinfo["sid"]), int(slot["slot_idx"]))
            if rid is not None and matched_slots.get(key) != int(rid):
                visible_wrong_slots += 1

    return {
        "matched_total": int(len(assignments)),
        "multi_structures": int(sum(v >= 2 for v in matched_per_struct.values())),
        "visible_wrong_slots": int(visible_wrong_slots),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)

    specs = [_structure_spec(PRIMARY_STRUCT[0], PRIMARY_STRUCT[1], random.randrange(4), bg, sc, primary=True)]
    while len(specs) < n_structs:
        width, teeth = random.choice(SECONDARY_STRUCTS)
        specs.append(_structure_spec(width, teeth, random.randrange(4), bg, sc, primary=False))

    placed_result = _place_structures(specs, h, w, bg, sc)
    if placed_result is None:
        return None
    struct_grid, placed = placed_result

    inp = np.array(struct_grid, dtype=int, copy=True)
    red_mask = np.zeros_like(inp, dtype=bool)

    primary = next((s for s in placed if s["primary"]), None)
    if primary is None:
        return None

    primary_result = _place_primary_matches(inp, primary, bg, red_mask)
    if primary_result is None:
        return None
    primary_matched, visible_decoy = primary_result

    secondary_matched = _place_secondary_matches(inp, placed, bg, red_mask)
    if secondary_matched is None:
        return None

    matched = primary_matched + secondary_matched
    matched_count = len(matched)

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched_count)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched_count)
    if min_dist > max_dist or min_dist < VISIBLE_DECOYS_PER_GRID:
        return None
    total_distractors = random.randint(min_dist, max_dist)
    if total_distractors < VISIBLE_DECOYS_PER_GRID:
        return None
    background_distractors = total_distractors - VISIBLE_DECOYS_PER_GRID

    if _would_match_any_slot(placed, visible_decoy["bbox"], visible_decoy["shape"], inp, bg):
        return None

    if not _place_background_distractors(inp, placed, bg, background_distractors):
        return None

    expected = _render_expected_output(inp, matched, bg)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    metrics = _behavior_metrics(inp)
    if metrics["multi_structures"] < 1 or metrics["visible_wrong_slots"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D10/0+D11/0")
