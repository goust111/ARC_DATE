# Combo variant of 8b9c3697 (index 12)
# combo_spec: D5/1+D11/0
# Name: 视线内干扰保留

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    Original behavior:
#    - A comb structure searched outward from its interior slot and the generator
#      guaranteed that only the true matching red sat on that ray.
#    - Any red not chosen for docking was erased in the output.
#    Combo behavior:
#    - A red rectangle may sit directly on the opening ray yet still be invalid
#      because its rectangle shape does not equal the slot implied by the teeth.
#    - Such invalid reds, including the visible aligned ones, remain red in place.
#    Trace:
#    - Suppose a downward-opening structure expects a 2x2 docked rectangle.
#    - A true 2x2 red sits lower on the same columns and moves upward into the slot,
#      leaving a black trail.
#    - Another nearby structure has a 1x3 opening span, but the red on its ray is
#      only 1x2. The old D11-only behavior would reject it yet erase it; this combo
#      still rejects it, but keeps it visible in the output. That is the new joint
#      effect: shape-aware filtering plus preservation of the rejected red.
# b) Complexity calibration
#    The variant uses one forced unmatched structure and one forced visible decoy
#    per grid. This is enough to make the rule observable in 2-4 examples without
#    flooding the board with many special cases. Shapes still come from the small
#    original menu, so humans can compare repeated slot widths across structures.
# c) Implicit-assumption audit of the original main()
#    - "First visible red is a valid candidate" no longer holds because a visible
#      decoy can occupy the opening line.
#    - "Unmatched reds can be cleared globally before redrawing matches" no longer
#      holds because preserving rejected reds is part of the rule.
#    - "Any aligned component can be shifted into the slot" no longer holds; the
#      component must be a solid rectangle of the exact inferred size.
#    The solver therefore parses each structure, infers its dock rectangle, matches
#    only exact-shape rectangles on the opening side with a clear corridor, and
#    starts output from the full input grid so unmatched reds survive.
# d) Placement legality
#    Structures still use padding and never overlap. The forced visible decoy is
#    placed outside the structure bbox, on the opening side, with a pure background
#    corridor from the slot to the decoy. Other distractors avoid structure bboxes,
#    avoid touching other reds, and are rejected if they would accidentally become
#    valid matches for any structure.
# e) Visual observability
#    Every generated sample contains at least one red rectangle directly in front of
#    an unmatched opening that stays red in the output. In the same grid, at least
#    one matched red moves and paints black. This contrast makes the combo readable:
#    some aligned reds move, some aligned reds do not, and the non-movers persist.
# f) Visual cleanliness
#    Only one mandatory visible decoy is forced, with 4-neighborhood spacing from
#    other reds. Remaining distractors mostly stay away from the opening rays, so
#    the board still shows clear cavities, clear trails, and a small number of
#    preserved red non-movers rather than dense clutter around every structure.
# g) Color reasoning
#    No extra color semantics are introduced. Red means candidate movable matter,
#    black marks traveled path, and the structure color is unchanged. The inference
#    burden comes from geometry and persistence, not from hidden color remapping.
# h) New solver assumptions and generator guarantees
#    - Structures remain exact rotations of the known comb template.
#    - True movers are still solid rectangles with bbox equal to the inferred dock
#      shape, and each valid mover has an all-background corridor to its slot.
#    - Visible decoys intersect an opening ray but intentionally fail shape match.
#    - Paths of true movers do not cross preserved distractors. The generator enforces
#      these conditions and rejects any sample where the shape-aware match count or
#      visibility signature is wrong.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4
UNMATCHED_STRUCTURES = 1

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8
VISIBLE_DECOYS_PER_GRID = 1

MAX_RETRIES = 300


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(box[2] - box[0] + 1), int(box[3] - box[1] + 1)


def _shift_bbox(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return int(box[0] + dr), int(box[1] + dc), int(box[2] + dr), int(box[3] + dc)


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

    teeth = int(len(tooth_cols))
    start = int(tooth_cols[0])
    red_h = int(min(teeth, 2))

    dock_canonical = np.zeros_like(canonical, dtype=bool)
    dock_canonical[2:2 + red_h, start:start + teeth] = True
    dock_local = np.rot90(dock_canonical, k=(4 - rot_to_base) % 4)
    dock_bbox = tuple(map(int, bbox_coords(np.argwhere(dock_local))))

    return {
        "dir": open_dir,
        "shape": _bbox_hw(dock_bbox),
        "dock_bbox_local": dock_bbox,
    }


def _red_infos(grid: np.ndarray) -> list[dict]:
    red = int(Color.RED)
    infos: list[dict] = []
    for coords in components_mask(grid == red, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        h, w = _bbox_hw(box)
        infos.append(
            {
                "coords": pts,
                "bbox": box,
                "shape": (h, w),
                "is_rect": len(pts) == h * w,
            }
        )
    return infos


def _structure_infos(grid: np.ndarray, bg: int) -> list[dict]:
    red = int(Color.RED)
    struct_mask = (grid != bg) & (grid != red)
    infos: list[dict] = []
    for sid, coords in enumerate(components_mask(struct_mask, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = box
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[r - r0, c - c0] = True
        meta = _analyze_local_structure(local_mask)
        infos.append(
            {
                "sid": int(sid),
                "coords": set(pts),
                "bbox": box,
                "dir": meta["dir"],
                "shape": meta["shape"],
                "dock_bbox": _shift_bbox(meta["dock_bbox_local"], r0, c0),
            }
        )
    return infos


def _candidate_distance(
    grid: np.ndarray,
    bg: int,
    struct_info: dict,
    red_bbox: tuple[int, int, int, int],
) -> int | None:
    sr0, sc0, sr1, sc1 = struct_info["bbox"]
    tr0, tc0, tr1, tc1 = struct_info["dock_bbox"]
    rr0, rc0, rr1, rc1 = red_bbox
    dr, dc = struct_info["dir"]

    if dr == 1:
        if rc0 != tc0 or rc1 != tc1 or rr0 <= sr1:
            return None
        corridor = grid[tr1 + 1:rr0, tc0:tc1 + 1]
    elif dr == -1:
        if rc0 != tc0 or rc1 != tc1 or rr1 >= sr0:
            return None
        corridor = grid[rr1 + 1:tr0, tc0:tc1 + 1]
    elif dc == 1:
        if rr0 != tr0 or rr1 != tr1 or rc0 <= sc1:
            return None
        corridor = grid[tr0:tr1 + 1, tc1 + 1:rc0]
    else:
        if rr0 != tr0 or rr1 != tr1 or rc1 >= sc0:
            return None
        corridor = grid[tr0:tr1 + 1, rc1 + 1:tc0]

    if corridor.size and np.any(corridor != bg):
        return None
    return int(abs(rr0 - tr0) + abs(rc0 - tc0))


def _solve_assignments(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, tuple[int, int], int]], list[dict], list[dict]]:
    red_infos = _red_infos(grid)
    struct_infos = _structure_infos(grid, bg)

    ranked: list[tuple[int, int, int]] = []
    for sinfo in struct_infos:
        best: tuple[int, int, int] | None = None
        for rid, rinfo in enumerate(red_infos):
            if not rinfo["is_rect"] or rinfo["shape"] != sinfo["shape"]:
                continue
            dist = _candidate_distance(grid, bg, sinfo, rinfo["bbox"])
            if dist is None:
                continue
            cand = (int(dist), int(rid), int(sinfo["sid"]))
            if best is None or cand < best:
                best = cand
        if best is not None:
            ranked.append(best)

    ranked.sort()
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], int]] = []
    for _dist, rid, sid in ranked:
        if rid in used_red:
            continue
        used_red.add(rid)
        red_box = red_infos[rid]["bbox"]
        dock_box = struct_infos[sid]["dock_bbox"]
        shift = (int(dock_box[0] - red_box[0]), int(dock_box[1] - red_box[1]))
        assignments.append((int(rid), shift, int(sid)))
    return assignments, red_infos, struct_infos


def _visible_from_structure(
    grid: np.ndarray,
    bg: int,
    struct_info: dict,
    red_bbox: tuple[int, int, int, int],
) -> bool:
    tr0, tc0, tr1, tc1 = struct_info["dock_bbox"]
    rr0, rc0, rr1, rc1 = red_bbox
    dr, dc = struct_info["dir"]

    if dr == 1:
        if rr0 <= tr1:
            return False
        lo = max(tc0, rc0)
        hi = min(tc1, rc1)
        for col in range(lo, hi + 1):
            if np.all(grid[tr1 + 1:rr0, col] == bg):
                return True
        return False
    if dr == -1:
        if rr1 >= tr0:
            return False
        lo = max(tc0, rc0)
        hi = min(tc1, rc1)
        for col in range(lo, hi + 1):
            if np.all(grid[rr1 + 1:tr0, col] == bg):
                return True
        return False
    if dc == 1:
        if rc0 <= tc1:
            return False
        lo = max(tr0, rr0)
        hi = min(tr1, rr1)
        for row in range(lo, hi + 1):
            if np.all(grid[row, tc1 + 1:rc0] == bg):
                return True
        return False
    if rc1 >= tc0:
        return False
    lo = max(tr0, rr0)
    hi = min(tr1, rr1)
    for row in range(lo, hi + 1):
        if np.all(grid[row, rc1 + 1:tc0] == bg):
            return True
    return False


def _visible_invalid_count(grid: np.ndarray, bg: int) -> int:
    assignments, red_infos, struct_infos = _solve_assignments(grid, bg)
    matched_rids = {rid for rid, _shift, _sid in assignments}
    count = 0
    for rid, rinfo in enumerate(red_infos):
        if rid in matched_rids:
            continue
        if any(_visible_from_structure(grid, bg, sinfo, rinfo["bbox"]) for sinfo in struct_infos):
            count += 1
    return int(count)


def _erase_unmatched_reference(grid: np.ndarray, bg: int) -> np.ndarray:
    assignments, red_infos, _struct_infos = _solve_assignments(grid, bg)
    out = np.array(grid, dtype=int)
    red = int(Color.RED)
    black = int(Color.BLACK)
    h, w = out.shape
    out[out == red] = bg

    for rid, (shift_r, shift_c), _sid in assignments:
        coords = red_infos[rid]["coords"]
        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = black

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = red
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only shape-compatible red rectangles into comb sockets, while leaving every rejected red where it started.

    Input structure:
    - The dominant background surrounds several large comb/bracket structures made of one non-red, non-black color.
    - Red connected components are solid rectangles outside the structures; some are true movers and some are visible decoys on an opening ray.

    Rule:
    - Parse each structure to recover its opening direction and the exact dock rectangle implied by the teeth.
    - A red component docks only if it is a solid rectangle of that exact shape, sits on the opening side with the same cross-axis span, and has a clear background corridor to the slot.
    - Matched reds slide straight into the slot and paint their swept path black; every unmatched red, including visible aligned decoys, remains red in its original position.

    Key insight:
    A red block being visible from a cavity is no longer sufficient and disappearing is no longer guaranteed: the solver must reject false positives by shape, then preserve those rejected reds instead of erasing them.
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
    for rid, (shift_r, shift_c), _sid in assignments:
        coords = red_infos[rid]["coords"]
        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = black

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = red

    return out


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = black
    output_patch[2:2 + red_h, start:start + teeth] = red
    input_patch[2 + slide:2 + slide + red_h, start:start + teeth] = red

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "matched": True,
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "matched": False,
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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


def _shape_aware_match_count(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos_unused, _struct_infos_unused = _solve_assignments(grid, bg)
    return len(assignments)


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        lr0, lc0, lr1, lc1 = map(int, bbox_coords(coords_arr))
        local_mask = spec["structure"][lr0:lr1 + 1, lc0:lc1 + 1] == sc
        meta = _analyze_local_structure(local_mask)

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "dir": meta["dir"],
                "shape": meta["shape"],
                "dock_bbox": _shift_bbox(meta["dock_bbox_local"], r0 + lr0, c0 + lc0),
                "matched": bool(spec["matched"]),
            }
        )

    return inp, out, struct_grid, placed


def _visible_decoy_shapes(item: dict) -> list[tuple[int, int]]:
    dh, dw = item["shape"]
    dr, dc = item["dir"]
    out: list[tuple[int, int]] = []
    for rh, rw in DISTRACTOR_SHAPES:
        if (rh, rw) == (dh, dw):
            continue
        if dr != 0 and rw <= dw:
            out.append((int(rh), int(rw)))
        if dc != 0 and rh <= dh:
            out.append((int(rh), int(rw)))
    return out


def _strict_bg_path_ok(
    inp: np.ndarray,
    item: dict,
    top: int,
    left: int,
    rh: int,
    rw: int,
    bg: int,
) -> bool:
    tr0, tc0, tr1, tc1 = item["dock_bbox"]
    dr, dc = item["dir"]

    if dr == 1:
        path = inp[tr1 + 1:top, left:left + rw]
    elif dr == -1:
        path = inp[top + rh:tr0, left:left + rw]
    elif dc == 1:
        path = inp[top:top + rh, tc1 + 1:left]
    else:
        path = inp[top:top + rh, left + rw:tc0]
    return (not path.size) or bool(np.all(path == bg))


def _would_match_any_structure(
    structures: list[dict],
    red_bbox: tuple[int, int, int, int],
    red_shape: tuple[int, int],
    grid: np.ndarray,
    bg: int,
) -> bool:
    for item in structures:
        if red_shape != item["shape"]:
            continue
        if _candidate_distance(grid, bg, item, red_bbox) is not None:
            return True
    return False


def _place_visible_decoy(inp: np.ndarray, out: np.ndarray, placed: list[dict], bg: int) -> bool:
    red_mask = inp == int(Color.RED)
    blocked_adj = _dilate4(red_mask)
    h, w = inp.shape

    unmatched = [item for item in placed if not item["matched"]]
    random.shuffle(unmatched)

    for item in unmatched:
        sr0, sc0, sr1, sc1 = item["bbox"]
        tr0, tc0, tr1, tc1 = item["dock_bbox"]
        shapes = _visible_decoy_shapes(item)
        random.shuffle(shapes)
        options: list[tuple[int, int, int, int]] = []

        for rh, rw in shapes:
            if item["dir"][0] != 0:
                for left in range(tc0, tc1 - rw + 2):
                    rows = range(sr1 + 2, h - rh + 1) if item["dir"][0] == 1 else range(0, sr0 - rh)
                    for top in rows:
                        if top < 0 or top + rh > h:
                            continue
                        if np.any(inp[top:top + rh, left:left + rw] != bg):
                            continue
                        if blocked_adj[top:top + rh, left:left + rw].any():
                            continue
                        if not _strict_bg_path_ok(inp, item, top, left, rh, rw, bg):
                            continue
                        box = (top, left, top + rh - 1, left + rw - 1)
                        if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                            continue
                        options.append((top, left, rh, rw))
            else:
                for top in range(tr0, tr1 - rh + 2):
                    cols = range(sc1 + 2, w - rw + 1) if item["dir"][1] == 1 else range(0, sc0 - rw)
                    for left in cols:
                        if left < 0 or left + rw > w:
                            continue
                        if np.any(inp[top:top + rh, left:left + rw] != bg):
                            continue
                        if blocked_adj[top:top + rh, left:left + rw].any():
                            continue
                        if not _strict_bg_path_ok(inp, item, top, left, rh, rw, bg):
                            continue
                        box = (top, left, top + rh - 1, left + rw - 1)
                        if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                            continue
                        options.append((top, left, rh, rw))

        if options:
            top, left, rh, rw = random.choice(options)
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            paste(inp, patch, top, left, transparent=bg)
            paste(out, patch, top, left, transparent=bg)
            return True
    return False


def _place_background_distractors(
    inp: np.ndarray, out: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int
) -> bool:
    if count <= 0:
        return True

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
            top, left = loc
            box = (top, left, top + rh - 1, left + rw - 1)
            if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                continue
            paste(inp, patch, top, left, transparent=bg)
            paste(out, patch, top, left, transparent=bg)
            red_mask[top:top + rh, left:left + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = UNMATCHED_STRUCTURES
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist or min_dist < VISIBLE_DECOYS_PER_GRID:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if _shape_aware_match_count(inp, bg) != matched:
        return None

    for _ in range(VISIBLE_DECOYS_PER_GRID):
        if not _place_visible_decoy(inp, out, placed, bg):
            return None

    remaining = distractor_count - VISIBLE_DECOYS_PER_GRID
    if not _place_background_distractors(inp, out, struct_grid, placed, bg, sc, remaining):
        return None

    if _shape_aware_match_count(inp, bg) != matched:
        return None
    if _visible_invalid_count(inp, bg) < VISIBLE_DECOYS_PER_GRID:
        return None
    if np.array_equal(main(inp), _erase_unmatched_reference(inp, bg)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo 8b9c3697 D5/1+D11/0")
