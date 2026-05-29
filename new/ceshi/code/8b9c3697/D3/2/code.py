# Variant of 8b9c3697
# Dimension: D3 — 目标位置：结构类型与插槽 (dst)
# Change: 2-3个梳齿/支架结构，每个有4行深度（标准化形式：后壁+齿+侧壁+空腔），开口方向为上下左右之一，插槽宽度3-5格... → 碰撞停止（无预设插槽）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D3 "collision_stop" variant
# ====================================================
# source_task note:
# - The prompt cites task `195c6913` as the inspiration for the collision-stop idea.
# - A repository search in this workspace found the ARC JSON examples for `195c6913`,
#   but no local solver `.py` file to inspect. The implementation below therefore uses
#   the provided variant description plus the source task's visible I/O behavior as the
#   baseline for the "stop when the next step would collide" mechanic.
#
# spec:
#   preserved_from_original:
#     grid: 19-23 rows x 22-27 cols
#     colors:
#       background: one non-black, non-red color
#       structures: one shared non-bg, non-red, non-black color
#       moving pieces: red
#       trail: black
#     counts:
#       structures: 2-3
#       total_red_components: 5-11
#       distractors: 2-8
#     behavior:
#       matched reds move in a straight axis-aligned line
#       traversed cells turn black
#       unmatched reds disappear
#   changed_core:
#     old_docking:
#       structure had an internal cavity/specified socket
#       the red ended inside the structure bbox
#     new_docking:
#       there is no internal socket to target
#       each structure has an exposed "front face" made of solid teeth/posts
#       a matched red slides toward that face and stops at the last collision-free placement
#       the final red remains outside the structure bbox, flush against the structure
#   concrete_objects:
#     normalized_structure_opening_down:
#       row0: full back wall
#       row1: two side posts plus a centered solid tooth block
#       row2: two side posts plus the same centered tooth block
#       row3: only the centered tooth block
#     width/span options:
#       width 7 with contact span 3
#       width 8 with contact span 4
#       width 9 with contact span 3 or 5
#     matched_red_shape:
#       a 2 x span red rectangle aligned with the centered tooth block
#   behavior_trace:
#     example:
#       - Opening faces downward.
#       - Original task: a red rectangle below the structure would be aligned to a bg socket
#         inside rows 2-3 of the bbox.
#       - Variant: rows 1-3 in the center are solid, so the same red can no longer enter.
#         It moves upward until one more step would overlap row 3, then stops directly
#         below the front teeth.
#       - Difference: the stop cell is derived from collision with occupied structure
#         pixels, not from a precomputed socket location.
#   complexity_calibration:
#     - The structure is simpler than the original cavity comb, but the solver still has to
#       infer both the approach direction and the stopping distance from geometry.
#     - Contact spans are limited to 3/4/5 columns, repeated across examples, so the task
#       is visually learnable instead of exploding into many unrelated free parameters.
#     - Structures may still be unmatched, and extra red distractors remain, so the task
#       does not collapse into a trivial "move every red to the nearest structure" rule.
#   implicit_assumptions_audit:
#     - Old solver assumption: candidate socket cells were bg cells strictly inside a
#       structure bbox. This no longer holds, so all bbox-interior socket logic is removed.
#     - Old solver assumption: the final target was a single socket cell adjacent to an
#       anchor. This is replaced by iterative motion until the next step would collide.
#     - Old generator assumption: the output patch could be built by dropping the red into
#       the cavity. This is replaced by building a patch where the red docks outside.
#     - Old distractor safety rule: avoid rays from interior sockets. New rule: avoid the
#       visible rays cast outward from each structure's intended front face, and reject any
#       sample where solver-style visibility becomes ambiguous from any side.
#   placement_legality:
#     - Output-first generation reserves the full footprint of structure + trail + final red.
#     - Structures are placed with padding, so trails and final reds from different matches
#       do not overlap.
#     - Distractors are only placed on bg, away from reserved structure bboxes, away from
#       front-face rays, and away from existing red components.
#   observability:
#     - In every matched case, the final red ends outside the structure bbox rather than
#       inside it. This makes the D3 change directly visible in few-shot ARC examples.
#     - The approach direction still varies over the four cardinal rotations, so the
#       "collision stop" rule is supported by repeated rotated evidence.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_SPAN = ((7, 3), (8, 4), (9, 3), (9, 5))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 2

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 300

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_dir(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(int(k) % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _collect_red_data(g: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_val = int(Color.RED)
    red_comps = components_mask(g == red_val, connectivity=4)
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


def _collect_structures(g: np.ndarray, bg: int) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]], np.ndarray]:
    red_val = int(Color.RED)
    struct_mask = (g != int(bg)) & (g != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)
    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox, struct_mask


def _visibility_info(
    g: np.ndarray,
    bg: int,
) -> tuple[
    list[list[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[set[tuple[int, int]]],
    list[tuple[int, int, int, int]],
    list[tuple[tuple[int, int], int, tuple[int, int], int]] | None,
    list[set[int]],
]:
    red_coords, red_bbox, red_at = _collect_red_data(g)
    struct_coords, struct_bbox, _struct_mask = _collect_structures(g, bg)

    if not red_coords or not struct_coords:
        return red_coords, red_bbox, struct_coords, struct_bbox, None, [set() for _ in struct_coords]

    h, w = g.shape
    red_val = int(Color.RED)
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], int]] = []
    struct_visible: list[set[int]] = []

    for sid, pts in enumerate(struct_coords):
        best = None
        visible_rids: set[int] = set()
        for dr, dc in _DIRS:
            hits: dict[int, list[int]] = defaultdict(lambda: [0, 10**9])
            for r, c in pts:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == red_val:
                    rid = int(red_at[(int(rr), int(cc))])
                    visible_rids.add(int(rid))
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    rec[1] = min(int(rec[1]), int(dist))
            if not hits:
                continue
            rid_best = max(hits, key=lambda rid: (int(hits[rid][0]), -int(hits[rid][1])))
            cnt, mind = hits[rid_best]
            cand = ((int(cnt), -int(mind)), int(rid_best), (int(dr), int(dc)), int(sid))
            if best is None or cand > best:
                best = cand
        struct_best.append(best)
        struct_visible.append(visible_rids)

    return red_coords, red_bbox, struct_coords, struct_bbox, struct_best, struct_visible


def _move_until_collision(
    coords: list[tuple[int, int]],
    move_vec: tuple[int, int],
    struct_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], int]:
    h, w = struct_mask.shape
    mv_r, mv_c = map(int, move_vec)
    cur = [(int(r), int(c)) for r, c in coords]
    steps = 0
    while True:
        nxt = [(int(r + mv_r), int(c + mv_c)) for r, c in cur]
        blocked = False
        for rr, cc in nxt:
            if not (0 <= rr < h and 0 <= cc < w):
                blocked = True
                break
            if bool(struct_mask[rr, cc]):
                blocked = True
                break
        if blocked:
            return cur, int(steps)
        cur = nxt
        steps += 1


def _solver_assignments(g: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], np.ndarray, list[tuple[int, tuple[int, int]]]]:
    red_coords, _red_bbox, _struct_coords, _struct_bbox, struct_best, _visible = _visibility_info(g, bg)
    assert red_coords, "expected at least one red component"
    assert struct_best is not None, "expected at least one non-bg non-red structure"

    _struct_coords2, _struct_bbox2, struct_mask = _collect_structures(g, bg)
    scored = [item for item in struct_best if item is not None]
    scored.sort(reverse=True, key=lambda t: t[0])

    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int]]] = []
    for _score, rid, dirv, _sid in scored:
        if int(rid) in used_red:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1]))))
    return red_coords, struct_mask, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide visible red rectangles toward protruding structures until the next step would collide, leaving a black trail.

    Input structure:
    - bg is the most common color. Large non-bg, non-red structures are solid comb/bracket-like shapes with an exposed face rather than an internal socket.
    - Red components are rectangular blobs placed outside the structures; some are aligned with a structure face, others are distractors.

    Rule:
    - For each structure, cast straight rays outward from exposed structure faces to find a visible red component.
    - Move that red component back toward the structure along the same axis, stopping at the last position that does not overlap the structure.
    - Paint all intermediate occupied cells black, and erase every red component that was not matched to a structure.

    Key insight: The destination is no longer a predefined cavity cell; it is the collision-determined final placement immediately before the moving red would hit the structure.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    assert red_val in set(map(int, np.unique(g)))

    red_coords, struct_mask, assignments = _solver_assignments(g, bg)

    out = g.copy()
    out[out == red_val] = int(bg)

    for rid, dirv in assignments:
        coords = red_coords[int(rid)]
        move_vec = (-int(dirv[0]), -int(dirv[1]))
        final_coords, steps = _move_until_collision(coords, move_vec, struct_mask)

        for step in range(int(steps)):
            off_r = int(step * move_vec[0])
            off_c = int(step * move_vec[1])
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = black_val

        for rr, cc in final_coords:
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = red_val

    return out


def _structure_array(width: int, span: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.full((4, width), bg, dtype=int)
    front = np.zeros((4, width), dtype=bool)
    start = (int(width) - int(span)) // 2

    g[0, :] = sc
    g[1, 1] = g[2, 1] = sc
    g[1, width - 2] = g[2, width - 2] = sc
    g[1:4, start:start + span] = sc
    front[3, start:start + span] = True
    return g, front


def _matched_patches(width: int, span: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)
    red_h = 2

    base, front = _structure_array(width, span, bg, sc)
    patch_h = 4 + int(slide) + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    front_mask = np.zeros((patch_h, width), dtype=bool)
    paste(structure, base, 0, 0, transparent=bg)
    paste(front_mask, front.astype(int), 0, 0, transparent=0)
    front_mask = front_mask.astype(bool)

    input_patch = structure.copy()
    output_patch = structure.copy()
    start = (int(width) - int(span)) // 2

    for off in range(1, int(slide) + 1):
        output_patch[4 + off:4 + off + red_h, start:start + span] = black_val
    output_patch[4:4 + red_h, start:start + span] = red_val
    input_patch[4 + slide:4 + slide + red_h, start:start + span] = red_val

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "front_mask": np.rot90(front_mask, k=rot_k),
        "front_dir": _rotate_dir((1, 0), rot_k),
    }


def _unmatched_patches(width: int, span: int, rot_k: int, bg: int, sc: int) -> dict:
    structure, front = _structure_array(width, span, bg, sc)
    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(structure, k=rot_k),
        "output": np.rot90(structure, k=rot_k),
        "front_mask": np.rot90(front, k=rot_k),
        "front_dir": _rotate_dir((1, 0), rot_k),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        dr, dc = item["front_dir"]
        for r, c in item["front_coords"]:
            rr, cc = int(r + dr), int(c + dc)
            while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                forbidden[rr, cc] = True
                rr += int(dr)
                cc += int(dc)
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _assignment_count_matches(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
    try:
        _red_coords, _struct_mask, assignments = _solver_assignments(grid, bg)
    except AssertionError:
        return False
    return len(assignments) == int(expected_matched)


def _place_structures(specs: list[dict], h: int, w: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
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
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        front_arr = np.argwhere(spec["front_mask"])
        front_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in front_arr]

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "front_coords": front_coords,
                "front_dir": spec["front_dir"],
            }
        )

    return inp, out, struct_grid, placed


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True

    red_mask = inp == int(Color.RED)
    shapes = list(DISTRACTOR_SHAPES)

    for _ in range(int(count)):
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
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = int(n_structs - unmatched)
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for idx in range(n_structs):
        width, span = random.choice(STRUCT_WIDTH_SPAN)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, span, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, span, rot_k, bg, sc))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _assignment_count_matches(inp, bg, matched):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    if not _assignment_count_matches(inp, bg, matched):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D3/2")
