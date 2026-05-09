# Combo variant of 8b9c3697 (index 24)
# combo_spec: D2/3+D3/2
# Name: 空心框架碰撞套环

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D2/3 + D3/2 combo
# ==========================================
# concrete fusion:
# - D2/3 changes the moving source from solid red rectangles into hollow red frames.
# - D3/2 removes the old interior socket. The target is now a solid structure that only
#   provides a collision surface; the frame keeps sliding until the next step would hit it.
# - To make the two changes cooperate rather than sit side by side, each non-red structure
#   is chosen as a protruding bar with a thicker stopper base. The frame hole matches the
#   bar thickness, so the frame can slide along the bar's axis and stop flush against the
#   stopper. The result reads as a hollow collar docking onto a protruding peg.
#
# behavior trace:
# - Original task, opening downward:
#   a solid red rectangle below a comb moved upward into a pre-existing cavity inside the
#   structure bbox, and the trail filled that cavity path in black.
# - This combo:
#   a hollow red frame starts on the open side of a protruding bar and moves straight toward
#   the bar's base. Because there is no socket, the solver does not target a bbox-interior
#   bg cell. It keeps advancing the whole frame until one more step would make the border
#   hit the thicker stopper. The final placement is therefore collision-defined and sits
#   flush against the structure's exposed tip.
# - The difference is visible in two ways at once: the moving object is no longer solid,
#   and the destination is derived from collision geometry rather than from an internal slot.
#
# complexity calibration:
# - Structures use only four repeated frame-hole sizes: 2x2, 2x3, 3x2, 3x3. This keeps the
#   "which frame fits which protrusion" relation visually learnable across a few examples.
# - Distractors are also hollow red frames, so the solver cannot rely on "solid vs hollow"
#   to classify targets. Matching still depends on clean line-of-sight from a structure face.
# - The final output always contains red rings enclosing non-red posts, which is visually
#   very different from the original solid red docking result.
#
# implicit assumptions audit:
# - Old assumption: a valid target cell is a bg socket strictly inside a structure bbox.
#   Broken by D3/2. All socket/bbox-interior logic is removed.
# - Old assumption: the moved red component can be anchored by one face cell that lands on
#   a known socket. Broken by D2/3+D3/2. The new solver instead moves the full component
#   step by step until the next step would collide with structure pixels.
# - New assumption: matched red components have a hole whose thickness matches the matched
#   protruding bar. The generator enforces this by pairing each matched frame's inner size
#   to the bar cross-section and letting the stopper define the collision stop.
# - New assumption: each matched structure sees at most one red component on a clean ray.
#   The generator enforces this by reserving all outward rays from every structure face and
#   rejecting samples whose solver assignment count changes after distractors are added.
#
# placement legality:
# - Output-first generation places full matched patches that already contain structure,
#   black trail, and final red frame, so no later step has to "discover" free space.
# - Distractor frames are placed by reserving their entire bounding boxes, not only their
#   visible red borders. This prevents accidental nesting of one hollow frame around another
#   object through its transparent hole.
# - Structures are padded apart, and the generator validates the final sample with
#   `main(inp) == out`.
#
# observability and cleanliness:
# - In every matched example, the output red component is still hollow and ends flush
#   against a protruding non-red bar. That collar-on-peg pattern is the central cue.
# - Black trail marks only the moving frame border, so the motion path stays legible rather
#   than flooding the interior with black.
# - Because post sizes repeat and directions rotate, the rule is inferable from 2-3 pairs:
#   "find the red frame on a straight ray from a post, slide it until it wraps around it."


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

POST_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))  # hollow-frame hole sizes
DISTRACTOR_HOLES = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 3

TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 300

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR_TO_ROT = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
_SLOT_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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


def _frame_array(inner_h: int, inner_w: int, bg: int, color: int) -> np.ndarray:
    frame = np.full((int(inner_h) + 2, int(inner_w) + 2), int(color), dtype=int)
    frame[1:-1, 1:-1] = int(bg)
    return frame


def _structure_array(inner_h: int, inner_w: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    outer_h = int(inner_h) + 2
    base_w = 2
    stem_len = int(inner_w) + 2
    total_w = base_w + stem_len
    g = np.full((outer_h, total_w), int(bg), dtype=int)
    front = np.zeros((outer_h, total_w), dtype=bool)
    g[:, :base_w] = int(sc)
    g[1:1 + int(inner_h), base_w:base_w + stem_len] = int(sc)
    front[1:1 + int(inner_h), base_w + stem_len - 1] = True
    return g, front


def _collect_red_data(
    g: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
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
]:
    red_coords, red_bbox, red_at = _collect_red_data(g)
    struct_coords, struct_bbox, _struct_mask = _collect_structures(g, bg)

    if not red_coords or not struct_coords:
        return red_coords, red_bbox, struct_coords, struct_bbox, None

    h, w = g.shape
    red_val = int(Color.RED)
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], int]] = []

    for sid, pts in enumerate(struct_coords):
        best = None
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
        if best is not None:
            struct_best.append(best)

    return red_coords, red_bbox, struct_coords, struct_bbox, struct_best


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
    red_coords, _red_bbox, _struct_coords, _struct_bbox, struct_best = _visibility_info(g, bg)
    assert red_coords, "expected at least one red component"
    assert struct_best is not None, "expected at least one non-bg non-red structure"

    _struct_coords2, _struct_bbox2, struct_mask = _collect_structures(g, bg)
    scored = list(struct_best)
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
    Slide visible hollow red frames straight toward protruding structures until the next step would collide, leaving a black outline trail.

    Input structure:
    - bg is the most common color. Non-bg, non-red components are protruding peg/bar structures in one shared color.
    - Red components are hollow rectangular frames. Some lie on a clear axis-aligned ray from a post face; the rest are distractors.

    Rule:
    - For each structure, cast rays outward from its exposed tip to find a visible red frame.
    - Move that frame back toward the structure along the same axis, one cell at a time, stopping at the last placement that does not overlap the structure.
    - Paint every intermediate frame placement black and erase every red frame that was not matched.

    Key insight: The destination is determined by collision geometry, and because the moving object is hollow, the stop position is the last flush collar placement before the frame border would hit the stopper/base.
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


def _matched_patches(inner_h: int, inner_w: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red_val = int(Color.RED)

    frame_red = _frame_array(inner_h, inner_w, bg, red_val)
    base_structure, front = _structure_array(inner_h, inner_w, bg, sc)
    outer_h, outer_w = frame_red.shape
    struct_h, struct_w = base_structure.shape
    patch_w = int(struct_w) + int(slide) + int(outer_w)

    structure = np.full((struct_h, patch_w), int(bg), dtype=int)
    input_patch = np.full((struct_h, patch_w), int(bg), dtype=int)
    front_mask = np.zeros((struct_h, patch_w), dtype=bool)

    paste(structure, base_structure, 0, 0, transparent=bg)
    paste(input_patch, base_structure, 0, 0, transparent=bg)
    front_mask[: front.shape[0], : front.shape[1]] = front

    paste(input_patch, frame_red, 0, int(struct_w + slide), transparent=bg)
    output_patch = main(input_patch)

    assert len(_solver_assignments(input_patch, bg)[2]) == 1
    assert np.any(output_patch == int(Color.BLACK))
    assert np.any(output_patch == red_val)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "front_mask": np.rot90(front_mask, k=rot_k),
        "front_dir": _rotate_dir((0, 1), rot_k),
    }


def _unmatched_patches(inner_h: int, inner_w: int, rot_k: int, bg: int, sc: int) -> dict:
    structure, front = _structure_array(inner_h, inner_w, bg, sc)
    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(structure, k=rot_k),
        "output": np.rot90(structure, k=rot_k),
        "front_mask": np.rot90(front, k=rot_k),
        "front_dir": _rotate_dir((0, 1), rot_k),
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


def _slot_candidates(slot: str, patch_h: int, patch_w: int, h: int, w: int) -> list[tuple[int, int]]:
    patch_h = int(patch_h)
    patch_w = int(patch_w)
    if patch_h > h or patch_w > w:
        return []

    candidates: list[tuple[int, int]] = []
    if slot == "top":
        r_vals = range(0, min(2, h - patch_h) + 1)
        for r0 in r_vals:
            for c0 in range(0, w - patch_w + 1):
                candidates.append((int(r0), int(c0)))
    elif slot == "bottom":
        base = h - patch_h
        r_lo = max(0, base - 2)
        for r0 in range(r_lo, base + 1):
            for c0 in range(0, w - patch_w + 1):
                candidates.append((int(r0), int(c0)))
    elif slot == "left":
        c_vals = range(0, min(2, w - patch_w) + 1)
        for c0 in c_vals:
            for r0 in range(0, h - patch_h + 1):
                candidates.append((int(r0), int(c0)))
    else:
        base = w - patch_w
        c_lo = max(0, base - 2)
        for c0 in range(c_lo, base + 1):
            for r0 in range(0, h - patch_h + 1):
                candidates.append((int(r0), int(c0)))

    random.shuffle(candidates)
    return candidates


def _place_structures(
    specs: list[tuple[str, dict]], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    struct_grid = np.full((h, w), int(bg), dtype=int)
    struct_blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for slot, spec in specs:
        loc = None
        for cand in _slot_candidates(slot, spec["output"].shape[0], spec["output"].shape[1], h, w):
            r0, c0 = cand
            patch = spec["output"]
            view = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
            if bool(((view != int(bg)) & (patch != int(bg))).any()):
                continue
            struct_patch = spec["structure"]
            struct_rows, struct_cols = np.where(struct_patch == int(sc))
            if len(struct_rows) == 0:
                continue
            blocked_hit = False
            for rr, cc in zip(struct_rows, struct_cols):
                if bool(struct_blocked[int(r0 + rr), int(c0 + cc)]):
                    blocked_hit = True
                    break
            if blocked_hit:
                continue
            loc = cand
            break
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == int(sc))
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        front_arr = np.argwhere(spec["front_mask"])
        front_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in front_arr]

        min_r = max(0, min(rows) - STRUCT_PADDING)
        max_r = min(h - 1, max(rows) + STRUCT_PADDING)
        min_c = max(0, min(cols) - STRUCT_PADDING)
        max_c = min(w - 1, max(cols) + STRUCT_PADDING)
        struct_blocked[min_r : max_r + 1, min_c : max_c + 1] = True

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "front_coords": front_coords,
                "front_dir": spec["front_dir"],
            }
        )

    return inp, out, struct_grid, placed


def _reserve_red_boxes(inp: np.ndarray) -> np.ndarray:
    blocked = np.zeros(inp.shape, dtype=bool)
    for coords in components_mask(inp == int(Color.RED), connectivity=4):
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    return blocked


def _place_distractors(inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    red_box_blocked = _reserve_red_boxes(inp)
    holes = list(DISTRACTOR_HOLES)

    for _ in range(int(count)):
        blocked = forbidden | bbox_blocked | _dilate4(red_box_blocked) | (inp != int(bg))
        canvas = np.full((h, w), int(bg), dtype=int)
        canvas[blocked] = int(sc)
        random.shuffle(holes)
        ok = False
        for inner_h, inner_w in holes:
            frame = _frame_array(inner_h, inner_w, bg, int(Color.RED))
            solid_box = np.full(frame.shape, int(Color.RED), dtype=int)
            loc = _place_patch(canvas, solid_box, bg, padding=0)
            if loc is None:
                continue
            paste(inp, frame, loc[0], loc[1], transparent=bg)
            red_box_blocked[loc[0] : loc[0] + frame.shape[0], loc[1] : loc[1] + frame.shape[1]] = True
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

    slots = random.sample(list(_SLOT_TO_DIR.keys()), n_structs)
    specs: list[tuple[str, dict]] = []
    for idx, slot in enumerate(slots):
        inner_h, inner_w = random.choice(POST_SHAPES)
        rot_k = _DIR_TO_ROT[_SLOT_TO_DIR[slot]]
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append((slot, _matched_patches(inner_h, inner_w, slide, rot_k, bg, sc)))
        else:
            specs.append((slot, _unmatched_patches(inner_h, inner_w, rot_k, bg, sc)))

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s[1]["output"] != bg), reverse=True)

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
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo 24")
