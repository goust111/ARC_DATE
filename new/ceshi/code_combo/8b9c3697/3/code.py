# Combo variant of 8b9c3697 (index 3)
# combo_spec: D1/3+D3/2
# Name: 最近距离碰撞停止

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo "nearest-distance collision-stop" variant
# ======================================================================
# a) Concretization and behavior trace:
#    - The source task had comb/bracket structures with an internal socket. A red rectangle only matched when
#      it lay exactly on the socket's row/column ray, then it slid into that cavity.
#    - D1/3 removes the strict same-row/same-column requirement. D3/2 removes the cavity target and replaces
#      it with "stop when the next step would hit structure pixels".
#    - In this combo, a matched red no longer needs exact socket alignment and no longer ends inside a cavity.
#      Example: a downward-facing solid comb has a 4-cell front span. A 2x4 red rectangle can start 6 cells
#      below it and shifted one column to the right. The old task would reject that offset red because the
#      socket ray is not perfectly aligned. The combo instead assigns it to that structure because it is the
#      nearest reachable side-contact candidate, moves it straight upward, paints the traversed placements
#      black, and stops with the red still outside the structure, flush against the front edge.
#    - The visible difference from the original task is therefore twofold in the same object: the matched red
#      is laterally offset in the input, and the final red stays outside the structure instead of docking into
#      an interior slot.
# b) Complexity calibration:
#    - The combo keeps motion axis-aligned so the task stays learnable from a few examples; only the matching
#      criterion and stopping condition change together.
#    - Lateral offsets are restricted to a small set (mostly 1-2 cells) and contact spans repeat across
#      examples (3/4/5 cells). This makes the "not strictly aligned, but still nearest and reachable" pattern
#      visible without exploding into arbitrary free-form geometry.
#    - Distractors remain present, but are only accepted when they do not become solver-selected reachable
#      reds. This preserves the need to distinguish matched reds from unmatched reds.
# c) Implicit assumptions audit:
#    - Original assumption: a valid target is a bg socket strictly inside the structure bbox. Broken here;
#      final positions are computed by collision with solid structure pixels instead.
#    - Original assumption: matching can be inferred by socket rays. Broken here; matching is now based on
#      nearest reachable side distance between a red bbox and a structure bbox, with only interval overlap on
#      the perpendicular axis required.
#    - Original assumption: each structure claims at most one visible red. The combo description is red-centric
#      ("each red moves toward its nearest structure direction"), so the solver assigns movements per red
#      component. The generator still constructs at most one intended matched red per structure to keep the
#      samples clean, but the solver itself does not rely on that stronger condition.
#    - New assumption introduced by the combo solver: a red can only reach a structure by straight motion if
#      its bbox projection overlaps that structure on the perpendicular axis. The generator enforces this for
#      matched reds and rejects distractor placements that would accidentally satisfy it.
# d) Placement legality:
#    - Structures are placed first with padding.
#    - Each matched red is generated from a final collision-stop placement adjacent to one chosen structure
#      side, then shifted outward by 4-8 cells to form the input. We explicitly verify that replaying the
#      solver-style motion from that input returns to the intended final placement and hits the intended
#      structure before anything else.
#    - Trails, start positions, final positions, and distractors are all kept non-overlapping.
# e) Visual observability:
#    - Every accepted matched example uses a non-zero lateral offset relative to the aligned contact span, so
#      the D1 change is directly visible.
#    - Every accepted matched example ends with the red outside the structure bbox and touching it by collision,
#      so the D3 change is also directly visible.
#    - Those two signals appear on the same moving object, making the fusion observable from 2-4 ARC examples.
# f) Visual cleanliness:
#    - Paths do not cross structure bboxes or each other. Distractors avoid reserved path regions and existing
#      reds, so red movers, black trails, and structure pixels remain easy to distinguish.
# g) Color reasoning and inferability:
#    - Colors keep their original roles: one background color, one shared structure color, red as the mover,
#      black as the trail. No hidden ordering by color value is used; all decisions come from visible geometry.
# h) Audit of solver assumptions against the generator:
#    - The solver assumes structures are connected non-bg non-red components; the generator uses only such
#      components.
#    - The solver assumes reachable matches require perpendicular projection overlap; intended matched reds are
#      constructed that way and distractors are checked against the solver so they remain unmatched.
#    - The solver assumes collision-stop motion is straight and independent of other reds; the generator clears
#      paths so matched motions are unambiguous and do not interfere with each other.


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_SPAN = ((7, 3), (8, 4), (9, 3), (9, 5))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 3

LATERAL_OFFSETS = (1, 2)
MIN_CONTACT_OVERLAP = 2

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MATCH_PLACEMENT_TRIES = 120
DISTRACTOR_PLACEMENT_TRIES = 160
MAX_RETRIES = 500
_GEN_BRANCH_PLANS = ((2, 1), (2, 2), (3, 2), (3, 3))
_GENERATE_CALL_COUNT = 0


def _rotate_dir(dirv: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = map(int, dirv)
    for _ in range(int(k) % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _translate_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return (
        int(bbox[0] + dr),
        int(bbox[1] + dc),
        int(bbox[2] + dr),
        int(bbox[3] + dc),
    )


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _bbox_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        mask[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = True
    return mask


def _rect_coords(top: int, left: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in range(int(top), int(top + h)) for c in range(int(left), int(left + w))]


def _move_until_collision(
    coords: list[tuple[int, int]],
    move_vec: tuple[int, int],
    struct_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], int, bool]:
    h, w = struct_mask.shape
    mv_r, mv_c = map(int, move_vec)
    cur = [(int(r), int(c)) for r, c in coords]
    steps = 0

    while True:
        nxt = [(int(r + mv_r), int(c + mv_c)) for r, c in cur]
        out_of_bounds = False
        collided = False
        for rr, cc in nxt:
            if not (0 <= rr < h and 0 <= cc < w):
                out_of_bounds = True
                break
            if bool(struct_mask[rr, cc]):
                collided = True
                break
        if collided:
            return cur, int(steps), True
        if out_of_bounds:
            return cur, int(steps), False
        cur = nxt
        steps += 1


def _path_mask(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    start_top: int,
    start_left: int,
    move_vec: tuple[int, int],
    steps: int,
) -> np.ndarray:
    h, w = grid_shape
    ph, pw = patch_shape
    mv_r, mv_c = map(int, move_vec)
    mask = np.zeros((h, w), dtype=bool)
    for step in range(int(steps)):
        r0 = int(start_top + step * mv_r)
        c0 = int(start_left + step * mv_c)
        mask[r0:r0 + ph, c0:c0 + pw] = True
    return mask


def _collect_red_data(
    g: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    red_val = int(Color.RED)
    red_comps = components_mask(g == red_val, connectivity=4)
    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    for coords in red_comps:
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
    return red_coords, red_bbox


def _collect_structures(
    g: np.ndarray,
    bg: int,
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]], np.ndarray]:
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


def _candidate_side_moves(
    red_bbox: tuple[int, int, int, int],
    struct_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, tuple[int, int]]]:
    rr0, rc0, rr1, rc1 = map(int, red_bbox)
    sr0, sc0, sr1, sc1 = map(int, struct_bbox)
    cands: list[tuple[int, int, tuple[int, int]]] = []

    col_overlap = _overlap_len(rc0, rc1, sc0, sc1)
    if int(rr1) < int(sr0) and col_overlap > 0:
        cands.append((int(sr0 - rr1 - 1), int(col_overlap), (1, 0)))
    if int(rr0) > int(sr1) and col_overlap > 0:
        cands.append((int(rr0 - sr1 - 1), int(col_overlap), (-1, 0)))

    row_overlap = _overlap_len(rr0, rr1, sr0, sr1)
    if int(rc1) < int(sc0) and row_overlap > 0:
        cands.append((int(sc0 - rc1 - 1), int(row_overlap), (0, 1)))
    if int(rc0) > int(sc1) and row_overlap > 0:
        cands.append((int(rc0 - sc1 - 1), int(row_overlap), (0, -1)))

    return cands


def _analyze_red_assignments(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_coords, red_bbox = _collect_red_data(g)
    struct_coords, struct_bbox, struct_mask = _collect_structures(g, bg)

    assignments: list[dict] = []
    for rid, bbox in enumerate(red_bbox):
        best = None
        for sid, sbbox in enumerate(struct_bbox):
            for gap, overlap, move_vec in _candidate_side_moves(bbox, sbbox):
                final_coords, steps, hit = _move_until_collision(red_coords[rid], move_vec, struct_mask)
                if not hit:
                    continue
                cand = (
                    int(gap),
                    -int(overlap),
                    int(sid),
                    int(move_vec[0]),
                    int(move_vec[1]),
                )
                if best is None or cand < best[0]:
                    best = (
                        cand,
                        {
                            "rid": int(rid),
                            "sid": int(sid),
                            "move_vec": (int(move_vec[0]), int(move_vec[1])),
                            "gap": int(gap),
                            "overlap": int(overlap),
                            "steps": int(steps),
                            "final_coords": [(int(r), int(c)) for r, c in final_coords],
                            "final_bbox": _translate_bbox(bbox, int(steps * move_vec[0]), int(steps * move_vec[1])),
                        },
                    )
        if best is not None:
            assignments.append(best[1])

    assignments.sort(key=lambda item: int(item["rid"]))
    return {
        "bg": int(bg),
        "red_coords": red_coords,
        "red_bbox": red_bbox,
        "structures": struct_coords,
        "struct_bbox": struct_bbox,
        "struct_mask": struct_mask,
        "assignments": assignments,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each reachable red rectangle straight toward its nearest structure side and stop on collision, leaving a black trail.

    Input structure:
    - The background is the dominant color, with 2-3 large non-background, non-red comb/bracket structures.
    - Red components are rectangular blobs placed outside those structures; some are laterally offset yet still overlap a reachable side projection, while others are distractors that cannot hit any structure by straight motion.

    Rule:
    - For every red component, compare the structures it can reach by straight north/south/east/west motion with perpendicular projection overlap, and choose the nearest such structure side.
    - Slide that red one cell at a time toward the chosen structure, turning every intermediate occupied placement black.
    - Stop at the last collision-free placement before the next step would overlap structure pixels; any red with no reachable structure disappears.

    Key insight:
    - The match is no longer determined by a pre-aligned socket ray, and the destination is no longer an interior slot: the red is selected by nearest reachable side distance and its final resting place is defined purely by collision with the structure boundary.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    assert red_val in set(map(int, np.unique(g)))

    analysis = _analyze_red_assignments(g)
    assert analysis["red_coords"], "expected at least one red component"
    assert analysis["structures"], "expected at least one non-bg non-red structure"

    out = g.copy()
    out[out == red_val] = int(bg)

    for item in analysis["assignments"]:
        rid = int(item["rid"])
        move_vec = (int(item["move_vec"][0]), int(item["move_vec"][1]))
        steps = int(item["steps"])
        coords = analysis["red_coords"][rid]

        for step in range(int(steps)):
            off_r = int(step * move_vec[0])
            off_c = int(step * move_vec[1])
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = int(black_val)

        for rr, cc in item["final_coords"]:
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(red_val)

    return out


def _structure_array(width: int, span: int, bg: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.full((4, width), bg, dtype=int)
    front = np.zeros((4, width), dtype=bool)
    start = (int(width) - int(span)) // 2

    g[0, :] = int(sc)
    g[1, 1] = int(sc)
    g[2, 1] = int(sc)
    g[1, width - 2] = int(sc)
    g[2, width - 2] = int(sc)
    g[1:4, start:start + span] = int(sc)
    front[3, start:start + span] = True
    return g, front


def _make_structure_spec(width: int, span: int, rot_k: int, bg: int, sc: int) -> dict:
    structure, front = _structure_array(width, span, bg, sc)
    structure = np.rot90(structure, k=rot_k)
    front = np.rot90(front.astype(int), k=rot_k).astype(bool)
    front_dir = _rotate_dir((1, 0), rot_k)
    front_bbox = tuple(map(int, bbox_coords(np.argwhere(front))))
    red_patch = np.rot90(np.full((2, span), int(Color.RED), dtype=int), k=rot_k)
    ph, pw = red_patch.shape

    if front_dir == (1, 0):
        aligned_final = (int(front_bbox[2] + 1), int(front_bbox[1]))
    elif front_dir == (-1, 0):
        aligned_final = (int(front_bbox[0] - ph), int(front_bbox[1]))
    elif front_dir == (0, 1):
        aligned_final = (int(front_bbox[0]), int(front_bbox[3] + 1))
    else:
        aligned_final = (int(front_bbox[0]), int(front_bbox[1] - pw))

    return {
        "structure": structure,
        "front_dir": (int(front_dir[0]), int(front_dir[1])),
        "front_bbox_local": front_bbox,
        "aligned_final_local": (int(aligned_final[0]), int(aligned_final[1])),
        "red_patch": red_patch,
        "span": int(span),
    }


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
) -> tuple[np.ndarray, list[dict]] | None:
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(struct_grid, spec["structure"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        placed.append(
            {
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "front_dir": spec["front_dir"],
                "front_bbox": _translate_bbox(spec["front_bbox_local"], r0, c0),
                "aligned_final_top_left": (
                    int(spec["aligned_final_local"][0] + r0),
                    int(spec["aligned_final_local"][1] + c0),
                ),
                "red_patch": spec["red_patch"].copy(),
                "span": int(spec["span"]),
            }
        )

    return struct_grid, placed


def _candidate_match_positions(item: dict) -> list[dict]:
    dr, dc = map(int, item["front_dir"])
    ph, pw = item["red_patch"].shape
    base_r0, base_c0 = map(int, item["aligned_final_top_left"])
    span = int(item["span"])

    if (dr, dc) in ((1, 0), (-1, 0)):
        perp = (0, 1)
    else:
        perp = (1, 0)

    offsets: list[int] = []
    max_off = min(int(max(LATERAL_OFFSETS)), int(span - MIN_CONTACT_OVERLAP))
    for mag in range(1, int(max_off) + 1):
        offsets.extend([-int(mag), int(mag)])
    random.shuffle(offsets)

    candidates: list[dict] = []
    for slide in range(SLIDE_RANGE[0], SLIDE_RANGE[1] + 1):
        for offset in offsets:
            overlap = int(span - abs(int(offset)))
            if overlap < int(MIN_CONTACT_OVERLAP):
                continue
            final_r0 = int(base_r0 + perp[0] * offset)
            final_c0 = int(base_c0 + perp[1] * offset)
            start_r0 = int(final_r0 + dr * slide)
            start_c0 = int(final_c0 + dc * slide)
            candidates.append(
                {
                    "slide": int(slide),
                    "lateral_offset": int(offset),
                    "input_bbox": (
                        int(start_r0),
                        int(start_c0),
                        int(start_r0 + ph - 1),
                        int(start_c0 + pw - 1),
                    ),
                    "final_bbox": (
                        int(final_r0),
                        int(final_c0),
                        int(final_r0 + ph - 1),
                        int(final_c0 + pw - 1),
                    ),
                }
            )
    random.shuffle(candidates)
    return candidates


def _place_matched_reds(
    struct_grid: np.ndarray,
    placed: list[dict],
    matched_ids: list[int],
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    h, w = struct_grid.shape
    struct_mask = struct_grid != bg
    bbox_mask = _bbox_mask((h, w), [tuple(map(int, item["bbox"])) for item in placed])
    order = list(matched_ids)
    random.shuffle(order)

    matched_items: dict[int, dict] = {}

    def rec(
        pos: int,
        inp: np.ndarray,
        reserved_mask: np.ndarray,
        red_mask: np.ndarray,
    ) -> np.ndarray | None:
        if pos == len(order):
            return inp

        idx = int(order[pos])
        item = placed[idx]
        patch = item["red_patch"]
        ph, pw = patch.shape
        move_vec = (-int(item["front_dir"][0]), -int(item["front_dir"][1]))

        for cand in _candidate_match_positions(item)[:MATCH_PLACEMENT_TRIES]:
            sr0, sc0, sr1, sc1 = map(int, cand["input_bbox"])
            fr0, fc0, fr1, fc1 = map(int, cand["final_bbox"])

            if sr0 < 0 or sc0 < 0 or sr1 >= h or sc1 >= w:
                continue
            if fr0 < 0 or fc0 < 0 or fr1 >= h or fc1 >= w:
                continue
            if bool((inp[sr0:sr0 + ph, sc0:sc0 + pw] != bg).any()):
                continue
            if bool(struct_mask[fr0:fr0 + ph, fc0:fc0 + pw].any()):
                continue

            start_coords = _rect_coords(sr0, sc0, ph, pw)
            final_coords, steps, hit = _move_until_collision(start_coords, move_vec, struct_mask)
            if not hit or int(steps) != int(cand["slide"]):
                continue
            if tuple(map(int, bbox_coords(np.array(final_coords, dtype=int)))) != tuple(map(int, cand["final_bbox"])):
                continue

            path = _path_mask((h, w), patch.shape, sr0, sc0, move_vec, steps)
            full_mask = path.copy()
            full_mask[fr0:fr0 + ph, fc0:fc0 + pw] = True

            if bool((full_mask & reserved_mask).any()):
                continue
            if bool((path & red_mask).any()):
                continue
            if bool((full_mask & bbox_mask).any()):
                continue

            next_inp = inp.copy()
            paste(next_inp, patch, sr0, sc0, transparent=bg)
            next_red_mask = red_mask.copy()
            next_red_mask[sr0:sr0 + ph, sc0:sc0 + pw] = True

            matched_items[idx] = {
                "sid": int(idx),
                "patch": patch.copy(),
                "input_bbox": tuple(map(int, cand["input_bbox"])),
                "final_bbox": tuple(map(int, cand["final_bbox"])),
                "path_mask": path.copy(),
                "full_mask": full_mask.copy(),
                "lateral_offset": int(cand["lateral_offset"]),
                "slide": int(cand["slide"]),
                "move_vec": (int(move_vec[0]), int(move_vec[1])),
            }

            solved = rec(pos + 1, next_inp, reserved_mask | full_mask, next_red_mask)
            if solved is not None:
                return solved
            matched_items.pop(idx, None)

        return None

    solved = rec(
        0,
        struct_grid.copy(),
        np.zeros((h, w), dtype=bool),
        np.zeros((h, w), dtype=bool),
    )
    if solved is None:
        return None

    return solved, [matched_items[idx] for idx in matched_ids if idx in matched_items]


def _render_expected(inp: np.ndarray, matched_items: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg = int(get_bg(inp))
    out[out == int(Color.RED)] = int(bg)

    for item in matched_items:
        out[item["path_mask"]] = int(Color.BLACK)
        fr0, fc0, _fr1, _fc1 = map(int, item["final_bbox"])
        paste(out, item["patch"], fr0, fc0, transparent=bg)
    return out


def _assignment_input_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    analysis = _analyze_red_assignments(grid)
    return sorted(tuple(map(int, analysis["red_bbox"][item["rid"]])) for item in analysis["assignments"])


def _place_distractors(
    inp: np.ndarray,
    placed: list[dict],
    matched_items: list[dict],
    bg: int,
    sc: int,
    count: int,
) -> bool:
    h, w = inp.shape
    planned_boxes = sorted(tuple(map(int, item["input_bbox"])) for item in matched_items)
    bbox_mask = _bbox_mask((h, w), [tuple(map(int, item["bbox"])) for item in placed])
    reserved = np.zeros((h, w), dtype=bool)
    for item in matched_items:
        reserved |= item["full_mask"]

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(int(count)):
        success = False
        for _try in range(DISTRACTOR_PLACEMENT_TRIES):
            red_mask = inp == int(Color.RED)
            blocked = bbox_mask | reserved | (inp != bg) | _dilate4(red_mask)
            canvas = np.full((h, w), bg, dtype=int)
            canvas[blocked] = int(sc)

            random.shuffle(shapes)
            candidate_loc = None
            candidate_patch = None
            for rh, rw in shapes:
                patch = np.full((rh, rw), int(Color.RED), dtype=int)
                loc = _place_patch(canvas, patch, bg, padding=0)
                if loc is not None:
                    candidate_loc = loc
                    candidate_patch = patch
                    break
            if candidate_loc is None or candidate_patch is None:
                continue

            temp = inp.copy()
            paste(temp, candidate_patch, candidate_loc[0], candidate_loc[1], transparent=bg)
            if _assignment_input_boxes(temp) == planned_boxes:
                paste(inp, candidate_patch, candidate_loc[0], candidate_loc[1], transparent=bg)
                success = True
                break
        if not success:
            return False

    return True


def _combo_metrics(placed: list[dict], matched_items: list[dict]) -> dict:
    outside_stop = 0
    for item in matched_items:
        sb = tuple(map(int, placed[int(item["sid"])]["bbox"]))
        fb = tuple(map(int, item["final_bbox"]))
        disjoint = (
            int(fb[2]) < int(sb[0])
            or int(fb[0]) > int(sb[2])
            or int(fb[3]) < int(sb[1])
            or int(fb[1]) > int(sb[3])
        )
        if disjoint:
            outside_stop += 1

    return {
        "matched": int(len(matched_items)),
        "misaligned": int(sum(abs(int(item["lateral_offset"])) > 0 for item in matched_items)),
        "outside_stop": int(outside_stop),
        "max_lateral_offset": int(max((abs(int(item["lateral_offset"])) for item in matched_items), default=0)),
    }


def _visible_combo_case(metrics: dict) -> bool:
    matched = int(metrics["matched"])
    return matched > 0 and int(metrics["misaligned"]) == matched and int(metrics["outside_stop"]) == matched


def _gen_attempt_bundle(
    target_structs: int | None = None,
    target_matched: int | None = None,
) -> dict | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    n_structs = int(target_structs) if target_structs is not None else random.randint(*STRUCT_COUNT_RANGE)
    if n_structs >= 3:
        h = random.randint(max(GRID_H_RANGE[0], GRID_H_RANGE[1] - 2), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], GRID_W_RANGE[1] - 3), GRID_W_RANGE[1])
    else:
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
    if target_matched is None:
        unmatched = 1 if random.random() < UNMATCHED_PROB else 0
        matched = int(n_structs - unmatched)
    else:
        matched = int(target_matched)
        unmatched = int(n_structs - matched)
        if unmatched not in (0, 1):
            return None
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for _ in range(n_structs):
        width, span = random.choice(STRUCT_WIDTH_SPAN)
        rot_k = random.randrange(4)
        specs.append(_make_structure_spec(width, span, rot_k, bg, sc))
    specs.sort(key=lambda spec: np.count_nonzero(spec["structure"] != bg), reverse=True)

    placed_result = _place_structures(specs, h, w, bg, sc)
    if placed_result is None:
        return None
    struct_grid, placed = placed_result

    matched_ids = list(range(n_structs))
    random.shuffle(matched_ids)
    matched_ids = sorted(matched_ids[:matched])

    match_result = _place_matched_reds(struct_grid, placed, matched_ids, bg)
    if match_result is None:
        return None
    inp, matched_items = match_result

    planned_boxes = sorted(tuple(map(int, item["input_bbox"])) for item in matched_items)
    if _assignment_input_boxes(inp) != planned_boxes:
        return None

    if not _place_distractors(inp, placed, matched_items, bg, sc, distractor_count):
        return None
    if _assignment_input_boxes(inp) != planned_boxes:
        return None

    expected = _render_expected(inp, matched_items)
    metrics = _combo_metrics(placed, matched_items)
    if not _visible_combo_case(metrics):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return {
        "inp": inp,
        "expected": expected,
        "metrics": metrics,
    }


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    plan = _GEN_BRANCH_PLANS[_GENERATE_CALL_COUNT % len(_GEN_BRANCH_PLANS)]
    _GENERATE_CALL_COUNT += 1

    targeted_retries = min(120, MAX_RETRIES // 4)
    for _ in range(targeted_retries):
        bundle = _gen_attempt_bundle(target_structs=plan[0], target_matched=plan[1])
        if bundle is not None:
            return bundle["inp"]

    for _ in range(MAX_RETRIES - targeted_retries):
        bundle = _gen_attempt_bundle()
        if bundle is not None:
            return bundle["inp"]
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D1/3+D3/2")
