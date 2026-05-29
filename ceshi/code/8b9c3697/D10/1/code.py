# Variant of 8b9c3697
# Dimension: D10 — 结构数量与匹配基数 (multi_step)
# Change: 2-3个结构，每个结构最多匹配1个红色色块（1:1），可选地有1个未匹配结构（35%概率）... → 链式顺序处理

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg


# Design analysis for the D10 variant
# ===================================
# spec_variant:
#   visible_layout:
#     - Keep the original comb/bracket objects and rectangular red pieces.
#     - Place 2-3 structures in a left-to-right chain.
#     - At least one adjacent pair is arranged face-to-face: the left structure opens right,
#       the next structure opens left, and both see the same red rectangle in the corridor.
#     - The shared red is deliberately closer to the right structure, so the original
#       global-best assignment would prefer the right structure, while the variant must still
#       give it to the left structure because the chain is processed strictly left-to-right.
#   concrete_behavior_trace:
#     - Original assignment stage:
#       1. Each structure independently picks its best visible red.
#       2. All candidate pairs are globally sorted by score.
#       3. The closer right structure wins the shared red; the left structure loses it.
#     - Variant assignment stage:
#       1. Sort structures by their x-position (left edge, then top edge).
#       2. Process them one by one.
#       3. The left structure consumes the shared red first.
#       4. The right structure recomputes only over the remaining pool, so it becomes unmatched.
#     - The local slide-and-black-trail motion is unchanged; only the allocation stage is now
#       sequential and stateful.
#   complexity_calibration:
#     - The chain effect is visible because at least one red component is simultaneously visible
#       from multiple structures.
#     - The geometry is constrained to one clear face-to-face pair so the examples do not become
#       cluttered or ambiguous.
#     - Optional third structures stay independent; they keep the original task's multi-object
#       feel without weakening the shared-candidate signal.
#   implicit_assumption_audit:
#     - The original `struct_best` collection assumed a structure's best candidate could be chosen
#       once before any red was consumed. That is false in this variant.
#     - Filtering out already-used reds after global scoring is insufficient, because a later
#       structure may need to fall back to a different visible red or to no red at all.
#     - Therefore the solver must rescan visible reds for each structure after every prior
#       assignment updates the available pool.
#   placement_legality:
#     - Generator places non-overlapping structures with deterministic coordinates.
#     - Shared red lies in the open corridor between the face-to-face pair and fits both sockets.
#     - Distractors are still forbidden from all socket rays so they do not create accidental
#       alternative matches.
#   observability:
#     - The ordering rule is purely spatial and visible: left-to-right.
#     - No hidden color id ordering or invisible metadata is used.
#     - Samples differ from the original because the left structure can take a farther shared red
#       that the original global solver would have assigned to the closer right structure.
#   cleanliness:
#     - All active structures sit on a single horizontal band, making the chain relation readable.
#     - Only one shared corridor is introduced, so the input remains compact and legible.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_CHOICES = (2, 3)
SLIDE_RANGE = (4, 8)
PAIR_GAP_EXTRA = (1, 2)

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 400
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _rotate_dir(dr: int, dc: int, rot_k: int) -> tuple[int, int]:
    for _ in range(int(rot_k) % 4):
        dr, dc = -int(dc), int(dr)
    return int(dr), int(dc)


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start : start + teeth] = sc
    return g


def _structure_spec(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    base = _structure_array(width, teeth, bg, sc)
    red_h = min(teeth, 2)
    start = (width - teeth) // 2

    dock = np.zeros_like(base, dtype=bool)
    dock[2 : 2 + red_h, start : start + teeth] = True

    patch = np.rot90(base, k=rot_k)
    dock_mask = np.rot90(dock, k=rot_k)
    dock_coords = [(int(r), int(c)) for r, c in np.argwhere(dock_mask)]
    dock_bbox = tuple(map(int, bbox_coords(np.argwhere(dock_mask))))

    return {
        "width": int(width),
        "teeth": int(teeth),
        "rot_k": int(rot_k),
        "patch": patch,
        "shape": tuple(map(int, patch.shape)),
        "dir": _rotate_dir(1, 0, rot_k),
        "dock_rel": dock_coords,
        "dock_bbox_rel": dock_bbox,
    }


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            grid[int(r), int(c)] = int(color)


def _structure_coords_abs(spec: dict, top: int, left: int, sc: int) -> list[tuple[int, int]]:
    coords = np.argwhere(spec["patch"] == int(sc))
    return [(int(top + r), int(left + c)) for r, c in coords]


def _best_candidate_for_structure(
    g: np.ndarray,
    sid: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    used_red: set[int],
    red_val: int,
    bg: int,
) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    h, w = g.shape
    s = struct_coords[int(sid)]
    sr0, sc0, sr1, sc1 = map(int, struct_bbox[int(sid)])
    best = None

    for dr, dc in _DIRS:
        hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
        for r, c in s:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                continue

            rr, cc = int(nr), int(nc)
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                rr += int(dr)
                cc += int(dc)

            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(red_val):
                rid = int(red_at[(int(rr), int(cc))])
                if int(rid) in used_red:
                    continue
                rec = hits[rid]
                rec[0] += 1
                dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                if int(dist) < int(rec[1]):
                    rec[1] = int(dist)
                    rec[2] = (int(r), int(c))

        if not hits:
            continue

        rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1]), -int(k)))
        cnt, mind, anchor = hits[rid_best]
        cand = (int(cnt), -int(mind), -int(rid_best), (int(dr), int(dc)), anchor)
        if best is None or cand > best:
            best = cand

    if best is None:
        return None

    cnt, neg_mind, neg_rid, dirv, anchor = best
    return int(cnt), int(-neg_rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1]))


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


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(struct_grid[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[rr, cc] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    blocked = forbidden | (inp != int(bg))

    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    for _ in range(count):
        shapes = list(DISTRACTOR_SHAPES)
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            candidates: list[tuple[int, int]] = []
            for r0 in range(0, h - rh + 1):
                for c0 in range(0, w - rw + 1):
                    region = blocked[r0 : r0 + rh, c0 : c0 + rw]
                    if bool(region.any()):
                        continue
                    if r0 > 0 and bool((inp[r0 - 1, c0 : c0 + rw] == int(Color.RED)).any()):
                        continue
                    if r0 + rh < h and bool((inp[r0 + rh, c0 : c0 + rw] == int(Color.RED)).any()):
                        continue
                    if c0 > 0 and bool((inp[r0 : r0 + rh, c0 - 1] == int(Color.RED)).any()):
                        continue
                    if c0 + rw < w and bool((inp[r0 : r0 + rh, c0 + rw] == int(Color.RED)).any()):
                        continue
                    candidates.append((int(r0), int(c0)))
            if not candidates:
                continue
            r0, c0 = random.choice(candidates)
            inp[r0 : r0 + rh, c0 : c0 + rw] = int(Color.RED)
            blocked[max(0, r0 - 1) : min(h, r0 + rh + 1), max(0, c0 - 1) : min(w, c0 + rw + 1)] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _visible_red_sets(grid: np.ndarray) -> tuple[list[set[int]], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)

    red_comps = components_mask(g == int(red_val), connectivity=4)
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = int(rid)

    struct_mask = (g != int(bg)) & (g != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    visible: list[set[int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        seen: set[int] = set()
        for dr, dc in _DIRS:
            for r, c in pts:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(red_val):
                    seen.add(int(red_at[(int(rr), int(cc))]))
        visible.append(seen)

    shared_count = 0
    for rid in range(len(red_comps)):
        if sum(int(rid in s) for s in visible) >= 2:
            shared_count += 1
    return visible, int(shared_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide red components into comb sockets, but allocate them in chain order.

    Input structure:
    - Background is the most common color. Large non-bg, non-red comb/bracket structures sit on the grid.
    - Red rectangles lie outside the structures; some are visible from more than one structure.

    Rule:
    - For each structure, identify its interior socket cells and ray-cast outward to find visible red components.
    - Process structures strictly from left to right. Each structure picks its best visible red from the
      still-unused pool, then that red is removed before the next structure is evaluated.
    - The chosen red slides into the socket and paints a black trail; all unused reds disappear.

    Key insight: Matching is not a global best-first assignment anymore; availability changes after every
    earlier structure, so later structures must be solved against the remaining red pool only.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    assert int(red_val) in set(map(int, np.unique(g)))

    red_comps = components_mask(g == int(red_val), connectivity=4)
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

    struct_mask = (g != int(bg)) & (g != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    order = sorted(
        range(len(struct_coords)),
        key=lambda sid: (
            int(struct_bbox[int(sid)][1]),
            int(struct_bbox[int(sid)][0]),
            int(struct_bbox[int(sid)][3]),
            int(struct_bbox[int(sid)][2]),
        ),
    )

    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for sid in order:
        best = _best_candidate_for_structure(
            g,
            int(sid),
            struct_coords,
            struct_bbox,
            red_at,
            used_red,
            red_val,
            bg,
        )
        if best is None:
            continue
        _cnt, rid, dirv, anchor = best
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1]))))

    out = g.copy()
    out[out == int(red_val)] = int(bg)

    for rid, (dr, dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        _apply_slide(out, coords, int(shift_r), int(shift_c), red_val, black_val)

    return out


def _expected_output(inp: np.ndarray, matched: list[dict], bg: int) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    out[out == int(Color.RED)] = int(bg)
    for item in matched:
        dr, dc = map(int, item["dir"])
        slide = int(item["slide"])
        shift_r = -int(slide * dr)
        shift_c = -int(slide * dc)
        _apply_slide(out, item["coords"], shift_r, shift_c, int(Color.RED), int(Color.BLACK))
    return out


def _pair_layout(
    left_spec: dict,
    right_spec: dict,
    slide_left: int,
    slide_right: int,
    margin_left: int,
) -> tuple[int, int, int]:
    left_dock = left_spec["dock_bbox_rel"]
    right_dock = right_spec["dock_bbox_rel"]
    left_col = int(margin_left)
    shared_left = int(left_col + left_dock[1] + slide_left)
    right_col = int(shared_left + slide_right - right_dock[1])
    shared_right = int(shared_left + (left_dock[3] - left_dock[1]))
    return int(left_col), int(right_col), int(shared_right)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])

    n_structs = random.choice(STRUCT_COUNT_CHOICES)
    pair_choices = STRUCT_WIDTH_TEETH if n_structs == 2 else ((7, 1), (8, 2))
    pair_width, pair_teeth = random.choice(pair_choices)
    left_spec = _structure_spec(pair_width, pair_teeth, 1, bg, sc)   # opening right
    right_spec = _structure_spec(pair_width, pair_teeth, 3, bg, sc)  # opening left

    slide_left_hi = SLIDE_RANGE[1] if n_structs == 2 else 6
    slide_right_hi = SLIDE_RANGE[1] if n_structs == 2 else 5
    slide_left = random.randint(max(SLIDE_RANGE[0], 5), slide_left_hi)
    slide_right = random.randint(SLIDE_RANGE[0], min(slide_right_hi, slide_left - 1))
    margin_left = random.randint(1, 3)
    left_col, right_col, shared_right = _pair_layout(left_spec, right_spec, slide_left, slide_right, margin_left)

    structs: list[dict] = [
        {"name": "left_shared", "spec": left_spec, "top": 0, "left": int(left_col)},
        {"name": "right_shared", "spec": right_spec, "top": 0, "left": int(right_col)},
    ]

    matched: list[dict] = []
    if n_structs == 3:
        third_width, third_teeth = random.choice(((7, 1), (8, 2)))
        third_spec = _structure_spec(third_width, third_teeth, 1, bg, sc)  # opening right
        gap = 1
        third_left = int(right_col + right_spec["shape"][1] + gap)
        third_slide = random.randint(4, 5)
        structs.append({"name": "independent", "spec": third_spec, "top": 0, "left": int(third_left), "slide": int(third_slide)})

    max_h = max(int(item["spec"]["shape"][0]) for item in structs)
    min_h = max(int(GRID_H_RANGE[0]), int(max_h + 2))

    right_extent = max(int(item["left"] + item["spec"]["shape"][1]) for item in structs)
    if n_structs == 3:
        third = structs[2]
        third_dock = third["spec"]["dock_bbox_rel"]
        third_red_right = int(third["left"] + third_dock[3] + third["slide"] + 1)
        right_extent = max(int(right_extent), int(third_red_right))
    min_w = max(int(GRID_W_RANGE[0]), int(right_extent + 2))

    if min_h > int(GRID_H_RANGE[1]) or min_w > int(GRID_W_RANGE[1]):
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    row = random.randint(1, h - max_h - 1)
    for item in structs:
        item["top"] = int(row)

    inp = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for item in structs:
        spec = item["spec"]
        top = int(item["top"])
        left = int(item["left"])
        ph, pw = map(int, spec["shape"])
        if top < 0 or left < 0 or top + ph > h or left + pw > w:
            return None
        if bool((struct_grid[top : top + ph, left : left + pw] != int(bg)).any()):
            return None
        patch = spec["patch"]
        mask = patch != int(bg)
        inp[top : top + ph, left : left + pw][mask] = patch[mask]
        struct_grid[top : top + ph, left : left + pw][mask] = patch[mask]
        coords = _structure_coords_abs(spec, top, left, sc)
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"name": item["name"], "coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    left_item = structs[0]
    left_shared = _translate(
        _translate(left_item["spec"]["dock_rel"], left_item["top"], left_item["left"]),
        slide_left * left_item["spec"]["dir"][0],
        slide_left * left_item["spec"]["dir"][1],
    )
    if any(not (0 <= r < h and 0 <= c < w) for r, c in left_shared):
        return None
    _paint_cells(inp, left_shared, int(Color.RED))
    matched.append({"coords": left_shared, "dir": left_item["spec"]["dir"], "slide": int(slide_left)})

    right_item = structs[1]
    right_shared = _translate(
        _translate(right_item["spec"]["dock_rel"], right_item["top"], right_item["left"]),
        slide_right * right_item["spec"]["dir"][0],
        slide_right * right_item["spec"]["dir"][1],
    )
    if sorted(left_shared) != sorted(right_shared):
        return None

    if n_structs == 3:
        third_item = structs[2]
        third_private = _translate(
            _translate(third_item["spec"]["dock_rel"], third_item["top"], third_item["left"]),
            third_item["slide"] * third_item["spec"]["dir"][0],
            third_item["slide"] * third_item["spec"]["dir"][1],
        )
        if any(not (0 <= r < h and 0 <= c < w) for r, c in third_private):
            return None
        if any(int(inp[r, c]) != int(bg) for r, c in third_private):
            return None
        _paint_cells(inp, third_private, int(Color.RED))
        matched.append({"coords": third_private, "dir": third_item["spec"]["dir"], "slide": int(third_item["slide"])})

    matched_count = len(matched)
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched_count)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched_count)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)
    if not _place_distractors(inp, struct_grid, placed, bg, distractor_count):
        return None

    visible_sets, shared_count = _visible_red_sets(inp)
    if int(shared_count) < 1:
        return None

    expected = _expected_output(inp, matched, bg)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if n_structs >= 2:
        vis0 = visible_sets[0]
        vis1 = visible_sets[1]
        if len(vis0 & vis1) < 1:
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D10 variant")
