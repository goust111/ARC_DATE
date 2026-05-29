# Variant of 8b9c3697
# Dimension: D1 — 匹配条件：行列对齐射线投射 (match_cond)
# Change: 从结构内部插槽（socket）向四个基本方向发射射线，命中红色色块即确定匹配关系和滑入方向（s.col/row == d... → 最近距离匹配（取消对齐约束）

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis for the D1 variant:
# a) Concretization and behavior trace:
#    - The comb/bracket structures stay unchanged: a single opening side leads into a narrow cavity.
#    - In the original task, a matched red rectangle had to sit exactly on the opening ray, so the black
#      trail was purely horizontal or vertical. Here the matched red may be laterally offset from that ray.
#    - Concretely, if a downward-opening comb has its socket strip at row 8, cols 10..12, the old solver
#      only matched a red that already shared cols 10..12. The new solver instead compares every red
#      component to the socket strip by Manhattan distance and picks the closest one even if it starts at
#      row 14, cols 12..14. The red then moves diagonally/orthogonally toward the same opening and still
#      docks into the cavity, leaving a staircase-like black trail.
# b) Complexity calibration:
#    - The matching freedom is loosened, but only one degree of freedom is added: perpendicular offset from
#      the opening. The structures, colors, red shapes, and trail rule are kept fixed so the task does not
#      balloon into a multi-factor puzzle.
#    - Every generated matched red uses a non-zero lateral offset. This makes the D1 change visible in
#      nearly every sample instead of relying on rare diagonal cases.
# c) Implicit assumptions in the original solver that no longer hold:
#    - Ray-casting assumed the red lies on the same row/column as the socket; this is replaced.
#    - Final alignment assumed only one axis shifts; the variant must support two-axis translation.
#    - The unmatched-structure branch depended on "no visible red". Under nearest-distance matching every
#      structure would still have some nearest red, so the generator removes that branch and matches all
#      structures.
# d) Placement legality:
#    - Matched red starts are placed outside all structure bounding boxes, non-overlapping, and with paths
#      that do not cross structure pixels or other structures' bounding boxes.
#    - Distractors are accepted only if they do not change the nearest-distance assignment chosen by the
#      solver.
# e) Visual observability:
#    - The intended matched reds are always offset away from the socket column/row, so the rule difference
#      from the original ray-aligned task is visible from the input alone.
#    - The output keeps the same dock-into-cavity result, but the black trail now commonly contains an
#      obvious sideways component.
# f) Visual cleanliness:
#    - Structure padding is preserved and distractors are kept off structure bounding boxes and off matched
#      travel trails, so the red/black/structure roles stay visually separable.
# g) Color reasoning:
#    - No new color logic is introduced. Red remains the moving piece, black remains the trail, and a single
#      non-red structure color is preserved exactly as in the original task.
# h) New solver assumptions audited against the generator:
#    - The solver assumes each structure has a single dominant opening-facing socket group; the generator
#      still uses the original comb shapes, so that remains true.
#    - The solver assumes matched reds use the same rectangle footprint as the socket target; the generator
#      constructs matched reds from the original target geometry, so docking stays well-defined.
#    - The greedy "each red used once" assignment can still leave a structure unmatched in adversarial
#      inputs; the generator explicitly rejects such cases and only emits inputs where every structure keeps
#      its intended nearest red after distractors are added.


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)


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


def _intermediate_offsets(shift_r: int, shift_c: int) -> list[tuple[int, int]]:
    abs_r = abs(int(shift_r))
    abs_c = abs(int(shift_c))
    steps = int(max(abs_r, abs_c))
    step_r = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
    step_c = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)
    return [
        (
            int(step_r * min(step, abs_r)),
            int(step_c * min(step, abs_c)),
        )
        for step in range(steps)
    ]


def _opening_groups(
    grid: np.ndarray,
    struct_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    bg: int,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, bbox)
    groups: dict[tuple[int, int], set[tuple[int, int]]] = {d: set() for d in _DIRS}

    for r, c in struct_cells:
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) != int(bg):
                continue
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue

            rr, cc = int(nr), int(nc)
            clear_to_opening = True
            while r0 <= rr <= r1 and c0 <= cc <= c1:
                if int(grid[rr, cc]) != int(bg):
                    clear_to_opening = False
                    break
                rr += int(dr)
                cc += int(dc)

            if clear_to_opening:
                groups[(int(dr), int(dc))].add((int(nr), int(nc)))

    return {
        d: sorted(cells)
        for d, cells in groups.items()
        if cells
    }


def _component_socket_distance(
    red_cells: list[tuple[int, int]],
    socket_cells: list[tuple[int, int]],
) -> int:
    best = 10**9
    for r, c in red_cells:
        for rr, cc in socket_cells:
            dist = abs(int(r) - int(rr)) + abs(int(c) - int(cc))
            if dist < best:
                best = int(dist)
    return int(best)


def _socket_strip(
    opening_cells: list[tuple[int, int]],
    dirv: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = map(int, dirv)
    if (dr, dc) == (1, 0):
        edge = max(int(r) for r, _ in opening_cells)
        return [(int(r), int(c)) for r, c in opening_cells if int(r) == edge]
    if (dr, dc) == (-1, 0):
        edge = min(int(r) for r, _ in opening_cells)
        return [(int(r), int(c)) for r, c in opening_cells if int(r) == edge]
    if (dr, dc) == (0, 1):
        edge = max(int(c) for _, c in opening_cells)
        return [(int(r), int(c)) for r, c in opening_cells if int(c) == edge]
    edge = min(int(c) for _, c in opening_cells)
    return [(int(r), int(c)) for r, c in opening_cells if int(c) == edge]


def _extract_reds(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    for coords in red_comps:
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
    return red_coords, red_bbox


def _analyze_assignments(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)

    red_coords, red_bbox = _extract_reds(g)

    struct_mask = (g != int(bg)) & (g != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)

    structures: list[dict] = []
    struct_best: list[dict] = []

    for sid, coords in enumerate(struct_comps):
        pts = {(int(r), int(c)) for r, c in coords}
        bbox = tuple(map(int, bbox_coords(coords)))
        openings = _opening_groups(g, pts, bbox, bg)
        structures.append(
            {
                "sid": int(sid),
                "coords": pts,
                "bbox": bbox,
                "openings": openings,
            }
        )
        if not openings or not red_coords:
            continue

        best = None
        for dirv, opening_cells in openings.items():
            sockets = _socket_strip(opening_cells, dirv)
            socket_bbox = tuple(map(int, bbox_coords(np.array(sockets, dtype=int))))
            for rid, pts_red in enumerate(red_coords):
                dist = int(_component_socket_distance(pts_red, sockets))
                cand = (
                    int(dist),
                    -int(len(sockets)),
                    int(sid),
                    int(rid),
                    (int(dirv[0]), int(dirv[1])),
                    socket_bbox,
                )
                if best is None or cand < best:
                    best = cand

        if best is not None:
            dist, neg_socket_count, sid2, rid, dirv, socket_bbox = best
            struct_best.append(
                {
                    "sid": int(sid2),
                    "rid": int(rid),
                    "dir": (int(dirv[0]), int(dirv[1])),
                    "socket_bbox": socket_bbox,
                    "dist": int(dist),
                    "socket_count": int(-neg_socket_count),
                }
            )

    struct_best.sort(key=lambda item: (int(item["dist"]), -int(item["socket_count"]), int(item["sid"])))
    used_red: set[int] = set()
    assignments: list[dict] = []
    for item in struct_best:
        rid = int(item["rid"])
        if rid in used_red:
            continue
        used_red.add(rid)
        assignments.append(item)
    assignments.sort(key=lambda item: int(item["sid"]))

    return {
        "bg": int(bg),
        "red_coords": red_coords,
        "red_bbox": red_bbox,
        "structures": structures,
        "assignments": assignments,
    }


def _target_top_left(
    red_bbox: tuple[int, int, int, int],
    socket_bbox: tuple[int, int, int, int],
    dirv: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, red_bbox)
    sr0, sc0, sr1, sc1 = map(int, socket_bbox)
    height = int(r1 - r0)
    width = int(c1 - c0)
    dr, dc = map(int, dirv)

    if (dr, dc) == (1, 0):
        return int(sr0), int(sc0)
    if (dr, dc) == (-1, 0):
        return int(sr1 - height), int(sc0)
    if (dr, dc) == (0, 1):
        return int(sr0), int(sc0)
    return int(sr0), int(sc1 - width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the nearest red component into each large comb opening, leaving a black travel trace.

    Input structure:
    - The background is the dominant color, with large non-background, non-red comb/bracket structures.
    - Red rectangular components sit outside those structures and may be offset from the opening row or column.

    Rule:
    - For each structure, find the interior socket cells that face outward through its opening.
    - Match the structure with the red component whose Manhattan distance to those socket cells is minimal, even
      when the red is not axis-aligned with the opening.
    - Move the matched red into the cavity, painting all intermediate positions black; reds not selected by any
      structure disappear.

    Key insight:
    - The opening direction still determines the docking target, but the pairing now comes from nearest-distance
      comparison to the socket mouth instead of strict row/column ray alignment.
    """
    g = np.array(grid, dtype=int)
    RED = int(Color.RED)
    BLACK = int(Color.BLACK)

    assert int(RED) in set(map(int, np.unique(g)))

    analysis = _analyze_assignments(g)
    red_coords = analysis["red_coords"]
    red_bbox = analysis["red_bbox"]
    bg = int(analysis["bg"])

    assert red_coords, "expected at least one red component"
    assert analysis["structures"], "expected at least one non-bg non-red structure to dock into"

    out = g.copy()
    out[out == int(RED)] = int(bg)

    for item in analysis["assignments"]:
        rid = int(item["rid"])
        dirv = (int(item["dir"][0]), int(item["dir"][1]))
        socket_bbox = tuple(map(int, item["socket_bbox"]))

        coords = red_coords[rid]
        bbox = tuple(map(int, red_bbox[rid]))
        target_r0, target_c0 = _target_top_left(bbox, socket_bbox, dirv)
        shift_r = int(target_r0 - bbox[0])
        shift_c = int(target_c0 - bbox[1])

        for off_r, off_c in _intermediate_offsets(shift_r, shift_c):
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(RED)

    return out


# --------------- generator (D1 nearest-distance variant) ---------------

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
SLIDE_RANGE = (4, 8)
LATERAL_OFFSETS = (1, 2, 3)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MATCH_PLACEMENT_TRIES = 100
DISTRACTOR_PLACEMENT_TRIES = 120
MAX_RETRIES = 400


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = sc
    g[1, width - 2] = sc
    g[2:4, 1] = sc
    g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _make_structure_spec(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    base = _structure_array(width, teeth, bg, sc)
    red_h = min(int(teeth), 2)
    start = (width - teeth) // 2

    target_mask = np.zeros_like(base, dtype=bool)
    target_mask[2:2 + red_h, start:start + teeth] = True

    structure = np.rot90(base, k=rot_k)
    target_rot = np.rot90(target_mask.astype(int), k=rot_k) > 0

    coords = np.argwhere(structure == sc)
    struct_cells = {(int(r), int(c)) for r, c in coords}
    bbox = tuple(map(int, bbox_coords(coords)))
    openings = _opening_groups(structure, struct_cells, bbox, bg)
    assert openings

    opening_dir, opening_cells = max(openings.items(), key=lambda kv: len(_socket_strip(kv[1], kv[0])))
    sockets = _socket_strip(opening_cells, opening_dir)
    socket_bbox = tuple(map(int, bbox_coords(np.array(sockets, dtype=int))))
    target_bbox = tuple(map(int, bbox_coords(np.argwhere(target_rot))))
    th, tw = _bbox_shape(target_bbox)
    red_patch = np.full((th, tw), int(Color.RED), dtype=int)

    return {
        "structure": structure,
        "opening_dir": (int(opening_dir[0]), int(opening_dir[1])),
        "socket_bbox_local": socket_bbox,
        "target_bbox_local": target_bbox,
        "red_patch": red_patch,
    }


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    padding: int,
) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _bbox_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        mask[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = True
    return mask


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(struct_grid, spec["structure"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] != bg)
        global_coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in global_coords]
        cols = [c for _, c in global_coords]

        placed.append(
            {
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "opening_dir": spec["opening_dir"],
                "socket_bbox": _translate_bbox(spec["socket_bbox_local"], r0, c0),
                "target_bbox": _translate_bbox(spec["target_bbox_local"], r0, c0),
                "red_patch": spec["red_patch"].copy(),
            }
        )

    return struct_grid, placed


def _path_mask_for_move(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    start_top: int,
    start_left: int,
    target_top: int,
    target_left: int,
) -> np.ndarray:
    h, w = grid_shape
    ph, pw = patch_shape
    shift_r = int(target_top - start_top)
    shift_c = int(target_left - start_left)
    mask = np.zeros((h, w), dtype=bool)

    for off_r, off_c in _intermediate_offsets(shift_r, shift_c):
        r0 = int(start_top + off_r)
        c0 = int(start_left + off_c)
        mask[r0:r0 + ph, c0:c0 + pw] = True
    return mask


def _candidate_starts(item: dict) -> list[tuple[int, int]]:
    target_r0, target_c0 = map(int, item["target_bbox"][:2])
    dr, dc = map(int, item["opening_dir"])
    candidates: list[tuple[int, int]] = []

    for slide in range(SLIDE_RANGE[0], SLIDE_RANGE[1] + 1):
        offsets = list(LATERAL_OFFSETS)
        random.shuffle(offsets)
        for lateral in offsets:
            for sign in (-1, 1):
                off = int(sign * lateral)
                if (dr, dc) == (1, 0):
                    candidates.append((int(target_r0 + slide), int(target_c0 + off)))
                elif (dr, dc) == (-1, 0):
                    candidates.append((int(target_r0 - slide), int(target_c0 + off)))
                elif (dr, dc) == (0, 1):
                    candidates.append((int(target_r0 + off), int(target_c0 + slide)))
                else:
                    candidates.append((int(target_r0 + off), int(target_c0 - slide)))

    random.shuffle(candidates)
    return candidates


def _place_matched_reds(
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    h, w = struct_grid.shape
    bbox_mask = _bbox_mask((h, w), [item["bbox"] for item in placed])
    struct_mask = struct_grid != bg
    order = list(range(len(placed)))
    random.shuffle(order)

    matched_items: list[dict | None] = [None] * len(placed)

    def rec(
        pos: int,
        inp: np.ndarray,
        red_mask: np.ndarray,
        reserved_paths: np.ndarray,
    ) -> np.ndarray | None:
        if pos == len(order):
            return inp

        idx = order[pos]
        item = placed[idx]
        patch = item["red_patch"]
        ph, pw = patch.shape
        target_r0, target_c0 = map(int, item["target_bbox"][:2])
        other_bbox_mask = bbox_mask.copy()
        br0, bc0, br1, bc1 = map(int, item["bbox"])
        other_bbox_mask[br0:br1 + 1, bc0:bc1 + 1] = False

        for start_r0, start_c0 in _candidate_starts(item)[:MATCH_PLACEMENT_TRIES]:
            start_r1 = int(start_r0 + ph - 1)
            start_c1 = int(start_c0 + pw - 1)
            if start_r0 < 0 or start_c0 < 0 or start_r1 >= h or start_c1 >= w:
                continue
            if bool((inp[start_r0:start_r0 + ph, start_c0:start_c0 + pw] != bg).any()):
                continue
            if bool(other_bbox_mask[start_r0:start_r0 + ph, start_c0:start_c0 + pw].any()):
                continue
            dilated_red = _dilate4(red_mask)
            if bool(dilated_red[start_r0:start_r0 + ph, start_c0:start_c0 + pw].any()):
                continue

            path_mask = _path_mask_for_move(
                (h, w),
                patch.shape,
                start_r0,
                start_c0,
                target_r0,
                target_c0,
            )
            if bool((path_mask & reserved_paths).any()):
                continue
            if bool((path_mask & red_mask).any()):
                continue
            if bool((path_mask & struct_mask).any()):
                continue
            if bool((path_mask & other_bbox_mask).any()):
                continue

            next_inp = inp.copy()
            paste(next_inp, patch, start_r0, start_c0, transparent=bg)
            next_red_mask = red_mask.copy()
            next_red_mask[start_r0:start_r0 + ph, start_c0:start_c0 + pw] = True
            next_reserved = reserved_paths | path_mask

            matched_items[idx] = {
                "patch": patch.copy(),
                "input_bbox": (start_r0, start_c0, start_r1, start_c1),
                "target_bbox": (
                    target_r0,
                    target_c0,
                    target_r0 + ph - 1,
                    target_c0 + pw - 1,
                ),
                "path_mask": path_mask.copy(),
                "opening_dir": item["opening_dir"],
                "socket_bbox": item["socket_bbox"],
            }

            solved = rec(pos + 1, next_inp, next_red_mask, next_reserved)
            if solved is not None:
                return solved
            matched_items[idx] = None

        return None

    solved_inp = rec(
        0,
        struct_grid.copy(),
        np.zeros((h, w), dtype=bool),
        np.zeros((h, w), dtype=bool),
    )
    if solved_inp is None:
        return None
    return solved_inp, [item for item in matched_items if item is not None]


def _render_expected(inp: np.ndarray, matched_items: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg = int(get_bg(inp))
    out[out == int(Color.RED)] = int(bg)

    for item in matched_items:
        out[item["path_mask"]] = int(Color.BLACK)
        target_r0, target_c0 = map(int, item["target_bbox"][:2])
        paste(out, item["patch"], target_r0, target_c0, transparent=bg)
    return out


def _assignment_input_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    analysis = _analyze_assignments(grid)
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
    bbox_mask = _bbox_mask((h, w), [item["bbox"] for item in placed])
    trail_mask = np.zeros((h, w), dtype=bool)
    for item in matched_items:
        trail_mask |= item["path_mask"]

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        success = False
        for _try in range(DISTRACTOR_PLACEMENT_TRIES):
            red_mask = inp == int(Color.RED)
            blocked = bbox_mask | trail_mask | (inp != bg) | _dilate4(red_mask)
            canvas = np.full((h, w), bg, dtype=int)
            canvas[blocked] = sc

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


def _is_visible_variant_case(inp: np.ndarray) -> bool:
    analysis = _analyze_assignments(inp)
    red_boxes = analysis["red_bbox"]
    misaligned = 0
    for item in analysis["assignments"]:
        rb = tuple(map(int, red_boxes[item["rid"]]))
        sb = tuple(map(int, item["socket_bbox"]))
        dr, dc = map(int, item["dir"])
        if (dr, dc) in ((1, 0), (-1, 0)) and int(rb[1]) != int(sb[1]):
            misaligned += 1
        if (dr, dc) in ((0, 1), (0, -1)) and int(rb[0]) != int(sb[0]):
            misaligned += 1
    return int(misaligned) == len(analysis["assignments"])


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - n_structs)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - n_structs)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for _ in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        specs.append(_make_structure_spec(width, teeth, rot_k, bg, sc))
    specs.sort(key=lambda spec: np.count_nonzero(spec["structure"] != bg), reverse=True)

    placed_result = _place_structures(specs, h, w, bg)
    if placed_result is None:
        return None
    struct_grid, placed = placed_result

    matched_result = _place_matched_reds(struct_grid, placed, bg)
    if matched_result is None:
        return None
    inp, matched_items = matched_result

    expected_boxes = sorted(tuple(map(int, item["input_bbox"])) for item in matched_items)
    if _assignment_input_boxes(inp) != expected_boxes:
        return None

    if not _place_distractors(inp, placed, matched_items, bg, sc, distractor_count):
        return None

    if _assignment_input_boxes(inp) != expected_boxes:
        return None
    if not _is_visible_variant_case(inp):
        return None

    expected = _render_expected(inp, matched_items)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D1/3")
