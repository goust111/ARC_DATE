# Combo variant of 8b9c3697 (index 18)
# combo_spec: D6/0+D10/1
# Name: 链式处理紧凑裁剪

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
# - Base task: each comb-like structure independently proposes a visible red block,
#   then a global best-first pass assigns reds to structures. Output keeps the full
#   input canvas.
# - D10/1 changes the assignment phase: structures are processed left-to-right, and
#   every earlier claim removes one red from the pool before later structures are
#   evaluated.
# - D6/0 changes only the framing phase: after all structures have either claimed a
#   red or stayed unmatched, the finished full-canvas scene is tight-cropped to the
#   minimal rectangle containing every non-background output cell.
# - Small fused trace:
#   1. Two face-to-face structures see the same red corridor block.
#   2. The right structure is closer, so the original global rule would award it.
#   3. The combo rule still gives that red to the left structure because the chain
#      order is left-to-right.
#   4. The right structure then has no remaining visible match and stays empty.
#   5. Only after that full scene is rendered do we crop away the outer background.
#
# b) Complexity calibration
# - The generator always creates one spatially obvious shared-candidate pair, so the
#   chain effect is visible and not drowned out by unrelated distractors.
# - Optional third structures remain independent and preserve the original
#   multi-object feel without adding several competing shared corridors.
# - Tight cropping is enforced with a positive outer margin in the pre-crop output,
#   so the D6 change is visible in every sample instead of occasionally degenerating
#   to the original full-frame behavior.
#
# c) Implicit-assumption review of main()
# - Original assumption that each structure can choose its best red once up front is
#   invalid here; later structures must rescan against the remaining red pool.
# - Original assumption that output shape equals input shape is invalid; cropping
#   must happen strictly after all slide/trail drawing is complete.
# - `out[out == RED] = bg` remains valid because all input reds must disappear unless
#   explicitly redrawn as matched final placements.
# - The crop remains well-defined because structures always survive, so the final
#   foreground is never empty.
#
# d) Placement legality
# - Structures are arranged on one horizontal band, non-overlapping, and at least
#   one cell away from all four borders in the final full-canvas output.
# - The shared red block is constructed by exact coordinate equality between the left
#   structure's farther corridor position and the right structure's nearer corridor
#   position.
# - Distractors are placed off all structure rays and with a 1-cell buffer from
#   other red components, preventing accidental extra matches or merged red blobs.
#
# e) Visual observability
# - The ordering rule is purely spatial: left structure first, then the next one to
#   its right. No color ids, component ids, or other hidden quantities participate.
# - The crop rule is also visible: all outer background-only rows and columns vanish.
# - Across examples, humans can see that the earlier structure sometimes steals a red
#   that a nearer later structure also sees, which separates this combo from the
#   original task.
#
# f) Visual cleanliness
# - Only one shared corridor is introduced, keeping the scene readable.
# - Structures remain separated, and black trails stay on straight docking corridors,
#   so the crop does not collapse several unrelated signals into clutter.
#
# g) Degeneracy checks enforced by the generator
# - `main(inp)` must equal the intended cropped output.
# - The cropped chain-order output must differ from a cropped original-style
#   global-best assignment on the same input; otherwise the sample is rejected.
# - The pre-crop output must have a positive background margin on all four sides so
#   the crop is guaranteed to shrink the frame.
#
# h) New assumptions introduced by the fused solver
# - Structure order is inferred from visible x-position (`bbox` left edge, then top).
# - Shared-red samples rely on later structures possibly becoming unmatched after an
#   earlier claim; the generator guarantees this with a face-to-face pair.
# - Residual tie cases are avoided by generation; the solver still uses only visible
#   geometry (hit count, distance, then red position) for deterministic fallback.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_CHOICES = (2, 3)
SLIDE_RANGE = (4, 8)
THIRD_WIDTH_TEETH = ((7, 1), (8, 2))
THIRD_SLIDE_RANGE = (4, 5)
OUTPUT_MARGIN = 1

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 500
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Return the minimal crop containing all non-background cells."""
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0, "expected non-background output"
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _output_has_margin(grid: np.ndarray, bg: int, margin: int) -> bool:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return False
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    h, w = grid.shape
    return r0 >= margin and c0 >= margin and r1 < h - margin and c1 < w - margin


def _rotate_dir(dr: int, dc: int, rot_k: int) -> tuple[int, int]:
    for _ in range(int(rot_k) % 4):
        dr, dc = -int(dc), int(dr)
    return int(dr), int(dc)


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    """Build the normalized 4-row comb with opening facing down."""
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start : start + teeth] = sc
    return g


def _structure_spec(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    """Return rotated structure patch plus the relative dock shape it accepts."""
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


def _apply_slide(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    shift_r: int,
    shift_c: int,
    red_val: int,
    black_val: int,
) -> None:
    """Paint black intermediate placements and a red final placement."""
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


def _shift_to_socket(
    red_bbox: tuple[int, int, int, int], dirv: tuple[int, int], anchor: tuple[int, int]
) -> tuple[int, int]:
    """Return the translation that docks the structure-facing side onto the socket."""
    r0, c0, r1, c1 = map(int, red_bbox)
    dr, dc = map(int, dirv)
    ar, ac = map(int, anchor)
    tr, tc = int(ar + dr), int(ac + dc)

    if dr == 0 and dc == 1:
        return 0, int(tc - c0)
    if dr == 0 and dc == -1:
        return 0, int(tc - c1)
    if dr == 1 and dc == 0:
        return int(tr - r0), 0
    return int(tr - r1), 0


def _extract_red_data(
    g: np.ndarray, red_val: int
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_comps = components_mask(g == int(red_val), connectivity=4)
    assert red_comps, "expected at least one red component"

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


def _extract_structure_data(
    g: np.ndarray, bg: int, red_val: int
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    struct_mask = (g != int(bg)) & (g != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox


def _best_candidate_for_structure(
    g: np.ndarray,
    sid: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    red_bbox: list[tuple[int, int, int, int]],
    used_red: set[int],
    bg: int,
    red_val: int,
) -> tuple[tuple[int, int, int, int], int, tuple[int, int], tuple[int, int]] | None:
    """
    Return the best visible red for one structure.

    Score order is geometric and human-visible:
    1. more socket hits
    2. smaller minimum distance
    3. leftmost visible red
    4. uppermost visible red
    """
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

        rid_best = max(
            hits.keys(),
            key=lambda k: (
                int(hits[k][0]),
                -int(hits[k][1]),
                -int(red_bbox[int(k)][1]),
                -int(red_bbox[int(k)][0]),
            ),
        )
        cnt, mind, anchor = hits[rid_best]
        score = (
            int(cnt),
            -int(mind),
            -int(red_bbox[int(rid_best)][1]),
            -int(red_bbox[int(rid_best)][0]),
        )
        cand = (score, int(rid_best), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1])))
        if best is None or cand[0] > best[0]:
            best = cand

    return best


def _structure_order(struct_bbox: list[tuple[int, int, int, int]]) -> list[int]:
    return sorted(
        range(len(struct_bbox)),
        key=lambda sid: (
            int(struct_bbox[int(sid)][1]),
            int(struct_bbox[int(sid)][0]),
            int(struct_bbox[int(sid)][3]),
            int(struct_bbox[int(sid)][2]),
        ),
    )


def _chain_assignments(
    g: np.ndarray,
    bg: int,
    red_val: int,
    red_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for sid in _structure_order(struct_bbox):
        best = _best_candidate_for_structure(
            g, sid, struct_coords, struct_bbox, red_at, red_bbox, used_red, bg, red_val
        )
        if best is None:
            continue
        _score, rid, dirv, anchor = best
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1]))))
    return assignments


def _global_assignments(
    g: np.ndarray,
    bg: int,
    red_val: int,
    red_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    struct_best: list[tuple[tuple[int, int, int, int], int, tuple[int, int], tuple[int, int], int]] = []
    for sid in range(len(struct_coords)):
        best = _best_candidate_for_structure(
            g, sid, struct_coords, struct_bbox, red_at, red_bbox, set(), bg, red_val
        )
        if best is None:
            continue
        score, rid, dirv, anchor = best
        struct_best.append((score, int(rid), dirv, anchor, int(sid)))

    struct_best.sort(reverse=True, key=lambda item: item[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, dirv, anchor, _sid in struct_best:
        if int(rid) in used_red:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dirv[0]), int(dirv[1])), (int(anchor[0]), int(anchor[1]))))
    return assignments


def _render_full_output(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    red_coords, red_bbox, red_at = _extract_red_data(g, red_val)
    struct_coords, struct_bbox = _extract_structure_data(g, bg, red_val)

    if mode == "chain":
        assignments = _chain_assignments(g, bg, red_val, red_bbox, red_at, struct_coords, struct_bbox)
    elif mode == "global":
        assignments = _global_assignments(g, bg, red_val, red_bbox, red_at, struct_coords, struct_bbox)
    else:
        raise ValueError(f"unknown mode: {mode}")

    out = g.copy()
    out[out == int(red_val)] = int(bg)
    for rid, dirv, anchor in assignments:
        shift_r, shift_c = _shift_to_socket(red_bbox[int(rid)], dirv, anchor)
        _apply_slide(out, red_coords[int(rid)], int(shift_r), int(shift_c), red_val, black_val)
    return out


def _visible_red_sets(grid: np.ndarray) -> tuple[list[set[int]], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_val = int(Color.RED)

    red_comps = components_mask(g == int(red_val), connectivity=4)
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = int(rid)

    struct_coords, struct_bbox = _extract_structure_data(g, bg, red_val)
    visible: list[set[int]] = []
    for sid, pts in enumerate(struct_coords):
        r0, c0, r1, c1 = map(int, struct_bbox[int(sid)])
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


def _structure_coords_abs(spec: dict, top: int, left: int, sc: int) -> list[tuple[int, int]]:
    coords = np.argwhere(spec["patch"] == int(sc))
    return [(int(top + r), int(left + c)) for r, c in coords]


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
    inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, count: int
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
                    if bool(blocked[r0 : r0 + rh, c0 : c0 + rw].any()):
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


def _pair_layout(
    left_spec: dict, right_spec: dict, slide_left: int, slide_right: int, margin_left: int
) -> tuple[int, int]:
    left_dock = left_spec["dock_bbox_rel"]
    right_dock = right_spec["dock_bbox_rel"]
    left_col = int(margin_left)
    shared_left = int(left_col + left_dock[1] + slide_left)
    right_col = int(shared_left + slide_right - right_dock[1])
    return int(left_col), int(right_col)


def _expected_full_output(inp: np.ndarray, matched: list[dict], bg: int) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    out[out == int(Color.RED)] = int(bg)
    for item in matched:
        dr, dc = map(int, item["dir"])
        slide = int(item["slide"])
        shift_r = -int(slide * dr)
        shift_c = -int(slide * dc)
        _apply_slide(out, item["coords"], shift_r, shift_c, int(Color.RED), int(Color.BLACK))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide matched red components into comb sockets in chain order, then tight-crop the finished scene.

    Input structure:
    - The dominant color is background; large non-background, non-red components are comb/bracket structures.
    - Red rectangles sit outside the structures, and at least one corridor red may be visible from more than one structure.

    Rule:
    - Detect interior socket cells as background cells inside a structure bbox that touch the structure.
    - Sort structures from left to right and process them one by one; each structure picks its best visible red from the still-unused pool, slides it into the socket, and paints intermediate positions black.
    - Remove all unmatched reds, keep all structures fixed, and finally crop the completed output to the smallest rectangle containing every non-background cell.

    Key insight: The crucial change is stateful allocation before rendering the crop: an earlier structure can consume a shared red that a later structure would otherwise win, and only the fully resolved scene determines the cropped frame.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    full = _render_full_output(g, mode="chain")
    return _tight_crop_non_bg(full, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])

    n_structs = random.choice(STRUCT_COUNT_CHOICES)
    pair_choices = STRUCT_WIDTH_TEETH if n_structs == 2 else THIRD_WIDTH_TEETH
    pair_width, pair_teeth = random.choice(pair_choices)
    left_spec = _structure_spec(pair_width, pair_teeth, 1, bg, sc)   # opening right
    right_spec = _structure_spec(pair_width, pair_teeth, 3, bg, sc)  # opening left

    slide_left_hi = SLIDE_RANGE[1] if n_structs == 2 else 6
    slide_right_hi = SLIDE_RANGE[1] if n_structs == 2 else 5
    slide_left = random.randint(max(SLIDE_RANGE[0], 5), slide_left_hi)
    slide_right = random.randint(SLIDE_RANGE[0], min(slide_right_hi, slide_left - 1))

    margin_left = random.randint(OUTPUT_MARGIN, OUTPUT_MARGIN + 2)
    left_col, right_col = _pair_layout(left_spec, right_spec, slide_left, slide_right, margin_left)

    structs: list[dict] = [
        {"name": "left_shared", "spec": left_spec, "top": 0, "left": int(left_col)},
        {"name": "right_shared", "spec": right_spec, "top": 0, "left": int(right_col)},
    ]

    if n_structs == 3:
        third_width, third_teeth = random.choice(THIRD_WIDTH_TEETH)
        third_spec = _structure_spec(third_width, third_teeth, 1, bg, sc)  # opening right
        third_slide = random.randint(*THIRD_SLIDE_RANGE)
        third_left = int(right_col + right_spec["shape"][1] + 1)
        structs.append(
            {
                "name": "third_private",
                "spec": third_spec,
                "top": 0,
                "left": int(third_left),
                "slide": int(third_slide),
            }
        )

    max_h = max(int(item["spec"]["shape"][0]) for item in structs)
    right_extent = max(int(item["left"] + item["spec"]["shape"][1]) for item in structs)
    min_h = max(int(GRID_H_RANGE[0]), int(max_h + 2 * OUTPUT_MARGIN))

    if n_structs == 3:
        third = structs[2]
        third_dock = third["spec"]["dock_bbox_rel"]
        third_red_right = int(third["left"] + third_dock[3] + third["slide"] + 1)
        right_extent = max(int(right_extent), int(third_red_right))
    min_w = max(int(GRID_W_RANGE[0]), int(right_extent + OUTPUT_MARGIN))

    if min_h > int(GRID_H_RANGE[1]) or min_w > int(GRID_W_RANGE[1]):
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    row = random.randint(OUTPUT_MARGIN, h - max_h - OUTPUT_MARGIN)
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

    matched: list[dict] = []

    left_item = structs[0]
    left_shared = _translate(
        _translate(left_item["spec"]["dock_rel"], left_item["top"], left_item["left"]),
        slide_left * left_item["spec"]["dir"][0],
        slide_left * left_item["spec"]["dir"][1],
    )
    if any(not (0 <= r < h and 0 <= c < w) for r, c in left_shared):
        return None
    if any(int(inp[r, c]) != int(bg) for r, c in left_shared):
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
    if len(visible_sets) < 2 or len(visible_sets[0] & visible_sets[1]) < 1:
        return None

    expected_full = _expected_full_output(inp, matched, bg)
    if not _output_has_margin(expected_full, bg, OUTPUT_MARGIN):
        return None
    expected = _tight_crop_non_bg(expected_full, bg)

    if expected.shape[0] >= inp.shape[0] or expected.shape[1] >= inp.shape[1]:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    baseline = _tight_crop_non_bg(_render_full_output(inp, mode="global"), bg)
    if np.array_equal(baseline, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant 18")
