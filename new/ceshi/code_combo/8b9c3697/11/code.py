# Combo variant of 8b9c3697 (index 11)
# combo_spec: D9/1+D10/1
# Name: 链式固定步长推进

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# =====================================
# spec_variant:
#   visible_layout:
#     - Keep the original comb/bracket structures and rectangular red pieces.
#     - Force a readable left-to-right chain of three structures:
#       1. a tall left structure opening right,
#       2. a central structure opening down,
#       3. a rightmost independent structure opening up.
#     - The first and second structures are arranged so that the first one's future black
#       trail crosses the second one's only exact-distance ray, but the crossing cells are
#       background in the input. Therefore the second structure looks solvable from the input.
#   concrete_behavior_trace:
#     - Original task:
#       1. Any visible red can slide all the way into its socket; distance is whatever the
#          current alignment demands.
#       2. Structures do not need stepwise state updates; later matches are decided from the
#          same initial visibility picture.
#     - Fused variant:
#       1. Every structure has a coded slide length = back-wall width - 2.
#       2. A red is eligible only if it is exactly that many cells away along the opening ray.
#       3. Structures are processed left-to-right on an evolving board.
#       4. After structure 1 slides, its black trail occupies cells that were previously empty.
#       5. Structure 2 must rescan on that updated board, so its former exact-distance red is
#          now hidden behind the fresh black barrier and disappears in the final output.
#       6. Structure 3 stays independent and still succeeds, proving the rule is not simply
#          "only the first structure moves".
#   complexity_calibration:
#     - Only one engineered interaction pair is used, which makes the chain effect visible
#       without flooding the grid with competing explanations.
#     - Widths are kept in the original 7-9 range, so the new numeric rule is inferable from
#       repeated geometry while remaining low-arity.
#     - Two structures succeed with different widths, giving repeated evidence that wider
#       structures produce longer black trails.
#   implicit_assumption_audit:
#     - The original solver assumed the slide offset could be inferred from the chosen red's
#       current position. That is false here; offset must come from structure geometry instead.
#     - The original matching phase could score all structures once, then filter reused reds.
#       That is false here; later structures must be rescanned after each earlier slide because
#       new black cells can block a previously valid exact-distance ray.
#     - Recomputing structure components from the evolving grid would also be false, because
#       black trails are dynamic obstacles, not new structure pixels. The solver therefore keeps
#       the original structure geometry fixed and only updates the obstacle/red layer.
#   placement_legality:
#     - Generator places all three structures deterministically inside bounds, with disjoint
#       structure footprints and exact-distance red placements.
#     - The blocking trail crosses only open background cells on the second structure's ray; it
#       does not invade any structure body.
#     - Distractors are banned from all socket rays so they cannot create accidental exact hits.
#   observability:
#     - Left-to-right order is purely spatial and visible.
#     - The coded distance is visible from the longer side of each comb, not from color ids or
#       any hidden ordering.
#     - The second structure's red looks initially well-aimed, but vanishes because the first
#       structure is resolved first and leaves a blocking trail behind.
#   cleanliness:
#     - The interaction pair is isolated on the left/middle, while the third structure sits to
#       the right as a clean control case.
#     - Only one trail crosses another structure's ray, so the board stays readable.


GRID_H_RANGE = (20, 23)
GRID_W_RANGE = (25, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

PRIMARY_LEFT_SPEC = (9, 3)      # width, teeth; slide = 7
PRIMARY_BLOCKED_SPEC = (8, 2)   # width, teeth; slide = 6
THIRD_SPEC_CHOICES = ((7, 1), (8, 2), (9, 3))

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 10
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 6

MAX_RETRIES = 500
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

    return {
        "width": int(width),
        "teeth": int(teeth),
        "rot_k": int(rot_k),
        "patch": patch,
        "shape": tuple(map(int, patch.shape)),
        "dir": _rotate_dir(1, 0, rot_k),
        "dock_rel": dock_coords,
        "dock_bbox_rel": tuple(map(int, bbox_coords(np.argwhere(dock_mask)))),
        "slide": int(width - 2),
    }


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            grid[int(r), int(c)] = int(color)


def _place_structure(
    canvas: np.ndarray,
    struct_grid: np.ndarray,
    *,
    name: str,
    spec: dict,
    top: int,
    left: int,
    bg: int,
    sc: int,
) -> dict | None:
    ph, pw = map(int, spec["shape"])
    h, w = canvas.shape
    if top < 0 or left < 0 or int(top + ph) > h or int(left + pw) > w:
        return None
    if bool((struct_grid[int(top) : int(top + ph), int(left) : int(left + pw)] != int(bg)).any()):
        return None

    patch = spec["patch"]
    paste(canvas, patch, int(top), int(left), transparent=bg)
    paste(struct_grid, patch, int(top), int(left), transparent=bg)

    coords_arr = np.argwhere(patch == int(sc))
    coords = [(int(top + r), int(left + c)) for r, c in coords_arr]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    dock_abs = _translate(spec["dock_rel"], int(top), int(left))
    return {
        "name": name,
        "spec": spec,
        "top": int(top),
        "left": int(left),
        "coords": coords,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "dock_abs": dock_abs,
    }


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
                out[int(rr), int(cc)] = int(black_val)

    for r, c in coords:
        rr = int(r + shift_r)
        cc = int(c + shift_c)
        if 0 <= rr < h and 0 <= cc < w:
            out[int(rr), int(cc)] = int(red_val)


def _fixed_slide_from_bbox(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return int(max(r1 - r0 + 1, c1 - c0 + 1) - 2)


def _red_state(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    red_comps = components_mask(grid == int(Color.RED), connectivity=4)
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


def _best_candidate_for_structure(
    g: np.ndarray,
    sid: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    slide: int,
    red_val: int,
    bg: int,
) -> tuple[int, tuple[int, int], int] | None:
    h, w = g.shape
    s = struct_coords[int(sid)]
    sr0, sc0, sr1, sc1 = map(int, struct_bbox[int(sid)])
    best = None

    for dr, dc in _DIRS:
        hits: dict[int, list] = defaultdict(lambda: [0, None])
        for r, c in s:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                continue

            rr, cc = int(nr), int(nc)
            steps = 0
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                rr += int(dr)
                cc += int(dc)
                steps += 1

            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(red_val) and int(steps) == int(slide):
                rid = int(red_at[(int(rr), int(cc))])
                rec = hits[rid]
                rec[0] += 1
                if rec[1] is None:
                    rec[1] = (int(r), int(c))

        if not hits:
            continue

        rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(k)))
        cnt, anchor = hits[rid_best]
        cand = (int(cnt), -int(rid_best), (int(dr), int(dc)), anchor)
        if best is None or cand > best:
            best = cand

    if best is None:
        return None

    cnt, neg_rid, dirv, _anchor = best
    return int(-neg_rid), (int(dirv[0]), int(dirv[1])), int(cnt)


def _candidate_rids_for_structure(
    g: np.ndarray,
    sid: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
    red_at: dict[tuple[int, int], int],
    slide: int,
    red_val: int,
    bg: int,
) -> set[int]:
    h, w = g.shape
    s = struct_coords[int(sid)]
    sr0, sc0, sr1, sc1 = map(int, struct_bbox[int(sid)])
    seen: set[int] = set()

    for dr, dc in _DIRS:
        for r, c in s:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                continue

            rr, cc = int(nr), int(nc)
            steps = 0
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                rr += int(dr)
                cc += int(dc)
                steps += 1

            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(red_val) and int(steps) == int(slide):
                seen.add(int(red_at[(int(rr), int(cc))]))

    return seen


def _run_chain(grid: np.ndarray, *, dynamic: bool, return_trace: bool = False):
    g0 = np.array(grid, dtype=int)
    bg = int(get_bg(g0))
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    struct_mask = (g0 != int(bg)) & (g0 != int(red_val))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure"

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

    work = g0.copy()
    out = g0.copy()
    out[out == int(red_val)] = int(bg)

    trace = []
    frozen = g0.copy()
    source_grid = frozen

    for sid in order:
        if dynamic:
            source_grid = work

        red_coords, _red_bbox, red_at = _red_state(source_grid)
        slide = _fixed_slide_from_bbox(struct_bbox[int(sid)])
        visible = sorted(
            _candidate_rids_for_structure(
                source_grid,
                int(sid),
                struct_coords,
                struct_bbox,
                red_at,
                int(slide),
                red_val,
                bg,
            )
        )
        best = _best_candidate_for_structure(
            source_grid,
            int(sid),
            struct_coords,
            struct_bbox,
            red_at,
            int(slide),
            red_val,
            bg,
        )

        step_rec = {
            "sid": int(sid),
            "slide": int(slide),
            "visible_before": visible,
            "assigned": None,
        }

        if best is None:
            trace.append(step_rec)
            continue

        rid, (dr, dc), hit_count = best
        step_rec["assigned"] = {
            "rid": int(rid),
            "dir": (int(dr), int(dc)),
            "hit_count": int(hit_count),
        }
        trace.append(step_rec)

        shift_r = -int(slide * dr)
        shift_c = -int(slide * dc)
        coords = red_coords[int(rid)]

        _apply_slide(out, coords, int(shift_r), int(shift_c), red_val, black_val)
        if dynamic:
            _apply_slide(work, coords, int(shift_r), int(shift_c), red_val, black_val)

    return (out, trace) if return_trace else out


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
                if not (int(r0) < nr < int(r1) and int(c0) < nc < int(c1)):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[int(rr), int(cc)] = True
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
            inp[int(r0) : int(r0 + rh), int(c0) : int(c0 + rw)] = int(Color.RED)
            blocked[
                max(0, int(r0) - 1) : min(h, int(r0 + rh) + 1),
                max(0, int(c0) - 1) : min(w, int(c0 + rw) + 1),
            ] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _planned_output(inp: np.ndarray, active: list[dict], bg: int) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    out[out == int(Color.RED)] = int(bg)
    for item in active:
        dr, dc = map(int, item["spec"]["dir"])
        slide = int(item["spec"]["slide"])
        shift_r = -int(slide * dr)
        shift_c = -int(slide * dc)
        _apply_slide(out, item["input_red"], shift_r, shift_c, int(Color.RED), int(Color.BLACK))
    return out


def _fused_behavior_holds(inp: np.ndarray) -> bool:
    dynamic_out, dynamic_trace = _run_chain(inp, dynamic=True, return_trace=True)
    static_out, static_trace = _run_chain(inp, dynamic=False, return_trace=True)

    if np.array_equal(dynamic_out, static_out):
        return False
    if len(dynamic_trace) < 3 or len(static_trace) < 3:
        return False

    second_dyn = dynamic_trace[1]
    second_static = static_trace[1]
    if not second_static["visible_before"]:
        return False
    if second_static["assigned"] is None:
        return False
    if second_dyn["assigned"] is not None:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide exact-distance red rectangles into comb sockets in left-to-right chain order.

    Input structure:
    - The background is the most common color. Large non-bg, non-red comb/bracket structures
      appear in a readable spatial chain, and red rectangles sit outside them.
    - Each structure's longer side encodes a required slide length: width minus two.

    Rule:
    - For a structure to match a red, the red must lie on the opening ray and its near face must
      be exactly the encoded number of cells away; shorter or longer alignments do not count.
    - Process structures strictly from left to right on an evolving grid. After each successful
      slide, paint the traversed cells black and keep that new board state for the next structure.
    - Reds that are never used disappear.

    Key insight: The slide distance is read from structure geometry, and later structures must be
    rescanned after earlier trails are drawn because those fresh black cells can block a red that
    was visible in the input.
    """
    return _run_chain(grid, dynamic=True, return_trace=False)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])

    s1 = _structure_spec(PRIMARY_LEFT_SPEC[0], PRIMARY_LEFT_SPEC[1], 1, bg, sc)    # opening right
    s2 = _structure_spec(PRIMARY_BLOCKED_SPEC[0], PRIMARY_BLOCKED_SPEC[1], 0, bg, sc)  # opening down
    w3, t3 = random.choice(THIRD_SPEC_CHOICES)
    s3 = _structure_spec(w3, t3, 2, bg, sc)  # opening up

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    inp = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)

    left = random.randint(1, 2)
    top1 = random.randint(6, min(9, h - int(s1["shape"][0]) - 3))
    top2 = int(top1 - 2)
    left2 = int(left + 4)

    left3 = int(w - int(s3["shape"][1]) - 2)
    top3_min = max(int(s3["slide"] + 1), int(top1 + 5))
    top3_max = h - int(s3["shape"][0]) - 2
    if top3_min > top3_max:
        return None
    top3 = random.randint(int(top3_min), int(top3_max))

    placed: list[dict] = []
    for name, spec, top, col in (
        ("leader", s1, top1, left),
        ("blocked", s2, top2, left2),
        ("independent", s3, top3, left3),
    ):
        item = _place_structure(inp, struct_grid, name=name, spec=spec, top=top, left=col, bg=bg, sc=sc)
        if item is None:
            return None
        placed.append(item)

    # Input reds are placed exactly slide cells away in the opening direction.
    for item in placed:
        dr, dc = map(int, item["spec"]["dir"])
        slide = int(item["spec"]["slide"])
        input_red = _translate(item["dock_abs"], int(slide * dr), int(slide * dc))
        item["input_red"] = input_red
        if any(not (0 <= r < h and 0 <= c < w) for r, c in input_red):
            return None
        if any(int(inp[r, c]) != int(bg) for r, c in input_red):
            return None
        _paint_cells(inp, input_red, int(Color.RED))

    active = [placed[0], placed[2]]
    matched_count = len(placed)
    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched_count)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched_count)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)
    if not _place_distractors(inp, struct_grid, placed, bg, distractor_count):
        return None

    expected = _planned_output(inp, active, bg)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if not _fused_behavior_holds(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant")
