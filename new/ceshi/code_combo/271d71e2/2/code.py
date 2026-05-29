# Combo variant of 271d71e2 (index 2)
# combo_spec: D1/1+D12/0
# Name: L形路径按距离排序
#
# Design analysis
# ===============
# a) Concrete behavior and traced example
#    - Base task: one same-color target line, one straight translation axis.
#      Each container moves `min(distance_to_line, grey_count)` cells and turns
#      that many grey cells orange from the head-facing side inward.
#    - This combo changes two things together:
#      1) Each mover now has two same-color target lines: a turn line and a
#         final line. The head first travels along the head-facing axis to the
#         turn line, then along the orthogonal axis to the final line.
#      2) Multiple movers are no longer independent. We first compute every
#         mover's full L-path distance, sort movers from near to far by that
#         total distance, then apply them one by one on the evolving canvas.
#         Earlier movers occupy space and can stop later movers early.
#    - Small example:
#      * A right-headed container has turn distance 4 and second segment 6, so
#        total L distance is 10. With 10 grey cells it reaches the final line.
#      * A second container has total distance 14 and would normally move 14.
#        But if A is processed first and its final box sits on B's first
#        corridor, B stops as soon as its next step would overlap A's moved box.
#      * Difference from either single change alone:
#        - not straight-line, because the path is inferred from an ordered pair
#          of lines;
#        - not independent multi-move, because the later mover's realized steps
#          can be smaller than its own grey cap and smaller than its full L
#          distance specifically due to earlier occupancy.
#
# b) Complexity calibration
#    - The generator keeps exactly two movers. That is enough to show ordering
#      while avoiding crowded ambiguous scenes.
#    - Signal colors are distinct per mover. This is a visual aid only; the
#      solver still accepts arbitrary non-bg/non-border signal colors.
#    - At least one mover always has `base_steps > first_dist`, so a turn is
#      truly used instead of the scene degenerating into a straight move.
#    - The generator prefers early blocking but does not require it. The actual
#      hard requirement is that ordered occupancy removes at least one step from
#      the later mover, so the D12 interaction is never degenerate.
#
# c) Implicit-assumption audit against the original solver
#    - Original assumption: a valid destination is collinear with the head.
#      Broken here. We must instead identify an ordered `(turn, final)` pair.
#    - Original assumption: movement vector is `direction * steps`.
#      Broken here. We now need piecewise translation along an L path.
#    - Original assumption: per-container processing is independent.
#      Broken here. Later legality checks depend on earlier moved occupancy.
#    - Original assumption: `steps = min(distance, grey_count)` is final.
#      Broken here. That value is now only an upper bound; collision can cap it.
#
# d) Placement legality guarantees in the generator
#    - Objects are generated from explicit specs, then rejected unless:
#      * all input objects stay inside bounds;
#      * input groups do not overlap and keep a one-cell moat;
#      * each mover's independent path is clear with respect to the other
#        mover's input objects;
#      * ordered simulation produces at least one collision-limited mover.
#    - Distances are strictly ordered, so the visible ordering cue is unique.
#
# e) Visual observability
#    - Each mover shows three same-color lines: head, turn, final. That makes
#      the L structure visible to a human in a few examples.
#    - The ordering cue is also visible: one mover has a shorter total L path
#      and reaches its destination, while the farther mover stops short despite
#      still having visible unused path and remaining grey budget.
#
# f) Cleanliness
#    - The generator uses only two movers and rejects scenes where the final
#      moved groups overlap or touch.
#
# g) Color-change inferability
#    - Grey->orange charging is unchanged from the base task. The only new
#      structure is geometric: two-leg paths plus ordered occupancy.
#
# h) New solver assumptions introduced here
#    - For each color there are exactly three relevant same-shape line objects:
#      head, turn, final. The solver still searches generically, but the
#      generator enforces that clean pattern.
#    - Ties in total L distance are resolved by reading order. The generator
#      avoids ties.
#    - A later mover may overlap only its own turn/final marker cells with its
#      moving head; body cells still require background.
#
# i) Generator support for those assumptions
#    - Distinct colors remove cross-matcher ambiguity.
#    - Second-segment lengths are at least line-length+1, keeping turn/final
#      markers visually separate.
#    - Metrics reject degenerate samples where ordered simulation equals the
#      independent L-path result.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}
SIGNAL_COLORS = [1, 2, 3, 4, 8, 9]


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _dominant_bg(grid: np.ndarray) -> int:
    cnt = Counter(map(int, grid.flatten()))
    return int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))


def _fill_order_local(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _head_tl_for_box(r0: int, c0: int, r1: int, c1: int, head_side: int) -> tuple[int, int]:
    if head_side == 0:
        return (r0 - 1, c0)
    if head_side == 1:
        return (r1 + 1, c0)
    if head_side == 2:
        return (r0, c0 - 1)
    return (r0, c1 + 1)


def _draw_line(grid: np.ndarray, tl: tuple[int, int], head_side: int, outer_h: int, outer_w: int, color: int) -> None:
    r, c = tl
    if head_side in (0, 1):
        grid[r, c : c + outer_w] = int(color)
    else:
        grid[r : r + outer_h, c] = int(color)


def _line_cells_from_tl(tl: tuple[int, int], head_side: int, outer_h: int, outer_w: int) -> set[tuple[int, int]]:
    r, c = tl
    if head_side in (0, 1):
        return {(r, cc) for cc in range(c, c + outer_w)}
    return {(rr, c) for rr in range(r, r + outer_h)}


def _translate_along_l(
    first_dir: tuple[int, int],
    second_dir: tuple[int, int],
    first_dist: int,
    steps: int,
) -> tuple[int, int]:
    if steps <= first_dist:
        return (first_dir[0] * steps, first_dir[1] * steps)
    rem = steps - first_dist
    return (
        first_dir[0] * first_dist + second_dir[0] * rem,
        first_dir[1] * first_dist + second_dir[1] * rem,
    )


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return any(abs(int(r - rr)) <= 1 and abs(int(c - cc)) <= 1 for r, c in a for rr, cc in b)


def _same_component_splits(
    g: np.ndarray,
    border: int,
    head_shape: set[tuple[int, int]],
    head_pts: set[tuple[int, int]],
    comp_pts: set[tuple[int, int]],
) -> list[dict]:
    H, W = g.shape
    rem = comp_pts - head_pts
    out = []
    if not rem:
        return out
    mask_rem = np.zeros((H, W), dtype=bool)
    for rr, cc in rem:
        mask_rem[int(rr), int(cc)] = True
    labr, nr = label(mask_rem, structure=STRUCT4)
    for k in range(1, int(nr) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
        if not pts:
            continue
        adj = False
        for r, c in pts:
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(border):
                    adj = True
                    break
            if adj:
                break
        if adj:
            continue
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        r00, c00 = int(min(rs)), int(min(cs))
        shp = {(int(r - r00), int(c - c00)) for r, c in pts}
        if shp != head_shape:
            continue
        out.append({"pts": pts, "shape": shp, "tl": (r00, c00)})
    return out


def _collect_items(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _dominant_bg(g)

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    all_signal_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_signal_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        border_pts = list(_rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist = {}
        for pr, pc in comp0["pts"]:
            d = min(abs(int(pr - br)) + abs(int(pc - bc)) for br, bc in border_pts)
            dist[(int(pr), int(pc))] = int(d)
            if best_d is None or d < best_d:
                best_d = int(d)
        if best_d is None:
            return None
        head_pts = {p for p, d in dist.items() if int(d) == int(best_d)}
        coords = np.array(list(head_pts), dtype=int)
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        if hc1 < c0:
            direction = (0, -1)
        elif hc0 > c1:
            direction = (0, 1)
        elif hr1 < r0:
            direction = (-1, 0)
        elif hr0 > r1:
            direction = (1, 0)
        else:
            direction = (0, 0)
        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    items = []
    for idx, cont in enumerate(containers):
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None or head_info["dir"] == (0, 0):
            continue

        color = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        first_dir = tuple(head_info["dir"])

        dests = []
        for d in comp_by_color.get(color, []):
            if d["shape"] != head_shape or d["pts"] == comp_pts:
                continue
            dests.append(d)
        dests.extend(_same_component_splits(g, BORDER, head_shape, head_pts, comp_pts))
        if len(dests) < 2:
            continue

        best_key = None
        best_path = None
        for turn in dests:
            dr1 = int(turn["tl"][0] - head_tl[0])
            dc1 = int(turn["tl"][1] - head_tl[1])
            if first_dir[0] != 0:
                if dc1 != 0 or dr1 * first_dir[0] <= 0:
                    continue
                first_dist = abs(dr1)
                second_axis_vertical = False
            else:
                if dr1 != 0 or dc1 * first_dir[1] <= 0:
                    continue
                first_dist = abs(dc1)
                second_axis_vertical = True

            for final in dests:
                if final is turn:
                    continue
                dr2 = int(final["tl"][0] - turn["tl"][0])
                dc2 = int(final["tl"][1] - turn["tl"][1])
                if second_axis_vertical:
                    if dc2 != 0 or dr2 == 0:
                        continue
                    second_dir = (1 if dr2 > 0 else -1, 0)
                    second_dist = abs(dr2)
                else:
                    if dr2 != 0 or dc2 == 0:
                        continue
                    second_dir = (0, 1 if dc2 > 0 else -1)
                    second_dist = abs(dc2)
                total_dist = int(first_dist + second_dist)
                key = (total_dist, first_dist, second_dist, int(turn["tl"][0]), int(turn["tl"][1]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_path = (turn, final, first_dist, second_dir, second_dist, total_dist)

        if best_path is None:
            continue

        turn, final, first_dist, second_dir, second_dist, total_dist = best_path
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        head_side = DIR_TO_SIDE[first_dir]
        grey_local = {
            (1 + rr, 1 + cc)
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        }
        grey_order_local = [
            (1 + rr, 1 + cc)
            for rr, cc in _fill_order_local(inner.shape[0], inner.shape[1], head_side)
            if (1 + rr, 1 + cc) in grey_local
        ]

        items.append(
            {
                "id": idx,
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "color": color,
                "head_side": head_side,
                "head_pts": head_pts,
                "head_tl": head_tl,
                "turn_pts": set(turn["pts"]),
                "final_pts": set(final["pts"]),
                "first_dir": first_dir,
                "second_dir": second_dir,
                "first_dist": int(first_dist),
                "second_dist": int(second_dist),
                "total_dist": int(total_dist),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "grey_order_local": grey_order_local,
                "base_steps": min(int(total_dist), int(len(grey_order_local))),
                "actual_steps": 0,
                "blocked": False,
            }
        )
    return bg, items


def _erase_item(grid: np.ndarray, bg: int, item: dict) -> None:
    r0, c0, r1, c1 = item["box"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    for r, c in item["head_pts"]:
        if int(grid[int(r), int(c)]) == int(item["color"]):
            grid[int(r), int(c)] = int(bg)


def _can_place_step(occupancy: np.ndarray, bg: int, item: dict, step: int) -> bool:
    H, W = occupancy.shape
    tr, tc = _translate_along_l(item["first_dir"], item["second_dir"], item["first_dist"], step)
    r0, c0, r1, c1 = item["box"]
    nr0, nc0, nr1, nc1 = int(r0 + tr), int(c0 + tc), int(r1 + tr), int(c1 + tc)
    if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
        return False
    if np.any(occupancy[nr0 : nr1 + 1, nc0 : nc1 + 1] != int(bg)):
        return False
    allowed = item["turn_pts"] | item["final_pts"]
    for r, c in item["head_pts"]:
        rr = int(r + tr)
        cc = int(c + tc)
        if rr < 0 or cc < 0 or rr >= H or cc >= W:
            return False
        val = int(occupancy[rr, cc])
        if val != int(bg) and (rr, cc) not in allowed:
            return False
    return True


def _apply_item(grid: np.ndarray, item: dict, steps: int) -> None:
    r0, c0, _r1, _c1 = item["box"]
    tr, tc = _translate_along_l(item["first_dir"], item["second_dir"], item["first_dist"], steps)
    patch = item["patch"].copy()
    for pr, pc in item["grey_order_local"][:steps]:
        if int(patch[pr, pc]) == GREY:
            patch[pr, pc] = int(ORANGE)
    paste(grid, patch, int(r0 + tr), int(c0 + tc), clip=False)
    for r, c in item["head_pts"]:
        grid[int(r + tr), int(c + tc)] = int(item["color"])


def _simulate_ordered(grid: np.ndarray, items: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int).copy()
    items2 = [dict(item) for item in items]
    items2.sort(key=lambda item: (int(item["total_dist"]), int(item["box"][0]), int(item["box"][1])))
    for item in items2:
        steps = 0
        if int(item["base_steps"]) > 0:
            occupancy = out.copy()
            _erase_item(occupancy, bg, item)
            while steps < int(item["base_steps"]) and _can_place_step(occupancy, bg, item, steps + 1):
                steps += 1
            item["blocked"] = steps < int(item["base_steps"])
        item["actual_steps"] = int(steps)
        if steps == 0:
            continue
        _erase_item(out, bg, item)
        _apply_item(out, item, steps)
    return out, items2


def _simulate_independent(grid: np.ndarray, items: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    for item in items:
        steps = int(item["base_steps"])
        if steps == 0:
            continue
        _erase_item(out, bg, item)
        _apply_item(out, item, steps)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, items = _collect_items(g)
    out, _ = _simulate_ordered(g, items, bg)
    return out


def _spec_full_disp(spec: dict) -> tuple[int, int]:
    return _translate_along_l(spec["first_dir"], spec["second_dir"], spec["first_dist"], spec["total_dist"])


def _spec_input_box(spec: dict) -> tuple[int, int, int, int]:
    dr, dc = _spec_full_disp(spec)
    fr, fc = spec["final_r"], spec["final_c"]
    r = int(fr - dr)
    c = int(fc - dc)
    return (r, c, r + spec["outer_h"] - 1, c + spec["outer_w"] - 1)


def _spec_geometry(spec: dict) -> dict:
    r0, c0, r1, c1 = _spec_input_box(spec)
    head_tl = _head_tl_for_box(r0, c0, r1, c1, spec["head_side"])
    turn_tl = (
        head_tl[0] + spec["first_dir"][0] * spec["first_dist"],
        head_tl[1] + spec["first_dir"][1] * spec["first_dist"],
    )
    final_tl = (
        turn_tl[0] + spec["second_dir"][0] * spec["second_dist"],
        turn_tl[1] + spec["second_dir"][1] * spec["second_dist"],
    )
    final_head_tl = final_tl
    input_non_bg = (
        {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        | _line_cells_from_tl(head_tl, spec["head_side"], spec["outer_h"], spec["outer_w"])
        | _line_cells_from_tl(turn_tl, spec["head_side"], spec["outer_h"], spec["outer_w"])
        | _line_cells_from_tl(final_tl, spec["head_side"], spec["outer_h"], spec["outer_w"])
    )
    final_body = {
        (rr, cc)
        for rr in range(spec["final_r"], spec["final_r"] + spec["outer_h"])
        for cc in range(spec["final_c"], spec["final_c"] + spec["outer_w"])
    }
    final_head = _line_cells_from_tl(final_head_tl, spec["head_side"], spec["outer_h"], spec["outer_w"])
    return {
        **spec,
        "input_r": int(r0),
        "input_c": int(c0),
        "head_tl": head_tl,
        "turn_tl": turn_tl,
        "final_tl": final_tl,
        "input_non_bg": input_non_bg,
        "final_body": final_body,
        "final_head": final_head,
    }


def _spec_step_cells(spec: dict, step: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = _spec_input_box(spec)
    tr, tc = _translate_along_l(spec["first_dir"], spec["second_dir"], spec["first_dist"], step)
    body = {(rr + tr, cc + tc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
    head = {(r + tr, c + tc) for r, c in _line_cells_from_tl(spec["head_tl"], spec["head_side"], spec["outer_h"], spec["outer_w"])}
    return body, head


def _build_spec(final_r: int, final_c: int, outer_h: int, outer_w: int, head_side: int, first_dist: int, second_sign: int, color: int) -> dict:
    first_dir = SIDE_TO_DIR[head_side]
    second_dir = (0, second_sign) if head_side in (0, 1) else (second_sign, 0)
    min_second = outer_w + 1 if head_side in (0, 1) else outer_h + 1
    second_dist = min_second + random.randint(0, 2)
    total_dist = first_dist + second_dist
    inner_area = (outer_h - 2) * (outer_w - 2)
    grey = total_dist
    if inner_area < grey:
        raise ValueError("area too small for chosen distances")
    return _spec_geometry(
        {
            "final_r": int(final_r),
            "final_c": int(final_c),
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
            "head_side": int(head_side),
            "first_dir": first_dir,
            "second_dir": second_dir,
            "first_dist": int(first_dist),
            "second_dist": int(second_dist),
            "total_dist": int(total_dist),
            "grey": int(grey),
            "initial_orange": int(inner_area - grey),
            "color": int(color),
        }
    )


def _spec_inside(spec: dict, grid_h: int, grid_w: int) -> bool:
    cells = spec["input_non_bg"] | spec["final_body"] | spec["final_head"]
    return (
        min(r for r, _c in cells) >= 0
        and min(c for _r, c in cells) >= 0
        and max(r for r, _c in cells) < grid_h
        and max(c for _r, c in cells) < grid_w
    )


def _spec_to_grid(specs: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r = spec["input_r"]
        c = spec["input_c"]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        ih = oh - 2
        iw = ow - 2
        order = _fill_order_local(ih, iw, spec["head_side"])
        interior = np.full((ih, iw), GREY, dtype=int)
        for idx, (rr, cc) in enumerate(order):
            if idx < int(spec["initial_orange"]):
                interior[rr, cc] = int(ORANGE)
        grid[r, c : c + ow] = int(BORDER)
        grid[r + oh - 1, c : c + ow] = int(BORDER)
        grid[r : r + oh, c] = int(BORDER)
        grid[r : r + oh, c + ow - 1] = int(BORDER)
        grid[r + 1 : r + oh - 1, c + 1 : c + ow - 1] = interior
        _draw_line(grid, spec["head_tl"], spec["head_side"], oh, ow, spec["color"])
        _draw_line(grid, spec["turn_tl"], spec["head_side"], oh, ow, spec["color"])
        _draw_line(grid, spec["final_tl"], spec["head_side"], oh, ow, spec["color"])
    return grid


def _paths_clear_between(spec_a: dict, spec_b: dict) -> bool:
    a_input = spec_a["input_non_bg"]
    b_input = spec_b["input_non_bg"]
    return not (a_input & b_input or _touching(a_input, b_input))


def _sample_pair() -> tuple[np.ndarray, dict] | None:
    grid_h = random.randint(20, 26)
    grid_w = random.randint(20, 26)
    colors = random.sample(SIGNAL_COLORS, 2)
    specs = []
    for idx in range(2):
        long_path = idx == 1
        built = None
        for _ in range(60):
            head_side = random.choice([0, 1, 2, 3])
            outer_h = random.randint(5 if not long_path else 6, 7 if not long_path else 8)
            outer_w = random.randint(5, 7)
            first_dist = random.randint(3, 5) if not long_path else random.randint(6, 10)
            second_sign = random.choice([-1, 1])
            min_second = outer_w + 1 if head_side in (0, 1) else outer_h + 1
            second_dist = min_second + random.randint(0, 2)
            total_dist = first_dist + second_dist
            inner_area = (outer_h - 2) * (outer_w - 2)
            if inner_area < total_dist:
                continue
            final_r = random.randint(1, grid_h - outer_h - 1)
            final_c = random.randint(1, grid_w - outer_w - 1)
            spec = _spec_geometry(
                {
                    "final_r": final_r,
                    "final_c": final_c,
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "head_side": head_side,
                    "first_dir": SIDE_TO_DIR[head_side],
                    "second_dir": (0, second_sign) if head_side in (0, 1) else (second_sign, 0),
                    "first_dist": first_dist,
                    "second_dist": second_dist,
                    "total_dist": total_dist,
                    "grey": total_dist,
                    "initial_orange": inner_area - total_dist,
                    "color": colors[idx],
                }
            )
            if _spec_inside(spec, grid_h, grid_w):
                built = spec
                break
        if built is None:
            return None
        specs.append(built)

    specs.sort(key=lambda spec: int(spec["total_dist"]))
    spec_a, spec_b = specs
    if not _paths_clear_between(spec_a, spec_b):
        return None

    grid = _spec_to_grid([spec_a, spec_b], grid_h, grid_w)
    try:
        _ = main(grid)
    except Exception:
        return None
    metrics = _fusion_metrics(grid)
    if metrics["n_items"] != 2:
        return None
    if metrics["blocked_count"] < 1 or metrics["ordered_diff_cells"] <= 0:
        return None
    if metrics["turned_base_count"] < 1:
        return None
    if len(metrics["distances"]) != 2 or not (metrics["distances"][0] < metrics["distances"][1]):
        return None
    if metrics["actual_steps"][0] != metrics["base_steps"][0]:
        return None
    if metrics["actual_steps"][1] >= metrics["base_steps"][1]:
        return None
    return grid, metrics


def _apply_symmetry(grid: np.ndarray) -> np.ndarray:
    choice = random.choice(["id", "flip_h", "flip_v", "transpose"])
    if choice == "flip_h":
        return np.fliplr(grid)
    if choice == "flip_v":
        return np.flipud(grid)
    if choice == "transpose":
        return np.transpose(grid)
    return grid


def _fusion_metrics(grid: np.ndarray) -> dict:
    bg, items = _collect_items(grid)
    ordered, sim_items = _simulate_ordered(grid, items, bg)
    independent = _simulate_independent(grid, items, bg)
    blocked_count = sum(1 for item in sim_items if item["blocked"])
    turned_base_count = sum(1 for item in sim_items if int(item["base_steps"]) > int(item["first_dist"]))
    turned_actual_count = sum(1 for item in sim_items if int(item["actual_steps"]) > int(item["first_dist"]))
    moved_groups = []
    for item in sim_items:
        steps = int(item["actual_steps"])
        if steps == 0:
            body = {(rr, cc) for rr in range(item["box"][0], item["box"][2] + 1) for cc in range(item["box"][1], item["box"][3] + 1)}
            head = set(item["head_pts"])
        else:
            tr, tc = _translate_along_l(item["first_dir"], item["second_dir"], item["first_dist"], steps)
            body = {
                (rr + tr, cc + tc)
                for rr in range(item["box"][0], item["box"][2] + 1)
                for cc in range(item["box"][1], item["box"][3] + 1)
            }
            head = {(r + tr, c + tc) for r, c in item["head_pts"]}
        moved_groups.append(body | head)
    return {
        "n_items": len(sim_items),
        "blocked_count": blocked_count,
        "turned_base_count": turned_base_count,
        "turned_actual_count": turned_actual_count,
        "ordered_diff_cells": int(np.count_nonzero(ordered != independent)),
        "distances": tuple(int(item["total_dist"]) for item in sim_items),
        "base_steps": tuple(int(item["base_steps"]) for item in sim_items),
        "actual_steps": tuple(int(item["actual_steps"]) for item in sim_items),
        "moved_groups": moved_groups,
    }


def generate_input() -> np.ndarray:
    for _ in range(800):
        built = _sample_pair()
        if built is None:
            continue
        grid, _metrics = built
        grid2 = _apply_symmetry(grid)
        try:
            output = main(grid2)
        except Exception:
            continue
        metrics2 = _fusion_metrics(grid2)
        if metrics2["blocked_count"] < 1 or metrics2["ordered_diff_cells"] <= 0:
            continue
        if metrics2["turned_base_count"] < 1:
            continue
        return grid2
    raise RuntimeError("Failed to generate valid combo variant after 800 attempts")
