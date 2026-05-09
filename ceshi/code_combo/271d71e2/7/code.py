# Combo variant of 271d71e2 (index 7)
# combo_spec: D4/2+D11/1
# Name: 箭头头部对准目标点

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete structure and behavior
# - The container is unchanged: black rectangle border, grey/orange rectangular interior.
# - The head is no longer a full edge line. It is a fixed 3x3 arrow sprite:
#   U = 999 / .9. / .9.
#   D = .9. / .9. / 999
#   L = 9.. / 999 / 9..
#   R = ..9 / 999 / ..9
# - The target is no longer a same-shape marker. It is a single maroon point placed farther ahead
#   in the arrow's pointing direction.
# - The translation distance is measured from the arrow tip to the point, not from arrow bbox
#   top-left and not from a full line center. After moving k steps, the arrow tip advances by k and
#   is supposed to land on the point when grey count is not the limiting factor.
# - Small trace:
#   Original task: a top line head above a box and a top line destination above it imply "move up"
#   and "overlap line onto line".
#   D4/2 alone: an arrow attached on the left may still point down, so direction comes from shape.
#   D11/1 alone: a line head aims its center at a point.
#   Combo here: an arrow attached orthogonally to the motion axis points toward a single dot, and
#   the arrow tip rather than any bbox anchor is what aligns to that dot.
#   Example: a right-pointing arrow attached above the container has tip at (r, c). A point sits at
#   (r, c+5). If the container has 3 grey cells left, it moves right by 3 and the 3 head-most grey
#   cells turn orange; if it has >=5 grey cells, it moves right by 5 and the tip lands on the dot.
#
# b) Complexity calibration
# - The arrow vocabulary is still only four shapes, keeping shape parsing low-entropy.
# - To ensure this is visibly different from the source task, the generator only attaches arrows on
#   a side orthogonal to the motion axis. Therefore "where the head is attached" and "where the box
#   moves" are decoupled in every mover.
# - The target is always a point with at least one background cell between it and the arrow tip, so
#   the scene stays readable and head vs target remain distinct 4-connected components.
#
# c) Implicit-assumption audit against the original solver
# - Original assumption: direction can be inferred from head position relative to the container.
#   Broken. In the combo, the arrow may be above the box while motion is right.
# - Original assumption: destination has the same normalized shape as the head. Broken. The target
#   is a single pixel, so matching must be axis+direction based.
# - Original assumption: translation vector can be read from destination top-left minus head
#   top-left. Broken again: a point has no meaningful shared shape anchor with the arrow. The new
#   anchor is the arrow tip.
# - Original same-component split logic is mostly unnecessary because the generator enforces a gap
#   between arrow tip and point. The solver still contains a tiny fallback for malformed gap=0
#   inputs by trying to peel one extra pixel off an otherwise valid arrow component.
#
# d) Placement legality
# - Each generated assembly reserves the union of five regions: input container, input arrow,
#   destination point, moved container, moved arrow. Packing is done with a 1-cell background moat.
# - Since the point lies on the same ray that the tip follows, both the input and the final state
#   fit inside the same reservation rectangle.
#
# e) Visual observability
# - Across samples, the arrow is visibly not on the side of travel, forcing the solver to read its
#   shape. The target is visibly a dot, forcing the solver to infer point alignment instead of shape
#   overlap. These two changes are jointly observable in only 2-4 examples.
#
# f) Visual cleanliness
# - Only three foreground object types appear: bordered container, arrow head, point target.
#   The point is sparse, the arrow is compact, and assemblies are separated by background.
#
# g) Color-change inferability
# - No new colors are introduced. Grey-to-orange charging order still follows the movement
#   direction, so the only new inference burden is the fused geometry: arrow orientation + tip to
#   point alignment.
#
# New solver assumptions introduced here and matched by the generator
# - Head components are exact 3x3 arrow sprites, or that sprite plus one fused forward point in the
#   pathological gap=0 case.
# - Every mover has exactly one nearest forward point on its arrow ray; the generator verifies
#   solver/output agreement and retries on ambiguous layouts.
# - Motion direction, fill order, and point alignment are all keyed off the arrow direction, not
#   the attachment side. The generator constructs interiors using that same rule.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DIR_TO_VEC = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ORTH_ATTACH = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}

ARROW_SHAPES = {
    "U": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "D": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "L": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "R": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}
ARROW_TIPS = {
    "U": (0, 1),
    "D": (2, 1),
    "L": (1, 0),
    "R": (1, 2),
}
SHAPE_TO_DIR = {frozenset(v): k for k, v in ARROW_SHAPES.items()}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _interior_fill_order(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == "R":
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == "L":
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == "D":
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _arrow_tl_relative(outer_h: int, outer_w: int, attach_side: str, anchor: int) -> tuple[int, int]:
    if attach_side == "U":
        return (-3, anchor)
    if attach_side == "D":
        return (outer_h, anchor)
    if attach_side == "L":
        return (anchor, -3)
    return (anchor, outer_w)


def _stamp_arrow(grid: np.ndarray, tl: tuple[int, int], direction: str, color: int) -> None:
    r0, c0 = tl
    for dr, dc in ARROW_SHAPES[direction]:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _rect_union(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r0 for r0, _c0, _r1, _c1 in rects),
        min(c0 for _r0, c0, _r1, _c1 in rects),
        max(r1 for _r0, _c0, r1, _c1 in rects),
        max(c1 for _r0, _c0, _r1, c1 in rects),
    )


def _normalize_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0 = int(min(rs))
    c0 = int(min(cs))
    return {(int(r - r0), int(c - c0)) for r, c in pts}, (r0, c0)


def _parse_arrow_component(pts: set[tuple[int, int]]) -> dict | None:
    shape, tl = _normalize_pts(pts)
    direction = SHAPE_TO_DIR.get(frozenset(shape))
    if direction is not None:
        tip_local = ARROW_TIPS[direction]
        return {
            "direction": direction,
            "head_pts": set(pts),
            "head_tl": tl,
            "tip": (int(tl[0] + tip_local[0]), int(tl[1] + tip_local[1])),
            "fused_point": None,
        }

    parsed = []
    for p in pts:
        rem = set(pts)
        rem.remove(p)
        if len(rem) != 5:
            continue
        rem_shape, rem_tl = _normalize_pts(rem)
        direction = SHAPE_TO_DIR.get(frozenset(rem_shape))
        if direction is None:
            continue
        dr, dc = DIR_TO_VEC[direction]
        tip_local = ARROW_TIPS[direction]
        tip = (int(rem_tl[0] + tip_local[0]), int(rem_tl[1] + tip_local[1]))
        if p == (int(tip[0] + dr), int(tip[1] + dc)):
            parsed.append(
                {
                    "direction": direction,
                    "head_pts": rem,
                    "head_tl": rem_tl,
                    "tip": tip,
                    "fused_point": p,
                }
            )
    if len(parsed) == 1:
        return parsed[0]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored point destinations, charging grey cells to orange.

    Combo rule:
    - The source marker is a 3x3 arrow sprite whose orientation gives the movement direction.
    - The target is a single point farther along that ray.
    - Distance is measured from the arrow tip to the target point.
    - Steps = min(tip_to_point_distance, grey_count).
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

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

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            comp_by_color[int(col)].append({"pts": pts, "tl": (int(r0), int(c0))})

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

        parsed = _parse_arrow_component(set(comp0["pts"]))
        if parsed is None:
            return None
        return {
            "color": hc,
            "head_pts": set(parsed["head_pts"]),
            "head_tl": tuple(parsed["head_tl"]),
            "tip": tuple(parsed["tip"]),
            "dir": DIR_TO_VEC[parsed["direction"]],
            "dir_name": parsed["direction"],
            "comp_pts": set(comp0["pts"]),
            "fused_point": parsed["fused_point"],
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        tip = tuple(head_info["tip"])
        comp_pts = set(head_info["comp_pts"])
        direction = tuple(head_info["dir"])
        dir_name = str(head_info["dir_name"])

        dest_points: set[tuple[int, int]] = set()
        for d in comp_by_color.get(hc, []):
            if d["pts"] == comp_pts:
                continue
            if len(d["pts"]) == 1:
                dest_points.add(next(iter(d["pts"])))

        if head_info["fused_point"] is not None:
            dest_points.add(tuple(head_info["fused_point"]))

        best_dist = None
        best_dest = None
        for p in dest_points:
            tr0 = int(p[0] - tip[0])
            tc0 = int(p[1] - tip[1])
            if direction[0] != 0:
                if tc0 != 0 or tr0 * direction[0] <= 0:
                    continue
            else:
                if tr0 != 0 or tc0 * direction[1] <= 0:
                    continue
            dist = abs(tr0) + abs(tc0)
            if dist == 0:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = p
        if best_dest is None or best_dist is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(best_dist), int(len(grey_coords)))
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        order_local = _interior_fill_order(inner.shape[0], inner.shape[1], dir_name)
        to_fill = {
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr, cc in order_local[:steps]
            if int(inner[rr, cc]) == GREY
        }

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == hc:
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(hc)

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs: list[dict] = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        inner_area = inner_h * inner_w

        direction = random.choice(list(DIR_TO_VEC.keys()))
        attach_side = random.choice(ORTH_ATTACH[direction])
        gap_bg = random.randint(1, 8)
        travel = gap_bg + 1

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            steps = min(grey, travel)

        if attach_side in ("U", "D"):
            anchor = random.randint(0, outer_w - 3)
        else:
            anchor = random.randint(0, outer_h - 3)

        head_r, head_c = _arrow_tl_relative(outer_h, outer_w, attach_side, anchor)
        dr, dc = DIR_TO_VEC[direction]
        tip_local = ARROW_TIPS[direction]
        tip_r = int(head_r + tip_local[0])
        tip_c = int(head_c + tip_local[1])
        dest_r = int(tip_r + dr * travel)
        dest_c = int(tip_c + dc * travel)

        out_cont_r = int(dr * steps)
        out_cont_c = int(dc * steps)
        out_head_r = int(head_r + dr * steps)
        out_head_c = int(head_c + dc * steps)

        rects = [
            (0, 0, outer_h - 1, outer_w - 1),
            (head_r, head_c, head_r + 2, head_c + 2),
            (dest_r, dest_c, dest_r, dest_c),
            (out_cont_r, out_cont_c, out_cont_r + outer_h - 1, out_cont_c + outer_w - 1),
            (out_head_r, out_head_c, out_head_r + 2, out_head_c + 2),
        ]
        min_r, min_c, max_r, max_c = _rect_union(rects)
        shift_r = int(-min_r)
        shift_c = int(-min_c)

        specs.append(
            {
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "direction": direction,
                "attach_side": attach_side,
                "travel": int(travel),
                "steps": int(steps),
                "initial_orange": int(initial_orange),
                "grey": int(grey),
                "cont_tl": (shift_r, shift_c),
                "head_tl": (int(head_r + shift_r), int(head_c + shift_c)),
                "dest_pt": (int(dest_r + shift_r), int(dest_c + shift_c)),
                "out_cont_tl": (int(out_cont_r + shift_r), int(out_cont_c + shift_c)),
                "out_head_tl": (int(out_head_r + shift_r), int(out_head_c + shift_c)),
                "reserve_h": int(max_r - min_r + 1),
                "reserve_w": int(max_c - min_c + 1),
                "is_decoy": bool(is_decoy),
            }
        )

    total_area = sum(s["reserve_h"] * s["reserve_w"] for s in specs)
    max_res_h = max(s["reserve_h"] for s in specs)
    max_res_w = max(s["reserve_w"] for s in specs)
    base = max(10, int(total_area**0.5) + 4)
    lo_h = max(10, max_res_h + 2, min(26, base))
    lo_w = max(10, max_res_w + 2, min(26, base))
    hi_h = min(26, max(lo_h, base + 8))
    hi_w = min(26, max(lo_w, base + 8))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    place_order = sorted(
        range(len(specs)),
        key=lambda i: specs[i]["reserve_h"] * specs[i]["reserve_w"],
        reverse=True,
    )
    placements: dict[int, tuple[int, int]] = {}
    for idx in place_order:
        s = specs[idx]
        rh, rw = s["reserve_h"], s["reserve_w"]
        if rh > grid_h or rw > grid_w:
            return None
        placed = False
        for _ in range(600):
            r0 = random.randint(0, grid_h - rh)
            c0 = random.randint(0, grid_w - rw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + rh + 1)
            cc1 = min(grid_w, c0 + rw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[r0 : r0 + rh, c0 : c0 + rw] = True
                placements[idx] = (r0, c0)
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for i, s in enumerate(specs):
        pr, pc = placements[i]
        cr, cc = pr + s["cont_tl"][0], pc + s["cont_tl"][1]
        hr, hc = pr + s["head_tl"][0], pc + s["head_tl"][1]
        dr, dc = pr + s["dest_pt"][0], pc + s["dest_pt"][1]

        oh, ow = s["outer_h"], s["outer_w"]
        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = np.full((s["inner_h"], s["inner_w"]), GREY, dtype=int)
        fill_order = _interior_fill_order(s["inner_h"], s["inner_w"], s["direction"])
        for idx_cell in range(s["initial_orange"]):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
        grid[cr + 1 : cr + 1 + s["inner_h"], cc + 1 : cc + 1 + s["inner_w"]] = interior

        _stamp_arrow(grid, (hr, hc), s["direction"], HEAD_COLOR)
        grid[dr, dc] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for i, s in enumerate(specs):
        if s["steps"] == 0:
            continue

        pr, pc = placements[i]
        cr, cc = pr + s["cont_tl"][0], pc + s["cont_tl"][1]
        hr, hc = pr + s["head_tl"][0], pc + s["head_tl"][1]
        ncr, ncc = pr + s["out_cont_tl"][0], pc + s["out_cont_tl"][1]
        nhr, nhc = pr + s["out_head_tl"][0], pc + s["out_head_tl"][1]

        oh, ow = s["outer_h"], s["outer_w"]
        expected[cr : cr + oh, cc : cc + ow] = BG
        for adr, adc in ARROW_SHAPES[s["direction"]]:
            expected[hr + adr, hc + adc] = BG

        patch = np.full((oh, ow), BORDER, dtype=int)
        interior = np.full((s["inner_h"], s["inner_w"]), GREY, dtype=int)
        fill_order = _interior_fill_order(s["inner_h"], s["inner_w"], s["direction"])
        for idx_cell in range(s["initial_orange"] + s["steps"]):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
        patch[1:-1, 1:-1] = interior
        expected[ncr : ncr + oh, ncc : ncc + ow] = patch
        _stamp_arrow(expected, (nhr, nhc), s["direction"], HEAD_COLOR)

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
