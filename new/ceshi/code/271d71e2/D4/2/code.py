# Variant of 271d71e2
# Dimension: D4 — 源对象结构 (src)
# Change: 黑色边框矩形容器，内部为灰色/橙色填充，附带同色头部线段指示方向... → 头部为箭头形状而非线段

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete structure and behavior
# - The container stays the same: a black rectangular frame with grey/orange interior.
# - The moving marker is no longer a 1-cell-thick line. It is a fixed 3x3 arrow sprite:
#   U = 999 / .9. / .9.
#   D = .9. / .9. / 999
#   L = 9.. / 999 / 9..
#   R = ..9 / 999 / ..9
# - The destination marker is an identical arrow sprite of the same color, translated further
#   along the arrow's pointing direction.
# - To make "direction from shape" necessary, the arrow is attached only to a side orthogonal
#   to its motion axis: horizontal motion uses a top/bottom attachment; vertical motion uses a
#   left/right attachment. Therefore the old heuristic "head is above => move up" is false here.
# - Example trace:
#   Original logic: a head line above a container implies direction = up.
#   Variant logic: an arrow attached above may still point right, so the container moves right.
#   The solver must read the arrow shape, then find the matching arrow destination along that
#   direction, then translate exactly as before with the grey-count cap.
#
# b) Complexity calibration
# - The arrow has only four orientations and one fixed size, so the new degree of freedom is
#   visually obvious but still low-complexity.
# - Attachment side is restricted to the two orthogonal sides of the motion direction. This
#   avoids degenerate samples that still look like the original task and gives repeated evidence
#   in a few examples that position and direction are decoupled.
#
# c) Old implicit assumptions that break
# - In the original solver, `pick_head_for_container()` infers direction from the head bbox
#   relative to the container. That assumes the head always sits on the movement side.
# - It also assumes the head and destination share exactly the same extracted shape after taking
#   the subset of pixels closest to the border. With an arrow sprite, that "closest layer only"
#   extraction would lose most of the arrow and break matching.
# - This variant therefore identifies the full adjacent component as the head, normalizes its
#   shape, matches it against four fixed arrow templates, and derives direction from that match.
#
# d) Placement legality
# - Each generated object reserves a rectangle covering: input container, input arrow, destination
#   arrow, moved container, and moved arrow. Reserved rectangles are packed with a 1-cell margin,
#   so input and output states remain non-overlapping and within bounds.
# - The destination arrow is always at least one background cell away from the head arrow, which
#   keeps them as separate 4-connected components and avoids ambiguous same-color merging.
#
# e) Visual observability
# - Because the arrow is orthogonally attached, the changed mechanism is visible in every mover:
#   the head is not on the side of travel, yet its triangular/T-shaped form clearly points toward
#   the destination.
#
# f) Visual cleanliness
# - Only three object types remain on the board: container, source arrow, destination arrow.
#   The arrow is compact (3x3), monochrome, and separated from other assemblies by background.
#
# g) Color-change inferability
# - No new colors are introduced. Grey-to-orange charging order is unchanged, so the only novel
#   cue that the viewer has to infer is the arrow orientation itself.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

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
SHAPE_TO_DIR = {frozenset(v): k for k, v in ARROW_SHAPES.items()}


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored arrow destinations, charging grey cells to orange.

    Variant rule:
    - The source marker is a 3x3 arrow sprite, not a line.
    - Direction is derived from the arrow's normalized shape, not from which side of the
      container the arrow touches.
    - The container and arrow translate toward the nearest matching arrow destination along
      the arrow direction, capped by the number of grey interior cells.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
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
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {"pts": pts, "shape": shape, "tl": (int(r0), int(c0))}
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
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

        direction = SHAPE_TO_DIR.get(frozenset(comp0["shape"]))
        if direction is None:
            return None
        return {
            "color": hc,
            "head_pts": set(comp0["pts"]),
            "head_shape": comp0["shape"],
            "head_tl": comp0["tl"],
            "dir": DIR_TO_VEC[direction],
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        direction = head_info["dir"]

        best_dist = None
        best_dest = None
        for d in comp_by_color.get(hc, []):
            if d["pts"] == head_pts or d["shape"] != head_shape:
                continue
            tr0 = int(d["tl"][0] - head_tl[0])
            tc0 = int(d["tl"][1] - head_tl[1])
            if direction[0] != 0:
                if tc0 != 0 or tr0 * direction[0] <= 0:
                    continue
            else:
                if tr0 != 0 or tc0 * direction[1] <= 0:
                    continue
            dist = abs(tr0) + abs(tc0)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = d
        if best_dest is None:
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

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[r, c]) == int(hc):
                out[r, c] = int(bg)

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
        gap = random.randint(0, 6)
        travel = 4 + gap

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
        dest_r = head_r + dr * travel
        dest_c = head_c + dc * travel
        out_cont_r = dr * steps
        out_cont_c = dc * steps
        out_head_r = head_r + dr * steps
        out_head_c = head_c + dc * steps

        rects = [
            (0, 0, outer_h - 1, outer_w - 1),
            (head_r, head_c, head_r + 2, head_c + 2),
            (dest_r, dest_c, dest_r + 2, dest_c + 2),
            (out_cont_r, out_cont_c, out_cont_r + outer_h - 1, out_cont_c + outer_w - 1),
            (out_head_r, out_head_c, out_head_r + 2, out_head_c + 2),
        ]
        min_r, min_c, max_r, max_c = _rect_union(rects)
        shift_r = -min_r
        shift_c = -min_c

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "direction": direction,
                "attach_side": attach_side,
                "travel": travel,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "cont_tl": (shift_r, shift_c),
                "head_tl": (head_r + shift_r, head_c + shift_c),
                "dest_tl": (dest_r + shift_r, dest_c + shift_c),
                "out_cont_tl": (out_cont_r + shift_r, out_cont_c + shift_c),
                "out_head_tl": (out_head_r + shift_r, out_head_c + shift_c),
                "reserve_h": max_r - min_r + 1,
                "reserve_w": max_c - min_c + 1,
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

    place_order = sorted(range(len(specs)), key=lambda i: specs[i]["reserve_h"] * specs[i]["reserve_w"], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    for idx in place_order:
        s = specs[idx]
        rh, rw = s["reserve_h"], s["reserve_w"]
        if rh > grid_h or rw > grid_w:
            return None
        placed = False
        for _ in range(400):
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
        dr, dc = pr + s["dest_tl"][0], pc + s["dest_tl"][1]

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
        _stamp_arrow(grid, (dr, dc), s["direction"], HEAD_COLOR)

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
        for dr, dc in ARROW_SHAPES[s["direction"]]:
            expected[hr + dr, hc + dc] = BG

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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
