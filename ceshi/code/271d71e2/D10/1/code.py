# Variant of 271d71e2
# Dimension: D10 — 容器内部分区结构 (meta)
# Change: 内部分为连续的橙色区域（头侧）和灰色区域（远侧），二者之间是平直分界线... → 灰色像素随机分布

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


"""
Design analysis for the D10 variant
===================================

spec:
  source_task: none
  preserved_structure:
    - background stays 6; container border stays 0; head/destination stay 9.
    - Each container is still a black rectangular frame whose interior uses only 5/7.
    - Head and destination remain full-edge 1-cell-thick lines aligned with the movement axis.
    - Movement rule is unchanged: steps = min(head-destination distance, current grey count).
    - Fill order is unchanged: consume only GREY cells, sorted from the head-facing side inward.
  D10_change:
    - Old task: interior colors were deterministic by a straight partition in fill-order space:
      the first K ordered cells were orange, the rest grey.
    - New task: the number of grey cells is still fixed, but their positions are sampled across the
      whole interior. Grey/orange may interleave arbitrarily.
  concrete_trace:
    - Suppose head is on the left, so interior order is by increasing column, then decreasing row.
    - Old input with area 6 and grey_count 3 had orange at ordered indices [0,1,2] and grey at [3,4,5].
    - New input may instead place grey at [0,2,5]. If distance=2, the solver must:
      1. count three grey cells,
      2. move two steps,
      3. recolor only the first two grey cells in that order: [0,2] -> orange,
         leaving the far grey at [5].
    - This is visibly different from the old contiguous suffix pattern and prevents a solver from
      inferring movement capacity from a flat grey block.
  complexity_calibration:
    - Only one degree of freedom changes: the spatial arrangement of interior 5/7 cells.
    - Counts, motion axis, destination matching, and colors remain stable, so the task stays readable.
    - Generator forces moving mixed-color interiors to be non-suffix in fill order, making the
      D10 change visible in typical samples instead of accidentally degenerating back to the original.
  implicit_assumptions_review:
    - Original container detection iterated connected components of (GREY | ORANGE). That relies on
      the old partition making the whole interior 4-connected through mixed colors. Random mixing can
      split one container into several color islands, so that detection becomes invalid.
    - New solver detects containers from BLACK rectangle-frame components and validates that their
      interiors contain only GREY/ORANGE. This removes any dependency on interior connectivity.
    - Original generator encoded the old rule by painting an orange prefix of the fill order.
      New generator samples the grey positions directly and computes expected outputs by consuming
      only the ordered subset of currently-grey cells.
  placement_legality:
    - Assembly boxes are unchanged: destination + gap + head + container.
    - Because final moved positions stay inside the same assembly span, the existing 1-cell spacing
      between assembly bounding boxes still guarantees no overlap before or after movement.
  visual_observability:
    - Moving containers use both 5 and 7 whenever area > 1.
    - Their grey sets are forced to violate the old contiguous suffix pattern, so the viewer sees
      intermixed orange/grey before motion, then sees head-side grey cells selectively turn orange.
  visual_cleanliness:
    - No new marker types are introduced.
    - Objects remain axis-aligned rectangles/lines with a background moat, so the grid stays readable.
  color_rule:
    - No new color mapping is introduced; only the spatial arrangement of existing 5/7 changes.
  new_solver_assumptions_and_generator_guards:
    - Solver now assumes each container border is an isolated rectangular BLACK component.
      Generator already constructs only such frames and keeps assemblies separated.
    - Solver still assumes each container has an adjacent head line and a same-shaped destination line
      on the motion axis. Generator continues to enforce that exactly.
"""


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# head_side: 0=top, 1=bottom, 2=left, 3=right
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _sorted_grey_coords(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Local interior coords ordered from the head-facing side inward."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:    # right, movement (0,1)
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left, movement (0,-1)
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom, movement (1,0)
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:                 # top, movement (-1,0)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _build_interior(inner_h: int, inner_w: int, grey_cells: set[tuple[int, int]]) -> np.ndarray:
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    for r, c in grey_cells:
        interior[int(r), int(c)] = GREY
    return interior


def _sample_grey_cells(
    inner_h: int,
    inner_w: int,
    head_side: int,
    grey_count: int,
    *,
    force_scrambled: bool,
) -> set[tuple[int, int]]:
    area = inner_h * inner_w
    order = _interior_fill_order(inner_h, inner_w, head_side)
    suffix_indices = set(range(area - grey_count, area))

    if grey_count <= 0:
        return set()
    if grey_count >= area:
        return set(order)

    for _ in range(80):
        idxs = set(random.sample(range(area), grey_count))
        if force_scrambled and idxs == suffix_indices:
            continue
        return {order[i] for i in idxs}

    # Deterministic fallback that still breaks the old contiguous-suffix pattern.
    idxs = set(suffix_indices)
    if force_scrambled:
        idxs.remove(area - 1)
        idxs.add(0)
    return {order[i] for i in idxs}


def _is_old_contiguous_pattern(inner_h: int, inner_w: int, head_side: int, grey_cells: set[tuple[int, int]]) -> bool:
    order = _interior_fill_order(inner_h, inner_w, head_side)
    grey_indices = {idx for idx, cell in enumerate(order) if cell in grey_cells}
    area = inner_h * inner_w
    grey_count = len(grey_cells)
    return grey_indices == set(range(area - grey_count, area))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging grey cells to orange.

    D10 variant note:
    - Grey cells inside a container may be scattered rather than forming one flat far-side block.
    - Movement still depends only on grey count, and recoloring still follows head-side order over
      the currently-grey cells.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    for pts in components_mask(g == BORDER, connectivity=4):
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border_pts = list(_rect_border_coords(int(r0), int(c0), int(r1), int(c1)))
        if len(border_pts) != int(len(pts)):
            continue
        if any(int(g[r, c]) != BORDER for r, c in border_pts):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})

    assert containers

    out = g.copy()
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
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
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

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]

        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue
        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        rem = comp_pts - head_pts
        if rem:
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
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
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
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        if not dests:
            continue

        best = None
        best_dest = None
        for d in dests:
            tr0 = int(d["tl"][0] - head_tl[0])
            tc0 = int(d["tl"][1] - head_tl[1])
            if direction[0] != 0 and tc0 != 0:
                continue
            if direction[1] != 0 and tr0 != 0:
                continue
            if direction[0] == 1 and tr0 <= 0:
                continue
            if direction[0] == -1 and tr0 >= 0:
                continue
            if direction[1] == 1 and tc0 <= 0:
                continue
            if direction[1] == -1 and tc0 >= 0:
                continue
            dist = abs(tr0) + abs(tc0)
            if best is None or dist < best:
                best = dist
                best_dest = d
        if best_dest is None:
            continue

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        order = _sorted_grey_coords(grey_coords, direction)
        to_fill = set(order[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers

        while True:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            if is_decoy or inner_h * inner_w >= 2:
                break

        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
            grey_cells: set[tuple[int, int]] = set()
        else:
            grey = random.randint(1, inner_area - 1)
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(grey, h2d)
            grey_cells = _sample_grey_cells(
                inner_h,
                inner_w,
                head_side,
                grey,
                force_scrambled=True,
            )

        specs.append({
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": steps,
            "grey": grey,
            "grey_cells": grey_cells,
            "is_decoy": is_decoy,
        })

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append({
            **s,
            "total_h": total_h,
            "total_w": total_w,
            "cont_r": cont_r,
            "cont_c": cont_c,
        })

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))

    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            cr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            cr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[cr0:cr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        grey_cells = a["grey_cells"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = _build_interior(inner_h, inner_w, grey_cells)
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR

        gap = a["gap"]
        if hs == 0:
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        grey_cells = set(a["grey_cells"])
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        ordered_greys = [cell for cell in fill_order if cell in grey_cells]
        filled_now = set(ordered_greys[:steps])
        remaining_greys = grey_cells - filled_now

        interior = _build_interior(inner_h, inner_w, remaining_greys)
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    mixed_movers = [
        a for a in assemblies
        if (not a["is_decoy"]) and 0 < a["grey"] < a["inner_h"] * a["inner_w"]
    ]
    if not mixed_movers:
        return None
    if all(_is_old_contiguous_pattern(a["inner_h"], a["inner_w"], a["head_side"], a["grey_cells"]) for a in mixed_movers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
