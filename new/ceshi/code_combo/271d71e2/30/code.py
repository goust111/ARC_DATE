# Combo variant of 271d71e2 (index 30)
# combo_spec: D9/2+D16/2
# Name: 网格扩展背景色由容器数决定
#
# Design analysis
# ===============
# a_concretization_and_trace:
#   - Base task: each black frame with grey/orange interior moves toward its same-colored
#     head/destination line; moved distance is capped by remaining grey cells, and exactly
#     that many grey cells flip to orange from the head-facing side inward.
#   - D16 contribution: background is no longer fixed magenta. Instead the container count
#     determines the background color via a small visible mapping:
#       2 containers -> blue(1)
#       3 containers -> green(3)
#       4 containers -> yellow(4)
#   - D9 contribution needs a concreteization. In the original geometry, if a head stops on
#     the destination line then the frame itself always remains one cell inside the old grid,
#     so "expand only when overflow happens" would never trigger. To make the expansion rule
#     observable, we preserve the D9/2 single-variant concretization: when grey budget allows,
#     the object advances one extra cell beyond the original stop point. That puts the head
#     one cell beyond the destination and can push it outside the old canvas by exactly one
#     row/column. The solver therefore plans all moves first, computes how much padding is
#     needed on each side, then redraws on the expanded canvas.
#   - Small top-facing trace:
#       original task: dest row 0, head row 1, frame starts lower; full move lands head on
#       row 0, so the frame still fits.
#       combo task: same setup but allowed step count is gap+2. If grey >= gap+2, the frame
#       moves one more cell up, its head lands on row -1, output grows by one top row, and
#       that new row uses the container-count background color.
#   - Synergy: the newly created border area is not an arbitrary filler. Its color is exactly
#     the same count-driven background seen elsewhere, so the two dimensions interact in the
#     visible output rather than behaving like unrelated edits.
#
# b_complexity_calibration:
#   - Container count stays in 2..4, so the background mapping has only three cases and every
#     case is structurally easy to count.
#   - Expansion magnitude is at most one cell per expanding side, which is enough to be seen
#     but does not create large rescalings or crop ambiguity.
#   - The generator forces at least one mover to trigger expansion, while still allowing other
#     movers/decoys to show capped-by-grey and zero-move behavior from the base task.
#
# c_implicit_assumption_audit:
#   - Original `bg = dominant non-border color` is no longer the intended inference rule.
#     The solver must first find containers using only the invariant 0/5/7 rectangle logic,
#     then infer background from container count.
#   - Original `out = g.copy()` assumes output shape equals input shape. That fails once the
#     extra D9 step is allowed. We instead collect all moves, compute global min/max coords,
#     allocate an expanded canvas, and then erase/redraw using padded coordinates.
#   - Original `paste(..., clip=True)` would silently drop overflow pixels. In this combo we
#     expand first and then paste strictly inside bounds.
#   - Original `steps = abs(tr0)+abs(tc0)` assumes the head stops on the destination line.
#     Here the observable D9 concretization requires `base_steps = abs(tr0)+abs(tc0)+1`.
#   - Erasing original head/frame pixels now happens after the input grid is pasted into the
#     expanded canvas, because top/left padding shifts all legacy coordinates.
#
# d_placement_legality:
#   - Assemblies are still placed with at least one background cell of separation.
#   - Expanding movers are pinned against the corresponding input edge so the extra step
#     actually forces overflow in a predictable direction.
#   - Grey budget for an expanding mover is constrained to be at least `gap+2`, guaranteeing
#     the extra step is feasible.
#
# e_visual_observability:
#   - Across examples, a solver can count 2/3/4 containers and see the whole blank field use
#     blue/green/yellow respectively.
#   - Independently, at least one container produces an output that is one row/column larger
#     on the side it moved toward. The new strip is plain background, so the expansion rule is
#     readable without hidden metadata.
#
# f_cleanliness:
#   - No new marker colors are introduced beyond the small background palette.
#   - Expansion only adds empty space on outer borders, avoiding overlaps or extra layers in
#     the crowded central region.
#
# g_color_inference:
#   - The mapping uses only three colors and one obvious structural trigger: total container
#     count. No sorting by color id or any non-visual quantity is involved.
#
# h_variant_metric_for_postcheck:
#   - Measure `(n_containers, bg_color)` and require it to stay in `{2:1, 3:3, 4:4}`.
#   - Measure whether `main(inp).shape != inp.shape`; the generator should make this true for
#     essentially every accepted sample, otherwise the D9 part has degenerated.
#   - Also compare against the original solver on the same input: accepted samples should
#     change the output geometry, so `combo_main(inp) != original_main(inp)` should hold.
#
# i_new_solver_assumptions_to_generator_contract:
#   - The solver now assumes there are exactly 2..4 valid containers and that the input
#     background obeys the fixed mapping above.
#   - The generator enforces those counts, uses the same mapping, preserves assembly spacing,
#     and ensures every expanding case sits on the boundary it is supposed to overflow past.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BORDER = 0
BG_BY_CONTAINER_COUNT = {2: 1, 3: 3, 4: 4}
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _bg_for_container_count(n_containers: int) -> int:
    if n_containers not in BG_BY_CONTAINER_COUNT:
        raise ValueError(f"Unsupported container count: {n_containers}")
    return int(BG_BY_CONTAINER_COUNT[n_containers])


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _grey_fill_order(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _assembly_dims(outer_h: int, outer_w: int, head_side: int, gap: int) -> tuple[int, int, int, int]:
    if head_side == 0:
        return 1 + gap + 1 + outer_h, outer_w, gap + 2, 0
    if head_side == 1:
        return outer_h + 1 + gap + 1, outer_w, 0, 0
    if head_side == 2:
        return outer_h, 1 + gap + 1 + outer_w, 0, gap + 2
    return outer_h, outer_w + 1 + gap + 1, 0, 0


def _edge_constraint(grid_h: int, grid_w: int, total_h: int, total_w: int, head_side: int, expand_out: bool):
    if not expand_out:
        return None, None
    if head_side == 0:
        return 0, None
    if head_side == 1:
        return grid_h - total_h, None
    if head_side == 2:
        return None, 0
    return None, grid_w - total_w


def _find_containers(g: np.ndarray) -> list[dict]:
    H, W = g.shape
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
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    containers = _find_containers(g)
    assert containers
    bg = _bg_for_container_count(len(containers))

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
                    if val not in (bg, BORDER, GREY, ORANGE):
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

    ops = []
    min_r, max_r, min_c, max_c = 0, H - 1, 0, W - 1

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
            if d["shape"] != head_shape or d["pts"] == comp_pts:
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
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
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
        base_steps = abs(tr0) + abs(tc0) + 1

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(base_steps), int(len(grey_coords)))
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in _grey_fill_order(grey_coords, direction)[:steps]:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)

        new_r0 = int(r0 + tr)
        new_c0 = int(c0 + tc)
        moved_head = {(int(r + tr), int(c + tc)) for r, c in head_pts}
        min_r = min(min_r, new_r0, *(r for r, _c in moved_head))
        max_r = max(max_r, int(r1 + tr), *(r for r, _c in moved_head))
        min_c = min(min_c, new_c0, *(c for _r, c in moved_head))
        max_c = max(max_c, int(c1 + tc), *(c for _r, c in moved_head))

        ops.append(
            {
                "orig_box": (r0, c0, r1, c1),
                "head_pts": head_pts,
                "head_color": hc,
                "patch": patch,
                "new_tl": (new_r0, new_c0),
                "moved_head": moved_head,
            }
        )

    pad_top = max(0, -int(min_r))
    pad_left = max(0, -int(min_c))
    pad_bottom = max(0, int(max_r) - (H - 1))
    pad_right = max(0, int(max_c) - (W - 1))

    out = np.full((H + pad_top + pad_bottom, W + pad_left + pad_right), int(bg), dtype=int)
    paste(out, g, pad_top, pad_left)

    for op in ops:
        r0, c0, r1, c1 = op["orig_box"]
        rr0, cc0 = int(r0 + pad_top), int(c0 + pad_left)
        rr1, cc1 = int(r1 + pad_top), int(c1 + pad_left)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(bg)
        hc = int(op["head_color"])
        for r, c in op["head_pts"]:
            rr = int(r + pad_top)
            cc = int(c + pad_left)
            if int(out[rr, cc]) == hc:
                out[rr, cc] = int(bg)

        nr0, nc0 = op["new_tl"]
        paste(out, op["patch"], int(nr0 + pad_top), int(nc0 + pad_left))
        for r, c in op["moved_head"]:
            out[int(r + pad_top), int(c + pad_left)] = hc

    return out


def _gen_attempt(target_n_containers: int | None = None) -> np.ndarray | None:
    n_containers = random.randint(2, 4) if target_n_containers is None else int(target_n_containers)
    bg = _bg_for_container_count(n_containers)

    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    mover_ids = list(range(n_movers))
    random.shuffle(mover_ids)
    n_expand = random.randint(1, min(2, n_movers))
    expand_ids = set(mover_ids[:n_expand])

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)
        inner_h = random.randint(1, 7)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        expand_out = i in expand_ids
        if is_decoy:
            gap = random.randint(0, 5)
            grey = 0
            initial_orange = inner_area
            steps = 0
            expand_out = False
        else:
            max_gap = min(6, inner_area - 1)
            if max_gap < 0:
                return None
            if expand_out:
                if inner_area < 2:
                    inner_h = random.randint(1, 4)
                    inner_w = random.randint(2, 5)
                    inner_area = inner_h * inner_w
                    outer_h = inner_h + 2
                    outer_w = inner_w + 2
                if inner_area < 2:
                    return None
                gap = random.randint(0, min(6, inner_area - 2))
                grey = random.randint(gap + 2, inner_area)
            else:
                gap = random.randint(0, max_gap)
                grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            steps = min(grey, gap + 2)

        total_h, total_w, cont_r, cont_c = _assembly_dims(outer_h, outer_w, head_side, gap)
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "is_decoy": is_decoy,
                "expand_out": expand_out,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    total_area = sum(s["total_h"] * s["total_w"] for s in specs)
    max_asm_h = max(s["total_h"] for s in specs)
    max_asm_w = max(s["total_w"] for s in specs)

    lo_h = max(8, max_asm_h)
    lo_w = max(8, max_asm_w)
    hi_h = min(26, max(lo_h, int(total_area**0.5) + 8))
    hi_w = min(26, max(lo_w, int(total_area**0.5) + 8))
    if lo_h > hi_h or lo_w > hi_w:
        return None
    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * n_containers
    order = sorted(
        range(n_containers),
        key=lambda i: (0 if specs[i]["expand_out"] else 1, -(specs[i]["total_h"] * specs[i]["total_w"])),
    )

    for idx in order:
        s = specs[idx]
        th, tw = s["total_h"], s["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        fixed_r, fixed_c = _edge_constraint(grid_h, grid_w, th, tw, s["head_side"], s["expand_out"])
        placed = False
        for _ in range(600):
            r0 = fixed_r if fixed_r is not None else random.randint(0, grid_h - th)
            c0 = fixed_c if fixed_c is not None else random.randint(0, grid_w - tw)
            cr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            cr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if occupied[cr0:cr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    if any(p is None for p in placements):
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    for s, placement in zip(specs, placements):
        pr, pc = placement
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]
        inner_h, inner_w = s["inner_h"], s["inner_w"]
        initial_orange = s["initial_orange"]
        cr = pr + s["cont_r"]
        cc = pc + s["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    ops = []
    min_r, max_r, min_c, max_c = 0, grid_h - 1, 0, grid_w - 1
    for s, placement in zip(specs, placements):
        pr, pc = placement
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        steps = s["steps"]
        inner_h, inner_w = s["inner_h"], s["inner_w"]
        initial_orange = s["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]
        cr = pr + s["cont_r"]
        cc = pc + s["cont_c"]
        nr = cr + dr * steps
        nc = cc + dc * steps

        if hs == 0:
            head_pts = {(cr - 1 + dr * steps, c + dc * steps) for c in range(cc, cc + ow)}
        elif hs == 1:
            head_pts = {(cr + oh + dr * steps, c + dc * steps) for c in range(cc, cc + ow)}
        elif hs == 2:
            head_pts = {(r + dr * steps, cc - 1 + dc * steps) for r in range(cr, cr + oh)}
        else:
            head_pts = {(r + dr * steps, cc + ow + dc * steps) for r in range(cr, cr + oh)}

        min_r = min(min_r, nr, *(r for r, _c in head_pts))
        max_r = max(max_r, nr + oh - 1, *(r for r, _c in head_pts))
        min_c = min(min_c, nc, *(c for _r, c in head_pts))
        max_c = max(max_c, nc + ow - 1, *(c for _r, c in head_pts))

        ops.append(
            {
                "cr": cr,
                "cc": cc,
                "nr": nr,
                "nc": nc,
                "oh": oh,
                "ow": ow,
                "hs": hs,
                "steps": steps,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "initial_orange": initial_orange,
            }
        )

    pad_top = max(0, -min_r)
    pad_left = max(0, -min_c)
    pad_bottom = max(0, max_r - (grid_h - 1))
    pad_right = max(0, max_c - (grid_w - 1))
    expected = np.full((grid_h + pad_top + pad_bottom, grid_w + pad_left + pad_right), int(bg), dtype=int)
    paste(expected, grid, pad_top, pad_left)

    for op in ops:
        cr, cc, nr, nc = op["cr"], op["cc"], op["nr"], op["nc"]
        oh, ow = op["oh"], op["ow"]
        hs = op["hs"]
        steps = op["steps"]
        inner_h, inner_w = op["inner_h"], op["inner_w"]
        initial_orange = op["initial_orange"]

        if steps == 0:
            continue

        expected[cr + pad_top : cr + pad_top + oh, cc + pad_left : cc + pad_left + ow] = int(bg)
        if hs == 0:
            expected[cr - 1 + pad_top, cc + pad_left : cc + pad_left + ow] = int(bg)
        elif hs == 1:
            expected[cr + oh + pad_top, cc + pad_left : cc + pad_left + ow] = int(bg)
        elif hs == 2:
            expected[cr + pad_top : cr + pad_top + oh, cc - 1 + pad_left] = int(bg)
        else:
            expected[cr + pad_top : cr + pad_top + oh, cc + ow + pad_left] = int(bg)

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr + pad_top : nr + pad_top + oh, nc + pad_left : nc + pad_left + ow] = patch

        if hs == 0:
            expected[nr - 1 + pad_top, nc + pad_left : nc + pad_left + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh + pad_top, nc + pad_left : nc + pad_left + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr + pad_top : nr + pad_top + oh, nc - 1 + pad_left] = HEAD_COLOR
        else:
            expected[nr + pad_top : nr + pad_top + oh, nc + ow + pad_left] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None
    if output.shape == grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    target_n_containers = random.randint(2, 4)
    for _ in range(500):
        result = _gen_attempt(target_n_containers)
        if result is not None:
            return result
    raise RuntimeError(
        f"Failed to generate valid combo variant after 500 attempts for n_containers={target_n_containers}"
    )
