# Variant of 271d71e2
# Dimension: D13 — 步数计算的数学关系 (meta)
# Change: steps = min(head_dest_distance, grey_count) -> steps = max(head_dest_distance, grey_count)

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the D13 max-step variant
# --------------------------------------------
# a) Concrete behavior:
#    The visual vocabulary stays the same as the original task: magenta background,
#    black-bordered rectangles, grey/orange interiors, one adjacent maroon head line,
#    and one aligned maroon destination line on the movement axis.
#    The only rule change is the movement arithmetic:
#      distance = gap + 1
#      steps = max(distance, grey_count)
#    This has two visible consequences:
#    - If distance > grey_count, the container still travels all the way to the
#      destination even though there are not enough grey cells to "pay" for every
#      step. Only the existing grey cells can turn orange.
#    - If grey_count > distance, the container overshoots the destination. The
#      destination line can end up behind or under the moved container because the
#      solver keeps the same layering as the original task: erase old box/head,
#      keep destination in place, paste moved box, then stamp moved head.
#    Example trace:
#      original: distance=5, grey=2 -> steps=min(5,2)=2, box stops early, 2 greys fill.
#      variant : distance=5, grey=2 -> steps=max(5,2)=5, box reaches destination, only
#                the 2 grey cells fill, so the moved box still changes position more than
#                color count.
#      original: distance=2, grey=5 -> steps=min(2,5)=2, some grey remains.
#      variant : distance=2, grey=5 -> steps=max(2,5)=5, box moves past destination and
#                all 5 grey cells fill.
#
# b) Complexity calibration:
#    The structural objects are unchanged, so recognition difficulty stays close to
#    the seed task. To make the D13 change observable, the generator always includes
#    at least one distance-dominant mover and one grey-dominant mover in every grid.
#    This prevents accidental degeneration into mostly "equal" cases where max/min
#    would look the same.
#
# c) Solver assumption audit:
#    - Container detection only assumes interiors are subsets of {grey, orange}; that
#      still holds, including fully-orange movers.
#    - Head/destination matching assumes same-color aligned line components; unchanged.
#    - The original code reused one variable `steps` for both translation distance and
#      number of grey cells converted. That coupling depended on `steps <= grey_count`.
#      Under `max`, that assumption breaks, so the solver must separate:
#        movement_steps = max(distance, grey_count)
#        fill_count = min(movement_steps, grey_count) == grey_count
#    - The original generator treated grey=0 boxes as stationary decoys. That is no
#      longer valid because max(distance, 0) = distance > 0, so fully-orange boxes
#      also move if they have a destination marker.
#
# d) Placement legality:
#    Because a box can now move past its destination, the old input-only assembly bbox
#    is insufficient. The generator reserves a full swept-frame bbox that contains the
#    input box/head, the static destination, and the output box/head. Frames are placed
#    with a 1-cell background moat so both input and output stay non-overlapping.
#
# e) Visual observability:
#    Across examples, humans can now see both branches:
#    - some boxes reach the destination despite too few grey cells;
#    - some boxes pass beyond the destination because grey count is larger.
#    Since steps >= grey_count always holds in this variant, every moved box ends with
#    a fully-orange interior. The only remaining variability is how far it moves, which
#    is exactly the intended D13 focus.
#
# f) Visual cleanliness:
#    All objects remain axis-aligned rectangles/lines with the original palette. The
#    frame-based placement prevents different assemblies from interleaving in the swept
#    corridor, avoiding ambiguous overlaps between unrelated boxes.
#
# g) Color inference:
#    No new colors or color mappings are introduced. Grey still means "available charge"
#    and orange still means "charged". The difference is only the arithmetic that links
#    movement length to distance vs grey count.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

MAX_DISTANCE = 8
MAX_GREY = 8
MAX_FRAME_AXIS = 18


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant D13 changes only the step arithmetic:
    - distance = head-to-destination offset along the movement axis.
    - movement steps = max(distance, grey_count).
    - Grey-to-orange conversion still cannot exceed the actual grey cell count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

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

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
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
                adjacent_to_border = False
                for r, c in pts:
                    for dr, dc in DIR4:
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
                            adjacent_to_border = True
                            break
                    if adjacent_to_border:
                        break
                if adjacent_to_border:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00, c00 = int(min(rs)), int(min(cs))
                shp = {(int(r - r00), int(c - c00)) for r, c in pts}
                if shp != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        best_dest = None
        best = None
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
        distance = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        grey_count = int(len(grey_coords))
        steps = max(int(distance), grey_count)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        order = _grey_fill_order(grey_coords, direction)
        fill_count = min(steps, grey_count)
        to_fill = set(order[:fill_count])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _choose_branch_plan(n_containers: int) -> list[str]:
    plan = ["distance_dominant", "grey_dominant"]
    extras = ["distance_dominant", "grey_dominant", "equal"]
    while len(plan) < n_containers:
        plan.append(random.choice(extras))
    random.shuffle(plan)
    return plan


def _make_spec(branch: str) -> dict | None:
    for _ in range(200):
        head_side = random.randint(0, 3)
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        inner_area = inner_h * inner_w
        max_grey = min(inner_area, MAX_GREY)
        if max_grey < 1:
            continue

        if branch == "grey_dominant":
            if max_grey < 2:
                continue
            overshoot_lo = 2 if max_grey >= 3 else 1
            overshoot = random.randint(overshoot_lo, min(4, max_grey - 1))
            distance = random.randint(1, max_grey - overshoot)
            grey = distance + overshoot
        elif branch == "distance_dominant":
            distance = random.randint(1, MAX_DISTANCE)
            max_grey_here = min(max_grey, distance - 1)
            if max_grey_here < 0:
                continue
            if max_grey_here == 0:
                grey = 0
            elif random.random() < 0.25:
                grey = 0
            else:
                grey = random.randint(1, max_grey_here)
        else:
            distance = random.randint(1, min(max_grey, MAX_DISTANCE))
            grey = distance

        steps = max(distance, grey)
        axis_outer = outer_h if head_side in (0, 1) else outer_w
        if steps + axis_outer + 1 > MAX_FRAME_AXIS:
            continue

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "distance": distance,
            "gap": distance - 1,
            "steps": steps,
            "grey": grey,
            "initial_orange": inner_area - grey,
            "branch": branch,
        }
    return None


def _layout_for_spec(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    distance = spec["distance"]
    steps = spec["steps"]

    if hs == 0:
        return {**spec, "total_h": steps + oh + 1, "total_w": ow, "in_r": steps + 1, "in_c": 0, "out_r": 1, "out_c": 0}
    if hs == 1:
        return {**spec, "total_h": oh + steps + 1, "total_w": ow, "in_r": 0, "in_c": 0, "out_r": steps, "out_c": 0}
    if hs == 2:
        return {**spec, "total_h": oh, "total_w": steps + ow + 1, "in_r": 0, "in_c": steps + 1, "out_r": 0, "out_c": 1}
    return {**spec, "total_h": oh, "total_w": ow + steps + 1, "in_r": 0, "in_c": 0, "out_r": 0, "out_c": steps}


def _draw_input_assembly(grid: np.ndarray, assembly: dict, pr: int, pc: int) -> None:
    oh, ow = assembly["outer_h"], assembly["outer_w"]
    hs = assembly["head_side"]
    distance = assembly["distance"]
    inner_h, inner_w = assembly["inner_h"], assembly["inner_w"]
    initial_orange = assembly["initial_orange"]

    cr = pr + assembly["in_r"]
    cc = pc + assembly["in_c"]

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
        grid[cr - 1 - distance, cc : cc + ow] = HEAD_COLOR
    elif hs == 1:
        grid[cr + oh, cc : cc + ow] = HEAD_COLOR
        grid[cr + oh + distance, cc : cc + ow] = HEAD_COLOR
    elif hs == 2:
        grid[cr : cr + oh, cc - 1] = HEAD_COLOR
        grid[cr : cr + oh, cc - 1 - distance] = HEAD_COLOR
    else:
        grid[cr : cr + oh, cc + ow] = HEAD_COLOR
        grid[cr : cr + oh, cc + ow + distance] = HEAD_COLOR


def _build_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()

    for assembly, (pr, pc) in zip(assemblies, placements):
        oh, ow = assembly["outer_h"], assembly["outer_w"]
        hs = assembly["head_side"]
        inner_h, inner_w = assembly["inner_h"], assembly["inner_w"]

        cr = pr + assembly["in_r"]
        cc = pc + assembly["in_c"]
        nr = pr + assembly["out_r"]
        nc = pc + assembly["out_c"]

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1 : 1 + inner_h, 1 : 1 + inner_w] = ORANGE
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    branches = _choose_branch_plan(n_containers)

    assemblies = []
    for branch in branches:
        spec = _make_spec(branch)
        if spec is None:
            return None
        assemblies.append(_layout_for_spec(spec))

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(10, int(total_area ** 0.5) + 3)
    max_side = min(26, min_side + 6)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(max_asm_h + 2, min_side)
    lo_w = max(max_asm_w + 2, min_side)
    hi_h = max(lo_h, min(26, max_side + 2))
    hi_w = max(lo_w, min(26, max_side + 2))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(assemblies)
    order = sorted(
        range(len(assemblies)),
        key=lambda i: assemblies[i]["total_h"] * assemblies[i]["total_w"],
        reverse=True,
    )

    for idx in order:
        asm = assemblies[idx]
        th, tw = asm["total_h"], asm["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
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

    final_placements = [(int(r), int(c)) for r, c in placements if r is not None and c is not None]
    if len(final_placements) != len(assemblies):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for assembly, (pr, pc) in zip(assemblies, final_placements):
        _draw_input_assembly(grid, assembly, pr, pc)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, assemblies, final_placements)
    if not np.array_equal(output, expected):
        return None

    distance_dominant = sum(1 for a in assemblies if a["branch"] == "distance_dominant")
    grey_dominant = sum(1 for a in assemblies if a["branch"] == "grey_dominant")
    if distance_dominant < 1 or grey_dominant < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
