# Combo variant of 271d71e2 (index 5)
# combo_spec: D3/1+D5/2
# Name: 距离排序渐变着色
#
# Design analysis
# ===============
# a) Concrete behavior
# - Base geometry is unchanged: magenta background, black-bordered rectangular containers,
#   maroon head line on one side, aligned maroon destination line further along the motion axis.
# - The fused rule changes the charging logic in two coupled ways:
#   1. Which interior cells belong to the charged prefix is decided by Manhattan distance to the
#      destination-line midpoint, nearest first.
#   2. Charged cells do not all become orange. The charged prefix is painted by a visible cycle
#      2 -> 4 -> 7 -> 2 -> 4 -> 7 -> ...
# - Small right-moving example with a 3x3 interior:
#     legacy order from head side:      [(0,2),(1,2),(2,2),(0,1),...]
#     distance-to-dest-midpoint order:  [(1,2),(0,2),(2,2),(1,1),...]
#   If input already has 2 charged cells and gains 2 more:
#     legacy+orange output   would color cells {right column top/mid, then next} all as 7
#     fused output colors    [(1,2),(0,2),(2,2),(1,1)] as [2,4,7,2]
#   So both the chosen positions and the assigned colors differ from the source task.
#
# b) Complexity calibration
# - The color cycle uses only three colors, enough to infer a repeat without clutter.
# - The distance reference is the visible midpoint of the visible destination line; no hidden
#   metadata is used.
# - Generator-side filters force at least one mover to expose both dimensions at once:
#   distance order must differ from the original head-side sweep, and the charged prefix after
#   movement must be long enough to display the whole 3-color cycle.
#
# c) Implicit assumption audit
# - Original container detection assumed interiors were only {5,7}. That breaks once earlier
#   charged cells can be 2 or 4, so detection now accepts {5,2,4,7}.
# - Original head detection excluded only {bg,0,5,7}. That would misclassify red/yellow interior
#   pixels as heads, so all interior colors are excluded from head/destination search.
# - Original recoloring overwrote selected grey cells with constant 7. That no longer works
#   because the final color depends on prefix index. The solver must rebuild the whole interior
#   from "charged_count + order + cycle", not mutate cells in place.
# - Original generator encoded existing orange cells by legacy sweep order. Under the fused rule,
#   input charged cells must already be a prefix of the distance-sorted order; otherwise the
#   solver's continuation step would imply a different hidden history. The generator now enforces
#   that stronger invariant.
#
# d) Placement legality
# - Assemblies still follow the original linear layout: destination + gap + head + container.
# - A one-cell background moat is preserved between assemblies.
# - Motion remains inside each assembly footprint because the moved head only lands onto the
#   pre-existing destination line, exactly as in the base task.
#
# e) Visual observability
# - Humans can see the midpoint anchor because the destination is a full line.
# - Humans can see the cycle because the charged prefix uses a repeating 2/4/7 stripe.
# - The generator rejects samples where the fused output would match either degeneration:
#   distance-order + all-orange, or color-cycle + legacy head-side order.
#
# f) Visual cleanliness
# - No extra markers are introduced. Containers, heads, and destinations stay visually separate.
# - Only the interior palette and internal ordering change, so the scene remains readable.
#
# g) Change inferability
# - The cycle is structurally tied to a ranked prefix, not arbitrary per-cell recoloring.
# - The ranking is tied to a single visible reference point family: destination midpoint.
# - With 2-4 examples, the solver can infer both "which cells get charged first" and
#   "charged cells follow a repeating color sequence".

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
RED = 2
YELLOW = 4
ORANGE = 7
HEAD_COLOR = 9
CHARGE_CYCLE = (RED, YELLOW, ORANGE)
INTERIOR_COLORS = {GREY, *CHARGE_CYCLE}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# head on top -> container moves up; head on bottom -> moves down; etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _midpoint2_from_pts(pts: set[tuple[int, int]]) -> tuple[int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    return int(min(rs) + max(rs)), int(min(cs) + max(cs))


def _distance_fill_order_abs(
    cells: list[tuple[int, int]], ref2r: int, ref2c: int
) -> list[tuple[int, int]]:
    return sorted(
        cells,
        key=lambda p: (
            abs(2 * int(p[0]) - int(ref2r)) + abs(2 * int(p[1]) - int(ref2c)),
            int(p[0]),
            int(p[1]),
        ),
    )


def _distance_fill_order_local(inner_h: int, inner_w: int, head_side: int, gap: int) -> list[tuple[int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if head_side == 0:
        ref2r, ref2c = -2 * (gap + 2), outer_w - 1
    elif head_side == 1:
        ref2r, ref2c = 2 * (outer_h + gap + 1), outer_w - 1
    elif head_side == 2:
        ref2r, ref2c = outer_h - 1, -2 * (gap + 2)
    else:
        ref2r, ref2c = outer_h - 1, 2 * (outer_w + gap + 1)

    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return sorted(
        cells,
        key=lambda p: (
            abs(2 * (p[0] + 1) - ref2r) + abs(2 * (p[1] + 1) - ref2c),
            int(p[0]),
            int(p[1]),
        ),
    )


def _legacy_fill_order_local(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _paint_prefix(inner_h: int, inner_w: int, order: list[tuple[int, int]], charged_count: int, use_cycle: bool) -> np.ndarray:
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx, (r, c) in enumerate(order[:charged_count]):
        interior[r, c] = CHARGE_CYCLE[idx % len(CHARGE_CYCLE)] if use_cycle else ORANGE
    return interior


def _prefix_sets_differ(order_a: list[tuple[int, int]], order_b: list[tuple[int, int]], k: int) -> bool:
    return 0 < k < len(order_a) and set(order_a[:k]) != set(order_b[:k])


def _spec_exposes_combo(spec: dict) -> bool:
    if spec["is_decoy"] or spec["steps"] <= 0:
        return False
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    head_side = spec["head_side"]
    gap = spec["gap"]
    initial_charged = spec["initial_charged"]
    total_charged = initial_charged + spec["steps"]

    if total_charged < len(CHARGE_CYCLE):
        return False

    dist_order = _distance_fill_order_local(inner_h, inner_w, head_side, gap)
    legacy_order = _legacy_fill_order_local(inner_h, inner_w, head_side)
    return _prefix_sets_differ(dist_order, legacy_order, initial_charged) or _prefix_sets_differ(
        dist_order, legacy_order, total_charged
    )


def _solve_with_modes(grid: np.ndarray, *, use_distance_order: bool, use_cycle: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    mask = (g == GREY) | np.isin(g, list(CHARGE_CYCLE))
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, list(INTERIOR_COLORS)).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, *INTERIOR_COLORS})]

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
                    if val not in {int(bg), int(BORDER), *INTERIOR_COLORS}:
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
                if shp == head_shape:
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
        move_budget = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.sum(inner == GREY))
        initial_charged = int(inner.size - grey_count)
        steps = min(int(move_budget), grey_count)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        inner_h, inner_w = inner.shape
        if use_distance_order:
            ref2r, ref2c = _midpoint2_from_pts(best_dest["pts"])
            all_abs = [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr in range(inner_h) for cc in range(inner_w)]
            abs_order = _distance_fill_order_abs(all_abs, ref2r, ref2c)
            order = [(rr - (r0 + 1), cc - (c0 + 1)) for rr, cc in abs_order]
        else:
            head_side = DIR_TO_SIDE[direction]
            order = _legacy_fill_order_local(inner_h, inner_w, head_side)
        total_charged = initial_charged + steps

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _paint_prefix(inner_h, inner_w, order, total_charged, use_cycle)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - translation steps = min(head-to-destination distance, number of grey cells)
    - charged cells are the nearest cells to the destination midpoint
    - charged prefix colors cycle 2 -> 4 -> 7 repeatedly
    """
    return _solve_with_modes(grid, use_distance_order=True, use_cycle=True)


def _make_container_spec(is_decoy: bool, force_visible: bool) -> dict:
    head_side = random.randint(0, 3)
    if not is_decoy:
        if head_side in (0, 1):
            inner_h = random.randint(1, 8)
            inner_w = random.choice([3, 5]) if force_visible else random.randint(1, 5)
        else:
            inner_h = random.choice([3, 5, 7]) if force_visible else random.randint(1, 8)
            inner_w = random.randint(1, 5)
    else:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)

    inner_area = inner_h * inner_w
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if is_decoy:
        grey = 0
        initial_charged = inner_area
        gap = random.randint(0, 6)
        steps = 0
    else:
        if force_visible:
            gap = random.randint(1, min(4, 10))
            h2d = gap + 1
            grey_lo = 2
            grey_hi = min(inner_area, inner_area + h2d - len(CHARGE_CYCLE))
            if grey_hi < grey_lo:
                return _make_container_spec(is_decoy=is_decoy, force_visible=False)
            grey = random.randint(grey_lo, grey_hi)
            initial_charged = inner_area - grey
        else:
            gap = random.randint(0, 10)
            h2d = gap + 1
            grey = random.randint(1, inner_area)
            initial_charged = inner_area - grey
        steps = min(grey, h2d)

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "steps": steps,
        "initial_charged": initial_charged,
        "grey": grey,
        "is_decoy": is_decoy,
    }


def _build_specs() -> list[dict]:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    visible_slot = random.randrange(n_movers)
    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        specs.append(_make_container_spec(is_decoy=is_decoy, force_visible=(i == visible_slot)))

    if not any(_spec_exposes_combo(spec) for spec in specs):
        return []
    if not any((not spec["is_decoy"]) and spec["steps"] >= 2 for spec in specs):
        return []
    return specs


def _assembly_spec(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    gap = spec["gap"]
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
    return {**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}


def _place_assemblies(assemblies: list[dict], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
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
    return placements


def _line_starts(assembly: dict, cr: int, cc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    oh, ow = assembly["outer_h"], assembly["outer_w"]
    hs = assembly["head_side"]
    gap = assembly["gap"]
    if hs == 0:
        return (cr - 1, cc), (cr - 2 - gap, cc)
    if hs == 1:
        return (cr + oh, cc), (cr + oh + 1 + gap, cc)
    if hs == 2:
        return (cr, cc - 1), (cr, cc - 2 - gap)
    return (cr, cc + ow), (cr, cc + ow + 1 + gap)


def _build_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        hs = a["head_side"]
        steps = a["steps"]
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
        order = _distance_fill_order_local(inner_h, inner_w, hs, a["gap"])
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _paint_prefix(inner_h, inner_w, order, a["initial_charged"] + steps, True)
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
    specs = _build_specs()
    if not specs:
        return None

    assemblies = [_assembly_spec(spec) for spec in specs]
    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area**0.5) + 2)
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
    placements = _place_assemblies(assemblies, grid_h, grid_w)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        hs = a["head_side"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        order = _distance_fill_order_local(inner_h, inner_w, hs, a["gap"])
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = _paint_prefix(
            inner_h, inner_w, order, a["initial_charged"], True
        )

        head_start, dest_start = _line_starts(a, cr, cc)
        if hs in (0, 1):
            grid[head_start[0], head_start[1] : head_start[1] + ow] = HEAD_COLOR
            grid[dest_start[0], dest_start[1] : dest_start[1] + ow] = HEAD_COLOR
        else:
            grid[head_start[0] : head_start[0] + oh, head_start[1]] = HEAD_COLOR
            grid[dest_start[0] : dest_start[0] + oh, dest_start[1]] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    baseline_cycle_only = _solve_with_modes(grid, use_distance_order=False, use_cycle=True)
    baseline_distance_only = _solve_with_modes(grid, use_distance_order=True, use_cycle=False)
    if np.array_equal(output, baseline_cycle_only):
        return None
    if np.array_equal(output, baseline_distance_only):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
