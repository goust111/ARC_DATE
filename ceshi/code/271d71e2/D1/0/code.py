# Variant of 271d71e2
# Dimension: D1 — 路径方向类型 (path)
# Change: straight_ray — 容器沿头部指示的四个基本方向（上/下/左/右）直线平移... → 对角线移动

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, STRUCT8, bbox_coords, colors_present, components_mask, paste

# Design analysis for the diagonal-path variant
# --------------------------------------------
# variant_spec:
#   core_change:
#     original_path: container moves on one row or one column
#     variant_path: container moves on one of four 45-degree diagonals
#       {(-1,-1), (-1,1), (1,-1), (1,1)}
#   concrete_objects:
#     background: 6 (magenta), unchanged
#     container:
#       unchanged: black border rectangle with grey/orange rectangular interior
#     head:
#       old_form: full edge-length straight line attached to one side
#       new_form: 2-pixel diagonal segment attached to one corner
#       corner_templates:
#         nw: {(r0-1,c0), (r0,c0-1)}
#         ne: {(r0-1,c1), (r0,c1+1)}
#         sw: {(r1,c0-1), (r1+1,c0)}
#         se: {(r1,c1+1), (r1+1,c1)}
#       reason:
#         a long diagonal ray would place most head pixels farther from the box than the first
#         one, breaking the original "closest pixels are the head" assumption. Using a short
#         2-pixel diagonal keeps every head pixel equally close to the container, so the split
#         between head and target remains visually obvious and solver-friendly.
#     target:
#       shape: identical 2-pixel diagonal segment
#       placement: same diagonal ray as the head, shifted by h2d = gap + 1 diagonal steps
#       gap0_case:
#         with 8-connectivity the touching head+target can form one same-color component, just
#         like the original orthogonal task's touching line case under 4-connectivity
#   behavior_trace:
#     original_small_case:
#       top head -> destination above -> valid candidate requires tc == 0 and tr < 0
#       movement steps = min(abs(tr), grey_count)
#       grey fill order comes from the top edge inward
#     variant_small_case:
#       NW head -> destination NW -> valid candidate requires tr == tc < 0
#       movement steps = min(abs(tr), grey_count)
#       grey fill order comes from the NW corner inward, sorted by corner-distance layers
#     visible_difference:
#       both row and column change every time a mover translates; orange growth starts from a
#       corner instead of a full side, so the D1 change is visible in both geometry and color.
#   complexity_calibration:
#     - Only four diagonal directions exist, mirroring the original four orthogonal directions.
#     - Head/target shapes remain tiny and repeated across examples, so humans can match them.
#     - The target is still same-colored and identical in shape to the head, keeping the old
#       matching idea while making the path computation diagonal.
#   hidden_assumption_audit:
#     - Original code labeled head/destination components with STRUCT4. That silently assumes
#       touching markers connect orthogonally. In the variant, touching diagonal segments only
#       connect under STRUCT8, so component extraction must switch to 8-connectivity.
#     - Original destination filtering assumed exactly one moving axis:
#         if direction[0] != 0 and tc0 != 0: continue
#       That rejects every diagonal target. Fix: require abs(tr0) == abs(tc0) and matching signs.
#     - Original grey fill order used edge-oriented sort keys. With a corner head that would
#       recolor the wrong cells even when movement is correct. Fix: sort interior cells by
#       Manhattan distance to the head corner, then by a corner-specific tie-break.
#     - Original same-component split used STRUCT4 after removing the head. For diagonal gap=0
#       cases this would incorrectly tear apart the remaining diagonal target. Fix: use STRUCT8.
#   placement_legality:
#     generator constructs each scenario in local coordinates, computes the union of input and
#     output footprints, then reserves that entire axis-aligned bbox with a 1-cell moat. This
#     guarantees:
#       - all objects stay in bounds
#       - input objects do not overlap or diagonally touch unrelated ones
#       - motion corridors remain clear
#       - no unrelated same-color marker lies on a nearer competing diagonal ray
#   solver_assumptions_added_by_variant:
#     - Each head is one of the four fixed 2-pixel diagonal corner templates above.
#     - The matching target has the same 2-pixel diagonal shape and lies farther out on the same
#       diagonal ray.
#     - The generator enforces enough spacing that nearest-valid-target matching is unambiguous.

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _normalize_points(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (int(r0), int(c0))


def _shift_points(pts: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _head_pts_for_box(box: tuple[int, int, int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if direction == (-1, -1):
        return {(r0 - 1, c0), (r0, c0 - 1)}
    if direction == (-1, 1):
        return {(r0 - 1, c1), (r0, c1 + 1)}
    if direction == (1, -1):
        return {(r1, c0 - 1), (r1 + 1, c0)}
    return {(r1, c1 + 1), (r1 + 1, c1)}


def _head_local_points(outer_h: int, outer_w: int, direction: tuple[int, int]) -> set[tuple[int, int]]:
    if direction == (-1, -1):
        return {(-1, 0), (0, -1)}
    if direction == (-1, 1):
        return {(-1, outer_w - 1), (0, outer_w)}
    if direction == (1, -1):
        return {(outer_h - 1, -1), (outer_h, 0)}
    return {(outer_h - 1, outer_w), (outer_h, outer_w - 1)}


def _corner_fill_sort_key(
    inner_h: int, inner_w: int, direction: tuple[int, int], cell: tuple[int, int]
) -> tuple[int, int, int]:
    r, c = cell
    if direction == (-1, -1):
        return (int(r + c), int(r), int(c))
    if direction == (-1, 1):
        return (int(r + (inner_w - 1 - c)), int(r), int(-c))
    if direction == (1, -1):
        return (int((inner_h - 1 - r) + c), int(-r), int(c))
    return (int((inner_h - 1 - r) + (inner_w - 1 - c)), int(-r), int(-c))


def _corner_fill_order(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    cells.sort(key=lambda p: _corner_fill_sort_key(inner_h, inner_w, direction, p))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored diagonal destination markers, charging grey cells to orange.

    Variant:
    - The head and target are short diagonal segments attached to a corner.
    - Valid motion is diagonal only.
    - Grey cells are consumed from the head corner inward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

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

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT8)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            pts = {(int(r), int(c)) for r, c in coords}
            shape, tl = _normalize_points(pts)
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": tl})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(int(r), int(c)) for r, c, val in head_seeds if int(val) == int(hc)}

        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        directions = [d for d in DIAG_DIRS if _head_pts_for_box(box_, d) <= seed_pts]
        if len(directions) != 1:
            return None
        direction = directions[0]
        head_pts = _head_pts_for_box(box_, direction)
        if not head_pts <= comp0["pts"]:
            return None

        head_shape, head_tl = _normalize_points(head_pts)
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": head_tl,
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_

        head_info = pick_head_for_container(box_)
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]

        dests = []
        for comp in comp_by_color.get(hc, []):
            if comp["shape"] != head_shape or comp["pts"] == comp_pts:
                continue
            dests.append(comp)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT8)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if not pts:
                    continue
                shape, tl = _normalize_points(pts)
                if shape != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shape, "tl": tl})

        if not dests:
            continue

        best_dist = None
        best_dest = None
        for dest in dests:
            tr0 = int(dest["tl"][0] - head_tl[0])
            tc0 = int(dest["tl"][1] - head_tl[1])
            if tr0 == 0 or abs(tr0) != abs(tc0):
                continue
            if int(tr0 / abs(tr0)) != int(direction[0]):
                continue
            if int(tc0 / abs(tc0)) != int(direction[1]):
                continue
            dist = abs(tr0)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = dest
        if best_dest is None:
            continue

        steps = abs(int(best_dest["tl"][0] - head_tl[0]))
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_local = [
            (int(rr), int(cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_local)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        grey_local.sort(key=lambda p: _corner_fill_sort_key(inner.shape[0], inner.shape[1], direction, p))
        to_fill_local = set(grey_local[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if 0 <= r < H and 0 <= c < W and int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill_local:
            patch[int(rr + 1), int(cc + 1)] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[int(rrr), int(ccc)] = int(hc)

    return out


def _scenario_bbox(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    direction = spec["direction"]
    h2d = spec["h2d"]
    steps = spec["steps"]

    container_pts = {(r, c) for r in range(oh) for c in range(ow)}
    head_pts = _head_local_points(oh, ow, direction)
    dest_pts = _shift_points(head_pts, direction[0] * h2d, direction[1] * h2d)
    out_container_pts = _shift_points(container_pts, direction[0] * steps, direction[1] * steps)
    out_head_pts = _shift_points(head_pts, direction[0] * steps, direction[1] * steps)
    footprint = container_pts | head_pts | dest_pts | out_container_pts | out_head_pts

    rows = [r for r, _c in footprint]
    cols = [c for _r, c in footprint]
    min_r = int(min(rows))
    min_c = int(min(cols))
    max_r = int(max(rows))
    max_c = int(max(cols))

    spec = dict(spec)
    spec["min_r"] = min_r
    spec["min_c"] = min_c
    spec["asm_h"] = int(max_r - min_r + 1)
    spec["asm_w"] = int(max_c - min_c + 1)
    return spec


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for idx in range(n_containers):
        is_decoy = idx >= n_movers
        inner_h = random.randint(1, 7)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        direction = random.choice(DIAG_DIRS)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(0, 8)
            steps = min(grey, gap + 1)

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "direction": direction,
                "gap": gap,
                "h2d": gap + 1,
                "steps": steps,
                "grey": grey,
                "initial_orange": initial_orange,
                "is_decoy": is_decoy,
            }
        )

    specs = [_scenario_bbox(spec) for spec in specs]

    total_area = sum(spec["asm_h"] * spec["asm_w"] for spec in specs)
    min_side = max(10, int(total_area ** 0.5) + 4)
    max_asm_h = max(spec["asm_h"] for spec in specs)
    max_asm_w = max(spec["asm_w"] for spec in specs)
    lo_h = max(10, max_asm_h + 2)
    lo_w = max(10, max_asm_w + 2)
    hi_h = min(26, max(lo_h, min_side + 6))
    hi_w = min(26, max(lo_w, min_side + 6))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * n_containers

    order = sorted(range(n_containers), key=lambda i: specs[i]["asm_h"] * specs[i]["asm_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        ah, aw = spec["asm_h"], spec["asm_w"]
        if ah > grid_h or aw > grid_w:
            return None
        placed = False
        for _ in range(400):
            pr = random.randint(0, grid_h - ah)
            pc = random.randint(0, grid_w - aw)
            rr0 = max(0, pr - 1)
            cc0 = max(0, pc - 1)
            rr1 = min(grid_h, pr + ah + 1)
            cc1 = min(grid_w, pc + aw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[pr : pr + ah, pc : pc + aw] = True
                placements[idx] = (pr, pc)
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for spec, placement in zip(specs, placements):
        assert placement is not None
        pr, pc = placement
        direction = spec["direction"]
        oh, ow = spec["outer_h"], spec["outer_w"]
        inner_h, inner_w = spec["inner_h"], spec["inner_w"]
        shift_r = int(pr - spec["min_r"])
        shift_c = int(pc - spec["min_c"])
        cr, cc = int(shift_r), int(shift_c)

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for rr, cc2 in _corner_fill_order(inner_h, inner_w, direction)[: spec["initial_orange"]]:
            interior[int(rr), int(cc2)] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        head_pts = _shift_points(_head_local_points(oh, ow, direction), shift_r, shift_c)
        dest_pts = _shift_points(head_pts, direction[0] * spec["h2d"], direction[1] * spec["h2d"])
        for r, c in head_pts | dest_pts:
            if not (0 <= r < grid_h and 0 <= c < grid_w):
                return None
            grid[int(r), int(c)] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for spec, placement in zip(specs, placements):
        assert placement is not None
        pr, pc = placement
        direction = spec["direction"]
        oh, ow = spec["outer_h"], spec["outer_w"]
        inner_h, inner_w = spec["inner_h"], spec["inner_w"]
        steps = spec["steps"]
        shift_r = int(pr - spec["min_r"])
        shift_c = int(pc - spec["min_c"])
        cr, cc = int(shift_r), int(shift_c)
        head_pts = _shift_points(_head_local_points(oh, ow, direction), shift_r, shift_c)

        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        for r, c in head_pts:
            expected[int(r), int(c)] = BG

        nr = int(cr + direction[0] * steps)
        nc = int(cc + direction[1] * steps)
        patch = np.full((oh, ow), BORDER, dtype=int)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for rr, cc2 in _corner_fill_order(inner_h, inner_w, direction)[: spec["initial_orange"] + steps]:
            interior[int(rr), int(cc2)] = ORANGE
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        moved_head = _shift_points(head_pts, direction[0] * steps, direction[1] * steps)
        for r, c in moved_head:
            expected[int(r), int(c)] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
