# Variant of 271d71e2
# Dimension: D13 — 步数计算的数学关系 (meta)
# Change: steps = min(head_dest_distance, grey_count) → steps = max(0, grey_count - head_dest_distance)

from __future__ import annotations

# Design analysis for the D13 variant
# ----------------------------------
# variant_spec:
#   unchanged_visual_grammar:
#     - background is 6
#     - each active object is still a black rectangular container whose interior
#       is a head-facing orange prefix followed by grey cells
#     - a full-edge maroon head line sits next to one side of the container
#     - an identical maroon destination line lies farther away on the same axis
#   changed_math:
#     - original: distance = gap + 1, steps = min(distance, grey_count)
#     - variant:  distance = gap + 1, steps = max(0, grey_count - distance)
#   concrete_trace:
#     - example: grey_count = 8, distance = 5
#     - original behavior: move 5 and head lands on the destination
#     - variant behavior: move 3 and stop short of the destination; exactly the
#       first 3 head-side grey cells turn orange
#     - example: grey_count = 4, distance = 6
#     - original behavior: move 4
#     - variant behavior: move 0 even though grey exists, so the container stays
#       put; this branch is important because it makes the new arithmetic visible
#   complexity_calibration:
#     - only the arithmetic changes; object detection, matching, direction
#       inference, and fill ordering stay fixed to avoid changing unrelated
#       dimensions
#     - the generator deliberately mixes:
#         1. movers with distance < grey_count < 2 * distance, so steps are
#            positive but still smaller than distance, making "stop short" easy
#            to see and preventing overshoot
#         2. stationary grey containers with grey_count <= distance, so humans
#            can see that having grey cells no longer guarantees movement
#     - keeping the destination line unchanged preserves a clean visual cue for
#       the residual distance
#   hidden_assumptions_audit:
#     - solver line "steps = abs(tr0) + abs(tc0)" encoded the old assumption
#       that movement amount starts from destination distance; this must now be
#       treated as head_dest_distance only
#     - solver line "steps = min(steps, len(grey_coords))" encoded the old cap;
#       for the variant it becomes "max(0, grey_count - distance)"
#     - the later fill logic assumes steps <= grey_count; the new formula still
#       satisfies that because distance >= 1 on valid inputs
#     - erase/paste order assumes the moved container stays within the
#       head-to-destination corridor; the generator enforces steps < distance for
#       movers so the moved box never crosses the destination line
#   placement_legality:
#     - assemblies are placed using the same input bounding boxes as the source
#       task: destination + gap + head + container
#     - because generator movers satisfy steps < distance, the output position
#       remains inside that same bounding box, so pairwise non-overlap and
#       in-bounds placement can still be guaranteed with the original occupancy
#       strategy
#   visual_observability:
#     - every generated grid has at least one positive mover and at least one
#       grey-but-stationary container
#     - movers stop before the destination instead of snapping onto it, which is
#       visibly different from the source task on the same geometry
#   cleanliness_and_color_reasoning:
#     - no new colors or markers are introduced
#     - the only color change remains grey -> orange from the head side inward,
#       so the task remains readable from structure rather than color ids
#   new_main_assumptions_to_guarantee:
#     - each container still has exactly one axis-aligned matching destination
#     - same-color head/destination components remain isolated from other
#       assemblies by at least one background cell
#     - generated movers satisfy positive difference without overshoot

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _variant_steps(grey_count: int, head_dest_distance: int) -> int:
    """Positive difference rule for D13."""
    return max(0, int(grey_count) - int(head_dest_distance))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging
    grey cells to orange.

    Variant rule:
    - head_dest_distance = gap + 1
    - steps = max(0, grey_count - head_dest_distance)
    - if grey_count <= head_dest_distance, the container does not move
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    # --- detect containers as rectangles enclosing GREY/ORANGE with BORDER frame ---
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

    # head + destination are in some non-bg, non-border color (not GREY/ORANGE)
    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    # Precompute components for each head color.
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

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal manhattan distance to this container border.
        border_pts = list(rect_border_coords(r0, c0, r1, c1))
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

        # direction from head bbox relative to container box
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

        # add split-from-same-component destinations (when dest touches head by color)
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
        head_dest_distance = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = _variant_steps(len(grey_coords), head_dest_distance)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:  # (-1,0)
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:steps])

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


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Return interior cell coords in the solver's head-facing fill order."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:  # right, direction (0,1): key (-col, row)
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left, direction (0,-1): key (col, -row)
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom, direction (1,0): key (-row, -col)
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # top, direction (-1,0): key (row, col)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _random_inner_dims(min_area: int) -> tuple[int, int]:
    while True:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        if inner_h * inner_w >= min_area:
            return inner_h, inner_w


def _build_spec(role: str) -> dict | None:
    min_area = 4 if role == "mover" else 1
    inner_h, inner_w = _random_inner_dims(min_area=min_area)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if role == "mover":
        grey = random.randint(3, min(inner_area, 21))
        dist_lo = grey // 2 + 1
        dist_hi = min(grey - 1, 11)
        if dist_lo > dist_hi:
            return None
        head_dest_distance = random.randint(dist_lo, dist_hi)
        steps = _variant_steps(grey, head_dest_distance)
        assert 0 < steps < head_dest_distance
    elif role == "stationary":
        grey = random.randint(1, min(inner_area, 11))
        head_dest_distance = random.randint(grey, min(grey + 4, 11))
        steps = 0
    else:  # fully orange decoy
        grey = 0
        head_dest_distance = random.randint(1, 6)
        steps = 0

    return {
        "role": role,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "inner_area": inner_area,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "head_dest_distance": head_dest_distance,
        "gap": head_dest_distance - 1,
        "steps": steps,
        "grey": grey,
        "initial_orange": inner_area - grey,
    }


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    include_full = n_containers >= 3 and random.random() < 0.35

    roles = ["stationary"]
    if include_full:
        roles.append("full")
    while len(roles) < n_containers:
        roles.append("mover")
    random.shuffle(roles)

    specs = []
    for role in roles:
        spec = _build_spec(role)
        if spec is None:
            return None
        specs.append(spec)

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:  # top: dest above head above container
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:  # bottom: container above head above dest
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:  # left: dest left of head left of container
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:  # right: container left of head left of dest
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append(
            {
                **s,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

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
        for _ in range(400):
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
        initial_orange = a["initial_orange"]
        gap = a["gap"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

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

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
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
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
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

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid variant after 400 attempts")
