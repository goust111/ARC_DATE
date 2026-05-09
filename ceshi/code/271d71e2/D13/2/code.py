# Variant of 271d71e2
# Dimension: D13 — 步数计算的数学关系 (meta)
# Change: steps = min(head_dest_distance, grey_count)，其中 head_dest_distance = gap + 1
#         → steps = head_dest_distance % grey_count（取模；grey_count=0 时视为 0 步）
#
# Design analysis for this variant
# --------------------------------
# source_task:
#   requested_reference: 16de56c4
#   status: solver file not present in the workspace; only the JSON task data exists,
#           so there was no local solver implementation to read. The variant is
#           therefore implemented directly on top of 271d71e2 while keeping the
#           rest of the task structure unchanged.
#
# concrete_behavior:
#   unchanged_pixels:
#     - background stays 6
#     - containers are still black-bordered rectangles with grey/orange interiors
#     - the head and destination are still identical maroon lines on the movement axis
#     - grey->orange charging order is still from the head-facing side inward
#   changed_pixels:
#     - the only semantic change is how far a container moves.
#     - original: distance=7, grey_count=3 -> steps=min(7,3)=3
#     - variant:  distance=7, grey_count=3 -> steps=7%3=1
#     - so the same input now moves the container only 1 cell and converts only
#       the first grey cell in fill order, leaving a much larger residual gap
#       between the moved head and the destination.
#
# flow_impact:
#   solver:
#     - detection of containers, heads, destinations, and fill ordering remain valid
#     - only the distance->steps conversion changes
#   generator:
#     - must no longer sample grey/distance pairs as if min(...) were the target
#     - must deliberately create at least one container where modulo differs from min,
#       otherwise a generated example could accidentally behave like the original task
#
# complexity_calibration:
#   container_modes:
#     - mod_nonzero: distance >= grey_count and distance % grey_count in [1, grey_count-1]
#       This is the primary visible branch and is forced to appear at least once.
#     - distance_limited: grey_count > distance, so modulo equals distance.
#       This preserves a few "old-looking" examples for contrast.
#     - mod_zero: distance is an exact multiple of grey_count, so the container with
#       remaining grey cells does not move at all. This exposes the periodicity branch
#       without adding a new visual vocabulary.
#     - decoy_full_orange: grey_count=0, unchanged from the original task.
#   rationale:
#     - these few discrete modes keep the rule inferable from 2-4 examples
#     - forcing at least one mod_nonzero mover prevents degeneration
#     - limiting decoys/stationary-grey cases avoids clutter and ambiguity
#
# implicit_assumption_audit_of_main:
#   - container detection assumes interiors are still only grey/orange inside a black
#     rectangle. The variant keeps that unchanged.
#   - head/destination matching assumes same-colored aligned line sprites. Unchanged.
#   - destination choice assumes movement is axis-aligned and toward the nearest valid
#     matching line in that direction. Unchanged.
#   - the original `steps = min(distance, grey_count)` silently relied on grey_count
#     being the only cap. That is no longer true; step count must be derived from the
#     full distance before translation is computed.
#   - modulo needs a guard for grey_count=0. The original min() path handled this
#     naturally; the variant would otherwise raise ZeroDivisionError.
#
# placement_legality:
#   - the assembly footprint still reserves the full span from container to destination.
#   - because `distance % grey_count <= distance` whenever grey_count>0, any moved
#     container remains inside the same reserved corridor, so non-overlap and bounds
#     checks from the original placement strategy still work.
#
# visual_observability:
#   - the visible cue is not a hidden number; it is the residual gap left after moving
#     and the shorter orange-charging run inside the container.
#   - examples force at least one case where the container has enough grey cells but
#     still moves only a small remainder, which is visibly distinct from "move until
#     grey runs out".
#
# cleanliness_and_color_inference:
#   - no new colors are introduced
#   - no extra markers are added
#   - the grid remains readable because the variant is expressed only through the
#     amount of translation and the count of newly orange cells

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
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _step_count(head_dest_distance: int, grey_count: int) -> int:
    distance = int(head_dest_distance)
    grey = int(grey_count)
    if grey <= 0:
        return 0
    return int(distance % grey)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant rule:
    - Steps = head-to-destination distance modulo the number of gray interior cells.
    - When gray_count == 0, the container does not move.
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

    # Detect containers as rectangles enclosing GREY/FILL with BORDER frame.
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
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
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
        cands = comp_by_color.get(hc, [])
        comp0 = None
        for comp in cands:
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

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
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        # Add split-from-same-component destinations when head and dest touch.
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
        grey_count = len(grey_coords)
        steps = _step_count(head_dest_distance, grey_count)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:  # (-1, 0)
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
    """Return interior coords in fill order from the head-facing side inward."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:    # right, direction (0, 1): key (-col, row)
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left, direction (0, -1): key (col, -row)
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom, direction (1, 0): key (-row, -col)
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:                 # top, direction (-1, 0): key (row, col)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _choose_mover_spec(mode: str) -> dict | None:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if mode == "mod_nonzero":
        if inner_area < 2:
            return None
        grey = random.randint(2, min(inner_area, 6))
        remainder = random.randint(1, grey - 1)
        max_q = max(1, min(3, (12 - remainder) // grey))
        quotient = random.randint(1, max_q)
        distance = quotient * grey + remainder
    elif mode == "mod_zero":
        if inner_area < 1:
            return None
        grey_hi = min(inner_area, 6)
        if grey_hi < 1:
            return None
        grey = random.randint(1 if grey_hi == 1 else 2, grey_hi)
        max_q = max(1, min(3, 12 // grey))
        quotient = random.randint(1, max_q)
        distance = quotient * grey
    elif mode == "distance_limited":
        distance = random.randint(1, 8)
        if inner_area <= distance:
            return None
        grey = random.randint(distance + 1, inner_area)
    else:
        raise ValueError(f"unknown mover mode: {mode}")

    steps = _step_count(distance, grey)
    initial_orange = inner_area - grey
    head_len = outer_w if head_side in (0, 1) else outer_h
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": distance - 1,
        "distance": distance,
        "steps": steps,
        "initial_orange": initial_orange,
        "grey": grey,
        "head_len": head_len,
        "is_decoy": False,
        "mode": mode,
    }


def _choose_decoy_spec() -> dict:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    head_len = outer_w if head_side in (0, 1) else outer_h
    gap = random.randint(0, 6)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "distance": gap + 1,
        "steps": 0,
        "initial_orange": inner_area,
        "grey": 0,
        "head_len": head_len,
        "is_decoy": True,
        "mode": "decoy_full_orange",
    }


def _sample_specs() -> list[dict] | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    mover_modes = ["mod_nonzero"]
    while len(mover_modes) < n_movers:
        mover_modes.append(random.choices(
            ["distance_limited", "mod_nonzero", "mod_zero"],
            weights=[4, 3, 1],
            k=1,
        )[0])
    random.shuffle(mover_modes[1:])

    specs: list[dict] = []
    has_mod_difference = False
    for mode in mover_modes:
        spec = _choose_mover_spec(mode)
        if spec is None:
            return None
        specs.append(spec)
        if spec["grey"] > 0 and spec["steps"] != min(spec["distance"], spec["grey"]):
            has_mod_difference = True

    for _ in range(n_decoys):
        specs.append(_choose_decoy_spec())

    random.shuffle(specs)
    if not has_mod_difference:
        return None
    if not any((not s["is_decoy"]) and s["steps"] > 0 for s in specs):
        return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

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
    placements: list[tuple[int, int]] = []
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
                occupied[r0:r0 + th, c0:c0 + tw] = True
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

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc:cc + ow] = BORDER
        grid[cr + oh - 1, cc:cc + ow] = BORDER
        grid[cr:cr + oh, cc] = BORDER
        grid[cr:cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1:cr + 1 + inner_h, cc + 1:cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc:cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc:cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc:cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc:cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr:cr + oh, cc - 1] = HEAD_COLOR
            grid[cr:cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr:cr + oh, cc + ow] = HEAD_COLOR
            grid[cr:cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

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

        expected[cr:cr + oh, cc:cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc:cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc:cc + ow] = BG
        elif hs == 2:
            expected[cr:cr + oh, cc - 1] = BG
        else:
            expected[cr:cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr:nr + oh, nc:nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc:nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc:nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr:nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr:nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid modular-step variant after 300 attempts")
