# Variant of 271d71e2
# Dimension: D8 — 变换类型 (transform)
# Change: recolor — 灰色(5)变为橙色(7)，容器整体平移，目标线保持不动
#   -> 橙色变回灰色（逆向变换）
#
# Design analysis for this variant
# - Reference-task review:
#   - source_task is None, so there is no external solver to inspect. The baseline is
#     the original 271d71e2 solver/generator plus its ARC examples/spec.
# - a) Concrete behavior and trace:
#   - Pixel layout stays the same as the original task: magenta background, black
#     rectangular container, maroon head line flush with one side, matching maroon
#     destination line further along the same axis.
#   - The only intended semantic change is the color direction inside the container.
#     Original: the container moves toward the destination and "charges" by turning
#     grey cells on the head-facing side into orange. Variant: the container still
#     moves toward the destination, but it "discharges" by turning already-orange
#     cells on the head-facing side back into grey.
#   - Small trace, head on the left, inner row order from left to right:
#     - Original input interior: [7,7,5,5], distance=2, grey_count=2.
#       Original output after 2 steps: [7,7,7,7].
#     - Variant input interior: [7,7,5,5], distance=2, orange_count=2.
#       Variant output after 2 steps: [5,5,5,5].
#     - Difference: movement amount now depends on orange supply, and the modified
#       cells are the orange prefix nearest the head instead of the grey prefix.
# - b) Complexity calibration:
#   - I keep the same object vocabulary, rectangle sizes, head/destination geometry,
#     and 2-4 assemblies. This preserves the original matching difficulty and avoids
#     introducing new ambiguous cues.
#   - To make D8 visible, generator inputs intentionally contain nonzero orange in
#     movers, often with both orange and grey present. This prevents degeneration into
#     all-grey or unchanged interiors where the inverse transform would be hard to see.
#   - I still include 0-1 non-moving decoys, but their interior is fully grey, which
#     is the natural "already discharged" analogue of the original fully-orange decoy.
# - c) Implicit-assumption audit of main():
#   - Container detection based on GREY|ORANGE components remains valid because the
#     interior still uses only colors 5 and 7 inside a black frame.
#   - Head/destination detection is unchanged because D8 does not alter their color,
#     shape, adjacency, or alignment.
#   - The original solver assumed step capacity equals grey count and then blindly
#     painted chosen cells orange. In this variant that assumption is false:
#     capacity must equal orange count, and blindly assigning orange would destroy the
#     intended inverse behavior. The only required code change in the transformation
#     stage is therefore:
#       - collect orange cells instead of grey cells,
#       - cap steps by orange count,
#       - recolor chosen cells from ORANGE to GREY.
#   - The ordering logic itself is still usable because it is geometric, not color-
#     value-based. We reuse the same head-side ordering and apply it to orange cells.
# - d) Placement legality:
#   - Generator keeps the original assembly envelope: destination + gap + head +
#     container. Because movement is always toward destination and capped by the same
#     head/destination distance, the moved container and moved head stay inside that
#     envelope just as before.
#   - Assemblies are placed with a one-cell background moat, so separate objects do not
#     overlap and the solver's per-container head matching remains unambiguous.
# - e) Visual observability:
#   - In the original task, orange grows outward from the head side after movement.
#     Here, orange shrinks from that same side. Across 2-4 examples, a human can see
#     the same translation but the opposite color-flow direction.
#   - The variant-specific cue is visible even when step counts differ: the side next
#     to the head becomes greyer after movement instead of more orange.
# - f) Visual cleanliness:
#   - No extra markers are added. The grid remains sparse because assemblies are spaced
#     apart with background padding.
#   - Destination lines remain stationary, so each moving container still has a clear
#     before/after trajectory cue.
# - g) Color-change inferability:
#   - Only two interior colors participate, grey and orange, with a single reversible
#     mapping direction. The head-facing order gives the structural clue for which
#     orange cells revert first.
# - Post-change solver assumptions to guarantee in the generator:
#   - Movers must contain at least one orange cell, otherwise steps collapse to 0.
#   - For intended visual clarity, movers are generated with orange occupying the
#     head-side prefix of the canonical order; then the solver's "first orange cells"
#     rule produces a clean discharge front.

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


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_coords(coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(coords, key=lambda p: (-p[0], -p[1]))
    return sorted(coords, key=lambda p: (p[0], p[1]))


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Return local inner-cell order from the head-facing side inward."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return _ordered_coords(cells, SIDE_TO_DIR[head_side])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, discharging
    orange cells back to grey from the head-facing side.

    Steps = min(head-to-destination distance, number of orange interior cells).
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    # --- detect containers as rectangles enclosing GREY/ORANGE with BORDER frame ---
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

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal manhattan distance to this container border.
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
        orange_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == ORANGE
        ]
        steps = min(int(steps), int(len(orange_coords)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_discharge = set(_ordered_coords(orange_coords, direction)[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_discharge:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == ORANGE:
                patch[pr, pc] = int(GREY)
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
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            initial_orange = 0
            gap = random.randint(0, 6)
            steps = 0
        else:
            initial_orange = random.randint(1, inner_area)
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(initial_orange, h2d)

        specs.append({
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": steps,
            "initial_orange": initial_orange,
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

        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

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
        for idx_cell in range(steps, initial_orange):
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
    raise RuntimeError("Failed to generate valid after 300 attempts")
