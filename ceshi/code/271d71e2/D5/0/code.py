# Variant of 271d71e2
# Dimension: D5 — 灰色像素着色顺序 (multi_step)
# Change: ordered_processing — 灰色像素按从头侧向内的行/列顺序逐个变为橙色... → 从中心向外扩散
#
# Design analysis for this variant
# a) Concrete behavior:
#    - Objects stay the same as the source task: magenta background, black bordered
#      rectangle container, maroon head line, same-colored maroon destination line.
#    - Movement is unchanged: container + head move toward the destination by
#      min(head-to-dest distance, grey_count) cells.
#    - Only the charging order changes. The source task converts grey cells from the
#      head-facing edge inward. This variant instead starts from the geometric center
#      of the interior and expands by 4-neighbor BFS.
#    - For even interior sizes, the "center" is the 1/2/4 cells with minimum
#      Manhattan distance to the geometric midpoint; these are the BFS seeds.
#      This keeps the rule local and visible, while remaining deterministic.
#    - Example on a 5x3 interior with 3 pre-existing orange center-prefix cells:
#      original logic with head on top would charge the next top-side grey first;
#      variant logic charges the next closest grey to the central seed set first.
#      Movement steps remain identical, only which cells become orange differs.
#
# b) Complexity calibration:
#    - The new rule should be visible, so mover interiors are biased toward odd,
#      2D sizes (3x3, 5x3, 5x5, ...), giving an unambiguous center and observable
#      concentric growth.
#    - The generator rejects samples where the new output would match the original
#      head-side order on every mover, preventing silent degeneration.
#
# c) Old assumptions that no longer hold:
#    - In the source solver, fill order depended on movement direction. That is no
#      longer valid; direction is still used for translation but not for coloring.
#    - The source solver sorted only currently-grey cells. That remains safe here
#      only if initial orange cells form a prefix of the new global order. The
#      generator explicitly guarantees this prefix property.
#    - Container detection, head detection, destination matching, and movement cap
#      logic depend only on geometry/colors, not on the old interior ordering, so
#      they can remain unchanged.
#
# d) Placement legality:
#    - Assemblies are placed as non-overlapping rectangles with a 1-cell magenta moat.
#      Each assembly already reserves the destination, gap, head, and container span,
#      so the moved container stays inside its own reserved footprint.
#    - This preserves no-overlap, in-bounds movement, and a clear path to the target.
#
# e) Visual observability:
#    - At least one mover per generated sample must produce a different output from
#      the original solver on the same input.
#    - At least one mover must also have steps >= 2, so the center-out expansion is
#      visible as a small region rather than a single isolated pixel change.
#
# f) Visual cleanliness:
#    - No extra markers are introduced.
#    - Containers remain separated by background, so the center-out orange blobs stay
#      legible and are not confused with neighboring objects.
#
# g) Color-change inferability:
#    - Colors are unchanged from the source task: grey means remaining charge,
#      orange means charged. The only new inference is spatial order, and that order
#      is anchored by the visible center of each container interior.

from __future__ import annotations

from collections import Counter, defaultdict, deque
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


def _center_seed_cells(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    """Cells closest to the geometric midpoint, sorted for deterministic BFS seeding."""
    target_r2 = inner_h - 1
    target_c2 = inner_w - 1
    best = None
    seeds: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            dist2 = abs(2 * r - target_r2) + abs(2 * c - target_c2)
            if best is None or dist2 < best:
                best = dist2
                seeds = [(r, c)]
            elif dist2 == best:
                seeds.append((r, c))
    seeds.sort()
    return seeds


def _center_out_order(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    """Interior fill order: start at the center seed set and expand by 4-neighbor BFS."""
    seeds = _center_seed_cells(inner_h, inner_w)
    q = deque(seeds)
    seen = set(seeds)
    order: list[tuple[int, int]] = []

    # Fixed neighbor order is only a tie-break; the visible rule is distance from center.
    nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while q:
        r, c = q.popleft()
        order.append((r, c))
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return order


def _head_side_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Original source-task order, kept for generator-side visibility checks."""
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


def _old_rule_final_orange_set(
    inner_h: int, inner_w: int, head_side: int, initial_orange: int, steps: int
) -> set[tuple[int, int]]:
    """Final orange cells if the original head-side ordering were applied to this variant input."""
    center_order = _center_out_order(inner_h, inner_w)
    orange_now = set(center_order[:initial_orange])
    old_fill: list[tuple[int, int]] = []
    for cell in _head_side_order(inner_h, inner_w, head_side):
        if cell in orange_now:
            continue
        old_fill.append(cell)
        if len(old_fill) == steps:
            break
    return orange_now | set(old_fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant rule:
    - Translation rule is unchanged from 271d71e2.
    - Grey cells become orange from the interior center outward by 4-neighbor BFS,
      rather than from the head-facing edge inward.
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

    # Detect containers as rectangles enclosing GREY/ORANGE with BORDER frame.
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

    # Precompute components for each head color.
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

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal Manhattan distance to this container border.
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
            if d["shape"] != head_shape or d["pts"] == comp_pts:
                continue
            dests.append(d)

        # Add split-from-same-component destinations (when dest touches head by color).
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
        center_order = _center_out_order(inner.shape[0], inner.shape[1])
        remaining_greys = [(rr, cc) for rr, cc in center_order if int(inner[rr, cc]) == GREY]
        steps = min(int(steps), int(len(remaining_greys)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_fill_local = set(remaining_greys[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill_local:
            if int(patch[rr + 1, cc + 1]) == GREY:
                patch[rr + 1, cc + 1] = int(ORANGE)
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
    visible_new_behavior = False
    prominent_move = False

    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        if is_decoy:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            inner_area = inner_h * inner_w
            initial_orange = inner_area
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
        else:
            inner_h = random.choice([3, 5, 7])
            inner_w = random.choice([3, 5])
            inner_area = inner_h * inner_w
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(grey, h2d)

            new_final = set(_center_out_order(inner_h, inner_w)[: initial_orange + steps])
            old_final = _old_rule_final_orange_set(inner_h, inner_w, head_side, initial_orange, steps)
            if steps >= 2:
                prominent_move = True
            if steps > 0 and new_final != old_final:
                visible_new_behavior = True

        outer_h = inner_h + 2
        outer_w = inner_w + 2
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
            }
        )

    if not visible_new_behavior or not prominent_move:
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
        gap = a["gap"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        center_order = _center_out_order(inner_h, inner_w)

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = center_order[idx_cell]
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
        center_order = _center_out_order(inner_h, inner_w)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = center_order[idx_cell]
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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
