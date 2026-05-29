from __future__ import annotations

# Variant of 271d71e2
# Dimension: D10 — 容器内部分区结构 (meta)
# Change: 内部分为连续的橙色区域（头侧）和灰色区域（远侧），二者之间是平直分界线... → 多色分区（三种或更多颜色）
#
# Design analysis
# - Concrete behavior:
#   Input containers still have the same black frame, maroon head line, maroon destination line,
#   and move on the head->destination axis. The change is entirely inside the frame:
#   each moving container contains full-row/full-column stripes in fill order:
#   orange stripe(s) nearest the head, then one grey stripe, then one or more blue stripes.
#   Orange has charge level 0, grey has level 1, blue has level 2.
#   Steps = min(head_to_dest_distance, sum(level(cell)) over the interior).
#   During each unit step, scan cells from the head-facing side inward and decrement the first
#   non-orange cell by one level, so grey->orange and blue->grey->orange.
#   Example on a 1x4 interior ordered from left to right:
#   [7,5,1,1] with distance 4 gives capacity 1+2+2=5, so the moved output uses 4 steps and the
#   interior becomes [7,7,7,5]. The original task would have stopped after 1 step because it only
#   counted grey cells.
# - Complexity calibration:
#   Only one extra charged color is introduced (blue) so the mapping is inferable from 2-4 ARC
#   examples. Every moving container includes orange, grey, and blue, and all movers are generated
#   so blue really affects the step count.
# - Old assumptions that had to change in main():
#   1. Container detection can no longer search only (grey|orange) connected components;
#      it must treat (orange|grey|blue) as legal interior.
#   2. Head-color detection must exclude blue as an interior color.
#   3. Movement cap is no longer len(grey_cells); it is the weighted charge sum.
#   4. Patch updates can no longer blindly recolor grey->orange; blue may first become grey.
# - Placement legality:
#   The generator keeps the original assembly geometry (destination, optional gap, head, container)
#   and a 1-cell magenta moat between assemblies. Since each mover satisfies steps <= head_to_dest
#   distance, the moved container/head always stays inside its own assembly bounds.
# - Visual observability / cleanliness:
#   Input partitions are full stripes, not random speckles, so the extra blue region is obvious.
#   At least one mover per sample uses an odd amount of blue charge, producing a visible blue->grey
#   partial conversion in the output; this makes the 2-step role of blue observable.
# - New implicit assumptions after the change:
#   main() assumes interior colors are exactly orange/grey/blue and that containers remain simple
#   bordered rectangles. generate_input() guarantees this, guarantees one head/destination partner
#   per container color, and guarantees every mover has grey_count < steps so the variant does not
#   collapse back to the original grey-only behavior.

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
BLUE = 1
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

LEVEL_TO_COLOR = {0: ORANGE, 1: GREY, 2: BLUE}
COLOR_TO_LEVEL = {v: k for k, v in LEVEL_TO_COLOR.items()}
INTERIOR_COLORS = tuple(COLOR_TO_LEVEL.keys())
MAX_GAP = 10
MAX_H2D = MAX_GAP + 1


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_cells(height: int, width: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    if direction == (0, 1):
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == (0, -1):
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == (1, 0):
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _interior_capacity(interior: np.ndarray) -> int:
    total = 0
    for val in interior.flatten():
        total += int(COLOR_TO_LEVEL[int(val)])
    return total


def _advance_interior(interior: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    out = np.array(interior, copy=True)
    order = _ordered_cells(out.shape[0], out.shape[1], direction)
    for _ in range(int(steps)):
        for rr, cc in order:
            level = int(COLOR_TO_LEVEL[int(out[rr, cc])])
            if level > 0:
                out[rr, cc] = int(LEVEL_TO_COLOR[level - 1])
                break
        else:
            break
    return out


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _ordered_cells(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _stripe_size(inner_h: int, inner_w: int, head_side: int) -> int:
    return inner_w if head_side in (0, 1) else inner_h


def _axis_len(inner_h: int, inner_w: int, head_side: int) -> int:
    return inner_h if head_side in (0, 1) else inner_w


def _build_striped_interior(
    inner_h: int,
    inner_w: int,
    head_side: int,
    orange_bands: int,
    grey_bands: int,
    blue_bands: int,
) -> np.ndarray:
    order = _interior_fill_order(inner_h, inner_w, head_side)
    stripe = _stripe_size(inner_h, inner_w, head_side)
    orange_count = orange_bands * stripe
    grey_count = grey_bands * stripe
    blue_count = blue_bands * stripe
    assert orange_count + grey_count + blue_count == inner_h * inner_w

    interior = np.full((inner_h, inner_w), BLUE, dtype=int)
    for idx in range(orange_count):
        rr, cc = order[idx]
        interior[rr, cc] = ORANGE
    for idx in range(orange_count, orange_count + grey_count):
        rr, cc = order[idx]
        interior[rr, cc] = GREY
    return interior


def _choose_motion(grey_count: int, blue_count: int) -> tuple[int, int, int, str]:
    capacity = grey_count + 2 * blue_count
    partial_hi = min(capacity - 1, MAX_H2D)
    if partial_hi >= grey_count + 1 and random.random() < 0.7:
        h2d = random.randint(grey_count + 1, partial_hi)
        return h2d - 1, h2d, capacity, "distance"

    if capacity < MAX_H2D:
        h2d = random.randint(capacity + 1, MAX_H2D)
        return h2d - 1, capacity, capacity, "capacity"

    h2d = random.randint(grey_count + 1, min(capacity, MAX_H2D))
    return h2d - 1, h2d, capacity, "distance"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    interior_mask = np.isin(g, INTERIOR_COLORS)
    for pts in components_mask(interior_mask, connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, INTERIOR_COLORS).all()):
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
                    if val not in (int(bg), int(BORDER), *INTERIOR_COLORS):
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
        h2d = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        steps = min(int(h2d), int(_interior_capacity(inner)))
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[1:-1, 1:-1] = _advance_interior(inner, direction, steps)
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
    partial_blue_visible = False
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        for _ in range(100):
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            axis_len = _axis_len(inner_h, inner_w, head_side)
            if is_decoy or axis_len >= 3:
                break
        else:
            return None

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        stripe = _stripe_size(inner_h, inner_w, head_side)

        if is_decoy:
            orange_bands = axis_len
            grey_bands = 0
            blue_bands = 0
            gap = random.randint(0, 6)
            steps = 0
            capacity = 0
            mode = "decoy"
        else:
            orange_bands = random.randint(1, axis_len - 2)
            grey_bands = 1
            blue_bands = axis_len - orange_bands - grey_bands
            if blue_bands < 1:
                return None

            grey_count = grey_bands * stripe
            blue_count = blue_bands * stripe
            gap, steps, capacity, mode = _choose_motion(grey_count, blue_count)
            if steps <= grey_count:
                return None
            if steps < capacity and ((steps - grey_count) % 2 == 1):
                partial_blue_visible = True

        interior = _build_striped_interior(inner_h, inner_w, head_side, orange_bands, grey_bands, blue_bands)
        grey_count = int(np.count_nonzero(interior == GREY))
        blue_count = int(np.count_nonzero(interior == BLUE))

        specs.append({
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": steps,
            "capacity": capacity,
            "interior": interior,
            "grey_count": grey_count,
            "blue_count": blue_count,
            "is_decoy": is_decoy,
            "mode": mode,
        })

    if not partial_blue_visible:
        return None

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
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER
        grid[cr + 1 : cr + 1 + a["inner_h"], cc + 1 : cc + 1 + a["inner_w"]] = a["interior"]

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
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
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _advance_interior(a["interior"], SIDE_TO_DIR[hs], steps)
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
