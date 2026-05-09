# Variant of 271d71e2
# Dimension: D2 — 移动距离编码方式 (dst)
# Change: distance_encoding — 步数 = min(头到目标距离, 灰色像素数)，两个约束取较小值
#   → 纯距离驱动（忽略灰色数量）
#
# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Visual objects stay the same as the base task:
#      magenta background, black rectangular containers, grey/orange interiors,
#      and a maroon head line plus a same-shaped maroon destination line.
#    - The changed behavior is only the distance encoding:
#      base task: move by min(distance_to_dest, grey_count), paint one grey per step.
#      D2 variant: always move by the full head-to-destination distance.
#      Every grey cell turns orange; extra travel after greys are exhausted does not
#      produce any additional coloring.
#    - Small trace:
#      input: a 3x2 interior has 1 grey near the head, destination is 4 cells away.
#      base: steps=min(4,1)=1, container stops early after moving 1 cell.
#      D2:   steps=4, container reaches the destination; the single grey becomes orange.
#      The visible difference is the final position, not just the coloring.
#    - Another trace:
#      input: distance=2, grey_count=5.
#      base: container reaches the destination but only 2 grey cells become orange.
#      D2:   container still reaches the destination, but all 5 grey cells become orange.
#      The visible difference is the final coloring, not the position.
#
# b) Complexity calibration
#    - Keep 2-4 containers, same rectangular geometry, same single maroon color.
#    - To avoid degeneration back to the base task, generation forces repeated support
#      for the new rule:
#      at least one container has distance > grey_count (position differs from base),
#      and when possible at least one container has distance < grey_count
#      (coloring differs from base).
#    - This preserves the original recognition problem (matching head to destination)
#      while making the D2 change visible in multiple ways.
#
# c) Implicit-assumption review of main()
#    - Container detection depends only on black frames around grey/orange interiors.
#      This still holds.
#    - Head detection assumes the closest maroon pixels to a container are the head,
#      and the destination has the same normalized shape on the movement axis.
#      This still holds.
#    - The base solver's only D2-specific assumption was:
#         steps = min(distance, len(grey_coords))
#      plus filling only the first `steps` grey cells.
#      In this variant both assumptions are invalid:
#         steps must remain the full distance,
#         and all grey cells must be converted regardless of distance.
#    - Erasing the original container/head and pasting the moved patch is still safe
#      because the generator keeps the whole motion corridor inside one assembly bbox.
#
# d) Placement legality
#    - The input assembly bbox from the original task already spans the full corridor:
#      destination + gap + head + container.
#      Because the moved container ends adjacent to the destination, the output footprint
#      stays inside this same bbox. So placing disjoint assembly bboxes is sufficient for
#      both input and output.
#    - The generator still validates every sampled input by comparing the hand-built
#      expected output against main(input); ambiguous placements are rejected.
#
# e) Visual observability
#    - Humans can infer the new rule from only a few examples because some containers
#      travel farther than their grey count would permit in the base task, while others
#      finish with fully orange interiors even when distance is smaller than grey_count.
#    - No hidden numeric ordering is introduced; all cues are geometric and color-visible.
#
# f) Visual cleanliness
#    - Containers remain separated by at least one background cell around each assembly.
#      Head/destination lines remain axis-aligned and flush with one container edge.
#      No extra overlays or multi-color encodings are introduced.
#
# g) Color-change inference
#    - The palette is unchanged: only grey (5) can turn into orange (7).
#      In this variant the rule simplifies to: if a container moves, every grey interior
#      cell becomes orange by the end.

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

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    D2 variant rule:
    - Each container always translates the full head-to-destination distance.
    - Grey count no longer caps movement.
    - All grey interior cells become orange.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    BORDER = 0
    GREY = 5
    FILL = 7
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
    for pts in components_mask((g == GREY) | (g == FILL), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, FILL]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, FILL})]

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
                    if val not in (int(bg), int(BORDER), int(GREY), int(FILL)):
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

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[patch == GREY] = int(FILL)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


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


def _draw_head(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 1, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 1] = color
    else:
        grid[cr : cr + oh, cc + ow] = color


def _draw_dest(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, gap: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 2 - gap, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh + 1 + gap, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 2 - gap] = color
    else:
        grid[cr : cr + oh, cc + ow + 1 + gap] = color


def _erase_head(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int) -> None:
    _draw_head(grid, cr, cc, oh, ow, head_side, BG)


def _mode_sequence(n_containers: int) -> list[str]:
    modes = ["overshoot"]
    if n_containers >= 2:
        modes.append("underfill")
    while len(modes) < n_containers:
        modes.append(random.choices(["overshoot", "underfill", "balanced"], weights=[4, 4, 1])[0])
    random.shuffle(modes)
    return modes


def _sample_spec(mode: str) -> dict | None:
    for _ in range(100):
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if mode == "overshoot":
            grey = random.randint(1, min(area, 4))
            min_gap = grey
            if min_gap > 10:
                continue
            gap = random.randint(min_gap, 10)
        elif mode == "underfill":
            if area < 2:
                continue
            max_gap = min(10, area - 2)
            if max_gap < 0:
                continue
            gap = random.randint(0, max_gap)
            distance = gap + 1
            grey = random.randint(distance + 1, area)
        else:
            max_dist = min(11, area)
            if max_dist < 1:
                continue
            distance = random.randint(1, max_dist)
            gap = distance - 1
            grey = distance

        initial_orange = area - grey
        return {
            "mode": mode,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": gap + 1,
            "initial_orange": initial_orange,
            "grey": grey,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)

    specs = []
    for mode in _mode_sequence(n_containers):
        spec = _sample_spec(mode)
        if spec is None:
            return None
        specs.append(spec)

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
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]

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

        _draw_head(grid, cr, cc, oh, ow, hs, HEAD_COLOR)
        _draw_dest(grid, cr, cc, oh, ow, hs, a["gap"], HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        nr = cr + dr * a["steps"]
        nc = cc + dc * a["steps"]

        expected[cr : cr + oh, cc : cc + ow] = BG
        _erase_head(expected, cr, cc, oh, ow, hs)

        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = ORANGE
        expected[nr : nr + oh, nc : nc + ow] = patch
        _draw_head(expected, nr, nc, oh, ow, hs, HEAD_COLOR)

    if not np.array_equal(output, expected):
        return None

    overshoot_count = sum(1 for a in assemblies if a["steps"] > a["grey"])
    underfill_count = sum(1 for a in assemblies if a["steps"] < a["grey"])
    if overshoot_count < 1:
        return None
    if len(assemblies) >= 2 and underfill_count < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
