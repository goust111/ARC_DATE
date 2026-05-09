# Variant of 271d71e2
# Dimension: D16 — 背景色设定 (meta)
# Change: 背景为magenta(6)，固定单色背景... → 背景为棋盘格图案

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

"""
Design analysis for the D16 variant

a) Concrete behavior and trace
   - Keep the original objects unchanged: black rectangular containers, grey/orange
     interiors, maroon head lines and matching maroon destination lines.
   - The only visual change is the background: every non-object cell belongs to a
     2-color checkerboard using magenta(6) and teal(8).
   - Input and output still show the same container motion. The new requirement is
     that cells vacated by a moving container/head must revert to the local
     checkerboard color at those coordinates, not to a single flood-filled color.
   - Small trace: suppose a right-moving container vacates a 1-cell-wide vertical
     band. In the original task that band becomes all 6. In this variant the band
     alternates 6/8 by row parity, so the restored strip is visibly patterned.

b) Complexity calibration
   - The checkerboard uses only 2 fixed colors and a single deterministic rule
     based on coordinate parity. This is visually obvious in 2-4 examples and does
     not add ambiguous object classes.
   - Object colors remain unchanged, so head/destination matching is still unique.

c) Hidden assumptions in the original solver that no longer hold
   - `bg = dominant non-border color` assumed one background color. False now.
   - Excluding only one `bg` value when collecting head colors would wrongly treat
     the second checkerboard color as a candidate head/destination color.
   - Erasing a source rectangle with `out[...] = bg` would destroy the pattern and
     produce an incorrect solid patch.
   - The variant therefore needs position-aware background reconstruction.

d) Placement legality
   - Generator keeps the original assembly layout, so every container moves only
     within its own reserved assembly box.
   - Input groups still keep a 1-cell object gap from each other. Background cells
     may touch freely because they are not objects.
   - The checkerboard spans the whole board before objects are stamped on top, so
     both solver detection and output reconstruction are well defined.

e) Visual observability
   - Human solvers can see that the background is globally alternating 6/8.
   - When an object moves, the vacated trail restores that same alternation. This
     is the only new behavior and is directly observable without numeric metadata.

f) Visual cleanliness
   - The checkerboard lives only on background cells. Object silhouettes and colors
     are unchanged, so containers and markers remain easy to segment.

g) Color-change inferability
   - No new object recoloring rule is introduced. The only extra colors are the two
     background colors, arranged by a simple spatial pattern rather than a hidden
     mapping.

Post-change solver assumptions and how the generator guarantees them
   - Background colors are exactly two non-object colors, here fixed to 6 and 8.
   - Those colors follow a global checkerboard parity rule; the generator fills the
     whole board that way before drawing any objects.
   - At least one moving container exists, and moved/vacated cells stay inside the
     grid because the original assembly sizing already enforces that.
"""

BORDER = 0
GREY = 5
BG_A = 6
ORANGE = 7
BG_B = 8
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _checker_color(r: int, c: int, even_color: int, odd_color: int) -> int:
    return int(even_color if ((int(r) + int(c)) & 1) == 0 else odd_color)


def _paint_checkerboard_rect(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    even_color: int,
    odd_color: int,
) -> None:
    """Paint inclusive rectangle with the inferred checkerboard pattern."""
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            grid[r, c] = _checker_color(r, c, even_color, odd_color)


def _checkerboard_grid(h: int, w: int, phase: int) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = BG_A if ((r + c + phase) & 1) == 0 else BG_B
    return grid


def _infer_checkerboard_colors(g: np.ndarray) -> tuple[int, int]:
    """
    Infer which color belongs to even parity and which belongs to odd parity.

    The task keeps the original object palette {0,5,7,9}, so the two remaining
    visible colors are the checkerboard background colors.
    """
    object_colors = {BORDER, GREY, ORANGE, HEAD_COLOR}
    cnt = Counter(map(int, g.flatten()))
    bg_candidates = [int(c) for c in cnt if int(c) not in object_colors]
    bg_candidates.sort(key=lambda c: (-cnt[c], c))
    if not bg_candidates:
        return BG_A, BG_A
    if len(bg_candidates) == 1:
        return int(bg_candidates[0]), int(bg_candidates[0])

    a, b = int(bg_candidates[0]), int(bg_candidates[1])
    match_ab = 0
    match_ba = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            val = int(g[r, c])
            if val != a and val != b:
                continue
            expected_ab = a if ((r + c) & 1) == 0 else b
            expected_ba = b if ((r + c) & 1) == 0 else a
            match_ab += int(val == expected_ab)
            match_ba += int(val == expected_ba)
    return (a, b) if match_ab >= match_ba else (b, a)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Return interior cell coords in fill order from the head-facing side inward."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging grey
    cells to orange. Unlike the base task, vacated cells restore a 2-color global
    checkerboard background instead of a solid background color.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg_even, bg_odd = _infer_checkerboard_colors(g)
    bg_colors = {int(bg_even), int(bg_odd)}

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
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude=bg_colors | {BORDER, GREY, ORANGE})]

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
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    val = int(g[rr, cc])
                    if val not in bg_colors | {BORDER, GREY, ORANGE}:
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
            mask_rem = np.zeros((h, w), dtype=bool)
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
                        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BORDER:
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
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:steps])

        _paint_checkerboard_rect(out, r0, c0, r1, c1, bg_even, bg_odd)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = _checker_color(int(r), int(c), bg_even, bg_odd)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < h and 0 <= ccc < w:
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
            grey = 0
            initial_orange = inner_area
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(0, 10)
            steps = min(grey, gap + 1)

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

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:
            total_h, total_w = 1 + gap + 1 + oh, ow
            cont_r, cont_c = gap + 2, 0
        elif hs == 1:
            total_h, total_w = oh + 1 + gap + 1, ow
            cont_r, cont_c = 0, 0
        elif hs == 2:
            total_h, total_w = oh, 1 + gap + 1 + ow
            cont_r, cont_c = 0, gap + 2
        else:
            total_h, total_w = oh, ow + 1 + gap + 1
            cont_r, cont_c = 0, 0

        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

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

    phase = random.randint(0, 1)
    grid = _checkerboard_grid(grid_h, grid_w, phase)

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
    bg_even = BG_A if (phase & 1) == 0 else BG_B
    bg_odd = BG_B if (phase & 1) == 0 else BG_A

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

        _paint_checkerboard_rect(expected, cr, cc, cr + oh - 1, cc + ow - 1, bg_even, bg_odd)
        if hs == 0:
            _paint_checkerboard_rect(expected, cr - 1, cc, cr - 1, cc + ow - 1, bg_even, bg_odd)
        elif hs == 1:
            _paint_checkerboard_rect(expected, cr + oh, cc, cr + oh, cc + ow - 1, bg_even, bg_odd)
        elif hs == 2:
            _paint_checkerboard_rect(expected, cr, cc - 1, cr + oh - 1, cc - 1, bg_even, bg_odd)
        else:
            _paint_checkerboard_rect(expected, cr, cc + ow, cr + oh - 1, cc + ow, bg_even, bg_odd)

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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
