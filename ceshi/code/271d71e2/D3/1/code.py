# Variant of 271d71e2
# Dimension: D3 — 轨迹着色模式 (trajectory)
# Change: recolor — 灰色像素从头侧向内逐个变为橙色(7)，连续着色... → 颜色渐变循环
#
# Design analysis
# ===============
# a) Concrete behavior
# - The object layout stays the same as the base task: magenta background, black rectangular
#   containers, gray interior cells, one 1-cell-thick maroon head line on a container side,
#   and an aligned maroon destination line farther along the motion axis.
# - The changed part is only the "charged trail" inside the container. In the base task,
#   every charged cell is orange (7). In this variant, the charged prefix still grows from
#   the head-facing side inward, but its colors follow a fixed visible cycle:
#   red (2) -> yellow (4) -> orange (7) -> red (2) -> ...
# - Example trace on a 1x5 interior, head on the left:
#   base input interior  = [7, 7, 5, 5, 5]
#   base output interior = [7, 7, 7, 7, 5]  after 2 steps
#   variant input        = [2, 4, 5, 5, 5]
#   variant output       = [2, 4, 7, 2, 5]  after 2 steps
#   The movement amount is unchanged; only the colors assigned to the charged prefix differ.
#
# b) Complexity calibration
# - A 3-color cycle is enough to be visibly different from the base task and still easy to
#   infer from 2-4 examples. Using more colors would add noise without adding structure.
# - The cycle order is fixed across the whole variant family. Humans infer it from train
#   examples; the programmatic solver hard-codes that family rule.
# - Generator-side observability checks ensure most samples contain at least one mover whose
#   final charged prefix is long enough to display all three colors, preventing regression to
#   "looks almost like the original orange-only task".
#
# c) Original solver assumptions that break under this variant
# - Container detection originally assumed interior colors were only {5, 7}. That fails once
#   already-charged cells can be 2 or 4, so detection must accept {5, 2, 4, 7}.
# - Head/destination color detection originally excluded only {bg, border, gray, orange}.
#   With multi-color interiors, it must exclude the full interior palette, otherwise red/yellow
#   charged cells would be misread as head colors.
# - The original recolor step directly overwrote selected gray cells with 7. That relies on the
#   charged color being constant. In this variant, each cell's final color depends on its prefix
#   index, so the moved patch must rebuild the interior from the ordered prefix length instead of
#   performing a single unconditional assignment.
#
# d) Placement legality
# - The generator keeps the same assembly geometry as the base task: destination + gap + head +
#   container, with a 1-cell background moat between assemblies.
# - Because motion stays within each assembly bounding box and the head only lands onto the
#   existing destination line, non-overlap and in-bounds guarantees remain valid after movement.
#
# e) Visual observability
# - The new rule is directly visible because charged cells are contiguous and ordered from the
#   head side, so the repeating 2/4/7 pattern appears as a clearly aligned stripe.
# - The generator rejects weak samples where every mover would show at most one or two charged
#   colors, because those would be too easy to confuse with arbitrary recoloring.
#
# f) Visual cleanliness
# - No extra markers are added. The same clean container/head/destination layout is preserved,
#   and only the interior palette changes.
#
# g) Color-change inferability
# - Only three colors participate in the cycle, and they appear in a positional prefix whose
#   ordering is already meaningful in the base task. That gives the solver a structural cue:
#   "next charged cell = next color in the repeating cycle", rather than an unstructured mapping.
#
# Reference task note
# -------------------
# I inspected the source-task solver for 195c6913 and only reused its relevant mechanism:
# repeated painting by cycling through a fixed ordered list of colors. I do not import any of
# its unrelated movement/collision logic here.

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


BG = 6
BORDER = 0
GREY = 5
RED = 2
YELLOW = 4
ORANGE = 7
HEAD_COLOR = 9
CHARGE_CYCLE = (RED, YELLOW, ORANGE)
INTERIOR_COLORS = {GREY, *CHARGE_CYCLE}

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _fill_order_from_direction(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Return interior-local coords ordered from the head-facing side inward."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == (0, 1):  # right
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == (0, -1):  # left
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == (1, 0):  # down
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # up
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _fill_order_from_direction(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _paint_cycle_prefix(inner_h: int, inner_w: int, head_side: int, charged_count: int) -> np.ndarray:
    """Build an interior whose first charged_count cells follow the fixed color cycle."""
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    order = _interior_fill_order(inner_h, inner_w, head_side)
    for idx, (r, c) in enumerate(order[:charged_count]):
        interior[r, c] = CHARGE_CYCLE[idx % len(CHARGE_CYCLE)]
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant rule:
    - Steps = min(head-to-destination distance, number of gray interior cells).
    - The charged interior prefix still grows from the head-facing side inward.
    - Instead of recoloring every newly charged cell to orange, the entire charged prefix
      follows the fixed cycle 2 -> 4 -> 7 -> 2 -> 4 -> 7 -> ...
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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

    # Detect containers as BORDER rectangles enclosing only interior palette cells.
    containers = []
    mask = (g == GREY) | np.isin(g, CHARGE_CYCLE)
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != int(BORDER) for r, c in rect_border_coords(r0, c0, r1, c1)):
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
        for r, c in rect_border_coords(r0, c0, r1, c1):
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
        grey_count = int(np.sum(inner == GREY))
        initial_charged = int(inner.size - grey_count)
        steps = min(int(steps), grey_count)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        inner_h, inner_w = inner.shape
        order = _fill_order_from_direction(inner_h, inner_w, direction)
        total_charged = int(initial_charged + steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BORDER, dtype=int)
        patch_inner = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx, (rr, cc) in enumerate(order[:total_charged]):
            patch_inner[rr, cc] = CHARGE_CYCLE[idx % len(CHARGE_CYCLE)]
        patch[1:-1, 1:-1] = patch_inner
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
    n_movers = n_containers - n_decoys
    if n_movers < 1:
        n_movers = 1
        n_decoys = n_containers - 1

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers

        # Keep the original size envelope but bias against tiny interiors so the cycle is visible.
        for _ in range(20):
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            if inner_h * inner_w >= (1 if is_decoy else 3):
                break
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            initial_charged = inner_area
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_charged = inner_area - grey
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(grey, h2d)

        specs.append(
            {
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
        )

    # Reject low-signal cases that would hide the new color-cycle behavior.
    movers = [s for s in specs if not s["is_decoy"]]
    if not any(s["steps"] >= 1 and s["initial_charged"] + s["steps"] >= 3 for s in movers):
        return None
    if not any(s["steps"] >= 2 for s in movers):
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

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        gap = a["gap"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_charged = a["initial_charged"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = _paint_cycle_prefix(inner_h, inner_w, hs, initial_charged)
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
        initial_charged = a["initial_charged"]
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
        patch[1:-1, 1:-1] = _paint_cycle_prefix(inner_h, inner_w, hs, initial_charged + steps)
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
