# Variant of 271d71e2
# Dimension: D4 — 源对象结构 (src)
# Change: 黑色边框矩形容器，内部为灰色/橙色填充，附带同色头部线段指示方向
#   → 容器内允许第三种颜色蓝色(1)作为固定障碍；蓝色不计步、不变色，只占据内部空间

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, paste


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   src_change:
#     old: container interior used only two roles: GREY(5)=remaining charge, ORANGE(7)=spent charge
#     new: interior uses three roles:
#       - GREY(5): chargeable cells, each one permits one step and turns ORANGE when spent
#       - ORANGE(7): already-spent charge
#       - BLUE(1): obstacle, never counts toward steps and never changes color
#   concrete_visualization:
#     - Outer object is still a solid black rectangular border.
#     - A full-edge maroon line still marks the movement direction, with an aligned maroon destination line.
#     - Inside the rectangle, blue cells are mixed into the same rectangular cavity as grey/orange cells.
#     - Output keeps every blue cell in the same local position inside the moved rectangle.
#     - Only grey cells change to orange, and only in the head-facing scan order.
#   behavior_trace:
#     example:
#       - head on left, head-to-dest distance = 4
#       - interior scan order from left side is: [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]
#       - if local colors are [orange, blue, grey, grey, blue, grey], then chargeable cells are
#         [(0,0),(2,0),(0,1),(2,1)] and remaining grey count is 3
#       - original task would have treated every non-orange slot as potential future charge
#       - variant instead moves only min(4,3)=3 steps and converts the three grey cells, while both
#         blue cells stay blue
#     difference_from_original:
#       - container detection can no longer start from a grey/orange connected component
#       - step budget depends only on grey count, not on all non-orange interior area
#       - fill order must skip blue cells rather than recolor them
#   complexity_calibration:
#     - The third color is fixed to BLUE(1) rather than varying per object, so the new role is visible
#       without introducing a second matching problem.
#     - Most moving containers contain at least one blue cell, and generation enforces that at least one
#       mover per grid has a blue cell interleaved before later grey cells, so the obstacle behavior is
#       actually observable instead of decorative.
#     - At least one mover per generated grid is also forced into a "blue-sensitive" regime where
#       head-distance > grey_count and blue_count > 0, so if blue were incorrectly counted as charge
#       the movement distance would be larger.
#   implicit_assumption_audit:
#     - Original container detection:
#         `components_mask((g == GREY) | (g == FILL))`
#       depended on the interior being only grey/orange and often connected. This breaks immediately once
#       blue cells can split grey/orange islands. The new solver detects black rectangular border components.
#     - Original head-seed filtering excluded only bg/border/grey/orange. With blue inside the border, those
#       interior-adjacent border neighbors would be mistaken for head seeds. The new solver excludes BLUE too.
#     - Original `inner.isin([GREY, FILL])` assumed exactly two interior colors. The new solver validates
#       against `{GREY, ORANGE, BLUE}`.
#     - Original fill logic was safe because every future-charge cell was grey. In the variant, blindly
#       advancing through all interior coordinates would recolor blue obstacles. The new logic sorts only
#       grey coordinates, so blue cells are skipped and preserved.
#   placement_legality:
#     - Assemblies keep the original structure: destination line + gap + head line + container.
#     - Objects are placed with a one-cell background moat around each assembly.
#     - Movement stays inside the assembly footprint because the footprint already spans both input and
#       destination-side extents.
#     - Border detection assumes black pixels belong only to rectangle frames; the generator maintains that.
#   visual_observability:
#     - Blue uses a high-contrast fixed color and sits inside the same bounded container as grey/orange,
#       so its role is visually tied to the charging process.
#     - Because blue remains unchanged while adjacent grey turns orange, the solver must visibly distinguish
#       "unchanged obstacle" from "spent charge".
#   cleanliness:
#     - No extra external markers are added; the scene stays close to the original task.
#     - The third color is confined to the container interior, avoiding overlaps or extra matching clutter.


BG = 6
BORDER = 0
BLUE = 1
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ALLOWED_INTERIOR = {BLUE, GREY, ORANGE}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _dir_sort_key(direction: tuple[int, int]):
    if direction == (0, 1):
        return lambda p: (-p[1], p[0])
    if direction == (0, -1):
        return lambda p: (p[1], -p[0])
    if direction == (1, 0):
        return lambda p: (-p[0], -p[1])
    return lambda p: (p[0], p[1])


def _interior_scan_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    cells.sort(key=_dir_sort_key(SIDE_TO_DIR[head_side]))
    return cells


def _build_interior(
    inner_h: int,
    inner_w: int,
    head_side: int,
    blue_cells: set[tuple[int, int]],
    orange_count: int,
) -> np.ndarray:
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for r, c in blue_cells:
        interior[r, c] = BLUE

    chargeable = [p for p in _interior_scan_order(inner_h, inner_w, head_side) if p not in blue_cells]
    for r, c in chargeable[:orange_count]:
        interior[r, c] = ORANGE
    return interior


def _detect_containers(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    lab, ncomp = label(g == BORDER, structure=STRUCT4)
    containers = []
    for k in range(1, int(ncomp) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border_pts = list(_rect_border_coords(int(r0), int(c0), int(r1), int(c1)))
        if len(border_pts) != len(coords):
            continue
        if any(int(g[r, c]) != BORDER for r, c in border_pts):
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, list(ALLOWED_INTERIOR)).all()):
            continue

        # Border components are containers only if some non-border non-bg signal sits immediately outside.
        has_signal = False
        for r, c in border_pts:
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in {bg, BORDER, BLUE, GREY, ORANGE}:
                        has_signal = True
                        break
            if has_signal:
                break
        if not has_signal:
            continue

        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})
    return containers


def _build_head_components(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    head_colors = [c for c in colors_present(g, exclude={bg, BORDER, BLUE, GREY, ORANGE})]
    for col in head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})
    return comp_by_color


def _pick_head_for_container(
    g: np.ndarray,
    box_: tuple[int, int, int, int],
    bg: int,
    comp_by_color: dict[int, list[dict]],
) -> dict | None:
    H, W = g.shape
    r0, c0, r1, c1 = box_
    head_seeds = []
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                val = int(g[rr, cc])
                if val not in {bg, BORDER, BLUE, GREY, ORANGE}:
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


def _container_stats(g: np.ndarray) -> list[dict]:
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))
    containers = _detect_containers(g, bg)
    comp_by_color = _build_head_components(g, bg)
    stats = []

    for cont in containers:
        box_ = cont["box"]
        head_info = _pick_head_for_container(g, box_, bg, comp_by_color)
        if head_info is None or head_info["dir"] == (0, 0):
            continue

        r0, c0, r1, c1 = box_
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        blue_count = int(np.sum(inner == BLUE))
        grey_count = int(np.sum(inner == GREY))

        hc = int(head_info["color"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape or d["pts"] == comp_pts:
                continue
            dests.append(d)

        rem = comp_pts - head_info["head_pts"]
        if rem:
            mask_rem = np.zeros(g.shape, dtype=bool)
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
                        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(BORDER):
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

        best = None
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

        if best is None:
            continue

        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        blue_local = [(int(rr), int(cc)) for rr in range(inner.shape[0]) for cc in range(inner.shape[1]) if int(inner[rr, cc]) == BLUE]
        blue_before_grey = False
        if grey_coords:
            grey_local_order = sorted(
                [(p[0] - (r0 + 1), p[1] - (c0 + 1)) for p in grey_coords],
                key=_dir_sort_key(direction),
            )
            last_grey_key = _dir_sort_key(direction)(grey_local_order[-1])
            blue_before_grey = any(_dir_sort_key(direction)((rr, cc)) < last_grey_key for rr, cc in blue_local)

        stats.append(
            {
                "box": box_,
                "direction": direction,
                "distance": int(best),
                "grey": grey_count,
                "blue": blue_count,
                "steps": min(int(best), grey_count),
                "blue_sensitive": blue_count > 0 and min(int(best), grey_count + blue_count) != min(int(best), grey_count),
                "blue_before_grey": blue_before_grey,
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant rule:
    - Steps = min(head-to-dest distance, number of GREY cells only).
    - GREY cells convert to ORANGE from the head-facing side inward.
    - BLUE cells inside the container are obstacles: they do not count and do not change.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = _detect_containers(g, bg)
    assert containers

    out = g.copy()
    comp_by_color = _build_head_components(g, bg)

    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_

        head_info = _pick_head_for_container(g, box_, bg, comp_by_color)
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

        order = sorted(grey_coords, key=_dir_sort_key(direction))
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


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        for _ in range(60):
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            inner_area = inner_h * inner_w
            if is_decoy or inner_area >= 2:
                break
        else:
            return None

        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            max_blue = max(0, inner_area - 1)
            blue_count = random.randint(0, max_blue) if max_blue > 0 else 0
            chargeable = inner_area - blue_count
            grey = 0
            initial_orange = chargeable
            gap = random.randint(0, 6)
            steps = 0
        else:
            max_blue = max(1, inner_area - 1)
            blue_count = random.randint(1, max_blue)
            chargeable = inner_area - blue_count
            if chargeable <= 0:
                return None
            grey = random.randint(1, chargeable)
            initial_orange = chargeable - grey
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(grey, h2d)

        fill_order = _interior_scan_order(inner_h, inner_w, head_side)
        blue_cells = set(random.sample(fill_order, blue_count)) if blue_count > 0 else set()

        chargeable_order = [p for p in fill_order if p not in blue_cells]
        if len(chargeable_order) != chargeable:
            return None

        fill_index = {p: i for i, p in enumerate(fill_order)}
        grey_positions = chargeable_order[initial_orange:]
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
                "blue_count": blue_count,
                "blue_cells": blue_cells,
                "distance": gap + 1,
                "has_blue_before_grey": any(
                    fill_index[b] < fill_index[g]
                    for b in blue_cells
                    for g in grey_positions
                ) if blue_cells and grey > 0 else False,
            }
        )

    if not any((not s["is_decoy"]) and s["blue_count"] > 0 for s in specs):
        return None
    if not any((not s["is_decoy"]) and s["has_blue_before_grey"] for s in specs):
        return None
    if not any((not s["is_decoy"]) and s["blue_count"] > 0 and s["distance"] > s["grey"] for s in specs):
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

        interior = _build_interior(inner_h, inner_w, hs, a["blue_cells"], initial_orange)
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

        moved_interior = _build_interior(inner_h, inner_w, hs, a["blue_cells"], initial_orange + steps)
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = moved_interior
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

    stats = _container_stats(grid)
    if not any(s["blue_sensitive"] for s in stats):
        return None
    if not any(s["blue_before_grey"] for s in stats if s["grey"] > 0):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid D4 variant after 400 attempts")
