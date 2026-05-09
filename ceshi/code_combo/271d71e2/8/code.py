# Combo variant of 271d71e2 (index 8)
# combo_spec: D4/3+D10/2
# Name: 多色填充加权步数

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, paste


# Design analysis
# ---------------
# concrete_visualization:
# - The outer object is still a black rectangular frame with a full-edge maroon head line and an
#   aligned maroon destination line.
# - The fusion happens entirely inside the frame. Interior cells can now be:
#     ORANGE(7): already-spent charge
#     GREY(5): 1-step charge, turns into ORANGE when consumed
#     BLUE(1): 2-step weighted obstacle, never changes color
# - To keep the rule human-visible, the generator uses stripe-like bands along the movement axis:
#   some head-side bands are ORANGE, and the remaining bands are a BLUE/GREY mix with at least one
#   BLUE band before a later GREY band. This keeps the scene tidy while making "skip blue, fill grey"
#   visible.
#
# behavior_trace:
# - Suppose the head is on the left, so the fill order is leftmost column to rightmost column.
# - A 1x4 interior `[ORANGE, BLUE, GREY, BLUE]` has:
#     grey_count = 1
#     blue_count = 2
#     weighted_capacity = 1 + 2*2 = 5
# - If head-to-destination distance is 4:
#     original task: steps = min(4, 1) = 1, so the container would move 1 and the GREY would turn ORANGE
#     fused task:    steps = min(4, 5) = 4, so the container moves 4
# - Color change under the fused task:
#     only the GREY cell is consumable, so it becomes ORANGE
#     both BLUE cells stay BLUE even though they contributed 4 weighted steps together
# - Net difference:
#     movement distance is larger than the number of recolored cells
#     BLUE affects distance computation but not recoloring
#
# complexity_calibration:
# - The extra role is fixed to BLUE rather than per-object color remapping, so examples stay readable.
# - The generator forces at least one mover where `combo_steps > grey_only_steps`, otherwise the sample
#   would collapse back to the original task.
# - It also forces at least one mover with BLUE before a later GREY band, so solvers must infer that
#   BLUE is not just decoration and that GREY filling skips over BLUE obstacles.
# - Quantitative anti-degeneration check:
#     `boost = combo_steps - grey_only_steps`
#   must be positive for generated movers, otherwise the sample behaves exactly like the original task.
#
# implicit_assumption_audit:
# - Original container detection started from connected `(GREY | ORANGE)` regions. That breaks here
#   because BLUE can separate those regions. The fused solver therefore detects black rectangular
#   borders directly.
# - Original head detection excluded only background / border / grey / orange. With BLUE inside the
#   frame, border-adjacent BLUE would be mistaken for a head seed unless BLUE is excluded too.
# - Original step cap was `len(grey_cells)`. That no longer holds: steps are capped by
#   `grey_count + 2 * blue_count`.
# - D10/2's single-variant logic would additionally recolor BLUE through BLUE->GREY->ORANGE. That is
#   explicitly wrong for this combo: BLUE contributes weight but stays BLUE.
# - Because movement and recoloring are now decoupled, the fill logic must use:
#     moved_steps = min(distance, grey + 2*blue)
#     recolored_greys = min(moved_steps, grey_count)
#
# placement_legality:
# - Assemblies keep the original geometry: destination line + gap + head line + container.
# - Each assembly is placed with a 1-cell magenta moat, so containers and head/destination markers do
#   not overlap across objects.
# - The moved container stays inside its assembly footprint because `steps <= head_to_dest_distance`.
#
# visual_observability / cleanliness:
# - Using full stripes instead of random speckles makes the three interior roles legible at a glance.
# - Because some BLUE stripes sit before later GREY stripes, outputs visibly show GREY farther inward
#   turning ORANGE while closer BLUE remains BLUE.
# - No hidden ordering uses color ids; the visible head direction determines both movement and fill
#   order.


BG = 6
BORDER = 0
BLUE = 1
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
INTERIOR_COLORS = {BLUE, GREY, ORANGE}
MAX_GAP = 10
MAX_H2D = MAX_GAP + 1


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


def _ordered_cells(height: int, width: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    cells.sort(key=_dir_sort_key(direction))
    return cells


def _interior_scan_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _ordered_cells(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _axis_len(inner_h: int, inner_w: int, head_side: int) -> int:
    return inner_h if head_side in (0, 1) else inner_w


def _stripe_size(inner_h: int, inner_w: int, head_side: int) -> int:
    return inner_w if head_side in (0, 1) else inner_h


def _build_banded_interior(
    inner_h: int,
    inner_w: int,
    head_side: int,
    band_colors: list[int],
) -> np.ndarray:
    axis_len = _axis_len(inner_h, inner_w, head_side)
    stripe = _stripe_size(inner_h, inner_w, head_side)
    assert len(band_colors) == axis_len

    order = _interior_scan_order(inner_h, inner_w, head_side)
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    for idx, color in enumerate(band_colors):
        start = idx * stripe
        stop = start + stripe
        for rr, cc in order[start:stop]:
            interior[rr, cc] = int(color)
    return interior


def _weighted_capacity(inner: np.ndarray) -> int:
    grey_count = int(np.count_nonzero(inner == GREY))
    blue_count = int(np.count_nonzero(inner == BLUE))
    return grey_count + 2 * blue_count


def _advance_combo_interior(interior: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    out = np.array(interior, copy=True)
    grey_cells = [(rr, cc) for rr in range(out.shape[0]) for cc in range(out.shape[1]) if int(out[rr, cc]) == GREY]
    grey_cells.sort(key=_dir_sort_key(direction))
    for rr, cc in grey_cells[: int(steps)]:
        out[rr, cc] = int(ORANGE)
    return out


def _detect_containers(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
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
        if inner.size == 0 or not bool(np.isin(inner, list(INTERIOR_COLORS)).all()):
            continue

        has_signal = False
        for r, c in border_pts:
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
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
    h, w = g.shape
    r0, c0, r1, c1 = box_
    head_seeds = []
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
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


def _best_destination(
    g: np.ndarray,
    box_: tuple[int, int, int, int],
    head_info: dict,
    comp_by_color: dict[int, list[dict]],
) -> tuple[dict | None, int | None]:
    h, w = g.shape
    hc = int(head_info["color"])
    head_pts = set(head_info["head_pts"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    comp_pts = set(head_info["comp_pts"])
    direction = head_info["dir"]

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
                    if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(BORDER):
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
    return best_dest, best


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

        best_dest, distance = _best_destination(g, box_, head_info, comp_by_color)
        if best_dest is None or distance is None:
            continue

        r0, c0, r1, c1 = box_
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.count_nonzero(inner == GREY))
        blue_count = int(np.count_nonzero(inner == BLUE))
        weighted_steps = min(int(distance), int(_weighted_capacity(inner)))
        grey_only_steps = min(int(distance), grey_count)
        fill_count = min(weighted_steps, grey_count)

        order = _ordered_cells(inner.shape[0], inner.shape[1], head_info["dir"])
        rank = {p: idx for idx, p in enumerate(order)}
        blue_local = [(rr, cc) for rr in range(inner.shape[0]) for cc in range(inner.shape[1]) if int(inner[rr, cc]) == BLUE]
        grey_local = [(rr, cc) for rr in range(inner.shape[0]) for cc in range(inner.shape[1]) if int(inner[rr, cc]) == GREY]
        blue_before_grey = False
        if blue_local and grey_local:
            min_blue_rank = min(rank[p] for p in blue_local)
            max_grey_rank = max(rank[p] for p in grey_local)
            blue_before_grey = min_blue_rank < max_grey_rank

        mode = "distance" if weighted_steps == int(distance) and int(distance) < int(_weighted_capacity(inner)) else "capacity"
        stats.append(
            {
                "box": box_,
                "direction": head_info["dir"],
                "distance": int(distance),
                "grey": grey_count,
                "blue": blue_count,
                "capacity": int(_weighted_capacity(inner)),
                "combo_steps": weighted_steps,
                "grey_only_steps": grey_only_steps,
                "fill_count": fill_count,
                "boost": weighted_steps - grey_only_steps,
                "blue_before_grey": blue_before_grey,
                "mode": mode,
            }
        )

    return stats


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

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
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        best_dest, distance = _best_destination(g, box_, head_info, comp_by_color)
        if best_dest is None or distance is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        steps = min(int(distance), int(_weighted_capacity(inner)))
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_info["head_pts"]:
            if int(out[int(r), int(c)]) == int(head_info["color"]):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[1:-1, 1:-1] = _advance_combo_interior(inner, direction, steps)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_info["head_pts"]:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < h and 0 <= ccc < w:
                out[rrr, ccc] = int(head_info["color"])

    return out


def _choose_visible_motion(grey_count: int, blue_count: int) -> tuple[int, int, int, str] | None:
    capacity = grey_count + 2 * blue_count
    feasible = []
    hi = min(capacity, MAX_H2D)
    if grey_count + 1 <= hi:
        feasible.append("distance")
    if capacity + 1 <= MAX_H2D:
        feasible.append("capacity")
    if not feasible:
        return None

    mode = random.choice(feasible)
    if mode == "distance":
        h2d = random.randint(grey_count + 1, hi)
    else:
        h2d = random.randint(capacity + 1, MAX_H2D)
    return h2d - 1, min(h2d, capacity), capacity, mode


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        if is_decoy:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
            gap = random.randint(0, 6)
            steps = 0
            capacity = 0
            grey_count = 0
            blue_count = 0
            mode = "decoy"
        else:
            for _ in range(120):
                inner_h = random.randint(1, 8)
                inner_w = random.randint(1, 5)
                axis_len = _axis_len(inner_h, inner_w, head_side)
                if axis_len < 3:
                    continue

                orange_bands = random.randint(0, axis_len - 2)
                tail_len = axis_len - orange_bands
                tail = [BLUE]
                for _mid in range(max(0, tail_len - 2)):
                    tail.append(random.choice([BLUE, GREY]))
                tail.append(GREY)
                band_colors = [ORANGE] * orange_bands + tail

                interior = _build_banded_interior(inner_h, inner_w, head_side, band_colors)
                grey_count = int(np.count_nonzero(interior == GREY))
                blue_count = int(np.count_nonzero(interior == BLUE))
                motion = _choose_visible_motion(grey_count, blue_count)
                if motion is None:
                    continue
                gap, steps, capacity, mode = motion
                break
            else:
                return None

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
                "capacity": capacity,
                "interior": interior,
                "grey_count": grey_count,
                "blue_count": blue_count,
                "is_decoy": is_decoy,
                "mode": mode,
            }
        )

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
        patch[1:-1, 1:-1] = _advance_combo_interior(a["interior"], SIDE_TO_DIR[hs], steps)
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
    if len(stats) != n_containers:
        return None

    moving_stats = [s for s in stats if s["combo_steps"] > 0]
    if len(moving_stats) != n_movers:
        return None
    if not any(s["boost"] > 0 for s in moving_stats):
        return None
    if not any(s["combo_steps"] > s["fill_count"] for s in moving_stats):
        return None
    if not any(s["blue_before_grey"] and s["boost"] > 0 for s in moving_stats):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 500 attempts")
