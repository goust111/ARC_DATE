# Variant of 271d71e2
# Dimension: D8 — 变换类型 (transform)
# Change: recolor — 灰色(5)变为橙色(7)，容器整体平移，目标线保持不动... → 容器缩小（边框收缩）
#
# Design analysis
# ---
# variant_spec:
#   core_change:
#     original: a full black frame translates; up to `steps` grey cells turn orange.
#     variant: the head still advances toward the same-color destination, but the frame
#       now spends one trailing layer of body per step. After each step the translated
#       frame drops its rear border line as orange trail, and the rear grey row/column
#       becomes the new border. The object therefore reaches the target while becoming
#       shorter along the movement axis.
#   concrete_pixels:
#     input:
#       - background is 6, border is 0, head/destination are 9.
#       - each container is still a clean rectangle with a 1-cell black frame.
#       - to make "shrink per step" observable and unambiguous, the interior is no
#         longer an arbitrary orange/grey prefix over individual cells; instead it is
#         split into full stripes along the movement axis:
#           - orange slab on the head-facing side
#           - grey slab on the trailing side
#       - movers always keep at least one orange stripe, so the shrunken container
#         still has non-empty interior in the output.
#     output:
#       - the destination line stays fixed.
#       - the head line moves exactly as before.
#       - the final black frame is translated toward the destination by `steps`,
#         but its size on the movement axis is reduced by `steps`.
#       - behind the final frame sits a contiguous orange trail made from the
#         trailing border lines that were shed during motion.
#   behavior_trace:
#     example: |
#       Move-right, inner size 3x4, orange_layers=1, grey_layers=3, gap=2.
#       Original logic: steps=min(3,3)=3, whole 5x6 box shifts right by 3,
#       and three grey cells near the head turn orange.
#       Variant logic: steps=min(3, grey_columns=3)=3. The head still reaches the
#       same destination line, but after each rightward step the translated left border
#       is dropped as orange trail and the frame contracts from the left. Final result:
#       width shrinks by 3, the surviving interior keeps its original rightmost
#       orange stripe, and three orange trail columns remain behind it.
#     delta:
#       - old rule changes cell colors inside a fixed-size rectangle
#       - new rule changes the rectangle geometry itself and externalizes the consumed
#         mass as orange trail
#   complexity_calibration:
#     - Fuel is measured in grey layers, not grey pixels, because consuming one
#       row/column per step must be geometrically feasible.
#     - Only two interior regions are used (orange head-side slab, grey tail slab),
#       which is enough to show the shrinking rule without introducing ambiguous
#       multi-segment fuel patterns.
#     - Multiple containers still appear, with both distance-capped and fuel-capped
#       movers, plus occasional fully-orange decoys.
#   implicit_assumptions_review:
#     - Original `grey_coords` + per-cell ordering assumed movement fuel was counted
#       per pixel; this breaks once one step removes a whole row/column.
#       Replaced by `_grey_layer_depth`, which measures contiguous full-grey layers
#       on the trailing side.
#     - Original patch placement assumed the container bbox size stayed constant.
#       Variant output requires recomputing a smaller bbox via `_simulate_shrink`.
#     - Original interior could be any GREY/FILL mixture. Variant solver now assumes
#       the fuel side is arranged in full trailing stripes; generator enforces that.
#     - Original "fill inside patch" mutates colors but preserves border positions.
#       Variant instead trims the interior array and repaints exterior trail cells.
#   placement_legality:
#     - The generator keeps the same assembly envelope: destination + gap + head +
#       original container. The shrunken output and its trail stay inside that envelope,
#       so 1-cell spacing between assemblies remains sufficient for both input and output.
#     - Movers are generated only when the movement-axis interior depth is at least 2,
#       guaranteeing a surviving interior stripe after shrinking.
#   visual_observability:
#     - The new rule is visible because the output box is genuinely smaller, and orange
#       appears outside the frame as a path. This is distinct from the original task,
#       where orange only spreads inside an unchanged rectangle.
#   visual_cleanliness:
#     - Only one external artifact is added: a straight orange stripe trail aligned with
#       the motion axis. No overlapping symbols or color-dependent tie-breakers are used.
#   color_inference:
#     - Colors keep their original roles: 0=frame, 6=background, 9=head/destination,
#       7=spent trail / pre-existing charged region, 5=remaining fuel.
#       The structural cue for orange changes is explicit: orange outside the frame
#       always marks consumed motion steps.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _head_cells_from_box(box: tuple[int, int, int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if direction == (1, 0):
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    if direction == (0, -1):
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    return [(r, c1 + 1) for r in range(r0, r1 + 1)]


def _grey_layer_depth(inner: np.ndarray, direction: tuple[int, int]) -> int:
    if direction == (-1, 0):
        idxs = range(inner.shape[0] - 1, -1, -1)
        depth = 0
        for rr in idxs:
            if bool(np.all(inner[rr, :] == GREY)):
                depth += 1
            else:
                break
        return depth
    if direction == (1, 0):
        depth = 0
        for rr in range(inner.shape[0]):
            if bool(np.all(inner[rr, :] == GREY)):
                depth += 1
            else:
                break
        return depth
    if direction == (0, -1):
        depth = 0
        for cc in range(inner.shape[1] - 1, -1, -1):
            if bool(np.all(inner[:, cc] == GREY)):
                depth += 1
            else:
                break
        return depth
    depth = 0
    for cc in range(inner.shape[1]):
        if bool(np.all(inner[:, cc] == GREY)):
            depth += 1
        else:
            break
    return depth


def _trim_inner(inner: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    if steps <= 0:
        return inner.copy()
    if direction == (-1, 0):
        return inner[:-steps, :].copy()
    if direction == (1, 0):
        return inner[steps:, :].copy()
    if direction == (0, -1):
        return inner[:, :-steps].copy()
    return inner[:, steps:].copy()


def _simulate_shrink(
    box: tuple[int, int, int, int], direction: tuple[int, int], steps: int
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    trail: list[tuple[int, int]] = []
    for _ in range(steps):
        r0 += direction[0]
        c0 += direction[1]
        r1 += direction[0]
        c1 += direction[1]
        if direction == (-1, 0):
            trail.extend((r1, c) for c in range(c0, c1 + 1))
            r1 -= 1
        elif direction == (1, 0):
            trail.extend((r0, c) for c in range(c0, c1 + 1))
            r0 += 1
        elif direction == (0, -1):
            trail.extend((r, c1) for r in range(r0, r1 + 1))
            c1 -= 1
        else:
            trail.extend((r, c0) for r in range(r0, r1 + 1))
            c0 += 1
    return (r0, c0, r1, c1), trail


def _stamp_container(out: np.ndarray, box: tuple[int, int, int, int], inner: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = BORDER
    out[r0 + 1 : r1, c0 + 1 : c1] = inner


def _build_striped_interior(inner_h: int, inner_w: int, head_side: int, orange_layers: int) -> np.ndarray:
    inner = np.full((inner_h, inner_w), GREY, dtype=int)
    if head_side == 0:
        inner[:orange_layers, :] = ORANGE
    elif head_side == 1:
        inner[inner_h - orange_layers :, :] = ORANGE
    elif head_side == 2:
        inner[:, :orange_layers] = ORANGE
    else:
        inner[:, inner_w - orange_layers :] = ORANGE
    return inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant rule:
    - Steps = min(head-to-destination distance, trailing grey stripe depth).
    - For each step, the container advances one cell, sheds its trailing border
      line as orange trail, and shrinks by one interior row/column from the rear.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

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
                    if val not in (int(bg), BORDER, GREY, ORANGE):
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
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        head_info = pick_head_for_container(box_)
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
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
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
        distance = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_layers = _grey_layer_depth(inner, direction)
        axis_depth = int(inner.shape[0] if direction[0] != 0 else inner.shape[1])
        steps = min(int(distance), int(grey_layers), max(0, axis_depth - 1))
        if steps <= 0:
            continue

        final_box, trail = _simulate_shrink(box_, direction, steps)
        final_inner = _trim_inner(inner, direction, steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        for r, c in trail:
            if 0 <= r < H and 0 <= c < W:
                out[int(r), int(c)] = ORANGE

        _stamp_container(out, final_box, final_inner)
        for r, c in _head_cells_from_box(final_box, direction):
            if 0 <= r < H and 0 <= c < W:
                out[int(r), int(c)] = int(hc)

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)
        if head_side in (0, 1):
            inner_h = random.randint(1 if is_decoy else 2, 8)
            inner_w = random.randint(1, 5)
            axis_depth = inner_h
        else:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1 if is_decoy else 2, 5)
            axis_depth = inner_w

        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            orange_layers = axis_depth
            grey_layers = 0
            gap = random.randint(0, 6)
        else:
            grey_layers = random.randint(1, axis_depth - 1)
            orange_layers = axis_depth - grey_layers
            gap = random.randint(0, 10)

        steps = min(gap + 1, grey_layers, axis_depth - 1)

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
                "orange_layers": orange_layers,
                "grey_layers": grey_layers,
                "is_decoy": is_decoy,
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
    min_side = max(8, int(total_area**0.5) + 3)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)
    lo_h = max(8, max_asm_h + 2)
    lo_w = max(8, max_asm_w + 2)
    hi_h = min(26, max(lo_h, min_side + 6))
    hi_w = min(26, max(lo_w, min_side + 6))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
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
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
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
        box = (cr, cc, cr + oh - 1, cc + ow - 1)
        inner = _build_striped_interior(a["inner_h"], a["inner_w"], hs, a["orange_layers"])

        _stamp_container(grid, box, inner)
        for r, c in _head_cells_from_box(box, SIDE_TO_DIR[hs]):
            grid[r, c] = HEAD_COLOR

        if hs == 0:
            grid[cr - 2 - a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh + 1 + a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        if a["steps"] <= 0:
            continue
        hs = a["head_side"]
        oh, ow = a["outer_h"], a["outer_w"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        box = (cr, cc, cr + oh - 1, cc + ow - 1)
        direction = SIDE_TO_DIR[hs]
        inner = _build_striped_interior(a["inner_h"], a["inner_w"], hs, a["orange_layers"])

        expected[cr : cr + oh, cc : cc + ow] = BG
        for r, c in _head_cells_from_box(box, direction):
            expected[r, c] = BG

        final_box, trail = _simulate_shrink(box, direction, a["steps"])
        final_inner = _trim_inner(inner, direction, a["steps"])
        for r, c in trail:
            expected[r, c] = ORANGE
        _stamp_container(expected, final_box, final_inner)
        for r, c in _head_cells_from_box(final_box, direction):
            expected[r, c] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
