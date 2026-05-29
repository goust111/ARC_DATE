# Combo variant of 271d71e2 (index 27)
# combo_spec: D3/3+D8/0+D9/0
# Name: 灰色消失缩小紧凑裁剪
#
# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Preserve the original object vocabulary on the input: magenta background 6,
#      black rectangular containers 0, striped orange/grey interiors 7/5, a maroon
#      head line 9 attached to one side, and a matching maroon destination line on
#      the motion axis.
#    - The fused rule is not "D3 holes inside a fixed box" plus "D8 orange trail".
#      Instead, each motion step consumes one full trailing grey stripe, so the box
#      itself shrinks by one row/column from the rear while moving forward.
#    - The consumed stripe follows D3/3 semantics: it disappears to background rather
#      than turning orange. Therefore there is no external orange trail.
#    - Example, move right, inner = 2x4 with columns `[5, 5, 7, 7]`, distance = 2:
#      original task -> same-size box shifts right by 2, two grey cells near the
#      head side recolor orange.
#      fused task -> the box shifts right by 2 and loses its two leftmost grey
#      columns; final inner becomes `2x2` all orange, outer width shrinks by 2,
#      and nothing is left behind. The final whole scene is then tightly cropped.
# b) Complexity calibration:
#    - Shrink-per-step is only geometrically unambiguous if fuel is arranged as full
#      trailing stripes, so movers use a simple two-region interior:
#      orange slab on the head side, grey slab on the trailing side.
#    - This keeps the rule visible across 2-4 examples without introducing many
#      categories. There are still distance-capped movers, fuel-capped movers, and
#      occasional fully-orange decoys.
# c) Hidden assumptions audited against the original solver:
#    - Original code assumed fuel was counted per grey pixel. That breaks once one
#      step removes a whole row/column, so the solver now measures trailing grey
#      layer depth instead of arbitrary grey count.
#    - Original code assumed the translated bbox kept the same size. That no longer
#      holds; the solver must compute a smaller final bbox and trim the interior.
#    - Container detection still assumes input interiors contain only grey/orange.
#      That remains valid because the erased-to-background effect appears only in
#      the output, never in the generated input.
#    - Tight crop must use the input background color, not "most common in output",
#      because after cropping the background might not dominate.
# d) Placement legality:
#    - The generator is output-first. It packs the already-shrunk final assemblies,
#      then reconstructs the larger input containers by inverting the shrink motion.
#    - Both input and output scenes are rendered and checked for overlap, bounds, and
#      a one-cell moat between assemblies.
#    - Reverse reconstruction needs more than `steps` extra room on the trailing side:
#      because the final box has already lost `steps` layers, inverting the motion
#      expands the original box by another `steps` cells past the final anchor.
#      The placement code therefore reserves `2 * steps` margin in the reverse
#      direction before embedding the packed final cluster into the larger input grid.
# e) Visual observability:
#    - Humans can see three coupled effects at once: motion toward the maroon target,
#      the container becoming shorter along the motion axis, and the output canvas
#      hugging the final objects.
#    - The absence of an orange external trail is itself informative: mass is burnt
#      away rather than deposited.
# f) Visual cleanliness:
#    - Only one motion axis per assembly, one marker color, and rectangular boxes are
#      used. Packing keeps assemblies separated so heads/destinations do not merge
#      into ambiguous clutter.
# g) Color inference:
#    - No new colors are introduced. Orange remains "already charged" material on the
#      head side, grey remains consumable fuel on the rear side, black is the frame,
#      maroon marks head/destination, and magenta is background.
# New solver assumptions introduced here:
#    - For movers, the trailing fuel must be contiguous full stripes so shrink depth is
#      well-defined. `generate_input()` enforces this explicitly.
#    - Each mover must retain at least one interior stripe after shrinking, so the
#      generator caps steps by `axis_depth - 1`.

from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys

import numpy as np
from scipy.ndimage import label

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
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


def _crop_to_content(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.array(grid, copy=True)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _grey_layer_depth(inner: np.ndarray, direction: tuple[int, int]) -> int:
    if direction == (-1, 0):
        depth = 0
        for rr in range(inner.shape[0] - 1, -1, -1):
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
        return np.array(inner, copy=True)
    if direction == (-1, 0):
        return np.array(inner[:-steps, :], copy=True)
    if direction == (1, 0):
        return np.array(inner[steps:, :], copy=True)
    if direction == (0, -1):
        return np.array(inner[:, :-steps], copy=True)
    return np.array(inner[:, steps:], copy=True)


def _simulate_shrink_box(
    box: tuple[int, int, int, int], direction: tuple[int, int], steps: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return (r0 - steps, c0, r1 - 2 * steps, c1)
    if direction == (1, 0):
        return (r0 + 2 * steps, c0, r1 + steps, c1)
    if direction == (0, -1):
        return (r0, c0 - steps, r1, c1 - 2 * steps)
    return (r0, c0 + 2 * steps, r1, c1 + steps)


def _invert_shrink_box(
    box: tuple[int, int, int, int], direction: tuple[int, int], steps: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return (r0 + steps, c0, r1 + 2 * steps, c1)
    if direction == (1, 0):
        return (r0 - 2 * steps, c0, r1 - steps, c1)
    if direction == (0, -1):
        return (r0, c0 + steps, r1, c1 + 2 * steps)
    return (r0, c0 - 2 * steps, r1, c1 - steps)


def _stamp_container(out: np.ndarray, box: tuple[int, int, int, int], inner: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = BORDER
    out[r0 + 1 : r1, c0 + 1 : c1] = inner


def _build_striped_interior(inner_h: int, inner_w: int, head_side: int, orange_layers: int) -> np.ndarray:
    inner = np.full((inner_h, inner_w), GREY, dtype=int)
    if orange_layers <= 0:
        return inner
    if head_side == 0:
        inner[:orange_layers, :] = ORANGE
    elif head_side == 1:
        inner[inner_h - orange_layers :, :] = ORANGE
    elif head_side == 2:
        inner[:, :orange_layers] = ORANGE
    else:
        inner[:, inner_w - orange_layers :] = ORANGE
    return inner


def _assembly_dims(inner_h: int, inner_w: int, head_side: int, head_dest_distance: int) -> tuple[int, int, int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if head_side == 0:
        return outer_h + 1 + head_dest_distance, outer_w, head_dest_distance + 1, 0
    if head_side == 1:
        return outer_h + 1 + head_dest_distance, outer_w, 0, 0
    if head_side == 2:
        return outer_h, outer_w + 1 + head_dest_distance, 0, head_dest_distance + 1
    return outer_h, outer_w + 1 + head_dest_distance, 0, 0


def _build_assembly_cells(
    inner: np.ndarray, head_side: int, cont_r: int, cont_c: int, head_dest_distance: int
) -> dict[tuple[int, int], int]:
    inner_h, inner_w = map(int, inner.shape)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    box = (cont_r, cont_c, cont_r + outer_h - 1, cont_c + outer_w - 1)
    direction = SIDE_TO_DIR[head_side]
    cells: dict[tuple[int, int], int] = {}

    for r, c in _rect_border_coords(*box):
        cells[(int(r), int(c))] = BORDER
    for rr in range(inner_h):
        for cc in range(inner_w):
            cells[(int(cont_r + 1 + rr), int(cont_c + 1 + cc))] = int(inner[rr, cc])

    for r, c in _head_cells_from_box(box, direction):
        cells[(int(r), int(c))] = HEAD_COLOR

    dr, dc = direction
    for r, c in _head_cells_from_box(box, direction):
        cells[(int(r + dr * head_dest_distance), int(c + dc * head_dest_distance))] = HEAD_COLOR
    return cells


def _coords_with_moat(cells: dict[tuple[int, int], int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _render_scene(shape: tuple[int, int], assembly_cells: list[dict[tuple[int, int], int]]) -> np.ndarray | None:
    H, W = shape
    grid = np.full((H, W), BG, dtype=int)
    for cells in assembly_cells:
        for (r, c), color in cells.items():
            if not (0 <= r < H and 0 <= c < W):
                return None
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = int(color)
    return grid


def _has_moat_violation(assembly_cells: list[dict[tuple[int, int], int]], shape: tuple[int, int]) -> bool:
    for i in range(len(assembly_cells)):
        coords_i = set(assembly_cells[i].keys())
        moat_i = _coords_with_moat(assembly_cells[i], shape)
        for j in range(i + 1, len(assembly_cells)):
            coords_j = set(assembly_cells[j].keys())
            if coords_i & coords_j:
                return True
            if moat_i & coords_j:
                return True
    return False


def _pack_rectangles(rects: list[tuple[int, int]], max_h: int, max_w: int):
    total_area = sum(h * w for h, w in rects)
    max_rect_h = max(h for h, _w in rects)
    max_rect_w = max(w for _h, w in rects)
    if max_rect_h > max_h or max_rect_w > max_w:
        return None

    for _ in range(160):
        cluster_h = random.randint(max_rect_h, max_h)
        cluster_w = random.randint(max_rect_w, max_w)
        if cluster_h * cluster_w < total_area + len(rects):
            continue
        occupied = np.zeros((cluster_h, cluster_w), dtype=bool)
        order = list(range(len(rects)))
        order.sort(key=lambda idx: rects[idx][0] * rects[idx][1], reverse=True)
        placements = [None] * len(rects)
        ok = True
        for idx in order:
            h, w = rects[idx]
            placed = False
            for _ in range(320):
                r0 = random.randint(0, cluster_h - h)
                c0 = random.randint(0, cluster_w - w)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(cluster_h, r0 + h + 1)
                cc1 = min(cluster_w, c0 + w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    occupied[r0 : r0 + h, c0 : c0 + w] = True
                    placements[idx] = (r0, c0)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return cluster_h, cluster_w, placements
    return None


def _solve_same_size(grid: np.ndarray) -> tuple[np.ndarray, int]:
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
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = np.array(g, copy=True)
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
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
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
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                            adj = True
                            break
                    if adj:
                        break
                if adj:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00 = int(min(rs))
                c00 = int(min(cs))
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

        distance = abs(int(best_dest["tl"][0] - head_tl[0])) + abs(int(best_dest["tl"][1] - head_tl[1]))
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_layers = _grey_layer_depth(inner, direction)
        axis_depth = int(inner.shape[0] if direction[0] != 0 else inner.shape[1])
        steps = min(int(distance), int(grey_layers), max(0, axis_depth - 1))
        if steps <= 0:
            continue

        final_box = _simulate_shrink_box(box_, direction, steps)
        final_inner = _trim_inner(inner, direction, steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        _stamp_container(out, final_box, final_inner)
        for r, c in _head_cells_from_box(final_box, direction):
            if 0 <= r < H and 0 <= c < W:
                out[int(r), int(c)] = int(hc)

    return out, bg


def main(grid: np.ndarray) -> np.ndarray:
    out, bg = _solve_same_size(np.array(grid, dtype=int))
    return _crop_to_content(out, bg)


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    if n_decoys >= n_containers:
        n_decoys = n_containers - 1

    specs = []
    for i in range(n_containers):
        head_side = random.randint(0, 3)
        is_decoy = i >= (n_containers - n_decoys)

        if head_side in (0, 1):
            inner_h = random.randint(1 if is_decoy else 2, 8)
            inner_w = random.randint(1, 5)
            axis_depth = inner_h
        else:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1 if is_decoy else 2, 5)
            axis_depth = inner_w

        if is_decoy:
            grey_layers = 0
            orange_layers = axis_depth
            travel = random.randint(1, 6)
            steps = 0
        else:
            grey_layers = random.randint(1, axis_depth - 1)
            orange_layers = axis_depth - grey_layers
            travel = random.randint(1, 10)
            steps = min(travel, grey_layers, axis_depth - 1)

        remain = int(travel - steps)
        final_inner_h = inner_h - steps if head_side in (0, 1) else inner_h
        final_inner_w = inner_w - steps if head_side in (2, 3) else inner_w
        final_h, final_w, final_cont_r, final_cont_c = _assembly_dims(
            final_inner_h, final_inner_w, head_side, remain
        )

        specs.append(
            {
                "head_side": head_side,
                "is_decoy": is_decoy,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "axis_depth": axis_depth,
                "grey_layers": grey_layers,
                "orange_layers": orange_layers,
                "travel": int(travel),
                "steps": int(steps),
                "remain": remain,
                "final_inner_h": int(final_inner_h),
                "final_inner_w": int(final_inner_w),
                "final_h": int(final_h),
                "final_w": int(final_w),
                "final_cont_r": int(final_cont_r),
                "final_cont_c": int(final_cont_c),
            }
        )

    if not any(int(spec["steps"]) > 0 for spec in specs):
        return None

    need_top = need_bottom = need_left = need_right = 0
    for spec in specs:
        steps = int(spec["steps"])
        direction = SIDE_TO_DIR[int(spec["head_side"])]
        if direction == (1, 0):
            need_top = max(need_top, 2 * steps)
        elif direction == (-1, 0):
            need_bottom = max(need_bottom, 2 * steps)
        elif direction == (0, 1):
            need_left = max(need_left, 2 * steps)
        else:
            need_right = max(need_right, 2 * steps)

    max_cluster_h = 26 - need_top - need_bottom - 2
    max_cluster_w = 26 - need_left - need_right - 2
    if max_cluster_h < 3 or max_cluster_w < 3:
        return None

    pack = _pack_rectangles([(s["final_h"], s["final_w"]) for s in specs], max_cluster_h, max_cluster_w)
    if pack is None:
        return None
    cluster_h, cluster_w, placements = pack

    base_h = need_top + need_bottom + cluster_h
    base_w = need_left + need_right + cluster_w
    min_extra_h = max(2, 8 - base_h)
    min_extra_w = max(2, 8 - base_w)
    room_h = 26 - base_h
    room_w = 26 - base_w
    if room_h < min_extra_h or room_w < min_extra_w:
        return None

    extra_h = random.randint(min_extra_h, min(8, room_h))
    extra_w = random.randint(min_extra_w, min(8, room_w))
    extra_top = random.randint(1, extra_h - 1)
    extra_bottom = extra_h - extra_top
    extra_left = random.randint(1, extra_w - 1)
    extra_right = extra_w - extra_left

    grid_h = base_h + extra_top + extra_bottom
    grid_w = base_w + extra_left + extra_right
    if not (8 <= grid_h <= 26 and 8 <= grid_w <= 26):
        return None

    offset_r = need_top + extra_top
    offset_c = need_left + extra_left

    input_assemblies = []
    output_assemblies = []
    for spec, (pr, pc) in zip(specs, placements):
        hs = int(spec["head_side"])
        direction = SIDE_TO_DIR[hs]
        steps = int(spec["steps"])

        final_inner = _build_striped_interior(
            int(spec["final_inner_h"]), int(spec["final_inner_w"]), hs, int(spec["orange_layers"])
        )
        final_cont_r = int(offset_r + pr + int(spec["final_cont_r"]))
        final_cont_c = int(offset_c + pc + int(spec["final_cont_c"]))
        final_box = (
            final_cont_r,
            final_cont_c,
            final_cont_r + final_inner.shape[0] + 1,
            final_cont_c + final_inner.shape[1] + 1,
        )
        input_box = _invert_shrink_box(final_box, direction, steps)

        input_inner = _build_striped_interior(
            int(spec["inner_h"]), int(spec["inner_w"]), hs, int(spec["orange_layers"])
        )
        input_assemblies.append(
            _build_assembly_cells(
                input_inner,
                hs,
                int(input_box[0]),
                int(input_box[1]),
                int(spec["travel"]),
            )
        )
        output_assemblies.append(
            _build_assembly_cells(
                final_inner,
                hs,
                final_cont_r,
                final_cont_c,
                int(spec["remain"]),
            )
        )

    shape = (grid_h, grid_w)
    if _has_moat_violation(input_assemblies, shape):
        return None
    if _has_moat_violation(output_assemblies, shape):
        return None

    grid = _render_scene(shape, input_assemblies)
    output_full = _render_scene(shape, output_assemblies)
    if grid is None or output_full is None:
        return None

    expected = _crop_to_content(output_full, BG)
    if expected.shape == grid.shape:
        return None

    try:
        actual_full, _ = _solve_same_size(grid)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual_full, output_full):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid combo variant after 500 attempts")
