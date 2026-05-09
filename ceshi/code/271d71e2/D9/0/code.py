# Variant of 271d71e2
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: same_size — 输出与输入尺寸相同，容器在原位被擦除并在新位置重绘... → 紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys

import numpy as np
from scipy.ndimage import label

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the D9 variant
# a) Concrete behavior:
#    The input still shows the original same-size scene: bordered black containers
#    with a maroon head line and a matching maroon destination line on the movement
#    axis. The movement rule is unchanged. The only behavioral change is after the
#    final same-size scene is computed: discard all outer background-only rows and
#    columns, returning the minimal rectangle that contains every non-background
#    pixel in the final scene. Example: if two containers move upward into a compact
#    cluster near the center, the original solver would leave a full magenta frame
#    around them; this variant removes that frame, so the output height/width are
#    determined by the moved objects rather than the input canvas.
# b) Complexity calibration:
#    The crop is global, not per-container. All original object types remain, so the
#    task is still primarily about movement/charging; the new inference burden is to
#    notice that the output canvas hugs the post-movement objects. The generator
#    keeps 2-4 assemblies and usually multiple movers so the crop is visible without
#    adding extra symbols or extra cases.
# c) Implicit assumptions in the original main():
#    - Container detection from GREY/ORANGE interiors still holds because the input
#      format is unchanged.
#    - Head/destination matching via same-color line shape still holds because the
#      generator preserves straight full-edge head lines and aligned destinations.
#    - The original code writes onto an output canvas the same size as the input.
#      That assumption becomes invalid only at the final return step, so the safe
#      adaptation is to keep the entire original solve procedure and crop once the
#      final scene has been assembled.
#    - Cropping must use the input background color; using "most common color" on the
#      output would be unsafe because a tight crop can make background non-dominant.
# d) Placement legality:
#    The generator is output-first. It first packs the moved objects into a compact
#    final cluster, then derives the input by shifting each movable container/head
#    backward by its step count while leaving the destination fixed. This guarantees
#    that the crop is based on the true post-movement layout. Reject-sampling checks
#    bounds, overlap, and a one-cell moat between distinct assemblies in both input
#    and output renderings.
# e) Visual observability:
#    The generator enforces that the cropped output is strictly smaller than the
#    input, usually with margins on multiple sides. Across examples a human can see
#    that the internal motion rule is the same as the base task, but the outer blank
#    magenta frame is removed.
# f) Visual cleanliness:
#    Assemblies are packed with at least one background cell between them in the
#    final cluster, and reject-sampling removes input/output placements that become
#    crowded after the backward shift. This avoids unreadable overlaps or accidental
#    merged groups.
# g) Color-change inference:
#    No new color logic is introduced. Grey-to-orange charging remains the only
#    color transition, so D9 changes only the canvas shape, not the symbol system.
# Reference-task note:
#    The prompt cites e87109e9 as the source of the tight-crop idea. In this
#    workspace the JSON data for e87109e9 exists, but no readable solver .py exists,
#    so the implementation below uses the source-task idea (tight crop) without
#    copying code from a missing solver.


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


def _crop_to_content(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.array(grid, copy=True)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _solve_same_size(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

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

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
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

    return out, bg


def main(grid: np.ndarray) -> np.ndarray:
    out, bg = _solve_same_size(np.array(grid, dtype=int))
    return _crop_to_content(out, bg)


def _final_assembly_dims(spec: dict) -> tuple[int, int, int, int]:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    rem = spec["remain"]
    if hs == 0:
        return oh + 1 + rem, ow, rem + 1, 0
    if hs == 1:
        return oh + 1 + rem, ow, 0, 0
    if hs == 2:
        return oh, ow + 1 + rem, 0, rem + 1
    return oh, ow + 1 + rem, 0, 0


def _build_assembly_cells(spec: dict, cont_r: int, cont_c: int, orange_count: int, dest_gap_from_head: int):
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    cells: dict[tuple[int, int], int] = {}

    for r, c in _rect_border_coords(cont_r, cont_c, cont_r + oh - 1, cont_c + ow - 1):
        cells[(int(r), int(c))] = BORDER

    fill_order = _interior_fill_order(inner_h, inner_w, hs)
    orange_set = set(fill_order[:orange_count])
    for rr in range(inner_h):
        for cc in range(inner_w):
            color = ORANGE if (rr, cc) in orange_set else GREY
            cells[(int(cont_r + 1 + rr), int(cont_c + 1 + cc))] = int(color)

    if hs == 0:
        for c in range(cont_c, cont_c + ow):
            cells[(int(cont_r - 1), int(c))] = HEAD_COLOR
            cells[(int(cont_r - 1 - dest_gap_from_head), int(c))] = HEAD_COLOR
    elif hs == 1:
        for c in range(cont_c, cont_c + ow):
            cells[(int(cont_r + oh), int(c))] = HEAD_COLOR
            cells[(int(cont_r + oh + dest_gap_from_head), int(c))] = HEAD_COLOR
    elif hs == 2:
        for r in range(cont_r, cont_r + oh):
            cells[(int(r), int(cont_c - 1))] = HEAD_COLOR
            cells[(int(r), int(cont_c - 1 - dest_gap_from_head))] = HEAD_COLOR
    else:
        for r in range(cont_r, cont_r + oh):
            cells[(int(r), int(cont_c + ow))] = HEAD_COLOR
            cells[(int(r), int(cont_c + ow + dest_gap_from_head))] = HEAD_COLOR

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

    for _ in range(120):
        cluster_h = random.randint(max_rect_h, max_h)
        cluster_w = random.randint(max_rect_w, max_w)
        if cluster_h * cluster_w < total_area + len(rects):
            continue
        occupied = np.zeros((cluster_h, cluster_w), dtype=bool)
        placements: list[tuple[int, int]] = []
        order = list(range(len(rects)))
        order.sort(key=lambda idx: rects[idx][0] * rects[idx][1], reverse=True)
        temp = [None] * len(rects)
        ok = True
        for idx in order:
            h, w = rects[idx]
            placed = False
            for _ in range(250):
                r0 = random.randint(0, cluster_h - h)
                c0 = random.randint(0, cluster_w - w)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(cluster_h, r0 + h + 1)
                cc1 = min(cluster_w, c0 + w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    occupied[r0 : r0 + h, c0 : c0 + w] = True
                    temp[idx] = (r0, c0)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            placements = [temp[i] for i in range(len(rects))]
            return cluster_h, cluster_w, placements
    return None


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    if n_decoys >= n_containers:
        n_decoys = n_containers - 1

    specs = []
    for i in range(n_containers):
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        is_decoy = i >= (n_containers - n_decoys)

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            travel = random.randint(1, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            travel = random.randint(1, 10)
            steps = min(grey, travel)

        remain = int(travel - steps)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        spec = {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "is_decoy": is_decoy,
            "grey": grey,
            "initial_orange": initial_orange,
            "travel": int(travel),
            "steps": int(steps),
            "remain": int(remain),
        }
        final_h, final_w, final_cont_r, final_cont_c = _final_assembly_dims(spec)
        spec["final_h"] = final_h
        spec["final_w"] = final_w
        spec["final_cont_r"] = final_cont_r
        spec["final_cont_c"] = final_cont_c
        specs.append(spec)

    if not any(int(s["steps"]) > 0 for s in specs):
        return None

    need_top = need_bottom = need_left = need_right = 0
    for spec in specs:
        dr, dc = SIDE_TO_DIR[spec["head_side"]]
        steps = int(spec["steps"])
        if dr == 1:
            need_top = max(need_top, steps)
        elif dr == -1:
            need_bottom = max(need_bottom, steps)
        elif dc == 1:
            need_left = max(need_left, steps)
        else:
            need_right = max(need_right, steps)

    max_cluster_h = 26 - need_top - need_bottom - 2
    max_cluster_w = 26 - need_left - need_right - 2
    if max_cluster_h < 3 or max_cluster_w < 3:
        return None

    pack = _pack_rectangles([(s["final_h"], s["final_w"]) for s in specs], max_cluster_h, max_cluster_w)
    if pack is None:
        return None
    cluster_h, cluster_w, placements = pack

    room_h = 26 - need_top - need_bottom - cluster_h
    room_w = 26 - need_left - need_right - cluster_w
    if room_h < 2 or room_w < 2:
        return None

    extra_h = random.randint(2, min(8, room_h))
    extra_w = random.randint(2, min(8, room_w))
    extra_top = random.randint(1, extra_h - 1)
    extra_bottom = extra_h - extra_top
    extra_left = random.randint(1, extra_w - 1)
    extra_right = extra_w - extra_left

    grid_h = cluster_h + need_top + need_bottom + extra_top + extra_bottom
    grid_w = cluster_w + need_left + need_right + extra_left + extra_right
    if not (8 <= grid_h <= 26 and 8 <= grid_w <= 26):
        return None

    offset_r = need_top + extra_top
    offset_c = need_left + extra_left

    input_assemblies = []
    output_assemblies = []
    for spec, (pr, pc) in zip(specs, placements):
        final_cr = offset_r + pr + spec["final_cont_r"]
        final_cc = offset_c + pc + spec["final_cont_c"]
        dr, dc = SIDE_TO_DIR[spec["head_side"]]
        input_cr = final_cr - dr * spec["steps"]
        input_cc = final_cc - dc * spec["steps"]

        input_assemblies.append(
            _build_assembly_cells(
                spec,
                int(input_cr),
                int(input_cc),
                int(spec["initial_orange"]),
                int(spec["travel"]),
            )
        )
        output_assemblies.append(
            _build_assembly_cells(
                spec,
                int(final_cr),
                int(final_cc),
                int(spec["initial_orange"] + spec["steps"]),
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

    shrink_h = int(grid.shape[0] - expected.shape[0])
    shrink_w = int(grid.shape[1] - expected.shape[1])
    if shrink_h <= 0 and shrink_w <= 0:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid tight-crop variant after 400 attempts")
