# Variant of 271d71e2
# Dimension: D15 — gap=0时的头-目标关系 (meta)
# Change: gap=0时头部和目标线相邻接触（形成连通分量），head_dest_distance=1... → gap值编码在容器边框厚度中

from __future__ import annotations

# Design analysis for the D15 variant
# -----------------------------------
# source_task: None, so there is no separate reference solver to inspect.
#
# a) Concrete visual change and behavior trace
# - Original task:
#   A container had a fixed 1-pixel black frame. The informative quantity was the
#   visible head→destination spacing: head_dest_distance = gap + 1.
# - Variant:
#   The visible head→destination spacing is no longer per-object information.
#   Instead, every object uses the same visible marker offset, while the black
#   frame thickness t encodes the old distance:
#       encoded_gap = t - 1
#       encoded_head_dest_distance = t
#   So a thicker border means the container moves farther, even though the
#   head/dest markers themselves are placed with the same visible spacing.
# - Small example:
#   Suppose two containers have identical grey counts >= 4 and identical marker
#   spacing, but one has border thickness 1 and the other thickness 4.
#   Original logic would move them equally because the visible head→dest distance
#   is equal. Variant logic moves them by 1 and 4 respectively because the frame
#   thickness, not the visible gap, is the distance cue.
#
# b) Complexity calibration
# - Thickness uses only 4 discrete values: 1..4. That is enough to be visually
#   obvious and repeated, without creating too many branches.
# - Head/dest marker spacing is constant across objects, which removes the old
#   cue and forces attention onto border thickness.
# - Grey-count capping is retained so the task still resembles the original.
#
# c) Hidden-assumption audit against the original main()
# - Original container detection assumed:
#     inner_bbox expanded by exactly 1 pixel gives the outer border.
#   This fails immediately once thickness varies. The variant solver therefore
#   detects the maximal uniform black frame thickness around each full grey/orange
#   interior rectangle.
# - Original movement distance used:
#     steps = min(actual head_tl to dest_tl offset, grey_count)
#   That no longer matches the task, because marker offset is deliberately held
#   constant. The variant solver must instead use the detected border thickness.
# - Original head splitting logic was safe for touch/no-touch cases and remains
#   safe here, so it is kept mostly unchanged.
#
# d) Placement legality
# - The generator builds each object as a local assembly patch whose bbox covers
#   both the input footprint and the output footprint. Global placement uses these
#   union patches, so objects cannot overlap in either state.
# - The fixed visible marker offset is chosen to be >= max thickness, so even the
#   thickest moving head reaches the destination line at most, and the container
#   body never swallows the destination marker.
#
# e) Visual observability
# - Across examples, head/dest spacing looks constant while border thickness
#   changes. Movement distance changes with thickness, which makes the new rule
#   visible from only a few examples.
# - At least one mover per generated grid is distance-capped by border thickness
#   rather than by the grey count, so the new control variable actually matters.
#
# f) Cleanliness
# - No extra symbols are added. The scene still consists of magenta background,
#   black frames, grey/orange interior, and maroon markers.
# - Thick frames remain clean rectangles, so the objects stay legible.
#
# g) Color-rule audit
# - No new color mapping is introduced. Only the original grey→orange charging
#   behavior remains.

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

MIN_THICKNESS = 1
MAX_THICKNESS = 4

# Marker top-left offset from head to destination along the movement axis.
# This is constant across all objects; only border thickness varies per object.
MARKER_DISTANCE = MAX_THICKNESS

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _line_bbox_for_container(outer_h: int, outer_w: int, head_side: int, marker_step: int) -> tuple[int, int, int, int]:
    if head_side == 0:
        row = -int(marker_step)
        return row, 0, row, outer_w - 1
    if head_side == 1:
        row = outer_h - 1 + int(marker_step)
        return row, 0, row, outer_w - 1
    if head_side == 2:
        col = -int(marker_step)
        return 0, col, outer_h - 1, col
    col = outer_w - 1 + int(marker_step)
    return 0, col, outer_h - 1, col


def _bbox_shift(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + int(dr), c0 + int(dc), r1 + int(dr), c1 + int(dc)


def _paint_bbox(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)


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


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    thickness: int,
    head_side: int,
    orange_count: int,
) -> None:
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    grid[int(top) : int(top + outer_h), int(left) : int(left + outer_w)] = int(BORDER)

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    fill_order = _interior_fill_order(inner_h, inner_w, head_side)
    for idx in range(int(orange_count)):
        rr, cc = fill_order[idx]
        interior[int(rr), int(cc)] = int(ORANGE)

    r0 = int(top + thickness)
    c0 = int(left + thickness)
    grid[r0 : r0 + inner_h, c0 : c0 + inner_w] = interior


def _draw_head_line(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, head_side: int) -> None:
    box = _line_bbox_for_container(outer_h, outer_w, head_side, 1)
    _paint_bbox(grid, _bbox_shift(box, int(top), int(left)), HEAD_COLOR)


def _max_black_frame_thickness(g: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int) -> int:
    H, W = g.shape
    thickness = 0
    while True:
        t = thickness + 1
        r0, c0 = ir0 - t, ic0 - t
        r1, c1 = ir1 + t, ic1 + t
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            break
        if all(int(g[r, c]) == int(BORDER) for r, c in rect_border_coords(r0, c0, r1, c1)):
            thickness = t
        else:
            break
    return int(thickness)


def _detect_containers(g: np.ndarray) -> list[dict]:
    containers = []
    seen = set()
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if pts.shape[0] != inner.size:
            continue
        if not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue

        thickness = _max_black_frame_thickness(g, int(ir0), int(ic0), int(ir1), int(ic1))
        if thickness < MIN_THICKNESS:
            continue

        r0, c0 = int(ir0 - thickness), int(ic0 - thickness)
        r1, c1 = int(ir1 + thickness), int(ic1 + thickness)
        outer = g[r0 : r1 + 1, c0 : c1 + 1]
        band_mask = np.ones(outer.shape, dtype=bool)
        band_mask[thickness : thickness + inner.shape[0], thickness : thickness + inner.shape[1]] = False
        if not bool((outer[band_mask] == int(BORDER)).all()):
            continue

        key = (r0, c0, r1, c1)
        if key in seen:
            continue
        seen.add(key)
        containers.append(
            {
                "outer_box": key,
                "inner_box": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "thickness": int(thickness),
            }
        )
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))
    out = g.copy()

    containers = _detect_containers(g)
    assert containers

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

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

    def pick_head_for_container(container: dict):
        r0, c0, r1, c1 = container["outer_box"]
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

    for container in containers:
        r0, c0, r1, c1 = container["outer_box"]
        ir0, ic0, ir1, ic1 = container["inner_box"]

        head_info = pick_head_for_container(container)
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
                rs = [r for r, _ in pts]
                cs = [c for _, c in pts]
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

        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        grey_coords = [
            (int(ir0 + rr), int(ic0 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(container["thickness"]), int(len(grey_coords)))

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
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _make_spec(is_decoy: bool, max_inner_h: int, max_inner_w: int) -> dict:
    thickness = random.randint(MIN_THICKNESS, MAX_THICKNESS)
    inner_h = random.randint(1, max_inner_h)
    inner_w = random.randint(1, max_inner_w)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)

    if is_decoy:
        grey = 0
        initial_orange = inner_area
        steps = 0
        cap_type = "decoy"
    else:
        prefer_distance_cap = inner_area > thickness and random.random() < 0.6
        if prefer_distance_cap:
            grey = random.randint(thickness, inner_area)
            cap_type = "distance"
        else:
            grey = random.randint(1, inner_area)
            cap_type = "grey" if grey < thickness else "distance"
        initial_orange = inner_area - grey
        steps = min(grey, thickness)

    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "inner_area": inner_area,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "thickness": thickness,
        "head_side": head_side,
        "grey": grey,
        "initial_orange": initial_orange,
        "steps": steps,
        "is_decoy": is_decoy,
        "cap_type": cap_type,
    }


def _build_assembly(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    thickness = int(spec["thickness"])
    head_side = int(spec["head_side"])
    initial_orange = int(spec["initial_orange"])
    steps = int(spec["steps"])
    dr, dc = SIDE_TO_DIR[head_side]

    input_container = (0, 0, outer_h - 1, outer_w - 1)
    input_head = _line_bbox_for_container(outer_h, outer_w, head_side, 1)
    dest = _line_bbox_for_container(outer_h, outer_w, head_side, 1 + MARKER_DISTANCE)
    output_container = _bbox_shift(input_container, dr * steps, dc * steps)
    output_head = _bbox_shift(input_head, dr * steps, dc * steps)

    boxes = [input_container, input_head, dest, output_container, output_head]
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)

    shift_r = -int(min_r)
    shift_c = -int(min_c)
    local_h = int(max_r - min_r + 1)
    local_w = int(max_c - min_c + 1)

    input_patch = np.full((local_h, local_w), BG, dtype=int)
    output_patch = np.full((local_h, local_w), BG, dtype=int)

    input_top = shift_r
    input_left = shift_c
    output_top = shift_r + dr * steps
    output_left = shift_c + dc * steps

    _paint_bbox(output_patch, _bbox_shift(dest, shift_r, shift_c), HEAD_COLOR)
    _paint_bbox(input_patch, _bbox_shift(dest, shift_r, shift_c), HEAD_COLOR)

    _draw_container(
        input_patch,
        input_top,
        input_left,
        inner_h,
        inner_w,
        thickness,
        head_side,
        initial_orange,
    )
    _draw_head_line(input_patch, input_top, input_left, outer_h, outer_w, head_side)

    _draw_container(
        output_patch,
        output_top,
        output_left,
        inner_h,
        inner_w,
        thickness,
        head_side,
        initial_orange + steps,
    )
    _draw_head_line(output_patch, output_top, output_left, outer_h, outer_w, head_side)

    return input_patch, output_patch


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    if n_containers == 2:
        max_inner_h, max_inner_w = 5, 4
    elif n_containers == 3:
        max_inner_h, max_inner_w = 4, 4
    else:
        max_inner_h, max_inner_w = 3, 4

    specs = []
    for i in range(n_containers):
        specs.append(_make_spec(is_decoy=(i >= n_movers), max_inner_h=max_inner_h, max_inner_w=max_inner_w))

    mover_thicknesses = [int(s["thickness"]) for s in specs if not s["is_decoy"]]
    if len(set(mover_thicknesses)) < min(2, len(mover_thicknesses)):
        return None
    if not any((not s["is_decoy"]) and int(s["grey"]) > int(s["thickness"]) for s in specs):
        return None

    assemblies = []
    for spec in specs:
        inp_patch, out_patch = _build_assembly(spec)
        if inp_patch.shape[0] > 26 or inp_patch.shape[1] > 26:
            return None
        assemblies.append({"spec": spec, "input": inp_patch, "output": out_patch})

    assemblies.sort(key=lambda a: int(a["input"].shape[0] * a["input"].shape[1]), reverse=True)

    total_area = sum(int(a["input"].shape[0] * a["input"].shape[1]) for a in assemblies)
    max_h = max(int(a["input"].shape[0]) for a in assemblies)
    max_w = max(int(a["input"].shape[1]) for a in assemblies)

    lo_h = max(8, max_h + 2)
    lo_w = max(8, max_w + 2)
    hi_h = min(26, max(lo_h, int(total_area ** 0.5) + 8))
    hi_w = min(26, max(lo_w, int(total_area ** 0.5) + 8))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []

    for asm in assemblies:
        h, w = asm["input"].shape
        if h > grid_h or w > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for asm, (r0, c0) in zip(assemblies, placements):
        paste(grid, asm["input"], int(r0), int(c0))
        paste(expected, asm["output"], int(r0), int(c0))

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
