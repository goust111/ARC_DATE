# Variant of 271d71e2
# Dimension: D2 — 移动距离编码方式 (dst)
# Change: distance_encoding — 步数 = min(头到目标距离, 灰色像素数)，两个约束取较小值... → 碰撞停止

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

"""
Design analysis for the D2 variant
=================================

Reference-task note
- The prompt points to source task `195c6913` for the collision-stop idea.
- I searched the workspace for a solver file for that task but only found the ARC JSON examples,
  not a local solver implementation, so the solver-code reading step is skipped because the file
  is unavailable in this environment.

a) Concrete behavior and flow impact
- Visual objects remain the same as the original task: magenta background, black rectangular
  containers with grey/orange interiors, a maroon head line attached to one side, and a same-shaped
  maroon destination line farther along the movement axis.
- The changed part is only how far a container moves.
- Original rule at the critical line:
  `steps = min(distance_to_dest, grey_count)`.
- New rule:
  `travel_steps = min(distance_to_dest, distance_to_first_blocking_container)`.
  Grey cells still turn orange from the head-facing side inward, but only
  `min(travel_steps, grey_count)` of them change. Grey no longer caps translation.
- Small example, moving right:
  input head at col 5, destination at col 11, so destination distance is 6.
  Another container begins at col 9, leaving 3 empty cells between head and blocker.
  Original: if grey_count = 2, movement is 2.
  Variant: movement is 3, because the mover keeps going until its head becomes adjacent to the
  blocking container. Only 2 grey cells turn orange.

b) Complexity calibration
- If every mover simply reached its destination, the variant would often look like "same task but
  farther", which is too weak.
- If blockers were allowed to be moving containers too, the rule would become order-dependent and
  visually messy.
- So the generator uses two repeated, learnable branches:
  `target_stop`: no blocker in front; the mover reaches the destination line.
  `blocker_stop`: a fully-orange stationary container sits before the destination; the mover stops
  adjacent to that container instead.
- Most movers are generated with `travel_steps > grey_count`, so the new D2 rule is visibly needed.

c) Hidden assumptions from the original solver that must change
- The original `steps = min(distance, len(grey_coords))` line assumes grey_count is a movement cap.
  This is false in the variant and must be split into:
  `travel_steps` for translation and `charge_steps` for recoloring.
- The original solver never looked at other containers as obstacles. The variant requires a new
  forward scan over other container boxes whose perpendicular spans overlap the mover.
- The solver still chooses the destination using the original head/dest matching logic. This logic
  assumes the intended destination is the nearest same-shaped maroon component in front. The
  generator therefore keeps blocker containers' own head/dest markers perpendicular to the mover,
  so they cannot be misread as its destination.
- The solver applies moves independently from the input grid. That would be unsafe if blockers
  moved too. To keep the D2 change isolated and the examples legible, the variant preserves one
  compatibility rule from the original task: fully-orange containers (`grey_count = 0`) remain
  stationary decoys. The generator uses those decoys as blockers.

d) Placement legality
- Each scene is built as an isolated patch and later packed into the grid with a 1-cell background
- margin, so unrelated scenes never overlap.
- In blocker scenes, the blocker is aligned with the mover along the movement axis so collision is
  unambiguous, and the blocker's own head/destination are placed on a perpendicular side.
- The output of each scene always fits inside that scene's input bounding box, because the mover
  only travels toward markers or blockers that are already inside the patch.

e) Visual observability
- The variant is visible in two distinct ways:
  1. many movers travel farther than their number of grey cells would allow in the original task;
  2. some movers clearly stop short of the destination because another container blocks the path.
- Both branches can be inferred from a small number of examples because the blocker sits directly
  in the corridor and the moved head ends flush against it.

f) Visual cleanliness
- Blocking scenes are limited to one mover plus one stationary blocker, avoiding chains of
  collisions.
- Independent scenes are separated by background, so the grid stays readable.

g) Color-change inferability
- No new colors are introduced.
- Orange growth still follows the same head-side ordering as the original task, so the only new
  inference burden is the stopping condition, not a new color rule.

h) New solver assumptions introduced by this variant
- A blocking container must already exist in the input, remain stationary, and overlap the mover on
  the perpendicular axis. The generator enforces this explicitly by making blockers fully orange.
- Destination matching still relies on identical head/destination line shape and same-color
  markers. The generator validates every candidate by comparing its hand-built expected output to
  `main(input)` and rejects ambiguous placements.
"""


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


def _head_cells(top: int, left: int, outer_h: int, outer_w: int, head_side: int) -> list[tuple[int, int]]:
    if head_side == 0:
        return [(top - 1, left + c) for c in range(outer_w)]
    if head_side == 1:
        return [(top + outer_h, left + c) for c in range(outer_w)]
    if head_side == 2:
        return [(top + r, left - 1) for r in range(outer_h)]
    return [(top + r, left + outer_w) for r in range(outer_h)]


def _translate_coords(coords: list[tuple[int, int]], dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(r + dr * steps, c + dc * steps) for r, c in coords]


def _interior_array(inner_h: int, inner_w: int, head_side: int, orange_count: int) -> np.ndarray:
    arr = np.full((inner_h, inner_w), GREY, dtype=int)
    order = _interior_fill_order(inner_h, inner_w, head_side)
    for r, c in order[:orange_count]:
        arr[r, c] = ORANGE
    return arr


def _paint_cells(canvas: dict[tuple[int, int], int], coords: list[tuple[int, int]], color: int) -> bool:
    for pos in coords:
        prev = canvas.get(pos)
        if prev is not None and int(prev) != int(color):
            return False
        canvas[pos] = int(color)
    return True


def _paint_container(
    canvas: dict[tuple[int, int], int],
    top: int,
    left: int,
    interior: np.ndarray,
) -> bool:
    inner_h, inner_w = interior.shape
    outer_h, outer_w = inner_h + 2, inner_w + 2
    coords = []
    colors = []
    for r in range(outer_h):
        for c in range(outer_w):
            rr, cc = top + r, left + c
            if r == 0 or r == outer_h - 1 or c == 0 or c == outer_w - 1:
                coords.append((rr, cc))
                colors.append(BORDER)
            else:
                coords.append((rr, cc))
                colors.append(int(interior[r - 1, c - 1]))
    for pos, color in zip(coords, colors):
        prev = canvas.get(pos)
        if prev is not None and int(prev) != int(color):
            return False
        canvas[pos] = int(color)
    return True


def _maps_to_patches(
    input_map: dict[tuple[int, int], int],
    output_map: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    keys = set(input_map) | set(output_map)
    min_r = min(r for r, _ in keys)
    max_r = max(r for r, _ in keys)
    min_c = min(c for _, c in keys)
    max_c = max(c for _, c in keys)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for (r, c), color in input_map.items():
        inp[r - min_r, c - min_c] = int(color)
    for (r, c), color in output_map.items():
        out[r - min_r, c - min_c] = int(color)
    return inp, out


def _collision_stop_steps(
    box_: tuple[int, int, int, int],
    direction: tuple[int, int],
    dest_steps: int,
    other_boxes: list[tuple[int, int, int, int]],
) -> int:
    r0, c0, r1, c1 = box_
    dr, dc = direction
    blocker_steps = None

    for br0, bc0, br1, bc1 in other_boxes:
        if direction == (0, 1):
            if max(r0, br0) > min(r1, br1):
                continue
            free = int(bc0 - (c1 + 1) - 1)
        elif direction == (0, -1):
            if max(r0, br0) > min(r1, br1):
                continue
            free = int((c0 - 1) - bc1 - 1)
        elif direction == (1, 0):
            if max(c0, bc0) > min(c1, bc1):
                continue
            free = int(br0 - (r1 + 1) - 1)
        else:
            if max(c0, bc0) > min(c1, bc1):
                continue
            free = int((r0 - 1) - br1 - 1)

        if free < 0:
            continue
        if blocker_steps is None or free < blocker_steps:
            blocker_steps = int(free)

    if blocker_steps is None:
        return int(dest_steps)
    return int(min(dest_steps, blocker_steps))


def main(grid: np.ndarray) -> np.ndarray:
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
            r0, c0, r1, c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_):
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

    all_boxes = [tuple(cont["box"]) for cont in containers]

    for cont in containers:
        box_ = tuple(cont["box"])
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

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        if not grey_coords:
            travel_steps = 0
            charge_steps = 0
        else:
            tr0 = int(best_dest["tl"][0] - head_tl[0])
            tc0 = int(best_dest["tl"][1] - head_tl[1])
            dest_steps = abs(tr0) + abs(tc0)
            other_boxes = [b for b in all_boxes if tuple(b) != box_]
            travel_steps = _collision_stop_steps(box_, direction, dest_steps, other_boxes)
            charge_steps = min(int(travel_steps), int(len(grey_coords)))

        dr, dc = direction
        tr = int(dr * travel_steps)
        tc = int(dc * travel_steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:charge_steps])

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


def _make_target_scene(force_diff: bool) -> dict | None:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    head_side = random.randint(0, 3)
    area = inner_h * inner_w
    travel = random.randint(2, 6)
    if force_diff:
        grey = random.randint(1, min(area, travel - 1))
    else:
        grey = random.randint(1, area)
    initial_orange = area - grey
    charge = min(travel, grey)

    outer_h, outer_w = inner_h + 2, inner_w + 2
    direction = SIDE_TO_DIR[head_side]
    input_map: dict[tuple[int, int], int] = {}
    output_map: dict[tuple[int, int], int] = {}

    interior_in = _interior_array(inner_h, inner_w, head_side, initial_orange)
    interior_out = _interior_array(inner_h, inner_w, head_side, initial_orange + charge)
    if not _paint_container(input_map, 0, 0, interior_in):
        return None
    if not _paint_container(output_map, direction[0] * travel, direction[1] * travel, interior_out):
        return None

    head = _head_cells(0, 0, outer_h, outer_w, head_side)
    dest = _translate_coords(head, direction[0], direction[1], travel)
    if not _paint_cells(input_map, head, HEAD_COLOR):
        return None
    if not _paint_cells(input_map, dest, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, dest, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, dest, HEAD_COLOR):
        return None

    inp, out = _maps_to_patches(input_map, output_map)
    return {
        "input": inp,
        "output": out,
        "containers": 1,
        "movers": [
            {
                "kind": "target",
                "travel_steps": int(travel),
                "grey": int(grey),
                "old_steps": int(min(travel, grey)),
                "dest_steps": int(travel),
            }
        ],
    }


def _make_decoy_scene() -> dict | None:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    head_side = random.randint(0, 3)
    area = inner_h * inner_w
    outer_h, outer_w = inner_h + 2, inner_w + 2
    direction = SIDE_TO_DIR[head_side]
    dest_steps = random.randint(2, 5)

    input_map: dict[tuple[int, int], int] = {}
    output_map: dict[tuple[int, int], int] = {}
    interior = _interior_array(inner_h, inner_w, head_side, area)

    if not _paint_container(input_map, 0, 0, interior):
        return None
    if not _paint_container(output_map, 0, 0, interior):
        return None

    head = _head_cells(0, 0, outer_h, outer_w, head_side)
    dest = _translate_coords(head, direction[0], direction[1], dest_steps)
    if not _paint_cells(input_map, head, HEAD_COLOR):
        return None
    if not _paint_cells(input_map, dest, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, head, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, dest, HEAD_COLOR):
        return None

    inp, out = _maps_to_patches(input_map, output_map)
    return {"input": inp, "output": out, "containers": 1, "movers": []}


def _make_blocker_scene(force_diff: bool) -> dict | None:
    inner_h = random.randint(1, 6)
    inner_w = random.randint(1, 4)
    head_side = random.randint(0, 3)
    area = inner_h * inner_w
    travel = random.randint(2, 5)
    if force_diff:
        grey = random.randint(1, min(area, travel - 1))
    else:
        grey = random.randint(1, area)
    initial_orange = area - grey
    charge = min(travel, grey)

    blocker_inner_h = random.randint(1, 6)
    blocker_inner_w = random.randint(1, 4)
    blocker_area = blocker_inner_h * blocker_inner_w

    outer_h, outer_w = inner_h + 2, inner_w + 2
    blocker_outer_h, blocker_outer_w = blocker_inner_h + 2, blocker_inner_w + 2
    direction = SIDE_TO_DIR[head_side]
    tail_gap = random.randint(1, 3)
    blocker_head_side = 0 if head_side in (2, 3) else 2
    blocker_dir = SIDE_TO_DIR[blocker_head_side]
    blocker_dest_steps = random.randint(2, 4)

    mover_top = 0
    mover_left = 0
    if head_side == 3:
        blocker_top, blocker_left = 0, outer_w + travel + 1
        axis_extent = blocker_outer_w
    elif head_side == 2:
        blocker_top, blocker_left = 0, -(blocker_outer_w + travel + 1)
        axis_extent = blocker_outer_w
    elif head_side == 1:
        blocker_top, blocker_left = outer_h + travel + 1, 0
        axis_extent = blocker_outer_h
    else:
        blocker_top, blocker_left = -(blocker_outer_h + travel + 1), 0
        axis_extent = blocker_outer_h

    dest_steps = travel + axis_extent + 1 + tail_gap
    mover_head = _head_cells(mover_top, mover_left, outer_h, outer_w, head_side)
    mover_dest = _translate_coords(mover_head, direction[0], direction[1], dest_steps)

    blocker_head = _head_cells(blocker_top, blocker_left, blocker_outer_h, blocker_outer_w, blocker_head_side)
    blocker_dest = _translate_coords(blocker_head, blocker_dir[0], blocker_dir[1], blocker_dest_steps)

    input_map: dict[tuple[int, int], int] = {}
    output_map: dict[tuple[int, int], int] = {}

    mover_interior_in = _interior_array(inner_h, inner_w, head_side, initial_orange)
    mover_interior_out = _interior_array(inner_h, inner_w, head_side, initial_orange + charge)
    blocker_interior = _interior_array(blocker_inner_h, blocker_inner_w, blocker_head_side, blocker_area)

    if not _paint_container(input_map, mover_top, mover_left, mover_interior_in):
        return None
    if not _paint_container(output_map, mover_top + direction[0] * travel, mover_left + direction[1] * travel, mover_interior_out):
        return None

    if not _paint_container(input_map, blocker_top, blocker_left, blocker_interior):
        return None
    if not _paint_container(output_map, blocker_top, blocker_left, blocker_interior):
        return None

    if not _paint_cells(input_map, mover_head, HEAD_COLOR):
        return None
    if not _paint_cells(input_map, mover_dest, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, _translate_coords(mover_head, direction[0], direction[1], travel), HEAD_COLOR):
        return None
    if not _paint_cells(output_map, mover_dest, HEAD_COLOR):
        return None

    if not _paint_cells(input_map, blocker_head, HEAD_COLOR):
        return None
    if not _paint_cells(input_map, blocker_dest, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, blocker_head, HEAD_COLOR):
        return None
    if not _paint_cells(output_map, blocker_dest, HEAD_COLOR):
        return None

    inp, out = _maps_to_patches(input_map, output_map)
    return {
        "input": inp,
        "output": out,
        "containers": 2,
        "movers": [
            {
                "kind": "blocker",
                "travel_steps": int(travel),
                "grey": int(grey),
                "old_steps": int(min(dest_steps, grey)),
                "dest_steps": int(dest_steps),
            }
        ],
    }


def _choose_scene_plan() -> list[str]:
    plans = [
        ["blocker"],
        ["target", "target"],
        ["blocker", "target"],
        ["blocker", "decoy"],
        ["target", "target", "decoy"],
        ["blocker", "target", "decoy"],
        ["blocker", "blocker"],
    ]
    weights = [4, 1, 4, 2, 1, 3, 2]
    return random.choices(plans, weights=weights, k=1)[0]


def _gen_attempt(return_meta: bool = False):
    plan = _choose_scene_plan()
    scenes = []
    movers_meta = []

    for kind in plan:
        force_diff = random.random() < 0.85
        if kind == "target":
            scene = _make_target_scene(force_diff=force_diff)
        elif kind == "blocker":
            scene = _make_blocker_scene(force_diff=force_diff)
        else:
            scene = _make_decoy_scene()
        if scene is None:
            return None
        scenes.append(scene)
        movers_meta.extend(scene["movers"])

    total_containers = sum(scene["containers"] for scene in scenes)
    if not (2 <= total_containers <= 4):
        return None
    if not movers_meta:
        return None

    order = sorted(range(len(scenes)), key=lambda i: int(scenes[i]["input"].shape[0] * scenes[i]["input"].shape[1]), reverse=True)
    scenes = [scenes[i] for i in order]

    total_area = sum(scene["input"].shape[0] * scene["input"].shape[1] for scene in scenes)
    max_h = max(scene["input"].shape[0] for scene in scenes)
    max_w = max(scene["input"].shape[1] for scene in scenes)

    min_side = max(8, int(total_area ** 0.5) + 2)
    lo_h = max(8, max_h + 2)
    lo_w = max(8, max_w + 2)
    hi_h = min(26, max(lo_h, min_side + 6))
    hi_w = min(26, max(lo_w, min_side + 6))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for scene in scenes:
        ph, pw = scene["input"].shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, grid_h - ph)
            c0 = random.randint(0, grid_w - pw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + ph + 1)
            cc1 = min(grid_w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    for scene, (r0, c0) in zip(scenes, placements):
        ph, pw = scene["input"].shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = scene["input"]
        expected[r0 : r0 + ph, c0 : c0 + pw] = scene["output"]

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    meta = {
        "plan": list(plan),
        "containers": int(total_containers),
        "movers": movers_meta,
    }
    if return_meta:
        return grid, meta
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
