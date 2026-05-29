# Combo variant of 271d71e2 (index 3)
# combo_spec: D2/2+D7/2
# Name: 碰撞停止遇障碍物

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
Design analysis for combo D2/2 + D7/2
=====================================

a) Concrete behavior and trace
- Kept from the base task: magenta background; black rectangular containers whose interiors are
  grey/orange; a maroon head line attached to one side; an equal maroon destination line farther
  along the motion axis.
- Base rule: `steps = min(head_dest_distance, grey_count)`. Grey count capped both motion and
  recoloring.
- Combo rule:
  `travel_steps = min(dest_steps, first_container_collision, first_obstacle_collision)`.
  `charge_steps = min(travel_steps, grey_count)`.
- Example, moving right:
  head-to-dest distance is 9, grey count is 3, a 2x2 obstacle would be hit after 4 moves, and a
  stationary container would be hit after 7 moves. The base task would move 3. The combo moves 4
  and recolors 3 cells: motion is no longer capped by grey, but it still stops before the
  obstacle because that collision happens before the blocker or the destination.

b) Complexity calibration
- The generator always includes two scene types in one grid:
  1. a fused scene with one mover, one stationary blocker container, and one obstacle in the same
     corridor; whichever of obstacle/container comes first is the true stop reason;
  2. a control scene with no blocker/obstacle where the mover travels farther than its grey count.
- This prevents degeneration into either single variant alone:
  the combo scene demonstrates collision reasoning, while the control scene demonstrates that grey
  is no longer the movement cap.
- Obstacle shape is fixed to a visible 2x2 block of color 8, keeping the rule learnable.

c) Hidden assumption audit against the original solver
- Original assumption: all non-bg/non-border/non-(5,7) colors are head/dest markers.
  Combo change: color 8 is now a non-marker obstacle, so it must be excluded from marker search.
- Original assumption: `steps = min(dist, len(grey_coords))`.
  Combo change: translation and recoloring must be split into `travel_steps` and `charge_steps`.
- Original assumption: path contains only background until the destination.
  Combo change: movement must scan for collisions with obstacles and with other container boxes.
- Original assumption: per-container independent stamping is safe.
  Combo compatibility rule: fully orange containers (`grey_count = 0`) are treated as stationary.
  The generator relies on that to make blocker containers unambiguous and non-order-dependent.

d) Placement legality
- The fused scene is built as an isolated patch with explicit coordinates for mover, blocker,
  obstacle, and both sets of markers. The final grid packs that patch plus a control patch with a
  1-cell background moat between patches.
- The blocker is placed on the mover's axis so that the moved rectangle/head would overlap it on
  the next step after the intended stop.
- The obstacle sits inside the mover's swept corridor, never on the initial pose, never on the
  destination line, and never overlapping the blocker.

e) Visual observability
- In the fused scene the maroon destination stays farther away, yet the output head stops before it
  because either a teal 2x2 block or another container is encountered first.
- In the control scene the mover clearly travels farther than the number of grey cells it had, so
  the old base rule is contradicted in the same example set.
- The human-visible cues are structural only: spatial ordering of destination, obstacle, and
  blocker; no hidden numeric ordering or color-id trick is used.

f) Visual cleanliness
- Each blocker container uses a perpendicular head/destination orientation, so its own markers do
  not lie on the mover's axis and are less likely to be misread as the mover's destination.
- Only one fused corridor appears per grid, avoiding chains of collisions.
- Independent scenes are separated by background and validated against the solver.

g) Change inferability
- The only new color is 8 for obstacles, used nowhere else.
- Orange growth keeps the original head-side ordering exactly, so the new inference burden is only:
  "movement continues until the earliest of destination, blocker, obstacle; recoloring follows the
  realized movement but cannot exceed available grey cells."

h) New solver assumptions and generator guarantees
- Solver assumption: other container boxes are static blockers.
  Generator guarantee: only fully-orange containers are placed in the mover corridor, so that
  assumption matches all generated blocker scenes.
- Solver assumption: the nearest valid same-shaped marker in front is the intended destination.
  Generator guarantee: blocker markers are perpendicular and every sample is accepted only if the
  hand-built expected output equals `main(input)`.
- Solver assumption: fully-orange containers stay put.
  Generator guarantee: every corridor blocker has zero grey cells.
"""


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
OBSTACLE = 8
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
    for r, c in _interior_fill_order(inner_h, inner_w, head_side)[:orange_count]:
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
    for r in range(outer_h):
        for c in range(outer_w):
            rr, cc = top + r, left + c
            color = BORDER if r in (0, outer_h - 1) or c in (0, outer_w - 1) else int(interior[r - 1, c - 1])
            prev = canvas.get((rr, cc))
            if prev is not None and int(prev) != int(color):
                return False
            canvas[(rr, cc)] = int(color)
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


def _obstacle_cells_for_stop(
    head_side: int,
    cont_top: int,
    cont_left: int,
    outer_h: int,
    outer_w: int,
    stop_steps: int,
    orth_offset: int,
) -> list[tuple[int, int]]:
    if head_side == 0:
        head_r = cont_top - 1
        near_r = head_r - (stop_steps + 1)
        return [(near_r - 1, cont_left + orth_offset + dc) for dc in range(2)] + [
            (near_r, cont_left + orth_offset + dc) for dc in range(2)
        ]
    if head_side == 1:
        head_r = cont_top + outer_h
        near_r = head_r + (stop_steps + 1)
        return [(near_r, cont_left + orth_offset + dc) for dc in range(2)] + [
            (near_r + 1, cont_left + orth_offset + dc) for dc in range(2)
        ]
    if head_side == 2:
        head_c = cont_left - 1
        near_c = head_c - (stop_steps + 1)
        return [(cont_top + orth_offset + dr, near_c - 1) for dr in range(2)] + [
            (cont_top + orth_offset + dr, near_c) for dr in range(2)
        ]
    head_c = cont_left + outer_w
    near_c = head_c + (stop_steps + 1)
    return [(cont_top + orth_offset + dr, near_c) for dr in range(2)] + [
        (cont_top + orth_offset + dr, near_c + 1) for dr in range(2)
    ]


def _collision_limited_steps(
    box_: tuple[int, int, int, int],
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    max_steps: int,
    blocker_mask: np.ndarray,
) -> int:
    if max_steps <= 0:
        return 0

    H, W = blocker_mask.shape
    r0, c0, r1, c1 = box_
    moving_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    moving_cells.extend(list(head_pts))
    dr, dc = direction

    for step in range(1, int(max_steps) + 1):
        tr = int(dr * step)
        tc = int(dc * step)
        for r, c in moving_cells:
            rr = int(r + tr)
            cc = int(c + tc)
            if not (0 <= rr < H and 0 <= cc < W):
                return step - 1
            if bool(blocker_mask[rr, cc]):
                return step - 1
    return int(max_steps)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))
    obstacle_mask = g == int(OBSTACLE)

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
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE, OBSTACLE})]

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

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE), int(OBSTACLE)):
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
            blocker_mask = obstacle_mask.copy()
            for br0, bc0, br1, bc1 in all_boxes:
                if (br0, bc0, br1, bc1) == box_:
                    continue
                blocker_mask[br0 : br1 + 1, bc0 : bc1 + 1] = True
            travel_steps = _collision_limited_steps(box_, head_pts, direction, dest_steps, blocker_mask)
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


def _make_target_scene() -> dict | None:
    for _ in range(200):
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 5)
        head_side = random.randint(0, 3)
        area = inner_h * inner_w
        travel = random.randint(2, 6)
        if area < 1 or travel < 2:
            continue
        grey = random.randint(1, min(area, travel - 1))
        if grey >= travel:
            continue
        initial_orange = area - grey
        charge = grey

        outer_h, outer_w = inner_h + 2, inner_w + 2
        direction = SIDE_TO_DIR[head_side]
        input_map: dict[tuple[int, int], int] = {}
        output_map: dict[tuple[int, int], int] = {}

        interior_in = _interior_array(inner_h, inner_w, head_side, initial_orange)
        interior_out = _interior_array(inner_h, inner_w, head_side, initial_orange + charge)
        if not _paint_container(input_map, 0, 0, interior_in):
            continue
        if not _paint_container(output_map, direction[0] * travel, direction[1] * travel, interior_out):
            continue

        head = _head_cells(0, 0, outer_h, outer_w, head_side)
        dest = _translate_coords(head, direction[0], direction[1], travel)
        moved_head = _translate_coords(head, direction[0], direction[1], travel)
        if not _paint_cells(input_map, head, HEAD_COLOR):
            continue
        if not _paint_cells(input_map, dest, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, moved_head, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, dest, HEAD_COLOR):
            continue

        inp, out = _maps_to_patches(input_map, output_map)
        if max(inp.shape) > 22:
            continue
        return {
            "input": inp,
            "output": out,
            "containers": 1,
            "movers": [
                {
                    "kind": "target",
                    "travel_steps": int(travel),
                    "charge_steps": int(charge),
                    "dest_steps": int(travel),
                    "grey": int(grey),
                    "base_steps": int(grey),
                    "stop_reason": "dest",
                }
            ],
        }
    return None


def _make_combo_scene(obstacle_first: bool) -> dict | None:
    for _ in range(300):
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 4)
        head_side = random.randint(0, 3)
        area = inner_h * inner_w
        actual = random.randint(1, 4)
        if area <= actual:
            continue
        grey = random.randint(actual + 1, area)
        if grey <= actual:
            continue
        initial_orange = area - grey

        blocker_inner_h = random.randint(1, 5)
        blocker_inner_w = random.randint(1, 4)
        blocker_area = blocker_inner_h * blocker_inner_w

        outer_h, outer_w = inner_h + 2, inner_w + 2
        blocker_outer_h, blocker_outer_w = blocker_inner_h + 2, blocker_inner_w + 2
        direction = SIDE_TO_DIR[head_side]
        blocker_head_side = 0 if head_side in (2, 3) else 2
        blocker_dir = SIDE_TO_DIR[blocker_head_side]
        blocker_dest_steps = random.randint(2, 4)

        blocker_axis_extent = blocker_outer_w if head_side in (2, 3) else blocker_outer_h
        orth_span = outer_w if head_side in (0, 1) else outer_h
        if orth_span < 2:
            continue
        orth_offset = random.randint(0, orth_span - 2)

        if obstacle_first:
            obstacle_stop = actual
            blocker_stop = actual + random.randint(2, 4)
            stop_reason = "obstacle"
        else:
            blocker_stop = actual
            obstacle_stop = blocker_stop + blocker_axis_extent + random.randint(1, 3)
            stop_reason = "blocker"

        mover_top = 0
        mover_left = 0
        if head_side == 3:
            blocker_top, blocker_left = 0, outer_w + blocker_stop + 1
        elif head_side == 2:
            blocker_top, blocker_left = 0, -(blocker_outer_w + blocker_stop + 1)
        elif head_side == 1:
            blocker_top, blocker_left = outer_h + blocker_stop + 1, 0
        else:
            blocker_top, blocker_left = -(blocker_outer_h + blocker_stop + 1), 0

        dest_steps = max(
            blocker_stop + blocker_axis_extent + 1 + random.randint(1, 3),
            obstacle_stop + 3 + random.randint(1, 3),
        )
        if dest_steps <= actual:
            continue

        mover_head = _head_cells(mover_top, mover_left, outer_h, outer_w, head_side)
        mover_dest = _translate_coords(mover_head, direction[0], direction[1], dest_steps)
        moved_head = _translate_coords(mover_head, direction[0], direction[1], actual)

        blocker_head = _head_cells(blocker_top, blocker_left, blocker_outer_h, blocker_outer_w, blocker_head_side)
        blocker_dest = _translate_coords(blocker_head, blocker_dir[0], blocker_dir[1], blocker_dest_steps)
        obstacle_cells = _obstacle_cells_for_stop(
            head_side,
            mover_top,
            mover_left,
            outer_h,
            outer_w,
            obstacle_stop,
            orth_offset,
        )

        input_map: dict[tuple[int, int], int] = {}
        output_map: dict[tuple[int, int], int] = {}

        mover_interior_in = _interior_array(inner_h, inner_w, head_side, initial_orange)
        mover_interior_out = _interior_array(inner_h, inner_w, head_side, initial_orange + actual)
        blocker_interior = _interior_array(blocker_inner_h, blocker_inner_w, blocker_head_side, blocker_area)

        if not _paint_container(input_map, mover_top, mover_left, mover_interior_in):
            continue
        if not _paint_container(output_map, mover_top + direction[0] * actual, mover_left + direction[1] * actual, mover_interior_out):
            continue
        if not _paint_container(input_map, blocker_top, blocker_left, blocker_interior):
            continue
        if not _paint_container(output_map, blocker_top, blocker_left, blocker_interior):
            continue

        if not _paint_cells(input_map, mover_head, HEAD_COLOR):
            continue
        if not _paint_cells(input_map, mover_dest, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, moved_head, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, mover_dest, HEAD_COLOR):
            continue

        if not _paint_cells(input_map, blocker_head, HEAD_COLOR):
            continue
        if not _paint_cells(input_map, blocker_dest, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, blocker_head, HEAD_COLOR):
            continue
        if not _paint_cells(output_map, blocker_dest, HEAD_COLOR):
            continue

        if not _paint_cells(input_map, obstacle_cells, OBSTACLE):
            continue
        if not _paint_cells(output_map, obstacle_cells, OBSTACLE):
            continue

        inp, out = _maps_to_patches(input_map, output_map)
        if max(inp.shape) > 22:
            continue
        return {
            "input": inp,
            "output": out,
            "containers": 2,
            "movers": [
                {
                    "kind": "combo",
                    "travel_steps": int(actual),
                    "charge_steps": int(actual),
                    "dest_steps": int(dest_steps),
                    "grey": int(grey),
                    "base_steps": int(min(dest_steps, grey)),
                    "stop_reason": stop_reason,
                    "obstacle_stop": int(obstacle_stop),
                    "blocker_stop": int(blocker_stop),
                }
            ],
        }
    return None


def _gen_attempt(return_meta: bool = False):
    combo_scene = _make_combo_scene(obstacle_first=bool(random.getrandbits(1)))
    target_scene = _make_target_scene()
    if combo_scene is None or target_scene is None:
        return None

    scenes = [combo_scene, target_scene]
    total_containers = sum(scene["containers"] for scene in scenes)
    if total_containers != 3:
        return None

    order = sorted(range(len(scenes)), key=lambda i: int(scenes[i]["input"].shape[0] * scenes[i]["input"].shape[1]), reverse=True)
    scenes = [scenes[i] for i in order]

    total_area = sum(scene["input"].shape[0] * scene["input"].shape[1] for scene in scenes)
    max_h = max(scene["input"].shape[0] for scene in scenes)
    max_w = max(scene["input"].shape[1] for scene in scenes)

    min_side = max(10, int(total_area ** 0.5) + 3)
    lo_h = max(10, max_h + 2)
    lo_w = max(10, max_w + 2)
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
        if ph > grid_h or pw > grid_w:
            return None
        placed = False
        for _ in range(400):
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
    movers_meta = []
    for scene, (r0, c0) in zip(scenes, placements):
        ph, pw = scene["input"].shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = scene["input"]
        expected[r0 : r0 + ph, c0 : c0 + pw] = scene["output"]
        movers_meta.extend(scene["movers"])

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    combo_meta = next((m for m in movers_meta if m["kind"] == "combo"), None)
    target_meta = next((m for m in movers_meta if m["kind"] == "target"), None)
    if combo_meta is None or target_meta is None:
        return None

    if not (combo_meta["travel_steps"] < combo_meta["base_steps"]):
        return None
    if not (target_meta["travel_steps"] > target_meta["base_steps"]):
        return None

    meta = {
        "containers": int(total_containers),
        "combo": combo_meta,
        "target": target_meta,
    }
    if return_meta:
        return grid, meta
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 400 attempts")
