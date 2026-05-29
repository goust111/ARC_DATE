# Combo variant of 271d71e2 (index 24)
# combo_spec: D2/2+D7/2+D12/0
# Name: 碰撞停止障碍物排序处理

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
- Base task: each bordered container independently moves toward its same-colored
  destination line by `min(head_to_dest_distance, grey_count)`.
- Combo behavior:
  1. `D2/2`: movement distance is no longer capped by grey count. Each mover
     tries to keep going until its head reaches the destination line.
  2. `D7/2`: static obstacle blocks (color 8) sit in the corridor and stop a
     mover one step before overlap.
  3. `D12/0`: movers are applied in ascending visible head-to-destination
     distance. Earlier movers are written onto the canvas before later movers
     probe their own motion, so an earlier stopped container becomes a new
     blocker for a later one.
- Concrete traced scene used by the generator:
  A horizontal mover on the left heads right but stops at a teal 2x2 obstacle.
  A vertical mover above has a longer destination distance, so it is processed
  second. Its input corridor is clear, but after A has moved, B's next step
  would overlap A's new box, so B also stops early. Grey->orange charging stays
  tied to realized movement steps, from the head-facing side inward.

b) Complexity calibration
- The generator always includes both fused effects in one scene:
  one mover blocked by a teal obstacle and one later mover blocked by the first
  mover's new position.
- Head/destination colors are chosen from a small repeated palette
  `{1,2,3,4,9}`. This is a support change to keep pair matching unambiguous
  once trajectories cross; it avoids accidental "which line belongs to which
  box?" ambiguity without adding a hidden variable.
- Only one ordering rule is used: ascending visible distance. Ties are avoided.

c) Hidden-assumption audit against the base solver
- Base line `steps = min(distance, grey_count)` is invalid now. The combo must
  split movement into:
  `base_distance` (target reach if unobstructed) and `charge_steps`
  (`min(actual_steps, grey_count)`).
- Base independent pasting is invalid. Later movers must inspect the current
  occupancy after earlier movers have already been applied.
- Base head-color detection treated every non-bg/non-border/non-grey/non-orange
  color as a possible signal. With D7, color 8 is no longer a head/destination
  color and must be excluded.
- Base code assumed final placement is safe once distance is chosen. Here every
  step must be tested against the live occupancy because blockers may be static
  obstacles, destination lines of other movers, or earlier moved containers.

d) Placement legality in the generator
- The generator constructs one deliberate two-mover scene with strict formulas
  so all cells remain in-bounds.
- Input objects are separated by background; B's input corridor is clear, and
  the A obstacle is placed only in A's corridor.
- The generator validates each candidate by comparing the hand-built ordered
  simulator against `main(input)` and rejects any ambiguous or degenerate scene.

e) Visual observability
- A human can see the teal obstacle directly in A's path and see A stop before
  the destination.
- A human can also see that B's path had no obstacle in the input yet B still
  stops after A has moved, which exposes the ordering dependency.
- Distance order is visually inferable because A's destination is nearer than
  B's in every generated scene.

f) Visual cleanliness
- The scene uses one horizontal mover, one vertical mover, and one teal block.
  This is enough to show the full combo without crowding the grid.
- Optional flip / transpose symmetries vary appearance while preserving the same
  readable structure.

g) Change inferability
- No new color remapping is introduced beyond the explicit obstacle color 8.
- Grey->orange still follows the original head-side order, so the only new
  burden is learning the stopping logic, not a new recoloring pattern.

h) New solver assumptions introduced here
- Ordered interaction matters only because earlier movers are written before
  later ones probe movement.
- A mover's head may overlap its own destination line, but its body may not
  overlap any non-background cell.
- Teal cells are pure obstacles and never participate in head/destination
  matching.

i) Generator guarantees for those assumptions
- Distances are strictly ordered (`A_distance < B_distance`).
- A is guaranteed obstacle-limited and B is guaranteed earlier-mover-limited.
- Each mover has at least as many grey cells as realized steps, so the visible
  orange growth equals realized movement steps exactly as described.

j) Main-loop assumption audit
- `_collect_items` assumes each container has exactly one nearest same-colored
  line component adjacent to its border that can be separated into head versus
  destination by border distance. The generator enforces this by using one
  signal color per mover and keeping unrelated signal lines disjoint.
- Destination matching assumes the valid mate lies collinear with the inferred
  direction and is the nearest same-shaped component in that direction. The
  generator keeps each mover's corridor isolated so no second valid mate exists.
- `_simulate_ordered` assumes every mover is processed once from the original
  input geometry, with order determined only by visible head-to-destination
  distance. The generator avoids distance ties in the interacting pair.
- `_step_blocker_kind` assumes any non-background occupancy encountered by the
  shifted body or head is a real blocker, except that the head may land on its
  own destination cells. The generator keeps body cells from ever needing to
  overlap destination lines and keeps obstacle color 8 out of signal matching.
- Ordered interaction also assumes earlier movers can change later paths without
  themselves moving again. The generator constructs one monotone collision chain:
  obstacle -> first mover stop -> second mover stop.
"""

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
OBSTACLE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}
SIGNAL_COLORS = [1, 2, 3, 4, 9]


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _fill_order_local(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _dominant_bg(grid: np.ndarray) -> int:
    cnt = Counter(map(int, grid.flatten()))
    return int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))


def _collect_items(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _dominant_bg(g)

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

    all_signal_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE, OBSTACLE})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_signal_colors:
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

    items = []
    for idx, cont in enumerate(containers):
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        head_info = pick_head_for_container(box_)
        if head_info is None or head_info["dir"] == (0, 0):
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = tuple(head_info["dir"])

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape or d["pts"] == comp_pts:
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

        best_dest = None
        best_dist = None
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
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = d
        if best_dest is None or best_dist is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        head_side = DIR_TO_SIDE[direction]
        grey_local = {
            (1 + rr, 1 + cc)
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        }
        grey_order_local = [
            (1 + rr, 1 + cc)
            for rr, cc in _fill_order_local(inner.shape[0], inner.shape[1], head_side)
            if (1 + rr, 1 + cc) in grey_local
        ]

        items.append(
            {
                "id": idx,
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "color": hc,
                "dir": direction,
                "head_pts": head_pts,
                "dest_pts": set(best_dest["pts"]),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "grey_order_local": grey_order_local,
                "grey_count": int(len(grey_order_local)),
                "base_distance": int(best_dist),
                "actual_steps": 0,
                "charge_steps": 0,
                "stop_reason": "none",
                "blocked": False,
            }
        )

    return bg, items


def _erase_item(grid: np.ndarray, bg: int, item: dict) -> None:
    r0, c0, r1, c1 = item["box"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    for r, c in item["head_pts"]:
        if int(grid[int(r), int(c)]) == int(item["color"]):
            grid[int(r), int(c)] = int(bg)


def _collision_kind(val: int) -> str:
    if int(val) == int(OBSTACLE):
        return "obstacle"
    return "container"


def _step_blocker_kind(occupancy: np.ndarray, bg: int, item: dict, step: int) -> str | None:
    H, W = occupancy.shape
    dr, dc = item["dir"]
    tr = int(dr * step)
    tc = int(dc * step)
    r0, c0, r1, c1 = item["box"]
    nr0, nc0, nr1, nc1 = int(r0 + tr), int(c0 + tc), int(r1 + tr), int(c1 + tc)
    if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
        return "wall"

    blocker = None
    for rr in range(nr0, nr1 + 1):
        for cc in range(nc0, nc1 + 1):
            val = int(occupancy[rr, cc])
            if val != int(bg):
                kind = _collision_kind(val)
                if kind == "obstacle":
                    return kind
                blocker = kind

    for r, c in item["head_pts"]:
        rr = int(r + tr)
        cc = int(c + tc)
        if rr < 0 or cc < 0 or rr >= H or cc >= W:
            return "wall"
        if (rr, cc) in item["dest_pts"]:
            continue
        val = int(occupancy[rr, cc])
        if val != int(bg):
            kind = _collision_kind(val)
            if kind == "obstacle":
                return kind
            blocker = kind
    return blocker


def _apply_item(grid: np.ndarray, item: dict, steps: int) -> None:
    r0, c0, _r1, _c1 = item["box"]
    dr, dc = item["dir"]
    tr = int(dr * steps)
    tc = int(dc * steps)
    charge_steps = min(int(steps), int(len(item["grey_order_local"])))
    patch = item["patch"].copy()
    for pr, pc in item["grey_order_local"][:charge_steps]:
        if int(patch[pr, pc]) == int(GREY):
            patch[pr, pc] = int(ORANGE)
    paste(grid, patch, int(r0 + tr), int(c0 + tc), clip=False)
    for r, c in item["head_pts"]:
        grid[int(r + tr), int(c + tc)] = int(item["color"])


def _simulate_ordered(grid: np.ndarray, items: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int).copy()
    items2 = [dict(item) for item in items]
    items2.sort(key=lambda item: (int(item["base_distance"]), int(item["box"][0]), int(item["box"][1])))

    for item in items2:
        occupancy = out.copy()
        _erase_item(occupancy, bg, item)

        steps = 0
        reason = "dest"
        while steps < int(item["base_distance"]):
            blocker = _step_blocker_kind(occupancy, bg, item, steps + 1)
            if blocker is not None:
                reason = blocker
                break
            steps += 1

        item["actual_steps"] = int(steps)
        item["charge_steps"] = int(min(steps, len(item["grey_order_local"])))
        item["stop_reason"] = reason if steps < int(item["base_distance"]) else "dest"
        item["blocked"] = steps < int(item["base_distance"])
        if steps == 0:
            continue
        _erase_item(out, bg, item)
        _apply_item(out, item, steps)
    return out, items2


def _simulate_independent_steps(grid: np.ndarray, items: list[dict], bg: int) -> dict[int, dict]:
    occupancy0 = np.array(grid, dtype=int)
    result = {}
    for item in items:
        occupancy = occupancy0.copy()
        _erase_item(occupancy, bg, item)
        steps = 0
        reason = "dest"
        while steps < int(item["base_distance"]):
            blocker = _step_blocker_kind(occupancy, bg, item, steps + 1)
            if blocker is not None:
                reason = blocker
                break
            steps += 1
        result[int(item["id"])] = {
            "actual_steps": int(steps),
            "stop_reason": reason if steps < int(item["base_distance"]) else "dest",
        }
    return result


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, items = _collect_items(g)
    out, _items = _simulate_ordered(g, items, bg)
    return out


def _line_cells(spec: dict, which: str) -> list[tuple[int, int]]:
    r = int(spec["r"])
    c = int(spec["c"])
    oh = int(spec["outer_h"])
    ow = int(spec["outer_w"])
    dist = int(spec["distance"])
    side = int(spec["head_side"])
    if which == "head":
        if side == 0:
            return [(r - 1, cc) for cc in range(c, c + ow)]
        if side == 1:
            return [(r + oh, cc) for cc in range(c, c + ow)]
        if side == 2:
            return [(rr, c - 1) for rr in range(r, r + oh)]
        return [(rr, c + ow) for rr in range(r, r + oh)]
    if which == "dest":
        if side == 0:
            return [(r - 1 - dist, cc) for cc in range(c, c + ow)]
        if side == 1:
            return [(r + oh + dist, cc) for cc in range(c, c + ow)]
        if side == 2:
            return [(rr, c - 1 - dist) for rr in range(r, r + oh)]
        return [(rr, c + ow + dist) for rr in range(r, r + oh)]
    raise ValueError(which)


def _spec_non_bg_cells(spec: dict) -> set[tuple[int, int]]:
    r = int(spec["r"])
    c = int(spec["c"])
    oh = int(spec["outer_h"])
    ow = int(spec["outer_w"])
    cells = {(rr, cc) for rr in range(r, r + oh) for cc in range(c, c + ow)}
    cells.update(_line_cells(spec, "head"))
    cells.update(_line_cells(spec, "dest"))
    return cells


def _fill_interior(spec: dict) -> np.ndarray:
    ih = int(spec["outer_h"] - 2)
    iw = int(spec["outer_w"] - 2)
    arr = np.full((ih, iw), GREY, dtype=int)
    orange_count = int(spec["initial_orange"])
    order = _fill_order_local(ih, iw, int(spec["head_side"]))
    for rr, cc in order[:orange_count]:
        arr[rr, cc] = int(ORANGE)
    return arr


def _specs_to_grid(specs: list[dict], obstacles: set[tuple[int, int]], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        grid[r, c : c + ow] = int(BORDER)
        grid[r + oh - 1, c : c + ow] = int(BORDER)
        grid[r : r + oh, c] = int(BORDER)
        grid[r : r + oh, c + ow - 1] = int(BORDER)
        grid[r + 1 : r + oh - 1, c + 1 : c + ow - 1] = _fill_interior(spec)
        for rr, cc in _line_cells(spec, "head"):
            grid[rr, cc] = int(spec["color"])
        for rr, cc in _line_cells(spec, "dest"):
            grid[rr, cc] = int(spec["color"])
    for rr, cc in obstacles:
        grid[rr, cc] = int(OBSTACLE)
    return grid


def _bounds_ok(cells: set[tuple[int, int]], grid_h: int, grid_w: int) -> bool:
    return all(0 <= r < grid_h and 0 <= c < grid_w for r, c in cells)


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr, cc in b:
            if abs(int(r - rr)) <= 1 and abs(int(c - cc)) <= 1:
                return True
    return False


def _fusion_metrics(grid: np.ndarray) -> dict:
    bg, items = _collect_items(grid)
    ordered_out, ordered_items = _simulate_ordered(grid, items, bg)
    indep = _simulate_independent_steps(grid, items, bg)
    ordered_by_id = {int(item["id"]): item for item in ordered_items}
    obstacle_blocked = sum(1 for item in ordered_items if item["stop_reason"] == "obstacle")
    container_blocked = sum(1 for item in ordered_items if item["stop_reason"] == "container")
    order_loss = sum(
        int(indep[idx]["actual_steps"]) - int(ordered_by_id[idx]["actual_steps"])
        for idx in ordered_by_id
    )
    return {
        "n_items": len(ordered_items),
        "obstacle_blocked": obstacle_blocked,
        "container_blocked": container_blocked,
        "order_loss": int(order_loss),
        "actual_steps": tuple(int(item["actual_steps"]) for item in ordered_items),
        "base_distance": tuple(int(item["base_distance"]) for item in ordered_items),
        "stop_reason": tuple(str(item["stop_reason"]) for item in ordered_items),
        "output": ordered_out,
    }


def _attempt_scene() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(20, 24)
    grid_w = random.randint(18, 24)
    colors = random.sample(SIGNAL_COLORS, 2)

    outer_h_a = random.randint(4, 6)
    outer_w_a = random.randint(5, 6)
    outer_h_b = random.randint(7, 10)
    outer_w_b = 3

    stop_a = random.randint(3, 5)
    stop_b = random.randint(3, 5)
    dist_a = stop_a + random.randint(2, 4)
    dist_b = max(dist_a + random.randint(2, 4), stop_b + random.randint(2, 4))

    area_a = (outer_h_a - 2) * (outer_w_a - 2)
    area_b = (outer_h_b - 2) * (outer_w_b - 2)
    if area_a < stop_a or area_b < stop_b:
        return None

    grey_a = random.randint(stop_a, area_a)
    grey_b = random.randint(stop_b, area_b)

    c_a_max = min(
        grid_w - outer_w_a - dist_a - 1,
        grid_w - outer_w_a - outer_w_b - 1,
    )
    if c_a_max < 1:
        return None
    c_a = random.randint(1, c_a_max)
    c_b = c_a + outer_w_a + 1

    r_b = random.randint(1, 3)
    r_a = r_b + outer_h_b + stop_b + 1
    if r_a + outer_h_a > grid_h:
        return None
    if r_b + outer_h_b + dist_b > grid_h - 1:
        return None

    spec_a = {
        "r": r_a,
        "c": c_a,
        "outer_h": outer_h_a,
        "outer_w": outer_w_a,
        "head_side": 3,
        "distance": dist_a,
        "color": colors[0],
        "initial_orange": area_a - grey_a,
    }
    spec_b = {
        "r": r_b,
        "c": c_b,
        "outer_h": outer_h_b,
        "outer_w": outer_w_b,
        "head_side": 1,
        "distance": dist_b,
        "color": colors[1],
        "initial_orange": area_b - grey_b,
    }
    specs = [spec_a, spec_b]

    all_cells = [_spec_non_bg_cells(spec) for spec in specs]
    if any(not _bounds_ok(cells, grid_h, grid_w) for cells in all_cells):
        return None
    if _touching(all_cells[0], all_cells[1]):
        return None

    obs_row0 = r_a + random.randint(0, outer_h_a - 2)
    obs_col0 = c_a + outer_w_a + stop_a + 1
    obstacles = {
        (obs_row0, obs_col0),
        (obs_row0 + 1, obs_col0),
        (obs_row0, obs_col0 + 1),
        (obs_row0 + 1, obs_col0 + 1),
    }
    if not _bounds_ok(obstacles, grid_h, grid_w):
        return None
    if any(obstacles & cells for cells in all_cells):
        return None

    grid = _specs_to_grid(specs, obstacles, grid_h, grid_w)
    metrics = _fusion_metrics(grid)
    expected = metrics["output"]

    if metrics["n_items"] != 2:
        return None
    if metrics["obstacle_blocked"] < 1 or metrics["container_blocked"] < 1:
        return None
    if metrics["order_loss"] <= 0:
        return None

    steps = metrics["actual_steps"]
    reasons = metrics["stop_reason"]
    if set(steps) != {int(stop_a), int(stop_b)}:
        return None
    if reasons[0] != "obstacle" or reasons[1] != "container":
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid, expected


def _apply_symmetry(grid: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choice = random.choice(["id", "flip_h", "flip_v", "transpose"])
    if choice == "flip_h":
        return np.fliplr(grid), np.fliplr(out)
    if choice == "flip_v":
        return np.flipud(grid), np.flipud(out)
    if choice == "transpose":
        return np.transpose(grid), np.transpose(out)
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _attempt_scene()
        if built is None:
            continue
        grid, expected = built
        grid2, expected2 = _apply_symmetry(grid, expected)
        try:
            output = main(grid2)
        except Exception:
            continue
        if not np.array_equal(output, expected2):
            continue
        metrics = _fusion_metrics(grid2)
        if metrics["obstacle_blocked"] < 1 or metrics["container_blocked"] < 1:
            continue
        if metrics["order_loss"] <= 0:
            continue
        return grid2
    raise RuntimeError("Failed to generate valid combo variant after 400 attempts")
