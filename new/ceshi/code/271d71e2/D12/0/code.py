# Variant of 271d71e2
# Dimension: D12 — 多容器处理顺序 (multi_step)
# Change: 各容器独立处理，互不影响（路径不交叉，无顺序依赖）... → 按距离排序处理

from __future__ import annotations

# Design analysis for the D12 variant
# ==================================
# a) Concrete behavior
#    - Objects stay the same: black-bordered containers with grey/orange interiors,
#      plus a head line and a matching destination line.
#    - The changed behavior is only in multi-container scenes: containers are no
#      longer solved independently. Every movable container first computes its
#      own uncapped travel amount exactly as in the base task:
#        base_steps = min(head_to_dest_distance, grey_count).
#    - Then movers are sorted by head-to-destination distance, nearest first.
#      They are applied one at a time to the current canvas. Each move is tested
#      step-by-step; the next step is legal only if the shifted container body
#      lands on background cells and the shifted head lands on background or on
#      its own destination line. If the next step would hit an already-placed
#      earlier mover, movement stops early.
#    - Visual consequence: some later containers stop before consuming all of
#      their nominal steps because an earlier, nearer container now occupies the
#      crossing region.
#
#    Small tracked example:
#      Original logic:
#        H wants to move right 3, V wants to move down 8. Their paths cross.
#        Base task processes both independently, so H moves 3 and V also moves 8.
#      Variant logic:
#        Sort by distance: H(3) before V(8).
#        H moves right 3 and occupies the crossing area.
#        V now tries to move down one step at a time; once its next box would
#        overlap H's new box, it stops early, e.g. after 5 instead of 8.
#      Difference:
#        The late mover's final position depends on earlier movers.
#
# b) Complexity calibration
#    - To keep the order effect visible, the generator always creates at least
#      one pair of orthogonally crossing movers.
#    - To avoid matching ambiguity after introducing crossing trajectories, the
#      generator assigns distinct visible signal colors to different head/dest
#      pairs. This is only a support change; the solver still accepts arbitrary
#      non-bg/non-border colors exactly like the original solver did.
#    - Only one ordering rule is used: ascending distance. No hidden parity or
#      color-based order is introduced.
#
# c) Original assumptions that break under D12
#    - The original `for cont in containers:` loop assumes each move can read and
#      write the output independently. That is no longer valid because later
#      motion depends on earlier occupancy.
#    - The original `steps = min(distance, grey_count)` is not sufficient anymore;
#      it now gives only an upper bound. A second cap from collision is needed.
#    - The original unconditional `paste` is safe only because paths never
#      interact. Under D12 we must probe legality before placing.
#
# d) Placement legality in the generator
#    - Input scenes keep all starting objects separated and within bounds.
#    - Head-to-destination corridors are clear in the input; interaction appears
#      only after earlier movers occupy the crossing region.
#    - The generator rejects scenes unless the ordered simulator and `main`
#      agree, and unless at least one container loses steps because of an earlier
#      mover.
#
# e) Visual observability
#    - The crossing geometry makes the new rule inferable from a few examples:
#      one mover reaches a crossing first, and another stops short even though it
#      still has grey cells and a visible destination farther away.
#    - Order depends only on visible distance from head to destination.
#
# f) Visual cleanliness
#    - The generator uses one deliberate crossing pair rather than many crowded
#      intersections.
#    - Optional symmetries (flip/transpose) vary appearance without increasing
#      clutter.
#
# g) Color-change inferability
#    - Grey-to-orange conversion remains identical to the base task: one cell per
#      realized step, from the head-facing side inward.
#    - Extra signal colors are only pair labels for head/destination lines, not a
#      hidden control variable.
#
# h) New solver assumptions introduced here
#    - Destination lines are static obstacles except for the mover's own head,
#      which may overlap its own destination line.
#    - Ties in distance are resolved deterministically by reading order. The
#      generator avoids meaningful tie cases in the interacting pair.
#    - Containers are detected from the original input and each container moves
#      at most once.
#
# i) Generator guarantees for those assumptions
#    - Distances in the interacting pair are strictly ordered.
#    - Crossing scenes are constructed so the earlier mover can realize all of
#      its base steps, while the later mover is collision-limited.
#    - Distinct signal colors keep head/destination matching unambiguous.

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}
SIGNAL_COLORS = [1, 2, 3, 4, 8, 9]


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

    all_signal_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
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
        grey_local = [
            (1 + rr, 1 + cc)
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        head_side = DIR_TO_SIDE[direction]
        grey_inner_order = _fill_order_local(inner.shape[0], inner.shape[1], head_side)
        grey_order_local = [(1 + rr, 1 + cc) for rr, cc in grey_inner_order if (1 + rr, 1 + cc) in set(grey_local)]
        base_steps = min(int(best_dist), int(len(grey_order_local)))

        items.append(
            {
                "id": idx,
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "color": hc,
                "dir": direction,
                "head_pts": head_pts,
                "head_tl": head_tl,
                "dest_pts": set(best_dest["pts"]),
                "dest_tl": tuple(best_dest["tl"]),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "grey_order_local": grey_order_local,
                "base_distance": int(best_dist),
                "base_steps": int(base_steps),
                "actual_steps": 0,
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


def _can_place_step(occupancy: np.ndarray, bg: int, item: dict, step: int) -> bool:
    H, W = occupancy.shape
    dr, dc = item["dir"]
    tr = int(dr * step)
    tc = int(dc * step)
    r0, c0, r1, c1 = item["box"]
    nr0, nc0, nr1, nc1 = int(r0 + tr), int(c0 + tc), int(r1 + tr), int(c1 + tc)
    if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
        return False
    if np.any(occupancy[nr0 : nr1 + 1, nc0 : nc1 + 1] != int(bg)):
        return False
    for r, c in item["head_pts"]:
        rr = int(r + tr)
        cc = int(c + tc)
        if rr < 0 or cc < 0 or rr >= H or cc >= W:
            return False
        val = int(occupancy[rr, cc])
        if val != int(bg) and (rr, cc) not in item["dest_pts"]:
            return False
    return True


def _apply_item(grid: np.ndarray, item: dict, steps: int) -> None:
    r0, c0, _r1, _c1 = item["box"]
    dr, dc = item["dir"]
    tr = int(dr * steps)
    tc = int(dc * steps)
    patch = item["patch"].copy()
    for pr, pc in item["grey_order_local"][:steps]:
        if int(patch[pr, pc]) == GREY:
            patch[pr, pc] = int(ORANGE)
    paste(grid, patch, int(r0 + tr), int(c0 + tc), clip=False)
    for r, c in item["head_pts"]:
        grid[int(r + tr), int(c + tc)] = int(item["color"])


def _simulate_ordered(grid: np.ndarray, items: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int).copy()
    items2 = [dict(item) for item in items]
    items2.sort(key=lambda item: (int(item["base_distance"]), int(item["box"][0]), int(item["box"][1])))

    for item in items2:
        steps = 0
        if int(item["base_steps"]) > 0:
            occupancy = out.copy()
            _erase_item(occupancy, bg, item)
            while steps < int(item["base_steps"]) and _can_place_step(occupancy, bg, item, steps + 1):
                steps += 1
            item["blocked"] = steps < int(item["base_steps"])
        item["actual_steps"] = int(steps)
        if steps == 0:
            continue
        _erase_item(out, bg, item)
        _apply_item(out, item, steps)
    return out, items2


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, items = _collect_items(g)
    out, _ = _simulate_ordered(g, items, bg)
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


def _path_cells(spec: dict) -> list[tuple[int, int]]:
    head = _line_cells(spec, "head")
    dest = _line_cells(spec, "dest")
    if spec["head_side"] in (0, 1):
        cols = [c for _r, c in head]
        r0 = min(r for r, _c in head)
        r1 = max(r for r, _c in head)
        d0 = min(r for r, _c in dest)
        d1 = max(r for r, _c in dest)
        low = min(r0, d0)
        high = max(r1, d1)
        cells = []
        for rr in range(low, high + 1):
            if rr in {r for r, _c in head} or rr in {r for r, _c in dest}:
                continue
            for cc in cols:
                cells.append((rr, cc))
        return cells
    rows = [r for r, _c in head]
    c0 = min(c for _r, c in head)
    c1 = max(c for _r, c in head)
    d0 = min(c for _r, c in dest)
    d1 = max(c for _r, c in dest)
    low = min(c0, d0)
    high = max(c1, d1)
    cells = []
    for cc in range(low, high + 1):
        if cc in {c for _r, c in head} or cc in {c for _r, c in dest}:
            continue
        for rr in rows:
            cells.append((rr, cc))
    return cells


def _spec_non_bg_cells(spec: dict) -> set[tuple[int, int]]:
    r = int(spec["r"])
    c = int(spec["c"])
    oh = int(spec["outer_h"])
    ow = int(spec["outer_w"])
    cells = {(rr, cc) for rr in range(r, r + oh) for cc in range(c, c + ow)}
    cells.update(_line_cells(spec, "head"))
    cells.update(_line_cells(spec, "dest"))
    return cells


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr, cc in b:
            if abs(int(r - rr)) <= 1 and abs(int(c - cc)) <= 1:
                return True
    return False


def _spec_to_grid(specs: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        ih = oh - 2
        iw = ow - 2
        fill_order = _fill_order_local(ih, iw, int(spec["head_side"]))
        interior = np.full((ih, iw), GREY, dtype=int)
        for idx in range(int(spec["initial_orange"])):
            rr, cc = fill_order[idx]
            interior[rr, cc] = int(ORANGE)
        grid[r, c : c + ow] = int(BORDER)
        grid[r + oh - 1, c : c + ow] = int(BORDER)
        grid[r : r + oh, c] = int(BORDER)
        grid[r : r + oh, c + ow - 1] = int(BORDER)
        grid[r + 1 : r + oh - 1, c + 1 : c + ow - 1] = interior
        for rr, cc in _line_cells(spec, "head"):
            grid[rr, cc] = int(spec["color"])
        for rr, cc in _line_cells(spec, "dest"):
            grid[rr, cc] = int(spec["color"])
    return grid


def _bounds_ok(spec: dict, grid_h: int, grid_w: int) -> bool:
    for r, c in _spec_non_bg_cells(spec):
        if r < 0 or c < 0 or r >= grid_h or c >= grid_w:
            return False
    return True


def _clear_paths(specs: list[dict]) -> bool:
    all_non_bg = [(_spec_non_bg_cells(spec), spec) for spec in specs]
    for i, (cells_i, spec_i) in enumerate(all_non_bg):
        for j, (cells_j, _spec_j) in enumerate(all_non_bg):
            if i == j:
                continue
            if cells_i & cells_j:
                return False
        path = set(_path_cells(spec_i))
        for j, (cells_j, _spec_j) in enumerate(all_non_bg):
            if i == j:
                continue
            if path & cells_j:
                return False
    return True


def _separated(specs: list[dict]) -> bool:
    all_cells = [_spec_non_bg_cells(spec) for spec in specs]
    for i in range(len(all_cells)):
        for j in range(i + 1, len(all_cells)):
            if _touching(all_cells[i], all_cells[j]):
                return False
    return True


def _variant_metrics(grid: np.ndarray) -> dict:
    bg, items = _collect_items(grid)
    _out, sim_items = _simulate_ordered(grid, items, bg)
    interaction_loss = sum(int(item["base_steps"]) - int(item["actual_steps"]) for item in sim_items)
    blocked_count = sum(1 for item in sim_items if item["blocked"])
    return {
        "n_items": len(sim_items),
        "blocked_count": blocked_count,
        "interaction_loss": interaction_loss,
        "distances": tuple(int(item["base_distance"]) for item in sim_items),
        "actual_steps": tuple(int(item["actual_steps"]) for item in sim_items),
        "base_steps": tuple(int(item["base_steps"]) for item in sim_items),
    }


def _attempt_scene() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(18, 22)
    grid_w = random.randint(18, 22)
    colors = random.sample(SIGNAL_COLORS, 2)

    outer_h_h = random.randint(4, 5)
    outer_w_h = random.randint(5, 7)
    outer_w_v = 3
    dist_h = 4
    dist_v = random.randint(max(6, outer_h_h + 2), 8)
    outer_h_v = random.randint(dist_v + 2, min(12, dist_v + 4))
    area_h = (outer_h_h - 2) * (outer_w_h - 2)
    area_v = (outer_h_v - 2) * (outer_w_v - 2)

    if area_h < dist_h or area_v < dist_v:
        return None

    grey_h = random.randint(dist_h, area_h)
    grey_v = random.randint(dist_v, area_v)

    c_h_max = grid_w - outer_w_h - dist_h - 2
    if c_h_max < 1:
        return None
    c_h = random.randint(1, c_h_max)

    r_h_min = outer_h_v + dist_v - outer_h_h - 1
    r_h_max = grid_h - outer_h_h - 3
    if r_h_min > r_h_max:
        return None
    r_h = random.randint(r_h_min, r_h_max)

    dest_row_v = r_h + outer_h_h + 1
    r_v = dest_row_v - outer_h_v - dist_v
    if r_v < 1:
        return None

    c_v = c_h + outer_w_h + 1
    if c_v + outer_w_v >= grid_w:
        return None

    spec_h = {
        "r": r_h,
        "c": c_h,
        "outer_h": outer_h_h,
        "outer_w": outer_w_h,
        "head_side": 3,
        "distance": dist_h,
        "color": colors[0],
        "initial_orange": area_h - grey_h,
    }
    spec_v = {
        "r": r_v,
        "c": c_v,
        "outer_h": outer_h_v,
        "outer_w": outer_w_v,
        "head_side": 1,
        "distance": dist_v,
        "color": colors[1],
        "initial_orange": area_v - grey_v,
    }
    specs = [spec_h, spec_v]

    if not all(_bounds_ok(spec, grid_h, grid_w) for spec in specs):
        return None
    if not _clear_paths(specs):
        return None
    if not _separated(specs):
        return None

    grid = _spec_to_grid(specs, grid_h, grid_w)
    bg, items = _collect_items(grid)
    expected, sim_items = _simulate_ordered(grid, items, bg)
    if not np.array_equal(main(grid), expected):
        return None
    metrics = _variant_metrics(grid)
    if metrics["n_items"] < 2 or metrics["blocked_count"] < 1 or metrics["interaction_loss"] <= 0:
        return None
    if len(metrics["distances"]) >= 2 and not (metrics["distances"][0] < metrics["distances"][1]):
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
        metrics = _variant_metrics(grid2)
        if metrics["blocked_count"] < 1 or metrics["interaction_loss"] <= 0:
            continue
        return grid2
    raise RuntimeError("Failed to generate valid D12 variant after 400 attempts")
