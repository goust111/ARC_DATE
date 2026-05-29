# Variant of 271d71e2
# Dimension: D1 — 路径方向类型 (path)
# Change: straight_ray — 容器沿头部指示的四个基本方向（上/下/左/右）直线平移... → 沿预设路径像素移动

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, paste


# Design analysis for the D1 path variant
# a) Concrete behavior:
#    Original 271d71e2 uses a head line and an aligned destination line of the
#    same color. The solver infers a cardinal direction from the head's side and
#    translates the whole container by `direction * steps`, where
#    `steps = min(head_to_dest_distance, grey_count)`.
#    In this variant the destination is no longer constrained to lie on the same
#    row/column as the head. Instead there is an explicit 1-cell-wide path of
#    color 8 between the head line and the destination line. The container still
#    keeps its orientation, but its translation after k steps is the k-th prefix
#    displacement of that path, measured from the unique anchor pixel at the
#    center of the head line. Example: for a top head, original logic would move
#    only "up by k"; the new logic may move "up, up, right, up", so after
#    k=3 the container is offset by (-2,+1), not by (-3,0).
# b) Complexity calibration:
#    To make the path rule visible and non-degenerate, every moving assembly uses
#    a bent 3-segment path: outward from the head, sideways, outward again. This
#    gives exactly two bends and keeps the number of route archetypes small. The
#    path color is fixed (8) so the solver's job is to identify and follow the
#    visible route, not to infer a new color mapping.
# c) Old implicit assumptions that break:
#    - `best_dest` filtering by collinearity (`tr0 == 0` or `tc0 == 0`) no longer
#      holds because destination lines may be offset in both row and column.
#    - `tr = dr * steps`, `tc = dc * steps` assumes every unit of movement is the
#      same cardinal step; this must be replaced by ordered path-prefix motion.
#    - The head line previously only needed a side; now it also needs a unique
#      anchor pixel. The generator enforces an odd head-line length so the center
#      cell is well-defined.
# d) Placement legality:
#    The generator builds each assembly with the full sweep bbox of the container
#    and head across the whole path, not just the input pose. That reserved bbox
#    is what gets packed into the grid, so two assemblies cannot overlap either
#    in the input or after movement. Paths are 4-connected, branch-free, and each
#    touches exactly one head anchor and one destination anchor.
# e) Visual observability:
#    Human solvers can see the black box, a maroon head line, a separate maroon
#    target line, and a teal bent route between them. Across examples the output
#    shows the container stopping either at the target or partway along the route
#    depending on available grey charge, so the path-following behavior is
#    visible without relying on hidden numeric conventions.
# f) Visual cleanliness:
#    The path is kept to one distinct color and one-pixel width. It starts from
#    the head-line center and ends at the destination-line center, avoiding the
#    rest of those lines, so the route does not visually merge with the markers.
# g) Color-change inference:
#    The original grey->orange charging rule is preserved exactly. The new path
#    color is structural only; it does not participate in recoloring, which
#    keeps the rule count low and the color semantics stable.
# h) New solver assumptions and generator guarantees:
#    - Head/destination lines are straight odd-length line components.
#    - Path color is 8 and each path component is a simple 4-connected chain.
#    - For each container, exactly one path component is adjacent to the head
#      anchor and to the matching destination anchor.
#    - The head side still determines only the grey fill order, not the route.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
PATH = 8
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


def _head_line_coords(r0: int, c0: int, outer_h: int, outer_w: int, side: int) -> list[tuple[int, int]]:
    if side == 0:
        return [(r0 - 1, c) for c in range(c0, c0 + outer_w)]
    if side == 1:
        return [(r0 + outer_h, c) for c in range(c0, c0 + outer_w)]
    if side == 2:
        return [(r, c0 - 1) for r in range(r0, r0 + outer_h)]
    return [(r, c0 + outer_w) for r in range(r0, r0 + outer_h)]


def _anchor_from_side(r0: int, c0: int, outer_h: int, outer_w: int, side: int) -> tuple[int, int]:
    if side == 0:
        return (r0 - 1, c0 + outer_w // 2)
    if side == 1:
        return (r0 + outer_h, c0 + outer_w // 2)
    if side == 2:
        return (r0 + outer_h // 2, c0 - 1)
    return (r0 + outer_h // 2, c0 + outer_w)


def _line_anchor_from_pts(pts: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not pts:
        return None
    rs = sorted({r for r, _c in pts})
    cs = sorted({c for _r, c in pts})
    if len(rs) == 1:
        if len(cs) % 2 == 0:
            return None
        row = rs[0]
        for c in range(cs[0], cs[-1] + 1):
            if (row, c) not in pts:
                return None
        return (row, cs[len(cs) // 2])
    if len(cs) == 1:
        if len(rs) % 2 == 0:
            return None
        col = cs[0]
        for r in range(rs[0], rs[-1] + 1):
            if (r, col) not in pts:
                return None
        return (rs[len(rs) // 2], col)
    return None


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


def _adjacent_to_path(point: tuple[int, int], path_pts: set[tuple[int, int]]) -> bool:
    pr, pc = point
    for dr, dc in DIR4:
        if (pr + dr, pc + dc) in path_pts:
            return True
    return False


def _ordered_path_vertices(
    start: tuple[int, int],
    goal: tuple[int, int],
    path_pts: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    allowed = set(path_pts)
    allowed.add(start)
    allowed.add(goal)

    q = deque([start])
    prev = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in allowed or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if goal not in prev:
        return None

    out: list[tuple[int, int]] = []
    cur = goal
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging grey cells to orange.

    Variant rule:
    - The head line still selects the interior fill order.
    - A visible PATH-colored chain connects the head-line center to the destination-line center.
    - Steps = min(path distance in 4-neighbor moves, grey interior cell count).
    - The container translates by following the first `steps` edges of that path.
    """
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
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    out = g.copy()

    head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE, PATH})]
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in head_colors:
        for coords in components_mask(g == int(col), connectivity=4):
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    path_components = []
    for coords in components_mask(g == PATH, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if pts:
            path_components.append(pts)

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val in (int(bg), int(BORDER), int(GREY), int(ORANGE), int(PATH)):
                        continue
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

        coords = np.array(list(comp0["pts"]), dtype=int)
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

        anchor = _line_anchor_from_pts(set(comp0["pts"]))
        if anchor is None:
            return None

        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in comp0["pts"]}
        return {
            "color": hc,
            "head_pts": set(comp0["pts"]),
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
            "anchor": anchor,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        head_anchor = tuple(map(int, head_info["anchor"]))

        if direction == (0, 0):
            continue

        best_dist = None
        best_vertices = None
        for dest in comp_by_color.get(hc, []):
            if dest["pts"] == comp_pts or dest["shape"] != head_shape:
                continue
            dest_anchor = _line_anchor_from_pts(set(dest["pts"]))
            if dest_anchor is None:
                continue
            for path_pts in path_components:
                if not _adjacent_to_path(head_anchor, path_pts):
                    continue
                if not _adjacent_to_path(dest_anchor, path_pts):
                    continue
                vertices = _ordered_path_vertices(head_anchor, dest_anchor, path_pts)
                if vertices is None:
                    continue
                dist = len(vertices) - 1
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_vertices = vertices

        if best_vertices is None or best_dist is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(best_dist), int(len(grey_coords)))
        tr = int(best_vertices[steps][0] - head_anchor[0])
        tc = int(best_vertices[steps][1] - head_anchor[1])

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
            if int(out[r, c]) == hc:
                out[r, c] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(hc)

    return out


def _path_vertices_from_anchor(
    start: tuple[int, int],
    side: int,
    seg1: int,
    seg2: int,
    seg3: int,
    turn_sign: int,
) -> list[tuple[int, int]]:
    dr0, dc0 = SIDE_TO_DIR[side]
    if side in (0, 1):
        dr1, dc1 = 0, int(turn_sign)
    else:
        dr1, dc1 = int(turn_sign), 0

    cur = (int(start[0]), int(start[1]))
    verts = [cur]
    for steps, (dr, dc) in ((seg1, (dr0, dc0)), (seg2, (dr1, dc1)), (seg3, (dr0, dc0))):
        for _ in range(int(steps)):
            cur = (int(cur[0] + dr), int(cur[1] + dc))
            verts.append(cur)
    return verts


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _build_assembly(spec: dict) -> dict:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    side = int(spec["head_side"])
    initial_orange = int(spec["initial_orange"])
    seg1 = int(spec["seg1"])
    seg2 = int(spec["seg2"])
    seg3 = int(spec["seg3"])
    turn_sign = int(spec["turn_sign"])

    cont_r0, cont_c0 = 0, 0
    cont_r1, cont_c1 = cont_r0 + outer_h - 1, cont_c0 + outer_w - 1
    head_pts = _head_line_coords(cont_r0, cont_c0, outer_h, outer_w, side)
    head_anchor = _anchor_from_side(cont_r0, cont_c0, outer_h, outer_w, side)
    vertices = _path_vertices_from_anchor(head_anchor, side, seg1, seg2, seg3, turn_sign)
    path_pts = vertices[1:-1]
    dest_anchor = vertices[-1]

    if side in (0, 1):
        half = outer_w // 2
        dest_pts = [(dest_anchor[0], dest_anchor[1] + dc) for dc in range(-half, half + 1)]
    else:
        half = outer_h // 2
        dest_pts = [(dest_anchor[0] + dr, dest_anchor[1]) for dr in range(-half, half + 1)]

    fill_order = _interior_fill_order(inner_h, inner_w, side)
    input_cells: dict[tuple[int, int], int] = {}
    for r, c in rect_border_coords(cont_r0, cont_c0, cont_r1, cont_c1):
        input_cells[(r, c)] = BORDER
    for idx, (rr, cc) in enumerate(fill_order):
        input_cells[(cont_r0 + 1 + rr, cont_c0 + 1 + cc)] = ORANGE if idx < initial_orange else GREY
    for p in head_pts:
        input_cells[p] = HEAD_COLOR
    for p in path_pts:
        input_cells[p] = PATH
    for p in dest_pts:
        input_cells[p] = HEAD_COLOR

    reserved = set(input_cells.keys())
    for vr, vc in vertices:
        tr = int(vr - head_anchor[0])
        tc = int(vc - head_anchor[1])
        reserved.update((r + tr, c + tc) for r in range(cont_r0, cont_r1 + 1) for c in range(cont_c0, cont_c1 + 1))
        reserved.update((r + tr, c + tc) for r, c in head_pts)

    all_pts = reserved | set(input_cells.keys()) | set(vertices)
    min_r = min(r for r, _c in all_pts)
    max_r = max(r for r, _c in all_pts)
    min_c = min(c for _r, c in all_pts)
    max_c = max(c for _r, c in all_pts)
    shift_r = -min_r
    shift_c = -min_c

    shifted_input = [(int(r + shift_r), int(c + shift_c), int(col)) for (r, c), col in input_cells.items()]
    shifted_head = _translate(head_pts, shift_r, shift_c)
    shifted_dest = _translate(dest_pts, shift_r, shift_c)
    shifted_vertices = _translate(vertices, shift_r, shift_c)
    shifted_box = (cont_r0 + shift_r, cont_c0 + shift_c)

    return {
        "res_h": int(max_r - min_r + 1),
        "res_w": int(max_c - min_c + 1),
        "input_cells": shifted_input,
        "box_tl": shifted_box,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_pts": shifted_head,
        "dest_pts": shifted_dest,
        "vertices": shifted_vertices,
        "head_side": side,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "initial_orange": initial_orange,
        "grey": int(spec["grey"]),
        "steps": min(int(spec["grey"]), int(len(vertices) - 1)),
    }


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.choice([2, 2, 3, 3, 4])
    n_decoys = random.choice([0, 0, 1])
    n_decoys = min(n_decoys, n_containers - 1)

    specs = []
    mover_count = 0
    for i in range(n_containers):
        is_decoy = i >= (n_containers - n_decoys)
        side = random.randint(0, 3)
        if side in (0, 1):
            inner_h = random.randint(1, 7)
            inner_w = random.choice([1, 3, 5])
        else:
            inner_h = random.choice([1, 3, 5])
            inner_w = random.randint(1, 5)

        area = inner_h * inner_w
        if is_decoy:
            grey = 0
            initial_orange = area
        else:
            grey = random.randint(1, area)
            initial_orange = area - grey
            mover_count += 1

        seg1 = random.randint(1, 3)
        seg2 = random.randint(2, 5)
        seg3 = random.randint(1, 3)

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "head_side": side,
                "grey": grey,
                "initial_orange": initial_orange,
                "seg1": seg1,
                "seg2": seg2,
                "seg3": seg3,
                "turn_sign": random.choice([-1, 1]),
            }
        )

    if mover_count == 0:
        return None

    assemblies = [_build_assembly(spec) for spec in specs]
    max_h = max(a["res_h"] for a in assemblies)
    max_w = max(a["res_w"] for a in assemblies)
    if max_h > 26 or max_w > 26:
        return None

    total_area = sum(a["res_h"] * a["res_w"] for a in assemblies)
    min_side = max(12, int(total_area**0.5) + 3)
    lo_h = max(max_h + 2, min_side)
    lo_w = max(max_w + 2, min_side)
    hi_h = min(26, lo_h + 6)
    hi_w = min(26, lo_w + 6)
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []

    for asm in assemblies:
        ah, aw = int(asm["res_h"]), int(asm["res_w"])
        if ah > grid_h or aw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - ah)
            c0 = random.randint(0, grid_w - aw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + ah + 1)
            cc1 = min(grid_w, c0 + aw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + ah, c0 : c0 + aw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for asm, (pr, pc) in zip(assemblies, placements):
        for r, c, col in asm["input_cells"]:
            grid[pr + r, pc + c] = int(col)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for asm, (pr, pc) in zip(assemblies, placements):
        steps = int(asm["steps"])
        if steps == 0:
            continue

        box_r, box_c = asm["box_tl"]
        outer_h = int(asm["outer_h"])
        outer_w = int(asm["outer_w"])
        side = int(asm["head_side"])
        inner_h = int(asm["inner_h"])
        inner_w = int(asm["inner_w"])
        initial_orange = int(asm["initial_orange"])
        vertices = asm["vertices"]
        head_pts = asm["head_pts"]

        start = vertices[0]
        cur = vertices[steps]
        tr = int(cur[0] - start[0])
        tc = int(cur[1] - start[1])

        gr = pr + box_r
        gc = pc + box_c
        expected[gr : gr + outer_h, gc : gc + outer_w] = BG
        for r, c in head_pts:
            expected[pr + r, pc + c] = BG

        fill_order = _interior_fill_order(inner_h, inner_w, side)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(initial_orange + steps):
            rr, cc = fill_order[idx]
            interior[rr, cc] = ORANGE
        patch = np.full((outer_h, outer_w), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[gr + tr : gr + tr + outer_h, gc + tc : gc + tc + outer_w] = patch

        for r, c in head_pts:
            expected[pr + r + tr, pc + c + tc] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid path-following variant after 400 attempts")
