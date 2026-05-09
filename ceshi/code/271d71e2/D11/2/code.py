# Variant of 271d71e2
# Dimension: D11 — 头部与目标线的形状关系 (meta)
# Change: 头部和目标线形状完全相同（同长度的水平或垂直线段），颜色相同（maroon/9）... → 目标线为L形或T形

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior
#    - The container, head line, movement axis, grey->orange charging order, and
#      color palette stay the same as the original task.
#    - The destination is no longer a copy of the head line. It is now a small
#      maroon L or T marker placed farther along the movement axis.
#    - The container still moves only along the head direction, but the moving
#      head aligns a specific anchor cell to the marker anchor:
#        * L marker: align the head endpoint that matches the arm side to the
#          elbow cell.
#        * T marker: align the head midpoint to the T-junction.
#    - Small trace:
#        original: top head line `99999`, destination `99999` above it, so the
#        solver uses line-top-left alignment.
#        variant: top head line `99999`, destination is `..9.. / .999.`
#        (a T with stem pointing away). The solver must use the middle head cell
#        and the T-junction, not the whole line shape. This changes the computed
#        travel distance while keeping the same charging rule.
# b) Complexity calibration
#    - Only two destination classes are used: L and T.
#    - Each class has a single visible anchor rule, so the rule is richer than
#      the original but still learnable from a few examples.
#    - Marker orientation is tied to movement direction, keeping the search
#      space small enough to avoid ambiguous matches.
# c) Original implicit assumptions that break here
#    - `if d["shape"] != head_shape`: invalid now, because destinations are not
#      head copies.
#    - `head_tl -> dest_tl` translation: invalid now, because alignment is by
#      elbow/junction anchor, not component top-left.
#    - "split same-color head/destination if they touch": the original code
#      depended on destination shape equality to recover a second line. The new
#      generator keeps at least one background cell between head and marker so
#      the solver can rely on separate components without hidden shape-equality
#      tricks.
# d) Placement legality
#    - The generator builds an assembly bbox from actual occupied cells of the
#      container, head, and L/T marker, then places those bboxes with a 1-cell
#      padding. Because the output container always moves inside that bbox toward
#      the marker, non-overlap remains valid in both input and output.
# e) Visual observability
#    - The change is visible: the maroon target is no longer a straight line,
#      and the moved head usually overlaps only part of the marker.
#    - T markers require odd head length so the midpoint is visually defined.
# f) Visual cleanliness
#    - Markers are small, separate connected components. There is no stacking of
#      multiple unrelated symbols in the same area, and gap >= 1 keeps head and
#      marker visually distinct in the input.
# g) Color inference
#    - No new colors are introduced. The only new signal is shape structure in
#      maroon, so the variant stays within ARC's visible reasoning regime.
# 9) New solver assumptions audited against the generator
#    - Every container has exactly one adjacent maroon head line.
#    - Every destination marker is a separate maroon component and is either:
#        * an L = axis stem + one side arm, anchor at the elbow
#        * a T = axis stem + 3-cell crossbar, anchor at the junction
#    - For T markers the corresponding head length is odd, guaranteed by the
#      generator via odd container span on the head axis.
#    - Candidate markers are matched only if the anchor shares the correct
#      cross-axis coordinate with the required head anchor cell, guaranteed by
#      construction in `_layout_from_spec`.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
PERP_DIRS = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}


def _ray(anchor: tuple[int, int], delta: tuple[int, int], length: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    dr, dc = delta
    return [(int(ar + dr * k), int(ac + dc * k)) for k in range(length)]


def _ray_len(pts: set[tuple[int, int]], anchor: tuple[int, int], delta: tuple[int, int]) -> int:
    r, c = anchor
    dr, dc = delta
    out = 0
    while (int(r), int(c)) in pts:
        out += 1
        r += dr
        c += dc
    return int(out)


def _marker_cells_from_spec(
    anchor: tuple[int, int],
    direction: tuple[int, int],
    marker_kind: str,
    stem_len: int,
    arm_dir: tuple[int, int] | None = None,
    arm_len: int | None = None,
) -> set[tuple[int, int]]:
    cells = set(_ray(anchor, direction, stem_len))
    if marker_kind == "T":
        perp_a, perp_b = PERP_DIRS[direction]
        cells.add((int(anchor[0] + perp_a[0]), int(anchor[1] + perp_a[1])))
        cells.add((int(anchor[0] + perp_b[0]), int(anchor[1] + perp_b[1])))
    else:
        assert arm_dir is not None and arm_len is not None
        cells.update(_ray(anchor, arm_dir, arm_len))
    return cells


def _analyze_marker_component(pts: set[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    analyses: dict[tuple[int, int], dict] = {}
    for direction in DIR4:
        dr, dc = direction
        perp_a, perp_b = PERP_DIRS[direction]
        candidates = []
        for anchor in pts:
            ar, ac = anchor
            if (int(ar + dr), int(ac + dc)) not in pts:
                continue
            if (int(ar - dr), int(ac - dc)) in pts:
                continue
            stem_len = _ray_len(pts, anchor, direction)

            pa = (int(ar + perp_a[0]), int(ac + perp_a[1]))
            pb = (int(ar + perp_b[0]), int(ac + perp_b[1]))
            if pa in pts and pb in pts:
                cells = set(_ray(anchor, direction, stem_len))
                cells.add(pa)
                cells.add(pb)
                if cells == pts:
                    candidates.append({"kind": "T", "anchor": anchor})

            for perp in (perp_a, perp_b):
                pr, pc = int(ar + perp[0]), int(ac + perp[1])
                rr, cc = int(ar - perp[0]), int(ac - perp[1])
                if (pr, pc) not in pts or (rr, cc) in pts:
                    continue
                arm_len = _ray_len(pts, anchor, perp)
                cells = set(_ray(anchor, direction, stem_len))
                cells.update(_ray(anchor, perp, arm_len))
                if cells == pts:
                    candidates.append({"kind": "L", "anchor": anchor, "perp": perp})

        dedup = []
        seen = set()
        for cand in candidates:
            key = (cand["kind"], cand["anchor"], cand.get("perp"))
            if key not in seen:
                seen.add(key)
                dedup.append(cand)
        if len(dedup) == 1:
            analyses[direction] = dedup[0]
    return analyses


def _head_anchor_for_marker(
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    marker: dict,
) -> tuple[int, int] | None:
    if direction[0] != 0:
        cells = sorted(head_pts, key=lambda p: (p[1], p[0]))
    else:
        cells = sorted(head_pts, key=lambda p: (p[0], p[1]))

    if marker["kind"] == "T":
        if len(cells) % 2 == 0:
            return None
        return cells[len(cells) // 2]

    perp = marker["perp"]
    if direction[0] != 0:
        return cells[0] if perp == (0, 1) else cells[-1]
    return cells[0] if perp == (1, 0) else cells[-1]


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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

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
            pts = {(int(r), int(c)) for r, c in coords}
            r0, c0, _r1, _c1 = bbox_coords(coords)
            comp_by_color[int(col)].append(
                {
                    "id": int(k),
                    "pts": pts,
                    "tl": (int(r0), int(c0)),
                    "markers": _analyze_marker_component(pts),
                }
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
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

        return {
            "color": hc,
            "head_pts": head_pts,
            "comp_id": comp0["id"],
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        best_marker = None
        best_head_anchor = None
        best_dist = None
        for comp in comp_by_color.get(hc, []):
            if int(comp["id"]) == int(head_info["comp_id"]):
                continue
            marker = comp["markers"].get(direction)
            if marker is None:
                continue
            head_anchor = _head_anchor_for_marker(head_pts, direction, marker)
            if head_anchor is None:
                continue

            hr, hc0 = head_anchor
            ar, ac = marker["anchor"]
            if direction[0] != 0:
                if int(ac) != int(hc0):
                    continue
                delta = int(ar - hr)
                if int(direction[0]) * delta <= 0:
                    continue
                dist = abs(delta)
            else:
                if int(ar) != int(hr):
                    continue
                delta = int(ac - hc0)
                if int(direction[1]) * delta <= 0:
                    continue
                dist = abs(delta)

            if best_dist is None or dist < best_dist:
                best_dist = int(dist)
                best_marker = marker
                best_head_anchor = head_anchor

        if best_marker is None or best_head_anchor is None or best_dist is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(best_dist), int(len(grey_coords)))

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
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(hc)

    return out


def _choose_marker_kinds(n_containers: int, n_movers: int) -> list[str]:
    kinds = []
    if n_containers >= 2:
        base = ["L", "T"]
        random.shuffle(base)
        kinds.extend(base[: min(2, n_containers)])
    while len(kinds) < n_containers:
        kinds.append(random.choice(["L", "T"]))
    random.shuffle(kinds)
    if n_movers >= 2:
        mover_kinds = kinds[:n_movers]
        if len(set(mover_kinds)) == 1:
            mover_kinds[0] = "L"
            mover_kinds[1] = "T"
            random.shuffle(mover_kinds)
            kinds[:n_movers] = mover_kinds
    return kinds


def _choose_inner_dims(marker_kind: str, head_side: int) -> tuple[int, int]:
    if marker_kind == "T" and head_side in (0, 1):
        inner_w = random.choice([1, 3, 5])
        inner_h = random.randint(1, 8)
    elif marker_kind == "T":
        inner_h = random.choice([1, 3, 5, 7])
        inner_w = random.randint(1, 5)
    else:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
    return int(inner_h), int(inner_w)


def _layout_from_spec(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    direction = SIDE_TO_DIR[hs]

    container_cells = {(r, c) for r in range(oh) for c in range(ow)}
    if hs == 0:
        head_cells = {(int(-1), c) for c in range(ow)}
    elif hs == 1:
        head_cells = {(int(oh), c) for c in range(ow)}
    elif hs == 2:
        head_cells = {(r, int(-1)) for r in range(oh)}
    else:
        head_cells = {(r, int(ow)) for r in range(oh)}

    if hs in (0, 1):
        head_order = sorted(head_cells, key=lambda p: (p[1], p[0]))
    else:
        head_order = sorted(head_cells, key=lambda p: (p[0], p[1]))

    if spec["marker_kind"] == "T":
        head_anchor = head_order[len(head_order) // 2]
    else:
        arm_dir = spec["arm_dir"]
        if hs in (0, 1):
            head_anchor = head_order[0] if arm_dir == (0, 1) else head_order[-1]
        else:
            head_anchor = head_order[0] if arm_dir == (1, 0) else head_order[-1]

    dist = int(spec["gap"] + 1)
    marker_anchor = (
        int(head_anchor[0] + direction[0] * dist),
        int(head_anchor[1] + direction[1] * dist),
    )
    marker_cells = _marker_cells_from_spec(
        marker_anchor,
        direction,
        spec["marker_kind"],
        spec["stem_len"],
        spec.get("arm_dir"),
        spec.get("arm_len"),
    )

    all_cells = container_cells | head_cells | marker_cells
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    shift_r, shift_c = -int(min_r), -int(min_c)

    def sh(pt: tuple[int, int]) -> tuple[int, int]:
        return (int(pt[0] + shift_r), int(pt[1] + shift_c))

    layout = dict(spec)
    layout["total_h"] = int(max_r - min_r + 1)
    layout["total_w"] = int(max_c - min_c + 1)
    layout["cont_r"] = int(shift_r)
    layout["cont_c"] = int(shift_c)
    layout["head_cells_local"] = {sh(p) for p in head_cells}
    layout["head_anchor_local"] = sh(head_anchor)
    layout["marker_anchor_local"] = sh(marker_anchor)
    layout["marker_cells_local"] = {sh(p) for p in marker_cells}
    return layout


def _build_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = np.array(grid, copy=True)
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]

        cr = int(pr + a["cont_r"])
        cc = int(pc + a["cont_c"])
        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        for r, c in a["head_cells_local"]:
            expected[int(pr + r), int(pc + c)] = BG

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(initial_orange + steps):
            rr, cc2 = fill_order[idx]
            interior[rr, cc2] = ORANGE

        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        for r, c in a["head_cells_local"]:
            rr = int(pr + r + dr * steps)
            cc3 = int(pc + c + dc * steps)
            expected[rr, cc3] = HEAD_COLOR

    return expected


def _gen_attempt(return_meta: bool = False):
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    marker_kinds = _choose_marker_kinds(n_containers, n_movers)
    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        marker_kind = marker_kinds[i]
        head_side = random.randint(0, 3)
        inner_h, inner_w = _choose_inner_dims(marker_kind, head_side)
        inner_area = int(inner_h * inner_w)
        outer_h = int(inner_h + 2)
        outer_w = int(inner_w + 2)

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            gap = random.randint(1, 5)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(1, 8)
            steps = min(grey, gap + 1)

        stem_len = random.randint(2, 4)
        spec = {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": steps,
            "initial_orange": initial_orange,
            "grey": grey,
            "is_decoy": is_decoy,
            "marker_kind": marker_kind,
            "stem_len": stem_len,
        }
        if marker_kind == "L":
            spec["arm_dir"] = random.choice(list(PERP_DIRS[SIDE_TO_DIR[head_side]]))
            spec["arm_len"] = random.randint(2, 3)
        specs.append(_layout_from_spec(spec))

    total_area = sum(a["total_h"] * a["total_w"] for a in specs)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in specs)
    max_asm_w = max(a["total_w"] for a in specs)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for a in specs:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
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
    for a, (pr, pc) in zip(specs, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]

        cr = int(pr + a["cont_r"])
        cc = int(pc + a["cont_c"])
        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(initial_orange):
            rr, cc2 = fill_order[idx]
            interior[rr, cc2] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        for r, c in a["head_cells_local"]:
            grid[int(pr + r), int(pc + c)] = HEAD_COLOR
        for r, c in a["marker_cells_local"]:
            grid[int(pr + r), int(pc + c)] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, specs, placements)
    if not np.array_equal(output, expected):
        return None

    if return_meta:
        return grid, {"assemblies": specs, "placements": placements, "expected": expected}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid variant after 500 attempts")
