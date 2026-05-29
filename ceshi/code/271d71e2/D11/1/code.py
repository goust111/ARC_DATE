# Variant of 271d71e2
# Dimension: D11 — 头部与目标线的形状关系 (meta)
# Change: 头部和目标线形状完全相同（同长度的水平或垂直线段），颜色相同（maroon/9）... → 目标是点而非线
#
# Design analysis for the D11 variant
# ===================================
# a) Concrete behavior and trace
# - Containers stay the same: black rectangular border, grey/orange interior, maroon head line
#   adjacent to one container side, magenta background.
# - The changed object is only the destination marker: instead of another full maroon line, the
#   target is one maroon pixel placed farther along the movement axis.
# - The target point is aligned with the head center, not with the whole head span. Therefore the
#   motion distance is measured from the head center cell to the target point.
# - Example: top-headed container with outer width 5 has a 5-pixel head line on row r-1 and center
#   column c+2. If the target point is at (r-4, c+2), the geometric distance is 3. The container
#   moves up by min(3, grey_count), and the first 3 grey cells from the top-facing interior order
#   turn orange.
# - Original logic at this stage: detect a same-colored destination component whose normalized shape
#   exactly matches the head line, then translate until the head line overlaps that line.
# - Variant logic: detect a same-colored single-pixel destination on the same axis as the head
#   center, then translate until the head center overlaps that point.
# - Difference: shape equality is no longer the matching cue; axis alignment plus center alignment
#   becomes the matching cue.
#
# b) Complexity calibration
# - If head lengths were even, "center aligns to a point" would be ambiguous. To keep the new rule
#   visually crisp and logically exact, the generator only creates odd head lengths. This is a
#   minimal compatibility adjustment caused directly by the D11 change.
# - The head remains a full edge line, so the task is not trivialized into point-to-point motion.
#   The solver still needs rectangle detection, head-side inference, distance capping by grey count,
#   and ordered grey-to-orange charging.
# - All containers still reuse color 9, so the solver cannot cheat by unique colors. The generator
#   must therefore keep the nearest aligned point per container unambiguous.
#
# c) Old implicit assumptions that break
# - In the original solver, destination candidates were filtered by `d["shape"] == head_shape`.
#   That assumption is false now because the destination is size 1 while the head is a line.
# - The old translation vector was based on destination top-left minus head top-left. For a point
#   target this is wrong: alignment must be computed from head center to target point.
# - The old same-component split logic was still useful for gap=0, but its acceptance criterion
#   must change from "same shape as head" to "single-pixel point not adjacent to the container".
#
# d) Placement legality
# - Each assembly still fits in a rectangular reservation: target point + gap + head + container.
#   Because motion is toward the target and capped by the point distance, both input and output stay
#   inside the same reserved box.
# - Assemblies are placed with a one-cell magenta moat, so containers, heads, and points do not
#   overlap other assemblies. Generator validation also discards ambiguous same-axis point matches.
#
# e) Visual observability
# - The difference from the source task is directly visible in every input: ahead of each head line
#   there is a single maroon dot instead of another full line.
# - Odd head lengths make "center aims at the dot" legible from only a few examples.
# - Samples include both capped-by-distance and capped-by-grey cases so the point target is not a
#   decorative marker; it genuinely controls movement distance.
#
# f) Visual cleanliness
# - The point marker uses the same maroon color as the head but much smaller shape, keeping the
#   scene sparse and readable. No extra colors or overlapping annotation layers are introduced.
#
# g) Color inferability
# - Color semantics are unchanged from the original task: maroon marks head/target, grey is charge
#   resource, orange is charged interior, black is border, magenta is background. The variant does
#   not add any hidden color-conditioned behavior.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# head on top -> moves up, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _head_center(head_pts: set[tuple[int, int]], direction: tuple[int, int]) -> tuple[int, int]:
    # The generator enforces odd line lengths, but the solver stays deterministic on even lengths
    # by using the lower median pixel along the varying axis.
    if direction[0] != 0:
        row = next(iter(head_pts))[0]
        cols = sorted(c for _r, c in head_pts)
        return int(row), int(cols[len(cols) // 2])
    col = next(iter(head_pts))[1]
    rows = sorted(r for r, _c in head_pts)
    return int(rows[len(rows) // 2]), int(col)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:  # right
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # top
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _head_side_from_direction(direction: tuple[int, int]) -> int:
    for side, vec in SIDE_TO_DIR.items():
        if vec == direction:
            return side
    raise ValueError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored point destinations, charging grey cells to orange.

    The head is still a full maroon line on one side of the container. The destination is now a
    single maroon point farther along the movement axis. Translation distance is measured from the
    head center cell to that point, then capped by the remaining grey count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

    # Detect containers as BORDER rectangles around GREY/ORANGE rectangles.
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
            comp_by_color[int(col)].append(
                {"pts": pts, "tl": (int(r0), int(c0)), "bbox": (int(r0), int(c0), int(r1), int(c1))}
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (bg, BORDER, GREY, ORANGE):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == hc}
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        # If target touches the head at gap=0, head pixels are still the closest same-color pixels
        # to the border; the farther split-off pixel is the destination point.
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

        return {
            "color": hc,
            "head_pts": head_pts,
            "head_center": _head_center(head_pts, direction),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_center = tuple(head_info["head_center"])
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dest_points: set[tuple[int, int]] = set()

        # Separate point components elsewhere in the grid.
        for d in comp_by_color.get(hc, []):
            if d["pts"] == comp_pts:
                continue
            if len(d["pts"]) != 1:
                continue
            dest_points.add(next(iter(d["pts"])))

        # Point target can also be fused with the head component when gap=0.
        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT4)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if len(pts) != 1:
                    continue
                p = next(iter(pts))
                # Ignore pathological pixels that still cling to the border side. Generator keeps
                # points off the border adjacency band, but this avoids misreading malformed inputs.
                adj_border = False
                for dr, dc in DIR4:
                    rr, cc = int(p[0] + dr), int(p[1] + dc)
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                        adj_border = True
                        break
                if not adj_border:
                    dest_points.add(p)

        best = None
        best_dest = None
        for p in dest_points:
            tr0 = int(p[0] - head_center[0])
            tc0 = int(p[1] - head_center[1])
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
            if dist == 0:
                continue
            if best is None or dist < best:
                best = dist
                best_dest = p
        if best_dest is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(best), int(len(grey_coords)))
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        head_side = _head_side_from_direction(direction)
        order_local = _interior_fill_order(inner.shape[0], inner.shape[1], head_side)
        to_fill = {
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr, cc in order_local[:steps]
            if int(inner[rr, cc]) == GREY
        }

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == hc:
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


def _sample_inner_dims(head_side: int) -> tuple[int, int]:
    # Odd length along the head span gives a unique center pixel for point alignment.
    if head_side in (0, 1):
        inner_h = random.randint(1, 8)
        inner_w = random.choice([1, 3, 5])
    else:
        inner_h = random.choice([1, 3, 5, 7])
        inner_w = random.randint(1, 5)
    return int(inner_h), int(inner_w)


def _assembly_geometry(outer_h: int, outer_w: int, head_side: int, gap: int) -> dict:
    if head_side == 0:  # top
        total_h = 1 + gap + 1 + outer_h
        total_w = outer_w
        cont_r = gap + 2
        cont_c = 0
        target = (0, outer_w // 2)
    elif head_side == 1:  # bottom
        total_h = outer_h + 1 + gap + 1
        total_w = outer_w
        cont_r = 0
        cont_c = 0
        target = (outer_h + gap + 1, outer_w // 2)
    elif head_side == 2:  # left
        total_h = outer_h
        total_w = 1 + gap + 1 + outer_w
        cont_r = 0
        cont_c = gap + 2
        target = (outer_h // 2, 0)
    else:  # right
        total_h = outer_h
        total_w = outer_w + 1 + gap + 1
        cont_r = 0
        cont_c = 0
        target = (outer_h // 2, outer_w + gap + 1)
    return {
        "total_h": int(total_h),
        "total_w": int(total_w),
        "cont_r": int(cont_r),
        "cont_c": int(cont_c),
        "target_local": (int(target[0]), int(target[1])),
    }


def _draw_container_and_markers(grid: np.ndarray, spec: dict, pr: int, pc: int):
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    initial_orange = spec["initial_orange"]

    cr = pr + spec["cont_r"]
    cc = pc + spec["cont_c"]

    grid[cr, cc : cc + ow] = BORDER
    grid[cr + oh - 1, cc : cc + ow] = BORDER
    grid[cr : cr + oh, cc] = BORDER
    grid[cr : cr + oh, cc + ow - 1] = BORDER

    fill_order = _interior_fill_order(inner_h, inner_w, hs)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx in range(initial_orange):
        rr, cc0 = fill_order[idx]
        interior[rr, cc0] = ORANGE
    grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

    if hs == 0:
        grid[cr - 1, cc : cc + ow] = HEAD_COLOR
    elif hs == 1:
        grid[cr + oh, cc : cc + ow] = HEAD_COLOR
    elif hs == 2:
        grid[cr : cr + oh, cc - 1] = HEAD_COLOR
    else:
        grid[cr : cr + oh, cc + ow] = HEAD_COLOR

    tr, tc = spec["target_local"]
    grid[pr + tr, pc + tc] = HEAD_COLOR


def _expected_output(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]

        if steps == 0:
            continue

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx in range(initial_orange + steps):
            rr, cc0 = fill_order[idx]
            interior[rr, cc0] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    assemblies = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)
        inner_h, inner_w = _sample_inner_dims(head_side)
        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            grey = 0
            initial_orange = inner_area
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey

        gap = random.randint(0, 10)
        distance = gap + 1
        steps = min(grey, distance)

        geom = _assembly_geometry(outer_h, outer_w, head_side, gap)
        assemblies.append(
            {
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "head_side": int(head_side),
                "gap": int(gap),
                "distance": int(distance),
                "steps": int(steps),
                "grey": int(grey),
                "initial_orange": int(initial_orange),
                "is_decoy": bool(is_decoy),
                **geom,
            }
        )

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
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
    for a, (pr, pc) in zip(assemblies, placements):
        _draw_container_and_markers(grid, a, pr, pc)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
