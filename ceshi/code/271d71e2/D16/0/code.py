# Variant of 271d71e2
# Dimension: D16 — 背景色设定 (meta)
# Change: 背景为magenta(6)，固定单色背景... → 背景色因区域不同而变化

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


"""
Design analysis for the D16 variant
==================================

variant_spec:
  region_layout:
    - The grid is split into two full-width horizontal background bands.
    - Rows [0, split) use TOP_BG = 6.
    - Rows [split, H) use BOTTOM_BG = 8.
    - No extra divider object is drawn; the visible color change itself is the cue.
  preserved_objects:
    - Containers remain black-bordered rectangles with grey/orange rectangular interiors.
    - Head and destination remain maroon full-edge lines.
    - Matching, direction inference, and grey-to-orange fill order stay the same.
  new_behavior:
    - A container belongs to the band containing its body.
    - In the top band, movement is unchanged: 1 grey charge moves 1 cell.
    - In the bottom band, movement is accelerated: 1 grey charge moves up to 2 cells.
    - Let distance = head-to-destination offset in cells.
    - Let speed = 1 (top) or 2 (bottom).
    - Actual travel = min(distance, speed * grey_count).
    - Number of newly orange cells = ceil(travel / speed).
    - Therefore bottom-band containers may move farther than the number of newly
      orange cells, which is the visible signature of the new rule.

behavior_trace:
  original_local_case:
    - bottom-style object with distance 4 and grey_count 2 would move 2 cells
      and turn 2 greys orange.
  variant_local_case:
    - in TOP_BG(6): still move 2 cells and turn 2 greys orange.
    - in BOTTOM_BG(8): move 4 cells and turn only 2 greys orange.
  observable_difference:
    - two otherwise similar containers in different bands no longer travel the
      same amount for the same number of consumed grey cells.

complexity_calibration:
  - Only two bands and two speeds are used, so the rule is learnable from a few examples.
  - Generator forces at least one mover in each band and at least one accelerated
    bottom-band mover where distance > grey_count, preventing regression to the
    original task.
  - Objects stay entirely inside one band, so the region cue is unambiguous and
    the solver does not need cross-band partial-speed logic.

implicit_assumption_audit:
  - Original code inferred one global background color and used it when erasing
    old container/head positions. That becomes incorrect once different rows have
    different background colors, because unconditional fills would paint the
    wrong color into whichever band the object vacated.
  - Original head-color discovery excluded only one background color; in this
    variant it must exclude both band colors.
  - Container detection via grey/orange interiors and black borders remains valid,
    because it never relied on the background being uniform.
  - Destination matching by same color / same shape / same axis also remains valid.

placement_legality:
  - Generator places each whole assembly (destination + gap + head + container)
    inside a single band, with a 1-cell padded non-overlap test.
  - Because the input assembly already spans the full travel corridor, keeping it
    inside one band guarantees the moved container and head also stay in that band.
  - Grid sizes and split rows are chosen only after assembly sizes are known.

visual_observability:
  - The background split is large, clean, and global, so humans can identify it
    immediately without reading color ids.
  - The accelerated rule is visible because at least one bottom-band mover travels
    farther than its count of newly orange cells would allow under the original rule.
  - Using only two background colors avoids visual clutter and hidden mappings.

new_solver_assumptions_checked_against_generator:
  - Each row exposes at least some background somewhere, allowing row-background
    inference. Generator keeps the grid wider than any assembly and uses sparse placement.
  - Each container lies in one band. Generator enforces this by region-constrained placement.
  - Only the two band colors act as backgrounds. Generator uses object colors
    {0,5,7,9} and band colors {6,8} exclusively.
"""


TOP_BG = 6
BOTTOM_BG = 8
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
REGION_SPEED = {TOP_BG: 1, BOTTOM_BG: 2}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _infer_row_backgrounds(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg_colors = colors_present(g, exclude={BORDER, GREY, ORANGE, HEAD_COLOR})
    if not bg_colors:
        raise AssertionError("No background colors detected")

    row_bg: list[int | None] = [None] * H
    for r in range(H):
        vals = [int(v) for v in g[r] if int(v) in bg_colors]
        if vals:
            row_bg[r] = int(Counter(vals).most_common(1)[0][0])

    last = None
    for r in range(H):
        if row_bg[r] is None:
            row_bg[r] = last
        else:
            last = row_bg[r]

    last = None
    for r in range(H - 1, -1, -1):
        if row_bg[r] is None:
            row_bg[r] = last
        else:
            last = row_bg[r]

    seen = [int(v) for v in row_bg if v is not None]
    if not seen:
        raise AssertionError("Failed to infer row backgrounds")
    fallback = int(Counter(seen).most_common(1)[0][0])
    final = [fallback if v is None else int(v) for v in row_bg]
    return final, [int(c) for c in bg_colors]


def _background_grid(shape: tuple[int, int], split: int) -> np.ndarray:
    H, W = shape
    bg = np.full((H, W), BOTTOM_BG, dtype=int)
    bg[:split, :] = TOP_BG
    return bg


def _movement_and_fill(distance: int, grey_count: int, speed: int) -> tuple[int, int]:
    travel = min(int(distance), int(speed * grey_count))
    fill_count = 0 if travel == 0 else int((travel + speed - 1) // speed)
    return int(travel), int(fill_count)


def _order_greys(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


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

    row_bg, bg_colors = _infer_row_backgrounds(g)
    bg_grid = np.repeat(np.array(row_bg, dtype=int)[:, None], W, axis=1)

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
    all_head_colors = [c for c in colors_present(g, exclude={BORDER, GREY, ORANGE, *bg_colors})]

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
                    if val not in (BORDER, GREY, ORANGE) and val not in bg_colors:
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
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
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
        distance = abs(int(best_dest["tl"][0] - head_tl[0])) + abs(int(best_dest["tl"][1] - head_tl[1]))
        region_color = int(row_bg[(r0 + r1) // 2])
        speed = int(REGION_SPEED.get(region_color, 1))
        travel, fill_count = _movement_and_fill(distance, len(grey_coords), speed)

        dr, dc = direction
        tr = int(dr * travel)
        tc = int(dc * travel)
        order = _order_greys(grey_coords, direction)
        to_fill = set(order[:fill_count])

        out[r0 : r1 + 1, c0 : c1 + 1] = bg_grid[r0 : r1 + 1, c0 : c1 + 1]
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg_grid[int(r), int(c)])

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            nr = int(r + tr)
            nc = int(c + tc)
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = hc

    return out


def _choose_regions(n_containers: int, n_movers: int) -> list[int]:
    regions = [0, 1]
    for _ in range(2, n_movers):
        regions.append(random.randint(0, 1))
    for _ in range(n_movers, n_containers):
        regions.append(random.randint(0, 1))
    random.shuffle(regions)
    return regions


def _build_specs() -> list[dict] | None:
    n_containers = random.randint(2, 4)
    n_decoys = 0 if n_containers == 2 else random.choice([0, 0, 1])
    if n_containers - n_decoys < 2:
        n_decoys = n_containers - 2
    n_movers = n_containers - n_decoys

    regions = _choose_regions(n_containers, n_movers)
    mover_flags = [True] * n_movers + [False] * n_decoys
    random.shuffle(mover_flags)

    mover_indices = [i for i, is_mover in enumerate(mover_flags) if is_mover]
    if not any(regions[i] == 0 for i in mover_indices):
        regions[mover_indices[0]] = 0
    if not any(regions[i] == 1 for i in mover_indices):
        regions[mover_indices[-1]] = 1

    bottom_movers = [i for i in mover_indices if regions[i] == 1]
    if not bottom_movers:
        idx = mover_indices[-1]
        regions[idx] = 1
        bottom_movers = [idx]

    accel_idx = random.choice(bottom_movers)
    specs = []

    for i in range(n_containers):
        is_mover = mover_flags[i]
        region = regions[i]
        speed = REGION_SPEED[TOP_BG if region == 0 else BOTTOM_BG]
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if not is_mover:
            grey = 0
            initial_orange = inner_area
            distance = random.randint(1, 6)
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            if i == accel_idx:
                max_dist = min(8, 2 * grey)
                if max_dist < grey + 1:
                    return None
                distance = random.randint(grey + 1, max_dist)
            else:
                distance = random.randint(1, 8)

        gap = distance - 1
        travel, fill_count = _movement_and_fill(distance, grey, speed)
        if is_mover and travel == 0:
            return None
        if i == accel_idx and not (travel > grey and fill_count == grey):
            return None

        if head_side in (0, 1):
            head_len = outer_w
        else:
            head_len = outer_h

        if head_side == 0:
            total_h = 1 + gap + 1 + outer_h
            total_w = outer_w
            cont_r = gap + 2
            cont_c = 0
        elif head_side == 1:
            total_h = outer_h + 1 + gap + 1
            total_w = outer_w
            cont_r = 0
            cont_c = 0
        elif head_side == 2:
            total_h = outer_h
            total_w = 1 + gap + 1 + outer_w
            cont_r = 0
            cont_c = gap + 2
        else:
            total_h = outer_h
            total_w = outer_w + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        specs.append(
            {
                "region": region,
                "speed": speed,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "inner_area": inner_area,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "head_len": head_len,
                "grey": grey,
                "initial_orange": initial_orange,
                "distance": distance,
                "gap": gap,
                "travel": travel,
                "fill_count": fill_count,
                "is_mover": is_mover,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    return specs


def _choose_grid(specs: list[dict]) -> tuple[int, int, int] | None:
    top_specs = [s for s in specs if s["region"] == 0]
    bottom_specs = [s for s in specs if s["region"] == 1]
    if not top_specs or not bottom_specs:
        return None

    total_area = sum(int(s["total_h"] * s["total_w"]) for s in specs)
    top_min = max(max(int(s["total_h"]) for s in top_specs) + 1, 6)
    bottom_min = max(max(int(s["total_h"]) for s in bottom_specs) + 1, 6)
    lo_h = top_min + bottom_min
    hi_h = min(26, max(lo_h, int(total_area**0.5) + 10))
    if lo_h > 26:
        return None
    grid_h = random.randint(lo_h, hi_h)
    split_lo = top_min
    split_hi = grid_h - bottom_min
    if split_lo > split_hi:
        return None
    split = random.randint(split_lo, split_hi)

    max_w = max(int(s["total_w"]) for s in specs)
    lo_w = max(max_w + 2, int(total_area**0.5) + 4)
    hi_w = min(26, max(lo_w, lo_w + 6))
    if lo_w > 26:
        return None
    grid_w = random.randint(lo_w, hi_w)
    return grid_h, grid_w, split


def _place_assemblies(specs: list[dict], grid_h: int, grid_w: int, split: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["total_h"] * specs[i]["total_w"]), reverse=True)

    for idx in order:
        s = specs[idx]
        th = int(s["total_h"])
        tw = int(s["total_w"])
        row_lo = 0 if s["region"] == 0 else split
        row_hi = split if s["region"] == 0 else grid_h
        if th > row_hi - row_lo or tw > grid_w:
            return None

        placed = False
        for _ in range(500):
            r0 = random.randint(row_lo, row_hi - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _draw_input(specs: list[dict], placements: list[tuple[int, int]], grid_h: int, grid_w: int, split: int) -> np.ndarray:
    grid = _background_grid((grid_h, grid_w), split)

    for s, (pr, pc) in zip(specs, placements):
        oh = int(s["outer_h"])
        ow = int(s["outer_w"])
        hs = int(s["head_side"])
        gap = int(s["gap"])
        inner_h = int(s["inner_h"])
        inner_w = int(s["inner_w"])
        initial_orange = int(s["initial_orange"])
        cr = int(pr + s["cont_r"])
        cc = int(pc + s["cont_c"])

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

    return grid


def _expected_output(specs: list[dict], placements: list[tuple[int, int]], grid: np.ndarray, split: int) -> np.ndarray:
    expected = np.array(grid, dtype=int).copy()
    bg_grid = _background_grid(grid.shape, split)

    for s, (pr, pc) in zip(specs, placements):
        if not s["is_mover"]:
            continue

        oh = int(s["outer_h"])
        ow = int(s["outer_w"])
        hs = int(s["head_side"])
        travel = int(s["travel"])
        fill_count = int(s["fill_count"])
        inner_h = int(s["inner_h"])
        inner_w = int(s["inner_w"])
        initial_orange = int(s["initial_orange"])
        dr, dc = SIDE_TO_DIR[hs]
        cr = int(pr + s["cont_r"])
        cc = int(pc + s["cont_c"])

        expected[cr : cr + oh, cc : cc + ow] = bg_grid[cr : cr + oh, cc : cc + ow]
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = bg_grid[cr - 1, cc : cc + ow]
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = bg_grid[cr + oh, cc : cc + ow]
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = bg_grid[cr : cr + oh, cc - 1]
        else:
            expected[cr : cr + oh, cc + ow] = bg_grid[cr : cr + oh, cc + ow]

        nr = int(cr + dr * travel)
        nc = int(cc + dc * travel)
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + fill_count):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE

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
    specs = _build_specs()
    if specs is None:
        return None

    dims = _choose_grid(specs)
    if dims is None:
        return None
    grid_h, grid_w, split = dims

    placements = _place_assemblies(specs, grid_h, grid_w, split)
    if placements is None:
        return None

    grid = _draw_input(specs, placements, grid_h, grid_w, split)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(specs, placements, grid, split)
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
