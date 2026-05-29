# Combo variant of 271d71e2 (index 22)
# combo_spec: D13/2+D16/0
# Name: 取模步数区域背景

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


"""
Design analysis for the combo variant
=====================================

fusion_rule:
  - Keep the original object vocabulary: black rectangular containers with grey/orange
    interiors, plus a same-colored head line and destination line.
  - Replace the original step rule with modulo:
      steps = head_dest_distance % grey_count, with grey_count==0 -> 0.
  - Replace the single global background with two full-width horizontal bands:
      top rows use 6, bottom rows use 8.
  - The bands do not change movement speed. Their role is visual structure plus
    correct restoration of vacated cells when a container moves.

concrete_behavior_trace:
  - Example A in the top band:
      distance=7, grey_count=3
      original task -> min(7,3)=3, move 3, fill 3 greys
      combo task    -> 7%3=1,     move 1, fill 1 grey
      difference    -> a much larger residual gap remains before the destination.
  - Example B in the bottom band:
      distance=6, grey_count=4
      original task -> min(6,4)=4
      combo task    -> 6%4=2
      difference    -> movement is still modulo-based even though the background band
        is different; after erasing the old position, the exposed cells must become 8
        rather than a single global background color.
  - This gives a genuine fusion: modulo controls motion, while region-aware background
    handling is necessary to produce the correct output image.

complexity_calibration:
  - Only two large regions are used. This keeps the background cue obvious and the
    color mapping small enough to infer from 2-4 examples.
  - Generator forces at least one positive-step modulo-remainder mover in each band,
    not just one globally. This prevents samples where one region only shows
    old-looking distance-limited behavior.
  - Additional movers may be:
      mod_nonzero     -> visible remainder behavior
      distance_limited-> looks like the old task, useful contrast
      mod_zero        -> exact multiple, grey remains but the object stays put
      full_orange     -> unchanged decoy with grey_count=0

implicit_assumption_audit_of_main:
  - Original code assumed one global background color for both detection and erasure.
    That is invalid now. The combo solver must infer row-wise backgrounds and erase
    old container/head pixels using the local band color.
  - Container detection via grey/orange interiors and black border still holds.
  - Head/destination matching by same color, same shape, and same axis still holds.
  - The movement cap is no longer grey_count through min(...); the solver must compute
    modulo on the full head-to-destination distance and guard against grey_count==0.
  - Because background is multi-color, head-color discovery must exclude all inferred
    background colors, not just a single dominant one.

placement_legality:
  - Each whole assembly (destination + gap corridor + head + container) is placed
    wholly inside one band.
  - Since steps = distance % grey_count always satisfies steps <= distance for grey>0,
    the moved container stays within the same reserved corridor.
  - Placement uses a padded occupancy check, so different assemblies stay separated by
    at least one background cell and remain in-bounds before and after movement.

visual_observability_and_cleanliness:
  - The horizontal split is broad and uncluttered, so humans can immediately see that
    the background is region-dependent.
  - The modulo signature is visible through shorter travel and fewer newly orange cells
    than the original rule would predict.
  - Regions are not overloaded with extra semantics like speed multipliers; this avoids
    a noisy interaction between multiple hidden rules and keeps the fusion readable.

generator_guarantees_for_new_assumptions:
  - Both background colors appear in every generated grid.
  - Each band contains at least one mover with steps>0 whose motion differs from the
    original min(...) rule, so both regions visibly support the modulo relation.
  - Grid width always exceeds any single assembly width, and sparse placement leaves
    enough visible background for row-background inference to be stable.
"""


TOP_BG = 6
BOTTOM_BG = 8
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


def _background_grid(shape: tuple[int, int], split: int) -> np.ndarray:
    h, w = shape
    bg = np.full((h, w), BOTTOM_BG, dtype=int)
    bg[:split, :] = TOP_BG
    return bg


def _infer_row_backgrounds(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg_colors = [int(c) for c in colors_present(g, exclude={BORDER, GREY, ORANGE, HEAD_COLOR})]
    if not bg_colors:
        raise AssertionError("No background colors detected")

    row_bg: list[int | None] = [None] * h
    for r in range(h):
        vals = [int(v) for v in g[r] if int(v) in bg_colors]
        if vals:
            row_bg[r] = int(Counter(vals).most_common(1)[0][0])

    last = None
    for r in range(h):
        if row_bg[r] is None:
            row_bg[r] = last
        else:
            last = row_bg[r]

    last = None
    for r in range(h - 1, -1, -1):
        if row_bg[r] is None:
            row_bg[r] = last
        else:
            last = row_bg[r]

    seen = [int(v) for v in row_bg if v is not None]
    if not seen:
        raise AssertionError("Failed to infer row backgrounds")
    fallback = int(Counter(seen).most_common(1)[0][0])
    final = [fallback if v is None else int(v) for v in row_bg]
    return final, bg_colors


def _step_count(distance: int, grey_count: int) -> int:
    distance = int(distance)
    grey_count = int(grey_count)
    if grey_count <= 0:
        return 0
    return int(distance % grey_count)


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


def _assembly_geometry(head_side: int, gap: int, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    if head_side == 0:
        return 1 + gap + 1 + outer_h, outer_w, gap + 2, 0
    if head_side == 1:
        return outer_h + 1 + gap + 1, outer_w, 0, 0
    if head_side == 2:
        return outer_h, 1 + gap + 1 + outer_w, 0, gap + 2
    return outer_h, outer_w + 1 + gap + 1, 0, 0


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    row_bg, bg_colors = _infer_row_backgrounds(g)
    bg_grid = np.repeat(np.array(row_bg, dtype=int)[:, None], w, axis=1)

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
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
                if 0 <= rr < h and 0 <= cc < w:
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
            mask_rem = np.zeros((h, w), dtype=bool)
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
                        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BORDER:
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

        distance = abs(int(best_dest["tl"][0] - head_tl[0])) + abs(int(best_dest["tl"][1] - head_tl[1]))
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = _step_count(distance, len(grey_coords))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_fill = set(_order_greys(grey_coords, direction)[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = bg_grid[r0 : r1 + 1, c0 : c1 + 1]
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == hc:
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
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = hc

    return out


def _choose_mover_spec(mode: str) -> dict | None:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if mode == "mod_nonzero":
        if inner_area < 2:
            return None
        grey = random.randint(2, min(inner_area, 6))
        remainder = random.randint(1, grey - 1)
        max_q = max(1, min(3, (12 - remainder) // grey))
        distance = random.randint(1, max_q) * grey + remainder
    elif mode == "mod_zero":
        grey = random.randint(1, min(inner_area, 6))
        max_q = max(1, min(3, 12 // grey))
        distance = random.randint(1, max_q) * grey
    elif mode == "distance_limited":
        distance = random.randint(1, 8)
        if inner_area <= distance:
            return None
        grey = random.randint(distance + 1, inner_area)
    else:
        raise ValueError(f"unknown mover mode: {mode}")

    steps = _step_count(distance, grey)
    initial_orange = inner_area - grey
    total_h, total_w, cont_r, cont_c = _assembly_geometry(head_side, distance - 1, outer_h, outer_w)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "distance": distance,
        "gap": distance - 1,
        "steps": steps,
        "grey": grey,
        "initial_orange": initial_orange,
        "is_decoy": False,
        "mode": mode,
        "total_h": total_h,
        "total_w": total_w,
        "cont_r": cont_r,
        "cont_c": cont_c,
    }


def _choose_decoy_spec() -> dict:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    distance = random.randint(1, 6)
    total_h, total_w, cont_r, cont_c = _assembly_geometry(head_side, distance - 1, outer_h, outer_w)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "distance": distance,
        "gap": distance - 1,
        "steps": 0,
        "grey": 0,
        "initial_orange": inner_area,
        "is_decoy": True,
        "mode": "full_orange",
        "total_h": total_h,
        "total_w": total_w,
        "cont_r": cont_r,
        "cont_c": cont_c,
    }


def _build_specs() -> list[dict] | None:
    n_containers = random.randint(2, 4)
    n_decoys = 0 if n_containers == 2 else random.choice([0, 0, 1])
    if n_containers - n_decoys < 2:
        n_decoys = n_containers - 2
    n_movers = n_containers - n_decoys

    mover_modes = ["mod_nonzero"]
    while len(mover_modes) < n_movers:
        mover_modes.append(random.choices(
            ["distance_limited", "mod_nonzero", "mod_zero"],
            weights=[4, 3, 1],
            k=1,
        )[0])
    random.shuffle(mover_modes)

    specs: list[dict] = []
    for mode in mover_modes:
        spec = _choose_mover_spec(mode)
        if spec is None:
            return None
        specs.append(spec)
    for _ in range(n_decoys):
        specs.append(_choose_decoy_spec())

    random.shuffle(specs)
    movers = [i for i, s in enumerate(specs) if not s["is_decoy"]]
    positive_movers = [i for i in movers if int(specs[i]["steps"]) > 0]
    moddiff_positive = [
        i for i in positive_movers
        if int(specs[i]["steps"]) != min(int(specs[i]["distance"]), int(specs[i]["grey"]))
    ]
    if len(positive_movers) < 2 or len(moddiff_positive) < 2:
        return None

    for s in specs:
        s["region"] = random.randint(0, 1)

    chosen = random.sample(moddiff_positive, 2)
    specs[chosen[0]]["region"] = 0
    specs[chosen[1]]["region"] = 1
    return specs


def _choose_grid(specs: list[dict]) -> tuple[int, int, int] | None:
    top_specs = [s for s in specs if int(s["region"]) == 0]
    bottom_specs = [s for s in specs if int(s["region"]) == 1]
    if not top_specs or not bottom_specs:
        return None

    total_area = sum(int(s["total_h"] * s["total_w"]) for s in specs)
    top_min = max(max(int(s["total_h"]) for s in top_specs) + 1, 6)
    bottom_min = max(max(int(s["total_h"]) for s in bottom_specs) + 1, 6)
    lo_h = top_min + bottom_min
    hi_h = min(26, max(lo_h, int(total_area ** 0.5) + 10))
    if lo_h > 26:
        return None
    grid_h = random.randint(lo_h, hi_h)

    split_lo = top_min
    split_hi = grid_h - bottom_min
    if split_lo > split_hi:
        return None
    split = random.randint(split_lo, split_hi)

    max_w = max(int(s["total_w"]) for s in specs)
    lo_w = max(max_w + 3, int(total_area ** 0.5) + 4)
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
        row_lo = 0 if int(s["region"]) == 0 else split
        row_hi = split if int(s["region"]) == 0 else grid_h
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
        steps = int(s["steps"])
        if steps == 0:
            continue

        oh = int(s["outer_h"])
        ow = int(s["outer_w"])
        hs = int(s["head_side"])
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

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
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
    raise RuntimeError("Failed to generate valid combo variant after 400 attempts")
