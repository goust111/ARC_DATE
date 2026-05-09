# Variant of 271d71e2
# Dimension: D9 — output grid shape (grid_shape)
# Change: same_size — output matches the input canvas with the moved container redrawn
#   -> output only the moved container interiors, packed into a cropped "dashboard" panel.
#
# Design analysis
# ---------------
# Reference task note:
# - The prompt cites a6f40cea as the crop inspiration source.
# - No standalone solver file for a6f40cea exists in the available workspace, so the
#   implementation below only uses its JSON I/O examples as the crop baseline:
#   the output is a tight panel that keeps only the relevant readings and drops the
#   surrounding scene.
#
# a) Concretization and behavior trace
# - Input stays the same as the original task: 2-4 black-bordered containers on a
#   magenta background, each with a maroon head line and a matching maroon destination
#   line. Each interior contains grey/orange cells.
# - Per container, the hidden state update is unchanged:
#   steps = min(head-to-destination distance, grey cell count), and the first `steps`
#   grey cells from the head-facing side become orange.
# - D9 changes only the final rendering step. Instead of redrawing the moved container
#   on the original canvas, we output only the final interior matrix of each container.
# - The final interiors are ordered by the input container positions in row-major
#   order `(top row, then left column)` and stacked vertically with one background row
#   between them. Width is the maximum interior width; shorter readings are left-aligned.
# - Small trace:
#   original: a 3x2 interior under a top head with cells
#     [[7,5],
#      [5,5],
#      [5,5]]
#   if steps = 2, the top-side fill order is row-major from the top:
#     (0,1), (1,0), (1,1), ...
#   final interior becomes
#     [[7,7],
#      [7,5],
#      [5,5]]
#   old task: redraw the whole bordered box at a translated location.
#   new task: keep only the 3x2 matrix above and place it into the packed output panel.
#
# b) Complexity calibration
# - The movement rule, head/destination matching, and fill ordering remain unchanged,
#   so the reasoning load still comes from understanding the original dynamics.
# - The new output is not too simple because the solver must now infer that spatial
#   placement on the large scene is irrelevant; only the post-move interior state is.
# - The packing rule uses a single stable ordering (row-major by input container box),
#   avoiding excess degrees of freedom while remaining visible across 2-4 objects.
#
# c) Implicit-assumption audit of the original solver
# - Original `out = g.copy()` and later erase/paste logic assumes the answer lives on
#   the same canvas. This is invalid under D9 and is replaced by panel packing.
# - Original movement translation (`tr`, `tc`) affects output position only. Under D9
#   we still need the translation distance to compute `steps`, but no longer need to
#   paste the moved box/head onto a full-size scene.
# - Original head/destination detection assumes the scene still contains the full
#   apparatus. That assumption remains valid because the input is unchanged.
# - Original fill ordering logic depended only on the movement direction, not on the
#   full-canvas redraw, so it remains valid and is reused directly on local interior
#   coordinates.
#
# d) Placement legality
# - Generator keeps the original scene layout: assemblies are separated by at least one
#   background cell and fit within the 8-26 input canvas.
# - Because each assembly already includes destination, gap, head, and container in the
#   movement axis, the translated container remains within that reserved assembly strip.
# - The packed output panel is also bounded explicitly: generator rejects samples whose
#   packed height would exceed 26.
#
# e) Visual observability
# - Human-visible difference from the original task is strong: outputs are narrow cropped
#   panels containing only grey/orange readings, with no black borders, no maroon lines,
#   and no full-scene background context.
# - The ordering is visible because multiple readings appear in a vertical list that
#   matches the top-to-bottom / left-to-right order of containers in the input scene.
#
# f) Visual cleanliness
# - A one-row background separator prevents different readings from visually merging.
# - Left alignment plus fixed panel width avoids irregular horizontal staggering.
# - Only the input contains overlapping semantic layers (container/head/destination);
#   the output contains just interior matrices, so the result is visually sparse.
#
# g) Color-change inferability
# - No new color mapping is introduced. Grey->orange conversion remains tied to the
#   original head-facing fill order, which is visible from the input configuration.
# - Output uses only background padding plus the original interior colors, keeping the
#   palette small and structurally explained.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[5]
for extra in (ROOT, ROOT / "solvers"):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Return local interior coords in the same order used by the solver's charge logic."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:  # right, direction (0,1)
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left, direction (0,-1)
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom, direction (1,0)
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # top, direction (-1,0)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _fill_from_direction(grey_cells: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_cells, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_cells, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_cells, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_cells, key=lambda p: (p[0], p[1]))


def _pack_interiors(readings: list[dict], bg: int) -> np.ndarray:
    assert readings
    total_h = sum(int(item["inner"].shape[0]) for item in readings) + (len(readings) - 1)
    max_w = max(int(item["inner"].shape[1]) for item in readings)
    out = np.full((total_h, max_w), int(bg), dtype=int)
    rr = 0
    for idx, item in enumerate(readings):
        inner = item["inner"]
        h, w = inner.shape
        out[rr : rr + h, :w] = inner
        rr += h
        if idx + 1 < len(readings):
            rr += 1
    return out


def _detect_final_interiors(grid: np.ndarray) -> tuple[int, list[dict]]:
    """
    Return background color and final container interiors.

    Each returned item has:
    - box: input container bbox (for ordering)
    - inner: post-move interior matrix (grey/orange only)
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # True background is dominant non-border color (border can dominate in some cases).
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
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1), "inner": inner.copy()})

    assert containers

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

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

    readings = []
    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        final_inner = cont["inner"].copy()

        head_info = pick_head_for_container(box_)
        if head_info is not None and head_info["dir"] != (0, 0):
            hc = int(head_info["color"])
            head_pts = set(head_info["head_pts"])
            head_shape = head_info["head_shape"]
            head_tl = head_info["head_tl"]
            comp_pts = set(head_info["comp_pts"])
            direction = head_info["dir"]

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

            if best_dest is not None:
                tr0 = int(best_dest["tl"][0] - head_tl[0])
                tc0 = int(best_dest["tl"][1] - head_tl[1])
                steps = abs(tr0) + abs(tc0)
                grey_cells = [
                    (rr, cc)
                    for rr in range(final_inner.shape[0])
                    for cc in range(final_inner.shape[1])
                    if int(final_inner[rr, cc]) == int(GREY)
                ]
                steps = min(int(steps), int(len(grey_cells)))
                for rr, cc in _fill_from_direction(grey_cells, direction)[:steps]:
                    final_inner[int(rr), int(cc)] = int(ORANGE)

        readings.append({"box": box_, "inner": final_inner})

    readings.sort(key=lambda item: (int(item["box"][0]), int(item["box"][1])))
    return bg, readings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each container logically as in the original task, but output only the
    final interior states as a vertically packed panel.
    """
    bg, readings = _detect_final_interiors(np.array(grid, dtype=int))
    return _pack_interiors(readings, bg)


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            initial_orange = inner_area
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(0, 10)
            h2d = gap + 1
            steps = min(grey, h2d)

        specs.append(
            {
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
            }
        )

    packed_h = sum(int(s["inner_h"]) for s in specs) + (n_containers - 1)
    packed_w = max(int(s["inner_w"]) for s in specs)
    if packed_h > 26 or packed_w > 26:
        return None

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:  # top: dest above head above container
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:  # bottom: container above head above dest
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:  # left: dest left of head left of container
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:  # right: container left of head left of dest
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area**0.5) + 2)
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
    panel_items = []

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        gap = a["gap"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
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

        final_interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            final_interior[r, c] = ORANGE
        panel_items.append({"box": (cr, cc, cr + oh - 1, cc + ow - 1), "inner": final_interior})

    try:
        output = main(grid)
    except Exception:
        return None

    panel_items.sort(key=lambda item: (int(item["box"][0]), int(item["box"][1])))
    expected = _pack_interiors(panel_items, BG)
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
